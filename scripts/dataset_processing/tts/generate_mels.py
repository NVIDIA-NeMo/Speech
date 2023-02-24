# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is to generate mel spectrograms from a Fastpitch model checkpoint. Please see general usage below. It runs
on GPUs by default, but you can add `--num-workers 5 --cpu` as an option to run on CPUs.

$ python scripts/dataset_processing/tts/generate_mels.py \
    --fastpitch-model-ckpt ./models/fastpitch/multi_spk/FastPitch--val_loss\=1.4473-epoch\=209.ckpt \
    --input-json-manifests /home/xueyang/HUI-Audio-Corpus-German-clean/test_manifest_text_normed_phonemes.json
    --output-json-manifest-root /home/xueyang/experiments/multi_spk_tts_de
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    BetaBinomialInterpolator,
    beta_binomial_prior_distribution,
)
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate mel spectrograms with pretrained FastPitch model, and create manifests for finetuning Hifigan.",
    )
    parser.add_argument(
        "--fastpitch-model-ckpt",
        required=True,
        type=Path,
        help="Specify a full path of a fastpitch model checkpoint with the suffix of either .ckpt or .nemo.",
    )
    parser.add_argument(
        "--input-json-manifests",
        nargs="+",
        required=True,
        type=Path,
        help="Specify a full path of a JSON manifest. You could add multiple manifests.",
    )
    parser.add_argument(
        "--output-json-manifest-root",
        required=True,
        type=Path,
        help="Specify a full path of output root that would contain new manifests.",
    )
    parser.add_argument(
        "--num-workers",
        default=-1,
        type=int,
        help="Specify the max number of concurrently Python workers processes. "
        "If -1 all CPUs are used. If 1 no parallel computing is used.",
    )
    parser.add_argument("--cpu", action='store_true', default=False, help="Generate mel spectrograms using CPUs.")
    args = parser.parse_args()
    return args


def __load_wav(audio_file):
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype='float32')
    return samples.transpose()


def __generate_mels(entry, spec_model, device, use_beta_binomial_interpolator, mel_root):
    # Generate a spectrograms (we need to use ground truth alignment for correct matching between audio and mels)
    audio = __load_wav(entry["audio_filepath"])
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)
    audio_len = torch.tensor(audio.shape[1], dtype=torch.long, device=device).unsqueeze(0)

    if spec_model.fastpitch.speaker_emb is not None and "speaker" in entry:
        speaker = torch.tensor([entry['speaker']]).to(device)
    else:
        speaker = None

    with torch.no_grad():
        if "normalized_text" in entry:
            text = spec_model.parse(entry["normalized_text"], normalize=False)
        else:
            text = spec_model.parse(entry['text'])

        text_len = torch.tensor(text.shape[-1], dtype=torch.long, device=device).unsqueeze(0)
        spect, spect_len = spec_model.preprocessor(input_signal=audio, length=audio_len)

        # Generate attention prior and spectrogram inputs for HiFi-GAN
        if use_beta_binomial_interpolator:
            beta_binomial_interpolator = BetaBinomialInterpolator()
            attn_prior = (
                torch.from_numpy(beta_binomial_interpolator(spect_len.item(), text_len.item()))
                .unsqueeze(0)
                .to(text.device)
            )
        else:
            attn_prior = (
                torch.from_numpy(beta_binomial_prior_distribution(text_len.item(), spect_len.item()))
                .unsqueeze(0)
                .to(text.device)
            )

        spectrogram = spec_model.forward(
            text=text, input_lens=text_len, spec=spect, mel_lens=spect_len, attn_prior=attn_prior, speaker=speaker,
        )[0]

        save_path = mel_root / f"{Path(entry['audio_filepath']).stem}.npy"
        np.save(save_path, spectrogram[0].to('cpu').numpy())
        entry["mel_filepath"] = str(save_path)

    return entry


def main():
    args = get_args()
    ckpt_path = args.fastpitch_model_ckpt
    input_manifest_filepaths = args.input_json_manifests
    output_json_manifest_root = args.output_json_manifest_root

    mel_root = output_json_manifest_root / "mels"
    mel_root.mkdir(exist_ok=True, parents=True)

    # load pretrained FastPitch model checkpoint
    suffix = ckpt_path.suffix
    if suffix == ".nemo":
        spec_model = FastPitchModel.restore_from(ckpt_path).eval()
    elif suffix == ".ckpt":
        spec_model = FastPitchModel.load_from_checkpoint(ckpt_path).eval()
    else:
        raise ValueError(f"Unsupported suffix: {suffix}")
    if not args.cpu:
        spec_model.cuda()
    device = spec_model.device

    use_beta_binomial_interpolator = spec_model.cfg.train_ds.dataset.get("use_beta_binomial_interpolator", False)

    for manifest in input_manifest_filepaths:
        logging.info(f"Processing {manifest}.")
        entries = []
        with open(manifest, "r") as fjson:
            for line in fjson:
                entries.append(json.loads(line.strip()))

        if device == "cpu":
            new_entries = Parallel(n_jobs=args.num_workers)(
                delayed(__generate_mels)(entry, spec_model, device, use_beta_binomial_interpolator, mel_root)
                for entry in entries
            )
        else:
            new_entries = []
            for entry in tqdm(entries):
                new_entry = __generate_mels(entry, spec_model, device, use_beta_binomial_interpolator, mel_root)
                new_entries.append(new_entry)

        mel_manifest_path = output_json_manifest_root / f"{manifest.stem}_mel{manifest.suffix}"
        with open(mel_manifest_path, "w") as fmel:
            for entry in new_entries:
                fmel.write(json.dumps(entry) + "\n")
        logging.info(f"Processing {manifest} is complete --> {mel_manifest_path}")


if __name__ == "__main__":
    main()
