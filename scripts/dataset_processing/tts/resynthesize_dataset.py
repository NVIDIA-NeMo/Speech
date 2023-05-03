# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
This script is a helper for resynthesizing TTS dataset using a pretrained text-to-spectrogram model.
Goal of resynthesis (as opposed to text-to-speech) is to use the largest amount of ground-truth features from existing speech data.
For example, for resynthesis we want to have the same pitch and durations instead of ones predicted by the model.
The results are to be used for some other task: vocoder finetuning, spectrogram enhancer training, etc.

Let's say we have the following toy dataset:
/dataset/manifest.json
/dataset/1/foo.wav
/dataset/2/bar.wav
/dataset/sup_data/pitch/1_foo.pt
/dataset/sup_data/pitch/2_bar.pt

manifest.json has two entries for "/dataset/1/foo.wav" and "/dataset/2/bar.wav"
(sup_data folder contains pitch files precomputed during training a FastPitch model on this dataset.)
(If you lost your sup_data - don't worry, we use TTSDataset class so they would be created on-the-fly)

Our script call is
$ python scripts/dataset_processing/tts/resynthesize_dataset.py \
    --model-path ./models/fastpitch/multi_spk/FastPitch--val_loss\=1.4473-epoch\=209.ckpt \
    --input-json-manifest "/dataset/manifest.json" \
    --input-sup-data-path "/dataset/sup_data/" \
    --output-folder "/output/" \
    --device "cuda:0" \
    --batch-size 1 \
    --num-workers 1

Then we get output dataset with following directory structure:
/output/manifest_mel.json
/output/mels/foo.npy
/output/mels/foo_gt.npy
/output/mels/bar.npy
/output/mels/bar_gt.npy

/output/manifest_mel.json has the same entries as /dataset/manifest.json but with new fields for spectrograms.
"mel_filepath" is path to the resynthesized spectrogram .npy, "mel_gt_filepath" is path to ground-truth spectrogram .npy

The output structure is similar to generate_mels.py script for compatibility reasons.
"""

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.parts.utils.helpers import process_batch, to_device_recursive


def chunks(iterable: Iterable, size: int) -> Iterator[List]:
    # chunks([1, 2, 3, 4, 5], size=2) -> [[1, 2], [3, 4], [5]]
    # assumes iterable does not have any `None`s
    args = [iter(iterable)] * size
    for chunk in itertools.zip_longest(*args, fillvalue=None):
        chunk = list(item for item in chunk if item is not None)
        if chunk:
            yield chunk


def load_model(path: Path, device: torch.device) -> SpectrogramGenerator:
    model = None
    if path.suffix == ".nemo":
        model = SpectrogramGenerator.restore_from(path, map_location=device)
    elif path.suffix == ".ckpt":
        model = SpectrogramGenerator.load_from_checkpoint(path, map_location=device)
    else:
        raise ValueError(f"Unknown checkpoint type {path.suffix} ({path})")

    return model.eval().to(device)


@dataclass
class TTSDatasetResynthesizer:
    """
    Reuses internals of a SpectrogramGenerator to resynthesize dataset using ground truth features.
    Default setup is FastPitch with learned alignment.
    If your use case requires different setup, you can either contribute to this script or subclass this class.
    """

    model: SpectrogramGenerator
    device: torch.device

    @torch.no_grad()
    def resynthesize_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resynthesizes a single batch.
        Takes a dict with main data and sup data.
        Outputs a dict with model outputs.
        """
        if not isinstance(self.model, FastPitchModel):
            raise NotImplementedError(
                "This script supports only FastPitch. Please implement resynthesizing routine for your desired model."
            )

        batch = to_device_recursive(batch, self.device)

        mels, mel_lens = self.model.preprocessor(input_signal=batch["audio"], length=batch["audio_lens"])

        reference_audio = batch.get("reference_audio", None)
        reference_audio_len = batch.get("reference_audio_lens", None)
        reference_spec, reference_spec_len = None, None
        if reference_audio is not None:
            reference_spec, reference_spec_len = self.model.preprocessor(
                input_signal=reference_audio, length=reference_audio_len
            )

        outputs_tuple = self.model.forward(
            text=batch["text"],
            durs=None,
            pitch=batch["pitch"],
            speaker=batch.get("speaker"),
            pace=1.0,
            spec=mels,
            attn_prior=batch.get("attn_prior"),
            mel_lens=mel_lens,
            input_lens=batch["text_lens"],
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_len,
        )
        names = self.model.fastpitch.output_types.keys()
        return {"spec": mels, "mel_lens": mel_lens, **dict(zip(names, outputs_tuple))}

    def resynthesized_batches(self) -> Iterator[Dict[str, Any]]:
        """
        Returns a generator of resynthesized batches.
        Each returned batch is a dict containing main data, sup data, and model output
        """
        self.model.setup_training_data(self.model._cfg["train_ds"])

        for batch_tuple in iter(self.model._train_dl):
            batch = process_batch(batch_tuple, sup_data_types_set=self.model._train_dl.dataset.sup_data_types)
            yield self.resynthesize_batch(batch)


def prepare_paired_mel_spectrograms(
    model_path: Path,
    input_json_manifest: Path,
    input_sup_data_path: Path,
    output_folder: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    model = load_model(model_path, device)

    dataset_config_overrides = {
        "dataset": {
            "manifest_filepath": str(input_json_manifest.absolute()),
            "sup_data_path": str(input_sup_data_path.absolute()),
        },
        "dataloader_params": {"batch_size": batch_size, "num_workers": num_workers, "shuffle": False},
    }
    model._cfg.train_ds = OmegaConf.merge(model._cfg.train_ds, DictConfig(dataset_config_overrides))
    resynthesizer = TTSDatasetResynthesizer(model, device)

    input_manifest = read_manifest(input_json_manifest)

    output_manifest = []
    output_json_manifest = output_folder / f"{input_json_manifest.stem}_mel{input_json_manifest.suffix}"
    output_mels_folder = output_folder / "mels"
    output_mels_folder.mkdir(exist_ok=True, parents=True)
    for batch, batch_manifest in tqdm(
        zip(resynthesizer.resynthesized_batches(), chunks(input_manifest, size=batch_size)), desc="Batch #"
    ):
        pred_mels = batch["spect"].cpu()  # key from fastpitch.output_types
        true_mels = batch["spec"].cpu()  # key from code above
        mel_lens = batch["mel_lens"].cpu().flatten()  # key from code above

        for i, (manifest_entry, length) in enumerate(zip(batch_manifest, mel_lens.tolist())):
            print(manifest_entry["audio_filepath"])
            filename = Path(manifest_entry["audio_filepath"]).stem

            # note that lengths match
            pred_mel = pred_mels[i, :, :length].clone().numpy()
            true_mel = true_mels[i, :, :length].clone().numpy()

            pred_mel_path = output_mels_folder / f"{filename}.npy"
            true_mel_path = output_mels_folder / f"{filename}_gt.npy"

            np.save(pred_mel_path, pred_mel)
            np.save(true_mel_path, true_mel)

            new_manifest_entry = {
                **manifest_entry,
                "mel_filepath": str(pred_mel_path),
                "mel_gt_filepath": str(true_mel_path),
            }
            output_manifest.append(new_manifest_entry)

    write_manifest(output_json_manifest, output_manifest, ensure_ascii=False)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Resynthesize TTS dataset using a pretrained text-to-spectrogram model",
    )
    parser.add_argument(
        "--model-path", required=True, type=Path, help="Path to a checkpoint (either .nemo or .ckpt)",
    )
    parser.add_argument(
        "--input-json-manifest", required=True, type=Path, help="Path to the input JSON manifest",
    )
    parser.add_argument(
        "--input-sup-data-path", required=True, type=Path, help="sup_data_path for the JSON manifest",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=Path,
        help="Path to the output folder. Will contain updated manifest and mels/ folder with spectrograms in .npy files",
    )
    parser.add_argument("--device", required=True, type=torch.device, help="Device ('cpu', 'cuda:0', ...)")
    parser.add_argument("--batch-size", required=True, type=int, help="Batch size in the DataLoader")
    parser.add_argument("--num-workers", required=True, type=int, help="Num workers in the DataLoader")
    return parser


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    prepare_paired_mel_spectrograms(**vars(arguments))
