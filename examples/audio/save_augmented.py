# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from itertools import islice
from pathlib import Path

import hydra
import lhotse
import numpy as np
import soundfile as sf
from lhotse import CutSet, MonoCut, Recording
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nemo.collections.audio.data.audio_to_audio_lhotse import LhotseAudioToTargetDataset
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig, get_lhotse_dataloader_from_config

"""
The purpose of this script is to save online-augmented data as provided by NeMo Lhotse dataloader.
The script piggybacks on a train_ds section of an existing model configuration file.

Intended use cases are: 1) preparing a validation set, 2) debugging.

Usage example:
$ python examples/audio/save_augmented.py \
    +input_cuts=some_path/cuts.jsonl \
    +output_cuts=some_other_path/cuts.gsm_and_clipping_augmented.jsonl \
    +keep_directory_structure=true \
    model.sample_rate=48000 \
    ++model.train_ds.rir_enabled=true \
    ++model.train_ds.rir_path=path/to/rir_manifest.jsonl

Assumptions:
- input data are described as a Lhotse CutSet in a JSONL file
   - consists of simple MonoCuts with Recording paths relative to the Cuts manifest
- the parent directory of the output cuts must exist

Requires additional config parameters `input_cuts` and `output_cuts`.
Produces:
- %output_cuts_parent_dir%/audio/
- %output_cuts_parent_dir%/%output_cuts_filename%.jsonl
where the audio folder contains the augmented and clean signals, respectively, with `.input.flac` and `.output.flac` suffixes.

If `keep_directory_structure` provided and is True, the script will preserve the directory structure of the input cuts.

Text is preserved from the input cuts if possible. 

Optional config parameter `num_samples` can be used to limit the number of samples to save (but not more than input dataloader size). 
If not specified, the dataloader is used until exhausted.
"""


def check_input_cuts(input_cuts_path: Path) -> None:
    """Validate that input cuts are well-formed MonoCuts with relative recording paths that exist on disk."""
    assert input_cuts_path.exists(), "input_cuts must exist"
    assert input_cuts_path.suffix == '.jsonl', "input_cuts must be a .jsonl file"
    assert input_cuts_path.parent.exists(), "input_cuts parent directory must exist"
    cuts = lhotse.CutSet.from_file(input_cuts_path)
    for i, cut in enumerate(cuts):
        assert isinstance(cut, MonoCut), f"{i}th cut is a {type(cut)}, not a MonoCut"
        assert len(cut.recording.sources) == 1, f"{i}th cut has {len(cut.recording.sources)} sources"
        assert cut.recording.sources[0].source is not None, f"{i}th cut has no audio source specified"

        recording_path = Path(cut.recording.sources[0].source)
        assert not recording_path.is_absolute(), f"{i}th cut's recording source is an absolute path: {recording_path}"

        recording_path_full = input_cuts_path.parent / recording_path
        assert recording_path_full.exists(), f"{i}th cut's recording source file does not exist: {recording_path_full}"


@hydra.main(config_path="conf", config_name="flow_matching_generative_finetuning.yaml")
def main(cfg: DictConfig):
    assert (
        cfg.get("input_cuts", None) is not None
    ), "input_cuts is required, please override (for example, +input_cuts=some_path/cuts.jsonl)"
    assert (
        cfg.get("output_cuts", None) is not None
    ), "output_cuts is required, please override (for example, +output_cuts=some_path/cuts.augmented.jsonl)"
    num_samples = cfg.get("num_samples", None)
    sample_rate = cfg.model.sample_rate
    keep_directory_structure = cfg.get("keep_directory_structure", False)

    input_cuts_path = Path(cfg.input_cuts)
    output_cuts_path = Path(cfg.output_cuts)
    check_input_cuts(input_cuts_path)  # throws an exception if they aren't ok

    assert output_cuts_path.parent.exists(), f"output_cuts parent directory must exist: {output_cuts_path.parent}"

    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "model.train_ds.cuts_path", str(input_cuts_path), force_add=True)
    OmegaConf.update(cfg, "model.train_ds.shuffle", False)  # ensure deterministic behavior
    OmegaConf.update(cfg, "model.train_ds.batch_size", 1)
    OmegaConf.update(cfg, "model.train_ds.shard_seed", 0, force_add=True)  # ensure deterministic behavior
    if cfg.model.train_ds.get("sample_rate", None) != sample_rate:
        OmegaConf.update(cfg, "model.train_ds.sample_rate", sample_rate, force_add=True)

    # Disable bucketing to preserve original cut ordering (DynamicBucketingSampler reorders by duration).
    # Also clear bucket params that would cause _auto_detect_bucketing_and_validate_batch_size to re-enable it.
    OmegaConf.update(cfg, "model.train_ds.use_bucketing", False, force_add=True)
    _defaults = LhotseDataLoadingConfig()
    for key in ("bucket_batch_size", "bucket_duration_bins"):
        OmegaConf.update(cfg, f"model.train_ds.{key}", getattr(_defaults, key), force_add=True)

    # Reset all filters to pass-through defaults — we want a 1:1 mapping from input to output cuts,
    # so no cuts should be silently dropped by model-config filter settings.
    for key in (
        "min_duration",
        "max_duration",
        "min_tps",
        "max_tps",
        "min_tokens",
        "max_tokens",
        "max_cer",
        "min_context_speaker_similarity",
    ):
        OmegaConf.update(cfg, f"model.train_ds.{key}", getattr(_defaults, key), force_add=True)

    dataloader = get_lhotse_dataloader_from_config(
        OmegaConf.create(cfg.model.train_ds), global_rank=0, world_size=1, dataset=LhotseAudioToTargetDataset()
    )

    cuts = lhotse.CutSet.from_file(input_cuts_path)
    if num_samples is None:
        num_samples = len(cuts)

    with CutSet.open_writer(output_cuts_path) as writer:
        for i, (sample, original_cut) in enumerate(
            tqdm(zip(islice(dataloader, num_samples), cuts), total=num_samples)
        ):
            # batch_size is 1, so we can access the first element
            input_audio = sample['input_signal'][0].numpy()
            output_audio = sample['target_signal'][0].numpy()

            # if necessary, apply negative gain to avoid clipping
            if (coeff := max(np.max(np.abs(input_audio)), np.max(np.abs(output_audio)))) > 1.0:
                input_audio = input_audio / coeff
                output_audio = output_audio / coeff

            if keep_directory_structure:
                # definitely a relative path because we checked for that earlier
                input_relative_path = Path(original_cut.recording.sources[0].source)

                input_path = output_cuts_path.parent / input_relative_path.with_suffix('.input.flac')
                output_path = output_cuts_path.parent / input_relative_path.with_suffix('.output.flac')

                # we know that `audio_dir` exists, but we need to create the parent directories
                input_path.parent.mkdir(exist_ok=True, parents=True)
                output_path.parent.mkdir(exist_ok=True, parents=True)
            else:
                (output_cuts_path.parent / 'audio').mkdir(exist_ok=True, parents=True)
                input_path = output_cuts_path.parent / 'audio' / f"{i:06}.input.flac"
                output_path = output_cuts_path.parent / 'audio' / f"{i:06}.output.flac"

            sf.write(input_path, input_audio, sample_rate, format='FLAC', subtype='PCM_24')
            sf.write(output_path, output_audio, sample_rate, format='FLAC', subtype='PCM_24')

            input_recording = Recording.from_file(input_path)
            input_recording.sources[0].source = str(input_path.relative_to(output_cuts_path.parent))
            output_recording = Recording.from_file(output_path)
            output_recording.sources[0].source = str(output_path.relative_to(output_cuts_path.parent))

            cut = MonoCut(
                id=input_recording.id, start=0, channel=0, duration=input_recording.duration, recording=input_recording
            )
            cut.target_recording = output_recording

            for optional_field_name in (
                'text',
                'original_text',
                'language',
            ):
                if (
                    hasattr(original_cut, optional_field_name)
                    and getattr(original_cut, optional_field_name) is not None
                ):
                    setattr(cut, optional_field_name, getattr(original_cut, optional_field_name))

            writer.write(cut)


if __name__ == "__main__":
    main()
