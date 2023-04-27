# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script is to compute speaker-level statistics, such as pitch mean & standard deviation, for a given
TTS training manifest.

This script should be run after extract_sup_data.py as it uses the precomputed supplemental features.

$ python <nemo_root_path>/scripts/dataset_processing/tts/compute_speaker_stats.py \
    --manifest_path=<data_root_path>/fastpitch_manifest.json \
    --sup_data_path=<data_root_path>/sup_data \
    --pitch_stats_path=<data_root_path>/pitch_stats.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
from nemo.collections.tts.torch.tts_data_types import Pitch
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute speaker level pitch statistics.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--sup_data_path", default=Path("sup_data"), type=Path, help="Path to base directory with supplementary data.",
    )
    parser.add_argument(
        "--pitch_stats_path",
        default=Path("pitch_stats.json"),
        type=Path,
        help="Path to output JSON file with speaker pitch statistics.",
    )
    args = parser.parse_args()
    return args


def _compute_stats(values: List[torch.Tensor]) -> Tuple[float, float]:
    values_tensor = torch.cat(values, dim=0)
    mean = values_tensor.mean().item()
    std = values_tensor.std(dim=0).item()
    return mean, std


def _get_sup_data_filepath(manifest_entry: dict, audio_dir: Path, sup_data_dir: Path) -> Path:
    """
    Get the absolute path of a supplementary data type for the input manifest entry.

    Example: audio_filepath "<audio_dir>/speaker1/audio1.wav" becomes "<sup_data_dir>/speaker1_audio1.pt"

    Args:
        manifest_entry: Manifest entry dictionary.
        audio_dir: base directory where audio is stored.
        sup_data_dir: base directory where supplementary data is stored.

    Returns:
        Path to the supplementary data file.
    """
    audio_path = Path(manifest_entry["audio_filepath"])
    rel_audio_path = audio_path.relative_to(audio_dir)
    rel_sup_data_path = rel_audio_path.with_suffix(".pt")
    sup_data_filename = str(rel_sup_data_path).replace(os.sep, "_")
    sup_data_filepath = sup_data_dir / sup_data_filename
    return sup_data_filepath


def main():
    args = get_args()
    manifest_path = args.manifest_path
    sup_data_path = args.sup_data_path
    pitch_stats_path = args.pitch_stats_path

    pitch_data_path = Path(os.path.join(sup_data_path, Pitch.name))
    if not os.path.exists(pitch_data_path):
        raise ValueError(
            f"Pitch directory {pitch_data_path} does not exist. Make sure 'sup_data_path' is correct "
            f"and that you have computed the pitch using extract_sup_data.py"
        )

    entries = read_manifest(manifest_path)

    audio_paths = [entry["audio_filepath"] for entry in entries]
    base_dir = get_base_dir(audio_paths)

    global_pitch_values = []
    speaker_pitch_values = defaultdict(list)
    for entry in tqdm(entries):
        pitch_path = _get_sup_data_filepath(manifest_entry=entry, audio_dir=base_dir, sup_data_dir=pitch_data_path)
        if not os.path.exists(pitch_path):
            logging.warning(f"Unable to find pitch file for {entry}")
            continue

        pitch = torch.load(pitch_path)
        # Filter out non-speech frames
        pitch = pitch[pitch != 0]
        global_pitch_values.append(pitch)
        if "speaker" in entry:
            speaker_id = entry["speaker"]
            speaker_pitch_values[speaker_id].append(pitch)

    global_pitch_mean, global_pitch_std = _compute_stats(global_pitch_values)
    pitch_stats = {"default": {"pitch_mean": global_pitch_mean, "pitch_std": global_pitch_std}}
    for speaker_id, pitch_values in speaker_pitch_values.items():
        pitch_mean, pitch_std = _compute_stats(pitch_values)
        pitch_stats[speaker_id] = {"pitch_mean": pitch_mean, "pitch_std": pitch_std}

    with open(pitch_stats_path, 'w', encoding="utf-8") as stats_f:
        json.dump(pitch_stats, stats_f, indent=4)


if __name__ == "__main__":
    main()
