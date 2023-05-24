# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script is to compute global and speaker-level feature statistics for a given TTS training manifest.

This script should be run after compute_features.py as it loads the precomputed feature data.

$ python <nemo_root_path>/scripts/dataset_processing/tts/compute_feature_stats.py \
    --feature_config_path=<nemo_root_path>/examples/tts/conf/features/feature_22050.yaml
    --manifest_path=<data_root_path>/manifest.json \
    --audio_dir=<data_root_path>/audio \
    --feature_dir=<data_root_path>/features \
    --stats_path=<data_root_path>/feature_stats.json

The output dictionary will contain the feature statistics for every speaker, as well as a "default" entry
with the global statistics.

For example:

{
    "default": {
        "pitch_mean": 100.0,
        "pitch_std": 50.0,
        "energy_mean": 7.5,
        "energy_std": 4.5
    },
    "speaker1": {
        "pitch_mean": 105.0,
        "pitch_std": 45.0,
        "energy_mean": 7.0,
        "energy_std": 5.0
    },
    "speaker2": {
        "pitch_mean": 110.0,
        "pitch_std": 30.0,
        "energy_mean": 5.0,
        "energy_std": 2.5
    }
}

"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS feature statistics.",
    )
    parser.add_argument(
        "--feature_config_path", required=True, type=Path, help="Path to feature config file.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--audio_dir", required=True, type=Path, help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--feature_dir", required=True, type=Path, help="Path to directory where feature data was stored.",
    )
    parser.add_argument(
        "--feature_names", default="pitch,energy", type=str, help="Comma separated list of features to process.",
    )
    parser.add_argument(
        "--mask_field",
        default="voiced_mask",
        type=str,
        help="If provided, stat computation will ignore non-masked frames.",
    )
    parser.add_argument(
        "--stats_path",
        default=Path("feature_stats.json"),
        type=Path,
        help="Path to output JSON file with dataset feature statistics.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite the output stats file if it exists.",
    )

    args = parser.parse_args()
    return args


def _compute_stats(values: List[torch.Tensor]) -> Tuple[float, float]:
    values_tensor = torch.cat(values, dim=0)
    mean = values_tensor.mean().item()
    std = values_tensor.std(dim=0).item()
    return mean, std


def main():
    args = get_args()

    feature_config_path = args.feature_config_path
    manifest_path = args.manifest_path
    audio_dir = args.audio_dir
    feature_dir = args.feature_dir
    feature_name_str = args.feature_names
    mask_field = args.mask_field
    stats_path = args.stats_path
    overwrite = args.overwrite

    if not manifest_path.exists():
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not audio_dir.exists():
        raise ValueError(f"Audio directory {audio_dir} does not exist.")

    if not feature_dir.exists():
        raise ValueError(
            f"Feature directory {feature_dir} does not exist. "
            f"Please check that the path is correct and that you ran compute_features.py"
        )

    if stats_path.exists():
        if overwrite:
            print(f"Will overwrite existing stats path: {stats_path}")
        else:
            raise ValueError(f"Stats path already exists: {stats_path}")

    feature_config = OmegaConf.load(feature_config_path)
    feature_config = instantiate(feature_config)
    featurizer_dict = feature_config.featurizers

    print(f"Found featurizers for {list(featurizer_dict.keys())}.")
    featurizers = featurizer_dict.values()

    feature_names = feature_name_str.split(",")
    # For each feature, we have a dictionary mapping speaker IDs to a list containing all features
    # for that speaker
    feature_stats = {name: defaultdict(list) for name in feature_names}

    entries = read_manifest(manifest_path)

    for entry in tqdm(entries):
        speaker = entry["speaker"]

        entry_dict = {}
        for featurizer in featurizers:
            feature_dict = featurizer.load(manifest_entry=entry, audio_dir=audio_dir, feature_dir=feature_dir)
            entry_dict.update(feature_dict)

        if mask_field:
            mask = entry_dict[mask_field]
        else:
            mask = None

        for feature_name in feature_names:
            values = entry_dict[feature_name]
            if mask is not None:
                values = values[mask]

            feature_stat_dict = feature_stats[feature_name]
            feature_stat_dict["default"].append(values)
            feature_stat_dict[speaker].append(values)

    stat_dict = defaultdict(dict)
    for feature_name in feature_names:
        mean_key = f"{feature_name}_mean"
        std_key = f"{feature_name}_std"
        feature_stat_dict = feature_stats[feature_name]
        for speaker_id, values in feature_stat_dict.items():
            speaker_mean, speaker_std = _compute_stats(values)
            stat_dict[speaker_id][mean_key] = speaker_mean
            stat_dict[speaker_id][std_key] = speaker_std

    with open(stats_path, 'w', encoding="utf-8") as stats_f:
        json.dump(stat_dict, stats_f, indent=4)


if __name__ == "__main__":
    main()
