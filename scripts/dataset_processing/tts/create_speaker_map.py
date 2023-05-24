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
This script takes a list of TTS manifests and creates a JSON mapping the input speaker names to
unique indices for multi-speaker TTS training.

To ensure that speaker names are unique across datasets, it is recommended that you prepend the speaker
names in your manifest with the name of the dataset.

$ python <nemo_root_path>/scripts/dataset_processing/tts/create_speaker_map.py \
    --manifest_path=manifest1.json \
    --manifest_path=manifest2.json \
    --speaker_map_path=speakers.json

Example output:

{
    "vctk_p225": 0,
    "vctk_p226": 1,
    "vctk_p227": 2,
    ...
}

"""

import argparse
import json
from pathlib import Path

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create mapping from speaker names to numerical speaker indices.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, action="append", help="Path to training manifest(s).",
    )
    parser.add_argument(
        "--speaker_map_path", required=True, type=Path, help="Path for output speaker index JSON",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite the output speaker file if it exists.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    manifest_paths = args.manifest_path
    speaker_map_path = args.speaker_map_path
    overwrite = args.overwrite

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise ValueError(f"Manifest {manifest_path} does not exist.")

    if speaker_map_path.exists():
        if overwrite:
            print(f"Will overwrite existing speaker path: {speaker_map_path}")
        else:
            raise ValueError(f"Speaker path already exists: {speaker_map_path}")

    speaker_set = set()
    for manifest_path in manifest_paths:
        entries = read_manifest(manifest_path)
        for entry in entries:
            speaker = str(entry["speaker"])
            speaker_set.add(speaker)

    speaker_list = list(speaker_set)
    speaker_list.sort()
    speaker_index_map = {speaker_list[i]: i for i in range(len(speaker_list))}

    with open(speaker_map_path, 'w', encoding="utf-8") as stats_f:
        json.dump(speaker_index_map, stats_f, indent=4)


if __name__ == "__main__":
    main()
