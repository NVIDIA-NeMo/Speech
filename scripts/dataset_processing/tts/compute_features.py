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
This script computes features for TTS models prior to training, such as pitch and energy.
The resulting features will be stored in the provided 'feature_dir'.

$ python <nemo_root_path>/scripts/dataset_processing/tts/compute_features.py \
    --feature_config_path=<nemo_root_path>/examples/tts/conf/features/feature_22050.yaml \
    --manifest_path=<data_root_path>/manifest.json \
    --audio_dir=<data_root_path>/audio \
    --feature_dir=<data_root_path>/features \
    --overwrite \
    --num_workers=1
"""

import argparse
from pathlib import Path

from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS features.",
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
        "--feature_dir", required=True, type=Path, help="Path to directory where feature data will be stored.",
    )
    parser.add_argument(
        "--dedupe_files",
        action=argparse.BooleanOptionalAction,
        help="If given, will only process the first manifest entry found for each audio file.",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, help="Whether to overwrite existing feature files.",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of parallel threads to use. If -1 all CPUs are used."
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    feature_config_path = args.feature_config_path
    manifest_path = args.manifest_path
    audio_dir = args.audio_dir
    feature_dir = args.feature_dir
    dedupe_files = args.dedupe_files
    overwrite = args.overwrite
    num_workers = args.num_workers

    if not manifest_path.exists():
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not audio_dir.exists():
        raise ValueError(f"Audio directory {audio_dir} does not exist.")

    feature_config = OmegaConf.load(feature_config_path)
    feature_config = instantiate(feature_config)
    featurizers = feature_config.featurizers

    entries = read_manifest(manifest_path)

    if dedupe_files:
        final_entries = []
        audio_filepath_set = set()
        for entry in entries:
            audio_filepath = entry["audio_filepath"]
            if audio_filepath in audio_filepath_set:
                continue
            final_entries.append(entry)
            audio_filepath_set.add(audio_filepath)
        entries = final_entries

    for feature_name, featurizer in featurizers.items():
        print(f"Computing: {feature_name}")
        Parallel(n_jobs=num_workers)(
            delayed(featurizer.save)(
                manifest_entry=entry, audio_dir=audio_dir, feature_dir=feature_dir, overwrite=overwrite
            )
            for entry in tqdm(entries)
        )


if __name__ == "__main__":
    main()
