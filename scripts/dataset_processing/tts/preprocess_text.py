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
This script is used to preprocess text before TTS model training. This is needed mainly for text normalization,
which is slow to rerun during training.

The output manifest will be the same as the input manifest but with final text stored in the 'normalized_text' field.

$ python <nemo_root_path>/scripts/dataset_processing/tts/preprocess_text.py \
    --input_manifest="<data_root_path>/manifest.json" \
    --output_manifest="<data_root_path>/manifest_processed.json" \
    --normalizer_config_path="<nemo_root_path>/examples/tts/conf/text/normalizer_en.yaml" \
    --lower_case \
    --num_workers=4 \
    --joblib_batch_size=16
"""

import argparse
from pathlib import Path

from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Process and normalize text data.",
    )
    parser.add_argument(
        "--input_manifest", required=True, type=Path, help="Path to input training manifest.",
    )
    parser.add_argument(
        "--output_manifest", required=True, type=Path, help="Path to output training manifest with processed text.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite the output manifest file if it exists.",
    )
    parser.add_argument(
        "--text_key", default="text", type=str, help="Input text field to normalize.",
    )
    parser.add_argument(
        "--normalized_text_key", default="normalized_text", type=str, help="Output field to save normalized text to.",
    )
    parser.add_argument(
        "--lower_case", action=argparse.BooleanOptionalAction, help="Whether to convert the final text to lower case.",
    )
    parser.add_argument(
        "--normalizer_config_path",
        required=False,
        type=Path,
        help="Path to config file for nemo_text_processing.text_normalization.normalize.Normalizer.",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of parallel threads to use. If -1 all CPUs are used."
    )
    parser.add_argument(
        "--joblib_batch_size", type=int, help="Batch size for joblib workers. Defaults to 'auto' if not provided."
    )
    parser.add_argument(
        "--max_entries", default=0, type=int, help="If provided, maximum number of entries in the manifest to process."
    )

    args = parser.parse_args()
    return args


def _process_entry(
    entry: dict,
    normalizer: Normalizer,
    text_key: str,
    normalized_text_key: str,
    lower_case: bool,
    lower_case_norm: bool,
) -> dict:
    text = entry[text_key]

    if normalizer is not None:
        if lower_case_norm:
            text = text.lower()
        text = normalizer.normalize(text, punct_pre_process=True, punct_post_process=True)

    if lower_case:
        text = text.lower()

    entry[normalized_text_key] = text

    return entry


def main():
    args = get_args()

    input_manifest_path = args.input_manifest
    output_manifest_path = args.output_manifest
    text_key = args.text_key
    normalized_text_key = args.normalized_text_key
    lower_case = args.lower_case
    num_workers = args.num_workers
    batch_size = args.joblib_batch_size
    max_entries = args.max_entries
    overwrite = args.overwrite

    if output_manifest_path.exists():
        if overwrite:
            print(f"Will overwrite existing manifest path: {output_manifest_path}")
        else:
            raise ValueError(f"Manifest path already exists: {output_manifest_path}")

    if args.normalizer_config_path:
        normalizer_config = OmegaConf.load(args.normalizer_config_path)
        normalizer = instantiate(normalizer_config)
        lower_case_norm = normalizer.input_case == "lower_cased"
    else:
        normalizer = None
        lower_case_norm = False

    entries = read_manifest(input_manifest_path)
    if max_entries:
        entries = entries[:max_entries]

    if not batch_size:
        batch_size = 'auto'

    output_entries = Parallel(n_jobs=num_workers, batch_size=batch_size)(
        delayed(_process_entry)(
            entry=entry,
            normalizer=normalizer,
            text_key=text_key,
            normalized_text_key=normalized_text_key,
            lower_case=lower_case,
            lower_case_norm=lower_case_norm,
        )
        for entry in tqdm(entries)
    )

    write_manifest(output_path=output_manifest_path, target_manifest=output_entries, ensure_ascii=False)


if __name__ == "__main__":
    main()
