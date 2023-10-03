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
This script is used to generate JSON manifests for mel-generator model training. The usage is below.

$ python scripts/dataset_processing/tts/thorsten_neutral/get_data.py \
    --data-root ~/experiments/thorsten_neutral \
    --manifests-root ~/experiments/thorsten_neutral \
    --data-version "22_10" \
    --min-duration 0.1 \
    --normalize-text
"""

import argparse
import json
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

from nemo.utils import logging

# Thorsten Müller published two neural voice datasets, 21.02 and 22.10.
THORSTEN_NEUTRAL = {
    "21_02": {
        "url": "https://zenodo.org/record/5525342/files/thorsten-neutral_v03.tgz?download=1",
        "dir_name": "thorsten-de_v03",
        "metadata": ["metadata.csv"],
    },
    "22_10": {
        "url": "https://zenodo.org/record/7265581/files/ThorstenVoice-Dataset_2022.10.zip?download=1",
        "dir_name": "ThorstenVoice-Dataset_2022.10",
        "metadata": ["metadata_train.csv", "metadata_dev.csv", "metadata_test.csv"],
    },
}


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download Thorsten Müller's neutral voice dataset and create manifests with predefined split. "
        "Thorsten Müller published two neural voice datasets, 21.02 and 22.10, where 22.10 provides better "
        "audio quality. Please choose one of the two for your TTS models. Details about the dataset are "
        "in https://github.com/thorstenMueller/Thorsten-Voice.",
    )
    parser.add_argument("--data-root", required=True, type=Path, help="where the resulting dataset will reside.")
    parser.add_argument("--manifests-root", required=True, type=Path, help="where the manifests files will reside.")
    parser.add_argument("--data-version", default="22_10", choices=["21_02", "22_10"], type=str)
    parser.add_argument("--min-duration", default=0.1, type=float)
    parser.add_argument("--max-duration", default=float('inf'), type=float)
    parser.add_argument("--val-size", default=100, type=int)
    parser.add_argument("--test-size", default=100, type=int)
    parser.add_argument(
        "--num-workers",
        default=-1,
        type=int,
        help="Specify the max number of concurrent Python worker processes. "
        "If -1 all CPUs are used. If 1 no parallel computing is used.",
    )
    parser.add_argument(
        "--normalize-text",
        default=False,
        action='store_true',
        help="Normalize original text and add a new entry 'normalized_text' to .json file if True.",
    )
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=float,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100.",
    )
    args = parser.parse_args()
    return args


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        logging.info(f"Downloading data: {source_url} --> {destination_path}")
        tmp_file_path = destination_path.with_suffix(".tmp")
        urllib.request.urlretrieve(source_url, filename=tmp_file_path)
        tmp_file_path.rename(destination_path)
    else:
        logging.info(f"Skipped downloading data because it exists: {destination_path}")


def __extract_file(filepath, data_dir):
    logging.info(f"Unzipping data: {filepath} --> {data_dir}")
    shutil.unpack_archive(filepath, data_dir)
    logging.info(f"Unzipping data is complete: {filepath}.")


def __save_json(json_file, dict_list):
    logging.info(f"Saving JSON split to {json_file}.")
    with open(json_file, "w") as f:
        for d in dict_list:
            f.write(json.dumps(d) + "\n")


def __text_normalization(json_file, num_workers=-1):
    text_normalizer_call_kwargs = {
        "punct_pre_process": True,
        "punct_post_process": True,
    }
    text_normalizer = Normalizer(
        lang="de", input_case="cased", overwrite_cache=True, cache_dir=str(json_file.parent / "cache_dir"),
    )

    def normalizer_call(x):
        return text_normalizer.normalize(x, **text_normalizer_call_kwargs)

    def add_normalized_text(line_dict):
        normalized_text = normalizer_call(line_dict["text"])
        line_dict.update({"normalized_text": normalized_text})
        return line_dict

    logging.info(f"Normalizing text for {json_file}.")
    with open(json_file, 'r', encoding='utf-8') as fjson:
        lines = fjson.readlines()
        # Note: you need to verify which backend works well on your cluster.
        # backend="loky" is fine on multi-core Ubuntu OS; backend="threading" on Slurm.
        dict_list = Parallel(n_jobs=num_workers)(
            delayed(add_normalized_text)(json.loads(line)) for line in tqdm(lines)
        )

    json_file_text_normed = json_file.parent / f"{json_file.stem}_text_normed{json_file.suffix}"
    with open(json_file_text_normed, 'w', encoding="utf-8") as fjson_norm:
        for dct in dict_list:
            fjson_norm.write(json.dumps(dct) + "\n")
    logging.info(f"Normalizing text is complete: {json_file} --> {json_file_text_normed}")


def __process_data(
    unzipped_dataset_path, metadata, min_duration, max_duration, val_size, test_size, seed_for_ds_split
):
    logging.info("Preparing JSON train/val/test splits.")

    entries = list()
    not_found_wavs = list()
    wrong_duration_wavs = list()

    for metadata_fname in metadata:
        meta_file = unzipped_dataset_path / metadata_fname
        with open(meta_file, 'r') as fmeta:
            for line in tqdm(fmeta):
                items = line.strip().split('|')
                wav_file_stem, text = items[0], items[1]
                wav_file = unzipped_dataset_path / "wavs" / f"{wav_file_stem}.wav"

                # skip audios if they do not exist.
                if not wav_file.exists():
                    not_found_wavs.append(wav_file)
                    logging.warning(f"Skipping {wav_file}: it is not found.")
                    continue

                # skip audios if their duration is out of range.
                duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
                duration = float(duration)
                if min_duration <= duration <= max_duration:
                    entry = {
                        'audio_filepath': str(wav_file),
                        'duration': duration,
                        'text': text,
                    }
                    entries.append(entry)
                elif duration < min_duration:
                    wrong_duration_wavs.append(wav_file)
                    logging.warning(f"Skipping {wav_file}: it is too short, less than {min_duration} seconds.")
                    continue
                else:
                    wrong_duration_wavs.append(wav_file)
                    logging.warning(f"Skipping {wav_file}: it is too long, greater than {max_duration} seconds.")
                    continue

    random.Random(seed_for_ds_split).shuffle(entries)
    train_size = len(entries) - val_size - test_size
    if train_size <= 0:
        raise ValueError("Not enough data for the train split.")

    logging.info("Preparing JSON train/val/test splits is complete.")
    train, val, test = (
        entries[:train_size],
        entries[train_size : train_size + val_size],
        entries[train_size + val_size :],
    )

    return train, val, test, not_found_wavs, wrong_duration_wavs


def main():
    args = get_args()
    data_root = args.data_root
    manifests_root = args.manifests_root
    data_version = args.data_version

    dataset_root = data_root / f"ThorstenVoice-Dataset-{data_version}"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # download and extract dataset
    dataset_url = THORSTEN_NEUTRAL[data_version]["url"]
    zipped_dataset_path = dataset_root / Path(dataset_url).name.split("?")[0]
    __maybe_download_file(dataset_url, zipped_dataset_path)
    __extract_file(zipped_dataset_path, dataset_root)

    # generate train/dev/test splits
    unzipped_dataset_path = dataset_root / THORSTEN_NEUTRAL[data_version]["dir_name"]
    entries_train, entries_val, entries_test, not_found_wavs, wrong_duration_wavs = __process_data(
        unzipped_dataset_path=unzipped_dataset_path,
        metadata=THORSTEN_NEUTRAL[data_version]["metadata"],
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        val_size=args.val_size,
        test_size=args.test_size,
        seed_for_ds_split=args.seed_for_ds_split,
    )

    # save json splits.
    train_json = manifests_root / "train_manifest.json"
    val_json = manifests_root / "val_manifest.json"
    test_json = manifests_root / "test_manifest.json"
    __save_json(train_json, entries_train)
    __save_json(val_json, entries_val)
    __save_json(test_json, entries_test)

    # save skipped audios that are not found into a file.
    if len(not_found_wavs) > 0:
        skipped_not_found_file = manifests_root / "skipped_not_found_wavs.list"
        with open(skipped_not_found_file, "w") as f_notfound:
            for line in not_found_wavs:
                f_notfound.write(f"{line}\n")

    # save skipped audios that are too short or too long into a file.
    if len(wrong_duration_wavs) > 0:
        skipped_wrong_duration_file = manifests_root / "skipped_wrong_duration_wavs.list"
        with open(skipped_wrong_duration_file, "w") as f_wrong_dur:
            for line in wrong_duration_wavs:
                f_wrong_dur.write(f"{line}\n")

    # normalize text if requested. New json file, train_manifest_text_normed.json, will be generated.
    if args.normalize_text:
        __text_normalization(train_json, args.num_workers)
        __text_normalization(val_json, args.num_workers)
        __text_normalization(test_json, args.num_workers)


if __name__ == "__main__":
    main()
