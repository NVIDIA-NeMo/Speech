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

# Disclaimer:
#   Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.

import argparse
import json
import os
import random
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from nemo_text_processing.text_normalization.normalize import Normalizer
from opencc import OpenCC

URL = "https://www.openslr.org/resources/93/data_aishell3.tgz"


def get_args():
    parser = argparse.ArgumentParser(
        description='Prepare SF_bilingual dataset and create manifests with predefined split'
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        help="where the dataset will reside",
        default="./DataChinese/sf_bilingual_speech_zh_en_vv1/SF_bilingual/",
    )
    parser.add_argument(
        "--manifests-path", type=Path, help="where the resulting manifests files will reside", default="./"
    )
    parser.add_argument("--val-size", default=0.01, type=float, help="eval set split")
    parser.add_argument("--test-size", default=0.01, type=float, help="test set split")
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=float,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100",
    )

    args = parser.parse_args()
    return args


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        tmp_file_path = destination_path.with_suffix('.tmp')
        urllib.request.urlretrieve(source_url, filename=str(tmp_file_path))
        tmp_file_path.rename(destination_path)


def __extract_file(filepath, data_dir):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_transcript(file_path: str):
    # Create directory for processed wav files
    Path(file_path / "processed").mkdir(parents=True, exist_ok=True)
    # Create zh-TW to zh-simplify converter
    cc = OpenCC('t2s')
    # Create normalizer
    text_normalizer = Normalizer(
        lang="zh", input_case="cased", overwrite_cache=True, cache_dir=str(file_path / "cache_dir"),
    )
    text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}
    normalizer_call = lambda x: text_normalizer.normalize(x, **text_normalizer_call_kwargs)
    entries = []
    i = 0
    SPEAKER_LEN = 7
    with open(file_path / "train" / "content.txt", encoding="utf-8") as fin:
        for line in fin:
            content = line.split()
            wav_name, text = content[0], "".join(content[1::2]) + "。"
            wav_name = wav_name.replace(u'\ufeff', '')
            speaker = wav_name[:SPEAKER_LEN]
            wav_file = file_path / "train" / "wav" / speaker / wav_name
            assert os.path.exists(wav_file), f"{wav_file} not found!"
            duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
            if float(duration) <= 3.0:  # filter out wav files shorter than 3 seconds
                continue
            processed_file = file_path / "processed" / wav_name
            # convert wav to mono 22050HZ, 16 bit (as SFSpeech dataset)
            subprocess.run(f"sox {wav_file} -r 22050 -c 1 -b 16 {processed_file}", shell=True)
            simplified_text = cc.convert(text)
            normalized_text = normalizer_call(simplified_text)
            entry = {
                'audio_filepath': os.path.abspath(processed_file),
                'duration': float(duration),
                'text': text,
                'normalized_text': normalized_text,
                'speaker': int(speaker[3:]),
            }

            i += 1
            entries.append(entry)
    return entries


def __process_data(dataset_path, val_size, test_size, seed_for_ds_split, manifests_dir):
    entries = __process_transcript(dataset_path)

    random.Random(seed_for_ds_split).shuffle(entries)

    train_size = 1.0 - val_size - test_size
    train_entries, validate_entries, test_entries = np.split(
        entries, [int(len(entries) * train_size), int(len(entries) * (train_size + val_size))]
    )

    assert len(train_entries) > 0, "Not enough data for train, val and test"

    def save(p, data):
        with open(p, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save(manifests_dir / "train_manifest.json", train_entries)
    save(manifests_dir / "val_manifest.json", validate_entries)
    save(manifests_dir / "test_manifest.json", test_entries)


def main():
    args = get_args()

    tarred_data_path = args.data_root / "data_aishell3.tgz"

    __maybe_download_file(URL, tarred_data_path)
    __extract_file(str(tarred_data_path), str(args.data_root))

    __process_data(
        args.data_root, args.val_size, args.test_size, args.seed_for_ds_split, args.manifests_path,
    )


if __name__ == "__main__":
    main()
