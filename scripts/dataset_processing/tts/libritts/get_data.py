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
#
# USAGE: python get_data.py --data-root=<where to put data> --data-set=<datasets_to_download> --num-workers=<number of parallel workers>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data-set=dev_clean,train_clean_100
import argparse
import fnmatch
import functools
import json
import multiprocessing
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download LibriTTS and create manifests')
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--data-sets", default="dev_clean", type=str)
parser.add_argument("--num-workers", default=4, type=int)
args = parser.parse_args()

URLS = {
    'TRAIN_CLEAN_100': "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    'TRAIN_CLEAN_360': "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
    'TRAIN_OTHER_500': "https://www.openslr.org/resources/60/train-other-500.tar.gz",
    'DEV_CLEAN': "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    'DEV_OTHER': "https://www.openslr.org/resources/60/dev-other.tar.gz",
    'TEST_CLEAN': "https://www.openslr.org/resources/60/test-clean.tar.gz",
    'TEST_OTHER': "https://www.openslr.org/resources/60/test-other.tar.gz",
}


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
    entries = []
    with open(file_path, encoding="utf-8") as fin:
        text = fin.readlines()[0].strip()

        # TODO(oktai15): add normalized text via Normalizer/NormalizerWithAudio
        wav_file = file_path.replace(".normalized.txt", ".wav")
        speaker_id = file_path.split('/')[-3]
        assert os.path.exists(wav_file), f"{wav_file} not found!"
        duration = subprocess.check_output(["soxi", "-D", wav_file])
        entry = {
            'audio_filepath': os.path.abspath(wav_file),
            'duration': float(duration),
            'text': text,
            'speaker': int(speaker_id),
        }

        entries.append(entry)

    return entries


def __process_data(data_folder, manifest_file, num_workers):
    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        # we will use normalized text provided by the original dataset
        for filename in fnmatch.filter(filenames, '*.normalized.txt'):
            files.append(os.path.join(root, filename))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript)
        results = p.imap(processing_func, files)
        for result in tqdm(results, total=len(files)):
            entries.extend(result)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    data_sets = args.data_sets
    num_workers = args.num_workers

    if data_sets == "ALL":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other"
    if data_sets == "mini":
        data_sets = "dev_clean,train_clean_100"
    for data_set in data_sets.split(','):
        filepath = data_root / f"{data_set}.tar.gz"
        print(f"Downloading data for {data_set}...")
        __maybe_download_file(URLS[data_set.upper()], filepath)
        print("Extracting...")
        __extract_file(str(filepath), str(data_root))

        print("Processing and building manifest.")
        __process_data(
            str(data_root / "LibriTTS" / data_set.replace("_", "-")),
            str(data_root / "LibriTTS" / f"{data_set}.json"),
            num_workers=num_workers,
        )


if __name__ == "__main__":
    main()
