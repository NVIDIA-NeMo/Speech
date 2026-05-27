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

# downloads the training/eval set for AISHELL Diarization.
# the training dataset is around 170GiB, to skip pass the --skip_train flag.

import argparse
import glob
import logging
import os
import tarfile
from pathlib import Path

import wget
from sox import Transformer

from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

train_url = "https://www.openslr.org/resources/111/train_{}.tar.gz"
train_datasets = ["S", "M", "L"]

eval_url = "https://www.openslr.org/resources/111/test.tar.gz"


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def __process_data(dataset_url: str, dataset_path: Path, manifest_output_path: Path):
    os.makedirs(dataset_path, exist_ok=True)
    tar_file_path = os.path.join(dataset_path, os.path.basename(dataset_url))
    if not os.path.exists(tar_file_path):
        wget.download(dataset_url, tar_file_path)
    extract_file(tar_file_path, str(dataset_path))
    wav_path = dataset_path / 'converted_wav/'
    extracted_dir = Path(tar_file_path).stem.replace('.tar', '')
    flac_path = dataset_path / (extracted_dir + '/wav/')
    __process_flac_audio(flac_path, wav_path)

    audio_files = [os.path.join(os.path.abspath(wav_path), file) for file in os.listdir(str(wav_path))]
    rttm_files = glob.glob(str(dataset_path / (extracted_dir + '/TextGrid/*.rttm')))
    rttm_files = [os.path.abspath(file) for file in rttm_files]

    audio_list = dataset_path / 'audio_files.txt'
    rttm_list = dataset_path / 'rttm_files.txt'
    with open(audio_list, 'w') as f:
        f.write('\n'.join(audio_files))
    with open(rttm_list, 'w') as f:
        f.write('\n'.join(rttm_files))
    create_manifest(
        str(audio_list),
        manifest_output_path,
        rttm_path=str(rttm_list),
    )


def __process_flac_audio(flac_path, wav_path):
    os.makedirs(wav_path, exist_ok=True)
    flac_files = os.listdir(flac_path)
    for flac_file in flac_files:
        # Convert FLAC file to WAV
        id = Path(flac_file).stem
        wav_file = os.path.join(wav_path, id + ".wav")
        if not os.path.exists(wav_file):
            Transformer().build(os.path.join(flac_path, flac_file), wav_file)


def main():
    parser = argparse.ArgumentParser(description="Aishell Data download")
    parser.add_argument("--data_root", default='./', type=str)
    parser.add_argument("--output_manifest_path", default='aishell_diar_manifest.json', type=str)
    parser.add_argument("--skip_train", help="skip downloading the training dataset", action="store_true")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True, parents=True)

    if not args.skip_train:
        for tag in train_datasets:
            dataset_url = train_url.format(tag)
            dataset_path = data_root / f'{tag}/'
            manifest_output_path = data_root / f'train_{tag}_manifest.json'
            __process_data(
                dataset_url=dataset_url, dataset_path=dataset_path, manifest_output_path=manifest_output_path
            )
    # create test dataset
    dataset_path = data_root / f'eval/'
    manifest_output_path = data_root / f'eval_manifest.json'
    __process_data(dataset_url=eval_url, dataset_path=dataset_path, manifest_output_path=manifest_output_path)


if __name__ == "__main__":
    main()
