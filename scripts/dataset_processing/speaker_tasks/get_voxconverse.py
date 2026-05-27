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

# downloads the training/eval set for VoxConverse.

import argparse
import logging
import os
import zipfile
from pathlib import Path

import wget

from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

dev_url = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
test_url = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
rttm_annotations_url = "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip"


def extract_file(filepath: Path, data_dir: Path):
    try:
        with zipfile.ZipFile(str(filepath), 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def _generate_manifest(data_root: Path, audio_path: Path, rttm_path: Path, manifest_output_path: Path):
    audio_list = str(data_root / 'audio_file.txt')
    rttm_list = str(data_root / 'rttm_file.txt')
    with open(audio_list, 'w') as f:
        f.write('\n'.join([str(os.path.join(rttm_path, x)) for x in os.listdir(audio_path)]))
    with open(rttm_list, 'w') as f:
        f.write('\n'.join([str(os.path.join(rttm_path, x)) for x in os.listdir(rttm_path)]))
    create_manifest(
        audio_list,
        str(manifest_output_path),
        rttm_path=rttm_list,
    )


def main():
    parser = argparse.ArgumentParser(description="VoxConverse Data download")
    parser.add_argument("--data_root", default='./', type=str)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True, parents=True)

    test_path = data_root / os.path.basename(test_url)
    dev_path = data_root / os.path.basename(dev_url)
    rttm_path = data_root / os.path.basename(rttm_annotations_url)

    if not os.path.exists(test_path):
        test_path = wget.download(test_url, str(data_root))
    if not os.path.exists(dev_path):
        dev_path = wget.download(dev_url, str(data_root))
    if not os.path.exists(rttm_path):
        rttm_path = wget.download(rttm_annotations_url, str(data_root))

    extract_file(test_path, data_root / 'test/')
    extract_file(dev_path, data_root / 'dev/')
    extract_file(rttm_path, data_root)

    _generate_manifest(
        data_root=data_root,
        audio_path=os.path.abspath(data_root / 'test/voxconverse_test_wav/'),
        rttm_path=os.path.abspath(data_root / 'voxconverse-master/test/'),
        manifest_output_path=data_root / 'test_manifest.json',
    )
    _generate_manifest(
        data_root=data_root,
        audio_path=os.path.abspath(data_root / 'dev/audio/'),
        rttm_path=os.path.abspath(data_root / 'voxconverse-master/dev/'),
        manifest_output_path=data_root / 'dev_manifest.json',
    )


if __name__ == "__main__":
    main()
