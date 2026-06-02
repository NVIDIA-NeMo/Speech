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
import glob
import json
import os
import os.path
import subprocess
import tarfile
import urllib.request
from typing import Optional

from nemo.utils.dependency import import_optional_dependency


def build_manifest(transcripts_path, manifest_path, data_dir, mount_dir, wav_path):
    """Build an AN4 manifest with local or mounted audio paths."""
    # create manifest with reference to this directory. This is useful when mounting the dataset.
    mount_dir = mount_dir if mount_dir else data_dir
    sox = import_optional_dependency("sox")
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in fin:
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line[: line.find('(') - 1].lower()
                transcript = transcript.replace('<s>', '').replace('</s>', '')
                transcript = transcript.strip()

                file_id = line[line.find('(') + 1 : -2]  # e.g. "cen4-fash-b"
                audio_path = os.path.join(
                    data_dir, wav_path, file_id[file_id.find('-') + 1 : file_id.rfind('-')], file_id + '.wav'
                )

                mounted_audio_path = os.path.join(
                    mount_dir, wav_path, file_id[file_id.find('-') + 1 : file_id.rfind('-')], file_id + '.wav'
                )
                duration = sox.file_info.duration(audio_path)

                # Write the metadata to the manifest
                metadata = {"audio_filepath": mounted_audio_path, "duration": duration, "text": transcript}
                json.dump(metadata, fout)
                fout.write('\n')


def download_an4(data_dir: str = "./", train_mount_dir: Optional[str] = None, test_mount_dir: Optional[str] = None):
    """
    Function to download the AN4 dataset. This hides pre-processing boilerplate for notebook ASR examples.

    Args:
        data_dir: Path to store the data.
        train_mount_dir: If you plan to mount the dataset, use this to prepend the mount directory to the
            audio filepath in the train manifest.
        test_mount_dir: If you plan to mount the dataset, use this to prepend the mount directory to the
            audio filepath in the test manifest.
    """
    print("******")
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
        an4_url = 'https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz'
        an4_path = os.path.join(data_dir, 'an4_sphere.tar.gz')
        urllib.request.urlretrieve(an4_url, an4_path)
        print(f"Dataset downloaded at: {an4_path}")
    else:
        print("Tarfile already exists.")
        an4_path = data_dir + '/an4_sphere.tar.gz'

    if not os.path.exists(data_dir + '/an4/'):
        tar = tarfile.open(an4_path)
        tar.extractall(path=data_dir)

        print("Converting .sph to .wav...")
        sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
        for sph_path in sph_list:
            wav_path = sph_path[:-4] + '.wav'
            cmd = ["sox", sph_path, wav_path]
            subprocess.run(cmd)
    print("Finished conversion.\n******")

    # Building Manifests
    print("******")
    train_transcripts = data_dir + '/an4/etc/an4_train.transcription'
    train_manifest = data_dir + '/an4/train_manifest.json'

    if not os.path.isfile(train_manifest):
        build_manifest(train_transcripts, train_manifest, data_dir, train_mount_dir, 'an4/wav/an4_clstk')
        print("Training manifest created.")

    test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
    test_manifest = data_dir + '/an4/test_manifest.json'
    if not os.path.isfile(test_manifest):
        build_manifest(test_transcripts, test_manifest, data_dir, test_mount_dir, 'an4/wav/an4test_clstk')
        print("Test manifest created.")
    print("***Done***")
