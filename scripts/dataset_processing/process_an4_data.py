# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import glob
import json
import logging
import os
import subprocess

import librosa

parser = argparse.ArgumentParser(description="AN4 dataset download and processing")
parser.add_argument("--data_root", required=True, default=None, type=str)
args = parser.parse_args()


def build_manifest(data_root, transcripts_path, manifest_path, wav_path):
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
                    data_root,
                    wav_path,
                    file_id[file_id.find('-') + 1 : file_id.rfind('-')],
                    file_id + '.wav',
                )

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript,
                }
                json.dump(metadata, fout)
                fout.write('\n')


def main():
    data_root = os.path.abspath(args.data_root)

    # Convert from .sph to .wav
    logging.info("Converting audio files to .wav...")
    sph_list = glob.glob(os.path.join(data_root, 'an4/**/*.sph'), recursive=True)
    for sph_path in sph_list:
        wav_path = sph_path[:-4] + '.wav'
        cmd = ['sox', sph_path, wav_path]
        subprocess.run(cmd)
    logging.info("Finished conversion.")

    # Build manifests
    logging.info("Building training manifest...")
    train_transcripts = os.path.join(data_root, 'an4/etc/an4_train.transcription')
    train_manifest = os.path.join(data_root, 'an4/train_manifest.json')
    train_wavs = os.path.join(data_root, 'an4/wav/an4_clstk')
    build_manifest(data_root, train_transcripts, train_manifest, train_wavs)
    logging.info("Training manifests created.")

    logging.info("Building test manifest...")
    test_transcripts = os.path.join(data_root, 'an4/etc/an4_test.transcription')
    test_manifest = os.path.join(data_root, 'an4/test_manifest.json')
    test_wavs = os.path.join(data_root, 'an4/wav/an4test_clstk')
    build_manifest(data_root, test_transcripts, test_manifest, test_wavs)
    logging.info("Test manifest created.")

    logging.info("Done with AN4 processing!")


if __name__ == '__main__':
    main()
