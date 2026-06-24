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

import argparse
import glob
import json
import re
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from nemo.utils.tar_utils import safe_extract


def get_args():
    parser = argparse.ArgumentParser(description='Download HiFiTTS and create manifests with predefined split')
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help='Directory into which to download and extract dataset. \{data-root\}/hi_fi_tts_v0 will be created.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        help='Choose to generate manifest for all or one of (train, test, split), note that this will still download the full dataset.',
    )

    args = parser.parse_args()
    return args


URL = "https://us.openslr.org/resources/109/hi_fi_tts_v0.tar.gz"


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        tmp_file_path = destination_path.with_suffix('.tmp')
        urllib.request.urlretrieve(source_url, filename=str(tmp_file_path))
        tmp_file_path.rename(destination_path)


def __extract_file(filepath, data_dir):
    try:
        with tarfile.open(filepath) as tar:
            safe_extract(tar, str(data_dir))
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_data(data_root, filelists):
    # Create manifests (based on predefined NVIDIA's split)
    for split in tqdm(filelists):
        manifest_target = data_root / f"{split}_manifest.json"
        print(f"Creating manifest for {split}.")

        entries = []
        for manifest_src in glob.glob(str(data_root / f"*_{split}.json")):
            try:
                search_res = re.search('.*\/([0-9]+)_manifest_([a-z]+)_.*.json', manifest_src)
                speaker_id = search_res.group(1)
                audio_quality = search_res.group(2)
            except Exception:
                print(f"Failed to find speaker id or audio quality for {manifest_src}, check formatting.")
                continue

            with open(manifest_src, 'r') as f_in:
                for input_json_entry in f_in:
                    data = json.loads(input_json_entry)

                    # Make sure corresponding wavfile exists
                    wav_path = data_root / data['audio_filepath']
                    assert wav_path.exists(), f"{wav_path} does not exist!"

                    entry = {
                        'audio_filepath': data['audio_filepath'],
                        'duration': data['duration'],
                        'text': data['text'],
                        'normalized_text': data['text_normalized'],
                        'speaker': int(speaker_id),
                        # Audio_quality is either clean or other.
                        # The clean set includes recordings with high sound-to-noise ratio and wide bandwidth.
                        # The books with noticeable noise or narrow bandwidth are included in the other subset.
                        # Note: some speaker_id's have both clean and other audio quality.
                        'audio_quality': audio_quality,
                    }
                    entries.append(entry)

        with open(manifest_target, 'w') as f_out:
            for m in entries:
                f_out.write(json.dumps(m) + '\n')


def main():
    args = get_args()

    split = ['train', 'dev', 'test'] if args.split == 'all' else list(args.split)

    tarred_data_path = args.data_root / "hi_fi_tts_v0.tar.gz"

    __maybe_download_file(URL, tarred_data_path)
    __extract_file(str(tarred_data_path), str(args.data_root))

    data_root = args.data_root / "hi_fi_tts_v0"
    __process_data(data_root, split)


if __name__ == '__main__':
    main()
