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
import json
import tarfile
import urllib.request
from pathlib import Path

import sox
import wget
from tqdm import tqdm

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )


def get_args():
    parser = argparse.ArgumentParser(description='Download LJSpeech and create manifests with predefined split')
    parser.add_argument("--data-root", required=True, type=Path)

    args = parser.parse_args()
    return args


URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
FILELIST_BASE = 'https://raw.githubusercontent.com/NVIDIA/tacotron2/master/filelists'


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


def __process_data(data_root):
    text_normalizer = Normalizer(
        lang="en", input_case="cased", overwrite_cache=True, cache_dir=data_root / "cache_dir",
    )
    text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}
    normalizer_call = lambda x: text_normalizer.normalize(x, **text_normalizer_call_kwargs)

    # Create manifests (based on predefined NVIDIA's split)
    filelists = ['train', 'val', 'test']
    for split in tqdm(filelists):
        # Download file list if necessary
        filelist_path = data_root / f"ljs_audio_text_{split}_filelist.txt"

        if not filelist_path.exists():
            wget.download(f"{FILELIST_BASE}/ljs_audio_text_{split}_filelist.txt", out=str(data_root))

        manifest_target = data_root / f"{split}_manifest.json"
        with open(manifest_target, 'w') as f_out:
            with open(filelist_path, 'r') as filelist:
                print(f"\nCreating {manifest_target}...")
                for line in tqdm(filelist):
                    basename = line[6:16]

                    text = line[21:].strip()
                    norm_text = normalizer_call(text)

                    # Make sure corresponding wavfile exists
                    wav_path = data_root / 'wavs' / f"{basename}.wav"
                    assert wav_path.exists(), f"{wav_path} does not exist!"

                    entry = {
                        'audio_filepath': str(wav_path),
                        'duration': sox.file_info.duration(wav_path),
                        'text': text,
                        'normalized_text': norm_text,
                    }

                    f_out.write(json.dumps(entry) + '\n')


def main():
    args = get_args()

    tarred_data_path = args.data_root / "LJSpeech-1.1.tar.bz2"

    __maybe_download_file(URL, tarred_data_path)
    __extract_file(str(tarred_data_path), str(args.data_root))

    data_root = args.data_root / "LJSpeech-1.1"

    __process_data(data_root)


if __name__ == '__main__':
    main()
