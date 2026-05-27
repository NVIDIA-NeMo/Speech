# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# USAGE: python get_aishell_data.py --data_root=<where to put data>

import argparse
import json
import logging
import os
import subprocess
import tarfile
import urllib.request

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Aishell Data download")
parser.add_argument("--data_root", required=True, default=None, type=str)
args = parser.parse_args()

URL = {"data_aishell": "http://www.openslr.org/resources/33/data_aishell.tgz"}


def __retrieve_with_progress(source: str, filename: str):
    """
    Downloads source to destination
    Displays progress bar
    Args:
        source: url of resource
        destination: local filepath
    Returns:
    """
    with open(filename, "wb") as f:
        response = urllib.request.urlopen(source)
        total = response.length

        if total is None:
            f.write(response.content)
        else:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for data in response:
                    f.write(data)
                    pbar.update(len(data))


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    source = URL[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        __retrieve_with_progress(source, filename=destination + ".tmp")
        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_all_files(filepath: str, data_root: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_root)
        audio_dir = os.path.join(data_dir, "wav")
        for subfolder, _, filelist in os.walk(audio_dir):
            for ftar in filelist:
                extract_file(os.path.join(subfolder, ftar), subfolder)
    else:
        logging.info("Skipping extracting. Data already there %s" % data_dir)


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def __process_data(data_folder: str, dst_folder: str):
    """
    To generate manifest
    Args:
        data_folder: source with wav files
        dst_folder: where manifest files will be stored
    Returns:

    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    transcript_file = os.path.join(data_folder, "transcript", "aishell_transcript_v0.8.txt")
    transcript_dict = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            audio_id, text = line.split(" ", 1)
            # remove white space
            text = text.replace(" ", "")
            transcript_dict[audio_id] = text

    data_types = ["train", "dev", "test"]
    vocab_count = {}
    for dt in data_types:
        json_lines = []
        audio_dir = os.path.join(data_folder, "wav", dt)
        for sub_folder, _, file_list in os.walk(audio_dir):
            for fname in file_list:
                audio_path = os.path.join(sub_folder, fname)
                audio_id = fname.strip(".wav")
                if audio_id not in transcript_dict:
                    continue
                text = transcript_dict[audio_id]
                for li in text:
                    vocab_count[li] = vocab_count.get(li, 0) + 1
                duration = subprocess.check_output("soxi -D {0}".format(audio_path), shell=True)
                duration = float(duration)
                json_lines.append(
                    json.dumps(
                        {
                            "audio_filepath": os.path.abspath(audio_path),
                            "duration": duration,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                )

        manifest_path = os.path.join(dst_folder, dt + ".json")
        with open(manifest_path, "w", encoding="utf-8") as fout:
            for line in json_lines:
                fout.write(line + "\n")

    vocab = sorted(vocab_count.items(), key=lambda k: k[1], reverse=True)
    vocab_file = os.path.join(dst_folder, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for v, c in vocab:
            f.write(v + "\n")


def main():
    data_root = args.data_root
    data_set = "data_aishell"
    logging.info("\n\nWorking on: {0}".format(data_set))
    file_path = os.path.join(data_root, data_set + ".tgz")
    logging.info("Getting {0}".format(data_set))
    __maybe_download_file(file_path, data_set)
    logging.info("Extracting {0}".format(data_set))
    data_folder = os.path.join(data_root, data_set)
    __extract_all_files(file_path, data_root, data_folder)
    logging.info("Processing {0}".format(data_set))
    __process_data(data_folder, data_folder)
    logging.info("Done!")


if __name__ == "__main__":
    main()
