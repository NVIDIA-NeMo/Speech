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

# USAGE: python get_hi-mia_data.py --data_root=<where to put data>

import argparse
import json
import logging as _logging
import os
import tarfile
import urllib.request
from glob import glob

import librosa as l
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from nemo.utils.tar_utils import safe_extract

parser = argparse.ArgumentParser(description="HI-MIA Data download")
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--log_level", default=20, type=int)
args = parser.parse_args()
logging = _logging.getLogger(__name__)
logging.addHandler(_logging.StreamHandler())
logging.setLevel(args.log_level)

URL = {
    "dev": "http://www.openslr.org/resources/85/dev.tar.gz",
    "test": "http://www.openslr.org/resources/85/test.tar.gz",
    "train": "http://www.openslr.org/resources/85/train.tar.gz",
}


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
    if not os.path.exists(destination) and not os.path.exists(os.path.splitext(destination)[0]):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        __retrieve_with_progress(source, filename=destination + ".tmp")
        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded {0}.".format(destination))
    elif os.path.exists(destination):
        logging.info("Destination {0} exists. Skipping.".format(destination))
    elif os.path.exists(os.path.splitext(destination)[0]):
        logging.warning(
            "Assuming extracted folder %s contains the extracted files from %s. Will not download.",
            os.path.basename(destination),
            destination,
        )
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
        with tarfile.open(filepath, encoding='utf-8') as tar:
            safe_extract(tar, data_dir)
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def __remove_tarred_files(filepath: str, data_dir: str):
    if os.path.exists(data_dir) and os.path.isfile(filepath):
        logging.info("Deleting %s" % filepath)
        os.remove(filepath)


def write_file(name, lines, idx):
    with open(name, "w") as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write("\n")
    logging.info("wrote %s", name)


def __process_data(data_folder: str, data_set: str):
    """
    To generate manifest
    Args:
        data_folder: source with wav files
    Returns:

    """
    fullpath = os.path.abspath(data_folder)
    filelist = glob(fullpath + "/**/*.wav", recursive=True)
    out = os.path.join(fullpath, data_set + "_all.json")
    utt2spk = os.path.join(fullpath, "utt2spk")
    utt2spk_file = open(utt2spk, "w")
    id = -2  # speaker id

    if os.path.exists(out):
        logging.warning(
            "%s already exists and is assumed to be processed. If not, please delete %s and rerun this script",
            out,
            out,
        )
        return

    speakers = []
    lines = []
    with open(out, "w") as outfile:
        for line in tqdm(filelist):
            line = line.strip()
            y, sr = l.load(line, sr=None)
            if sr != 16000:
                y, sr = l.load(line, sr=16000)
                l.output.write_wav(line, y, sr)
            dur = l.get_duration(y=y, sr=sr)
            if data_set == "test":
                speaker = line.split("/")[-1].split(".")[0].split("_")[0]
            else:
                speaker = line.split("/")[id]
            speaker = list(speaker)
            speaker = "".join(speaker)
            speakers.append(speaker)
            meta = {"audio_filepath": line, "duration": float(dur), "label": speaker}
            lines.append(meta)
            json.dump(meta, outfile)
            outfile.write("\n")
            utt2spk_file.write(line.split("/")[-1] + "\t" + speaker + "\n")

    utt2spk_file.close()

    if data_set != "test":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_idx, test_idx in sss.split(speakers, speakers):
            print(len(train_idx))

        out = os.path.join(fullpath, "train.json")
        write_file(out, lines, train_idx)
        out = os.path.join(fullpath, "dev.json")
        write_file(out, lines, test_idx)


def main():
    data_root = args.data_root
    for data_set in URL.keys():

        # data_set = 'data_aishell'
        logging.info("\n\nWorking on: {0}".format(data_set))
        file_path = os.path.join(data_root, data_set + ".tgz")
        logging.info("Getting {0}".format(data_set))
        __maybe_download_file(file_path, data_set)
        logging.info("Extracting {0}".format(data_set))
        data_folder = os.path.join(data_root, data_set)
        __extract_all_files(file_path, data_root, data_folder)
        __remove_tarred_files(file_path, data_folder)
        logging.info("Processing {0}".format(data_set))
        __process_data(data_folder, data_set)
        logging.info("Done!")


if __name__ == "__main__":
    main()
