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

# USAGE: python get_openslr_rir_data.py --data_root=<where to put data>
# Data is downloaded from OpenSLR's "Room Impulse Response and Noise Database"
# RIRs in multichannel files are separated into single channel files and
# a json file that can be used as in input to NeMo is created

import argparse
import glob
import json
import logging
import os
import subprocess
import urllib.request
from shutil import copy, move
from zipfile import ZipFile

from tqdm import tqdm

parser = argparse.ArgumentParser(description="OpenSLR RIR Data download and process")
parser.add_argument("--data_root", required=True, default=None, type=str)
args = parser.parse_args()

URLS = {
    "SLR28": ("http://www.openslr.org/resources/28/rirs_noises.zip"),
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
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        __retrieve_with_progress(source, filename=destination + ".tmp")
        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        with ZipFile(filepath, "r") as zipObj:
            zipObj.extractall(data_dir)
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def __process_data(data_folder: str, dst_folder: str, manifest_file: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
    Returns:
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    real_rir_list = os.path.join(data_folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list")
    rirfiles = []
    with open(real_rir_list, "r") as rir_f:
        for line in rir_f:
            rirfiles.append(os.path.join(data_folder, line.rstrip().split(" ")[4]))

    real_rir_folder = os.path.join(dst_folder, "real_rirs")
    if not os.path.exists(real_rir_folder):
        os.makedirs(real_rir_folder)
    # split multi-channel rir files to single channel
    for rir_f in rirfiles:
        n_chans = int(subprocess.check_output("soxi -c {0}".format(rir_f), shell=True))
        if n_chans == 1:
            copy(rir_f, real_rir_folder)
        else:
            for chan in range(1, n_chans + 1):
                chan_file_name = os.path.join(
                    real_rir_folder,
                    os.path.splitext(os.path.basename(rir_f))[0] + "-" + str(chan) + ".wav",
                )
                _ = subprocess.check_output(f"sox {rir_f} {chan_file_name} remix {chan}", shell=True)

    # move simulated rirs to processed
    if not os.path.exists(os.path.join(dst_folder, "simulated_rirs")):
        move(os.path.join(data_folder, "RIRS_NOISES", "simulated_rirs"), dst_folder)

    os.chdir(dst_folder)
    all_rirs = glob.glob("**/*.wav", recursive=True)
    with open(manifest_file, "w") as man_f:
        entry = {}
        for rir in all_rirs:
            rir_file = os.path.join(dst_folder, rir)
            duration = subprocess.check_output("soxi -D {0}".format(rir_file), shell=True)
            entry["audio_filepath"] = rir_file
            entry["duration"] = float(duration)
            entry["offset"] = 0
            entry["text"] = "_"
            man_f.write(json.dumps(entry) + "\n")

    print("Done!")


def main():
    data_root = os.path.abspath(args.data_root)
    data_set = "slr28"
    logging.getLogger().setLevel(logging.INFO)
    logging.info("\n\nWorking on: {0}".format(data_set))
    filepath = os.path.join(data_root, data_set + ".zip")
    logging.info("Getting {0}".format(data_set))
    __maybe_download_file(filepath, data_set.upper())
    logging.info("Extracting {0}".format(data_set))
    __extract_file(filepath, data_root)
    logging.info("Processing {0}".format(data_set))
    __process_data(
        data_root,
        os.path.join(os.path.join(data_root, "processed")),
        os.path.join(os.path.join(data_root, "processed", "rir.json")),
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()
