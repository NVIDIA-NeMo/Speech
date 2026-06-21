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

# USAGE: python get_demand_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download>
# where <datasets_to_download> can be: one or more of the 16 kHz noise profiles
# listed at https://zenodo.org/record/1227121#.Ygb4avXMKJk ,
# or ALL
# You can put more than one data_set comma-separated:
# --data_sets=DKITCHEN,DLIVING,NRIVER

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import urllib.request

parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--data_sets", default="ALL", type=str)

parser.add_argument('--log', dest='log', action='store_true', default=False)
args = parser.parse_args()

URLS = {
    'DKITCHEN': ("https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip"),
    'DLIVING': ("https://zenodo.org/record/1227121/files/DLIVING_16k.zip"),
    'DWASHING': ("https://zenodo.org/record/1227121/files/DWASHING_16k.zip"),
    'NFIELD': ("https://zenodo.org/record/1227121/files/NFIELD_16k.zip"),
    'NPARK': ("https://zenodo.org/record/1227121/files/NPARK_16k.zip"),
    'NRIVER': ("https://zenodo.org/record/1227121/files/NRIVER_16k.zip"),
    'OHALLWAY': ("https://zenodo.org/record/1227121/files/OHALLWAY_16k.zip"),
    'OMEETING': ("https://zenodo.org/record/1227121/files/OMEETING_16k.zip"),
    'OOFFICE': ("https://zenodo.org/record/1227121/files/OOFFICE_16k.zip"),
    'PCAFETER': ("https://zenodo.org/record/1227121/files/PCAFETER_16k.zip"),
    'PRESTO': ("https://zenodo.org/record/1227121/files/PRESTO_16k.zip"),
    'PSTATION': ("https://zenodo.org/record/1227121/files/PSTATION_16k.zip"),
    'SPSQUARE': ("https://zenodo.org/record/1227121/files/SPSQUARE_16k.zip"),
    'STRAFFIC': ("https://zenodo.org/record/1227121/files/STRAFFIC_16k.zip"),
    'TBUS': ("https://zenodo.org/record/1227121/files/TBUS_16k.zip"),
    'TCAR': ("https://zenodo.org/record/1227121/files/TCAR_16k.zip"),
    'TMETRO': ("https://zenodo.org/record/1227121/files/TMETRO_16k.zip"),
}


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
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    shutil.unpack_archive(filepath, data_dir)


def __create_manifest(dst_folder: str):
    """
    Create manifests for the noise files
    Args:
        file_path: path to a source transcript  with flac sources
        dst_folder: path where manifests will be created
    Returns:

        a list of metadata entries for processed files.
    """
    # Read directory
    # Get all wav file names
    # create line per wav file in manifest
    noise_name = os.path.basename(dst_folder)
    wav_files = glob.glob(dst_folder + "/*.wav")
    wav_files.sort()
    os.makedirs(os.path.join(os.path.dirname(dst_folder), "manifests"), exist_ok=True)
    with open(os.path.join(os.path.dirname(dst_folder), "manifests", noise_name + ".json"), "w") as mfst_f:
        for wav_f in wav_files:
            dur = subprocess.check_output(["soxi", "-D", wav_f])
            row = {"audio_filepath": wav_f, "text": "", "duration": float(dur)}
            mfst_f.write(json.dumps(row) + "\n")


def main():
    data_root = args.data_root
    data_sets = args.data_sets

    if args.log:
        print("here")
        logging.basicConfig(level=logging.INFO)
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if data_sets == "ALL":
        data_sets = URLS.keys()
    else:
        data_sets = data_sets.split(',')

    for data_set in data_sets:
        if data_set not in URLS.keys():
            raise ValueError(f"{data_sets} is not part of demand noise database")
        logging.info("\n\nWorking on: {0}".format(data_set))
        filepath = os.path.join(data_root, data_set + "_16k.zip")
        logging.info("Getting {0}".format(data_set))
        __maybe_download_file(filepath, data_set.upper())
        logging.info("Extracting {0}".format(data_set))
        __extract_file(filepath, data_root)
        logging.info("Processing {0}".format(data_set))
        __create_manifest(os.path.join(data_root, data_set))
    logging.info('Done!')


if __name__ == "__main__":
    main()
