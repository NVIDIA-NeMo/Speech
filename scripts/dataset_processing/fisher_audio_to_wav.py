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

# USAGE:
# python fisher_audio_to_wav.py \
#   --data_root=<FisherEnglishTrainingSpeech root> \
#   --dest_root=<destination dir root>
#
# Converts all .sph audio files in the Fisher dataset to .wav.
# Requires sph2pipe to be installed.
import argparse
import concurrent.futures
import glob
import logging
import os
import subprocess

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert Fisher .sph to .wav')
parser.add_argument(
    "--data_root",
    default=None,
    type=str,
    required=True,
    help="The path to the root Fisher dataset folder.",
)
parser.add_argument(
    "--dest_root",
    default=None,
    type=str,
    required=True,
    help="Path to the destination root directory.",
)
args = parser.parse_args()


def __convert_audio(in_path, out_path):
    """
    Helper function that's called per thread, converts sph to wav.
    Args:
        in_path: source sph file to convert
        out_path: destination for wav file
    """
    cmd = ["sph2pipe", "-f", "wav", "-p", in_path, out_path]
    subprocess.run(cmd)


def __process_set(data_root, dst_root):
    """
    Finds and converts all sph audio files in the given directory to wav.
    Args:
        data_folder: source directory with sph files to convert
        dst_root: where wav files will be stored
    """
    sph_list = glob.glob(data_root)

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # Set up and execute concurrent audio conversion
    tp = concurrent.futures.ProcessPoolExecutor(max_workers=64)
    futures = []

    for sph_path in tqdm(sph_list, desc="Submitting sph futures", unit="file"):
        audio_id, _ = os.path.splitext(os.path.basename(sph_path))
        out_path = os.path.join(dst_root, "{}.wav".format(audio_id))
        futures.append(tp.submit(__convert_audio, sph_path, out_path))

    pbar = tqdm(total=len(sph_list), desc="Converting sph files", unit="file")
    count = 0
    for f in concurrent.futures.as_completed(futures):
        count += 1
        pbar.update()
    tp.shutdown()
    pbar.close()


def main():
    data_root = args.data_root
    dest_root = args.dest_root

    logging.info("\n\nConverting audio for Part 1")
    __process_set(
        os.path.join(
            data_root,
            "LDC2004S13-Part1",
            "fisher_eng_tr_sp_d*",
            "audio",
            "*",
            "*.sph",
        ),
        os.path.join(dest_root, "LDC2004S13-Part1", "audio_wav"),
    )

    logging.info("\n\nConverting audio for Part 2")
    __process_set(
        os.path.join(
            data_root,
            "LDC2005S13-Part2",
            "fe_03_p2_sph*",
            "audio",
            "*",
            "*.sph",
        ),
        os.path.join(dest_root, "LDC2005S13-Part2", "audio_wav"),
    )


if __name__ == '__main__':
    main()
