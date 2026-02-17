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

import os
from argparse import ArgumentParser

import numpy as np

from nemo.collections.asr.parts.utils.vad_utils import prepare_manifest
from nemo.utils import logging


"""
This script is designed for inference of frame level Voice Activity Detection (VAD) 

This script serves three goals:
    (1) Write audio files to manifest
    (2) Split audio file for avoiding CUDA memory issue
    (3) Take care of joint of separate json line for an audio file

Usage:
python write_long_audio_manifest.py  --inp_dir=<FULL PATH OF FOLDER OF AUDIO FILES>  --split_duration=300 --window_length_in_sec=0.63 --num_worker=10

"""


def main():
    parser = ArgumentParser()
    parser.add_argument("--inp_dir", type=str, required=True, help="(full path) folder of files to be processed")
    parser.add_argument(
        "--inp_list", type=str, help="(full path) a file contains NAME of files inside inp_dir to be processed"
    )
    parser.add_argument("--out_dir", type=str, default=".", help="(full path) location to store generated json file")
    parser.add_argument("--manifest_name", type=str, default="generated_manifest", help="name of generated json file")
    parser.add_argument("--split_duration", type=int, required=True, help="max duration of each audio clip/line")
    parser.add_argument(
        "--window_length_in_sec",
        type=float,
        default=0.63,
        help="window length in sec for VAD context input , default is 0.63s",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for multiprocessing")

    args = parser.parse_args()

    if not args.inp_list:
        input_audios = []
        for root, dirs, files in os.walk(args.inp_dir):
            for basename in files:
                if basename.endswith('.wav'):
                    filename = os.path.join(root, basename)
                    input_audios.append(filename)
    else:
        name_list = np.loadtxt(args.inp_list, dtype='str')
        input_audios = [os.path.join(args.inp_dir, name + ".wav") for name in name_list]

    input_list = []
    for i in input_audios:
        input_list.append({'audio_filepath': i, "offset": 0, "duration": None})

    logging.info(f"Number of wav files to be processed: {len(input_audios)}")
    output_path = os.path.join(args.out_dir, args.manifest_name + '.json')

    logging.info("Split long audio file to avoid CUDA memory issue")
    logging.debug("Try smaller split_duration if you still have CUDA memory issue")

    config = {
        'input': input_list,
        'window_length_in_sec': args.window_length_in_sec,
        'split_duration': args.split_duration,
        'num_workers': args.num_workers,
        'prepared_manfiest_vad_input': output_path,
    }
    manifest_vad_input = prepare_manifest(config)
    logging.info(f"Done! Save to {manifest_vad_input}")


if __name__ == '__main__':
    main()
