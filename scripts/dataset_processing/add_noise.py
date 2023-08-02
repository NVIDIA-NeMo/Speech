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

# USAGE: python add_noise.py --input_manifest=<manifest file of original "clean" dataset>
#   --noise_manifest=<manifest file poinitng to noise data>
#   --out_dir=<destination directory for noisy audio and manifests>
#   --snrs=<list of snrs at which noise should be added to the audio>
#   --seed=<seed for random number generator>
#   --num_workers=<number of parallel workers>
# To be able to reproduce the same noisy dataset, use a fixed seed and num_workers=1

import argparse
import copy
import json
import multiprocessing
import os
import random

import numpy as np
import soundfile as sf

from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

rng = None
att_factor = 0.8
save_noise = False
sample_rate = 16000


def get_out_dir_name(out_dir, input_name, noise_name, snr):
    return os.path.join(out_dir, input_name, noise_name + "_" + str(snr) + "db")


def create_manifest(input_manifest, noise_manifest, snrs, out_path, save_noise):
    os.makedirs(os.path.join(out_path, "manifests"), exist_ok=True)
    for snr in snrs:
        out_dir = get_out_dir_name(
            out_path,
            os.path.splitext(os.path.basename(input_manifest))[0],
            os.path.splitext(os.path.basename(noise_manifest))[0],
            snr,
        )
        out_mfst = os.path.join(
            os.path.join(out_path, "manifests"),
            os.path.splitext(os.path.basename(input_manifest))[0]
            + "_"
            + os.path.splitext(os.path.basename(noise_manifest))[0]
            + "_"
            + str(snr)
            + "db"
            + ".json",
        )
        with open(input_manifest, "r") as inf, open(out_mfst, "w") as outf:
            for line in inf:
                row = json.loads(line.strip())
                row['audio_filepath'] = os.path.join(out_dir, os.path.basename(row['audio_filepath']))
                if save_noise:
                    file_ext = os.path.splitext(row['audio_filepath'])[1]
                    noise_filename = os.path.basename(row['audio_filepath']).replace(file_ext, "_noise" + file_ext)
                    row['noise_filepath'] = os.path.join(out_dir, noise_filename)
                outf.write(json.dumps(row) + "\n")


def process_row(row):
    audio_file = row['audio_filepath']
    global sample_rate
    data_orig = AudioSegment.from_file(audio_file, target_sr=sample_rate, offset=0)
    for snr in row['snrs']:
        min_snr_db = snr
        max_snr_db = snr
        global att_factor
        perturber = NoisePerturbation(
            manifest_path=row['noise_manifest'], min_snr_db=min_snr_db, max_snr_db=max_snr_db, rng=rng
        )
        out_dir = get_out_dir_name(
            row['out_dir'],
            os.path.splitext(os.path.basename(row['input_manifest']))[0],
            os.path.splitext(os.path.basename(row['noise_manifest']))[0],
            snr,
        )
        os.makedirs(out_dir, exist_ok=True)
        out_f = os.path.join(out_dir, os.path.basename(audio_file))
        if os.path.exists(out_f):
            continue
        data = copy.deepcopy(data_orig)
        perturber.perturb(data)

        max_level = np.max(np.abs(data.samples))

        norm_factor = att_factor / max_level
        new_samples = norm_factor * data.samples
        sf.write(out_f, new_samples.transpose(), sample_rate)

        global save_noise
        if save_noise:
            noise_samples = new_samples - norm_factor * data_orig.samples
            out_f_ext = os.path.splitext(out_f)[1]
            out_f_noise = out_f.replace(out_f_ext, "_noise" + out_f_ext)
            sf.write(out_f_noise, noise_samples.transpose(), sample_rate)


def add_noise(infile, snrs, noise_manifest, out_dir, num_workers=1):
    allrows = []

    with open(infile, "r") as inf:
        for line in inf:
            row = json.loads(line.strip())
            row['snrs'] = snrs
            row['out_dir'] = out_dir
            row['noise_manifest'] = noise_manifest
            row['input_manifest'] = infile
            allrows.append(row)
    pool = multiprocessing.Pool(num_workers)
    pool.map(process_row, allrows)
    pool.close()
    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="clean test set",
    )
    parser.add_argument("--noise_manifest", type=str, required=True, help="path to noise manifest file")
    parser.add_argument("--out_dir", type=str, required=True, help="destination directory for audio and manifests")
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument(
        "--attenuation_factor",
        default=0.8,
        type=float,
        help="Attenuation factor applied on the normalized noise-added samples before writing to wave",
    )
    parser.add_argument(
        "--save_noise", default=False, action="store_true", help="save the noise added to the input signal"
    )

    args = parser.parse_args()
    global sample_rate
    sample_rate = args.sample_rate
    global att_factor
    att_factor = args.attenuation_factor
    global save_noise
    save_noise = args.save_noise
    global rng
    rng = args.seed
    num_workers = args.num_workers

    add_noise(args.input_manifest, args.snrs, args.noise_manifest, args.out_dir, num_workers=num_workers)
    create_manifest(args.input_manifest, args.noise_manifest, args.snrs, args.out_dir, args.save_noise)


if __name__ == '__main__':
    main()
