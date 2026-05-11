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

"""
This script converts a filelist file where each line contains
<absolute path of wav file> to a manifest json file.
Optionally post processes the manifest file to create dev and train split for speaker embedding
training, also optionally segment an audio file in to segments of random DURATIONS and create those
wav files in CWD.

Args:
--filelist: path to file containing list of audio files
--manifest(optional): if you already have manifest file, but would like to process it for creating
    segments and splitting then use manifest ignoring filelist
--id: index of speaker label in filename present in filelist file that is separated by '/'
--out: output manifest file name
--split: if you would want to split the  manifest file for training purposes
    you may not need this for test set. output file names is <out>_<train/dev>.json, defaults to False
--create_segments: if you would want to segment each manifest line to segments of [1,2,3,4] sec or less
    you may not need this for test set, defaults to False
--min_spkrs_count: min number of samples per speaker to consider and ignore otherwise, defaults to 0 (all speakers)
"""

import argparse
import json
import os
import random

import librosa as l
import numpy as np
import soundfile as sf
import sox
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.contrib.concurrent import process_map
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

random.seed(42)

DURATIONS = sorted([3], reverse=True)
MIN_ENERGY = 0.01
CWD = os.getcwd()


def filter_manifest_line(manifest_line):
    split_manifest = []
    audio_path = manifest_line['audio_filepath']
    start = manifest_line.get('offset', 0)
    dur = manifest_line['duration']
    label = manifest_line['label']
    endname = os.path.splitext(audio_path.split(label, 1)[-1])[0]
    to_path = os.path.join(CWD, 'segments', label)
    to_path = os.path.join(to_path, endname[1:])
    os.makedirs(os.path.dirname(to_path), exist_ok=True)

    if dur >= min(DURATIONS):
        signal, sr = sf.read(audio_path)
        remaining_dur = dur - start

        segments = DURATIONS.copy()
        mode = int(remaining_dur // sum(DURATIONS))
        rem = remaining_dur % sum(DURATIONS)
        segments = mode * segments

        for val in DURATIONS:
            if rem >= val:
                segments.append(val)
                rem = rem - val

        for temp_dur in segments:
            segment_audio = signal[int(start * sr) : int(start * sr + temp_dur * sr)]
            if l.feature.rms(y=segment_audio).mean() > MIN_ENERGY:
                final_string = '_' + str(start) + '_' + str(temp_dur)
                final_string = final_string.replace('.', '-')
                to_file = to_path + final_string + '.wav'

                c_start = int(float(start * sr))
                c_end = c_start + int(float(temp_dur * sr))
                segment = signal[c_start:c_end]
                sf.write(to_file, segment, sr)

                meta = manifest_line.copy()
                meta['audio_filepath'] = to_file
                meta['offset'] = 0
                meta['duration'] = temp_dur
                split_manifest.append(meta)

            start = start + temp_dur

    return split_manifest


def count_and_consider_only(speakers, lines, min_count=10):
    """
    consider speakers only if samples per speaker is at least min_count
    """
    uniq_speakers, indices, counts = np.unique(speakers, return_index=True, return_counts=True)
    print("speaker count before filtering minimum number of speaker counts: ", len(uniq_speakers))
    required_speakers = {}
    for idx, count in enumerate(counts):
        if count >= min_count:
            required_speakers[uniq_speakers[idx]] = count

    print("speaker count after filtering minimum number of speaker counts: ", len(required_speakers))
    required_lines = []
    speakers_only = []
    for idx, speaker in enumerate(speakers):
        if speaker in required_speakers:
            required_lines.append(lines[idx])
            speakers_only.append(speaker)

    return speakers_only, required_lines


def write_file(name, lines, idx):
    with open(name, 'w', encoding='utf-8') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    print("wrote", name)


def read_file(filelist, id=-1):
    json_lines = []
    with open(filelist, 'r') as fo:
        lines = fo.readlines()
        lines = sorted(lines)
        for line in lines:
            line = line.strip()
            speaker = line.split('/')[id]
            speaker = list(speaker)
            speaker = ''.join(speaker)
            meta = {"audio_filepath": line, "offset": 0, "duration": None, "label": speaker}
            json_lines.append(meta)
    return json_lines


def get_duration(json_line):
    dur = json_line['duration']
    if dur is None:
        wav_path = json_line['audio_filepath']
        json_line['duration'] = sox.file_info.duration(wav_path)
    return json_line


def get_labels(lines):
    labels = []
    for line in lines:
        label = line['label']
        labels.append(label)
    return labels


def main(filelist, manifest, id, out, split=False, create_segments=False, min_count=10):
    if os.path.exists(out):
        os.remove(out)
    if filelist:
        lines = read_file(filelist=filelist, id=id)
        lines = process_map(get_duration, lines, chunksize=100)
        out_file = os.path.splitext(filelist)[0] + '_manifest.json'
        write_file(out_file, lines, range(len(lines)))
    else:
        lines = read_manifest(manifest)

    lines = process_map(get_duration, lines, chunksize=100)

    if create_segments:
        print(f"creating and writing segments to {CWD}")
        lines = process_map(filter_manifest_line, lines, chunksize=100)
        temp = []
        for line in lines:
            temp.extend(line)
        del lines
        lines = temp

    speakers = [x['label'] for x in lines]

    if min_count:
        speakers, lines = count_and_consider_only(speakers, lines, abs(min_count))

    write_file(out, lines, range(len(lines)))
    path = os.path.dirname(out)
    if split:
        speakers = [x['label'] for x in lines]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_idx, test_idx in sss.split(speakers, speakers):
            print("number of train samples after split: ", len(train_idx))

        out = os.path.join(path, 'train.json')
        write_file(out, lines, train_idx)
        out = os.path.join(path, 'dev.json')
        write_file(out, lines, test_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", help="path to filelist file", type=str, required=False, default=None)
    parser.add_argument("--manifest", help="manifest file name", type=str, required=False, default=None)
    parser.add_argument(
        "--id",
        help="field num seperated by '/' to be considered as speaker label from filelist file, can be ignored if manifest file is already provided with labels",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument("--out", help="manifest_file name", type=str, required=True)
    parser.add_argument(
        "--split",
        help="bool if you would want to split the manifest file for training purposes",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--create_segments",
        help="bool if you would want to segment each manifest line to segments of 4 sec or less",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--min_spkrs_count",
        default=0,
        type=int,
        help="min number of samples per speaker to consider and ignore otherwise",
    )

    args = parser.parse_args()

    main(
        args.filelist,
        args.manifest,
        args.id,
        args.out,
        args.split,
        args.create_segments,
        args.min_spkrs_count,
    )
