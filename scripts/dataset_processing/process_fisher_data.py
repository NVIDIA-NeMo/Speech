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

# USAGE:
# python process_fisher_data.py \
#   --audio_root=<audio (.wav) directory>
#   --transcript_root=<LDC Fisher dataset directory> \
#   --dest_root=<destination directory> \
#   --data_sets=LDC2004S13-Part1,LDC2005S13-Part2 \
#   --remove_noises
#
# Matches Fisher dataset transcripts to the corresponding audio file (.wav),
# and slices them into min_slice_duration segments with one speaker.
# Also performs some other processing on transcripts.
#
# Heavily derived from Patter's Fisher processing script.

import argparse
import glob
import json
import os
import re
from math import ceil, floor

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Fisher Data Processing")
parser.add_argument(
    "--audio_root",
    default=None,
    type=str,
    required=True,
    help="The path to the root of the audio (wav) data folder.",
)
parser.add_argument(
    "--transcript_root",
    default=None,
    type=str,
    required=True,
    help="The path to the root of the transcript data folder.",
)
parser.add_argument(
    "--dest_root",
    default=None,
    type=str,
    required=True,
    help="Path to the destination root directory.",
)

# Optional arguments
parser.add_argument(
    "--min_slice_duration",
    default=10.0,
    type=float,
    help="Minimum audio slice duration after processing.",
)
parser.add_argument(
    "--keep_low_conf",
    action="store_true",
    help="Keep all utterances with low confidence transcripts",
)
parser.add_argument(
    "--remove_noises",
    action="store_true",
    help="Removes transcripted noises such as [laughter].",
)
parser.add_argument(
    "--noises_to_emoji",
    action="store_true",
    help="Converts transcripts for noises to an emoji character.",
)
args = parser.parse_args()

# Total number of files before segmenting, and train/val/test splits
NUM_FILES = 5850 + 5849
TRAIN_END_IDX = int(NUM_FILES * 0.8)
VAL_END_IDX = int(NUM_FILES * 0.9)

# Known transcription errors and their fixes (from Mozilla)
TRANSCRIPT_BUGS = {
    "fe_03_00265-B-3353-3381": "correct",
    "fe_03_00991-B-52739-52829": "that's one of those",
    "fe_03_10282-A-34442-34484.wav": "they don't want",
    "fe_03_10677-B-10104-10641": "uh my mine yeah the german shepherd "
    + "pitbull mix he snores almost as loud "
    + "as i do",
    "fe_03_00027-B-39380-39405": None,
    "fe_03_11487-B-3109-23406": None,
    "fe_03_01326-A-30742-30793": None,
}

TRANSCRIPT_NUMBERS = {
    "401k": "four o one k",
    "f16": "f sixteen",
    "m16": "m sixteen",
    "ak47": "a k forty seven",
    "v8": "v eight",
    "y2k": "y two k",
    "mp3": "m p three",
    "vh1": "v h one",
    "90210": "nine o two one o",
    "espn2": "e s p n two",
    "u2": "u two",
    "dc3s": "d c threes",
    "book 2": "book two",
    "s2b": "s two b",
    "3d": "three d",
}

TAG_MAP = {
    "[laughter]": "🤣",
    "[laugh]": "🤣",
    "[noise]": "😕",
    "[sigh]": "😕",
    "[cough]": "😕",
    "[mn]": "😕",
    "[breath]": "😕",
    "[lipsmack]": "😕",
    "[[skip]]": "",
    "[pause]": "",
    "[sneeze]": "😕",
}


def __write_sample(dest, file_id, count, file_count, sample_rate, audio, duration, transcript):
    """
    Writes one slice to the given target directory.
    Args:
        dest: the destination directory
        file_id: name of the transcript/audio file for this block
        count: the count of segments in the file so far
        file_count: the total number of filse processed so far
        sample rate: sample rate of the audio data
        audio: audio data of the current sample
        duration: audio duration of the current sample
        transcript: transcript of the current sample
    """
    partition = __partition_name(file_count)
    audio_path = os.path.join(dest, partition, f"{file_id}_{count:03}.wav")

    # Write audio
    wavfile.write(audio_path, sample_rate, audio)

    # Write transcript info
    transcript = {
        "audio_filepath": audio_path,
        "duration": duration,
        "text": transcript,
    }

    # Append to manifest
    manifest_path = os.path.join(dest, f"manifest_{partition}.json")
    with open(manifest_path, 'a') as f:
        json.dump(transcript, f)
        f.write('\n')


def __normalize(utt):
    replace_table = str.maketrans(dict.fromkeys('()*;:"!&{},.-?'))
    utt = (
        utt.lower()
        .replace('[uh]', 'uh')
        .replace('[um]', 'um')
        .replace('<noise>', '[noise]')
        .replace('<spoken_noise>', '[vocalized-noise]')
        .replace('.period', 'period')
        .replace('.dot', 'dot')
        .replace('-hyphen', 'hyphen')
        .replace('._', ' ')
        .translate(replace_table)
    )
    utt = re.sub(r"'([a-z]+)'", r'\1', utt)  # Unquote quoted words
    return utt


def __process_utterance(file_id, trans_path, line, keep_low_conf, rem_noises, emojify):
    """
    Processes one utterance (one line of a transcript).
    Args:
        file_id: the ID of the transcript file
        trans_path: transcript path
        line: one line in the transcript file
        keep_low_conf: whether to keep low confidence lines
        rem_noises: whether to remove noise symbols
        emojify: whether to convert noise symbols to emoji, lower precedence
    """
    # Check for lines to skip (comments, empty, low confidence)
    if line.startswith('#') or not line.strip() or (not keep_low_conf and '((' in line):
        return None, None, None, None

    # Data and sanity checks
    line = line.split()

    t_start, t_end = float(line[0]), float(line[1])
    if (t_start < 0) or (t_end < t_start):
        print(f"Invalid time: {t_start} to {t_end} in {trans_path}")
        return None, None, None, None

    channel = line[2]
    idx = 0 if line[2] == 'A:' else 1

    if channel not in ('A:', 'B:'):
        print(f"Could not read channel info ({channel}) in {trans_path}")
        return None, None, None, None

    # Replacements as necessary
    line_id = '-'.join([file_id, channel[0], str(t_start * 10), str(t_end * 10)])

    content = TRANSCRIPT_BUGS.get(line_id, ' '.join(line[3:]))

    if content is None:
        return None, None, None, None

    for tag, newtag in TRANSCRIPT_NUMBERS.items():
        content = content.replace(tag, newtag)

    content = __normalize(content)

    if rem_noises:
        for k, _ in TAG_MAP.items():
            content = content.replace(k, '')
    elif emojify:
        for k, v in TAG_MAP.items():
            content = content.replace(k, v)

    return t_start, t_end, idx, content


def __process_one_file(
    trans_path,
    sample_rate,
    audio_data,
    file_id,
    dst_root,
    min_slice_duration,
    file_count,
    keep_low_conf,
    rem_noises,
    emojify,
):
    """
    Creates one block of audio slices and their corresponding transcripts.
    Args:
        trans_path: filepath to transcript
        sample_rate: sample rate of the audio
        audio_data: numpy array of shape [samples, channels]
        file_id: identifying label, e.g. 'fe_03_01102'
        dst_root: path to destination directory
        min_slice_duration: min number of seconds for an audio slice
        file_count: total number of files processed so far
        keep_low_conf: keep utterances with low-confidence transcripts
        rem_noises: remove noise symbols
        emojify: convert noise symbols into emoji characters
    """
    count = 0

    with open(trans_path, encoding="utf-8") as fin:
        fin.readline()  # Comment w/ corresponding sph filename
        fin.readline()  # Comment about transcriber

        transcript_buffers = ['', '']  # [A buffer, B buffer]
        audio_buffers = [[], []]
        buffer_durations = [0.0, 0.0]

        for line in fin:
            t_start, t_end, idx, content = __process_utterance(
                file_id, trans_path, line, keep_low_conf, rem_noises, emojify
            )

            if content is None or not content:
                continue

            duration = t_end - t_start

            # Append utterance to buffer
            transcript_buffers[idx] += content
            audio_buffers[idx].append(
                audio_data[
                    floor(t_start * sample_rate) : ceil(t_end * sample_rate),
                    idx,
                ]
            )
            buffer_durations[idx] += duration

            if buffer_durations[idx] < min_slice_duration:
                transcript_buffers[idx] += ' '
            else:
                # Write out segment and transcript
                count += 1
                __write_sample(
                    dst_root,
                    file_id,
                    count,
                    file_count,
                    sample_rate,
                    np.concatenate(audio_buffers[idx], axis=0),
                    buffer_durations[idx],
                    transcript_buffers[idx],
                )

                # Clear buffers
                transcript_buffers[idx] = ''
                audio_buffers[idx] = []
                buffer_durations[idx] = 0.0

            # Note: We drop any shorter "scraps" at the end of the file, if
            #   they end up shorter than min_slice_duration.


def __partition_name(file_count):
    if file_count >= VAL_END_IDX:
        return "test"
    elif file_count >= TRAIN_END_IDX:
        return "val"
    else:
        return "train"


def __process_data(
    audio_root,
    transcript_root,
    dst_root,
    min_slice_duration,
    file_count,
    keep_low_conf,
    rem_noises,
    emojify,
):
    """
    Converts Fisher wav files to numpy arrays, segments audio and transcripts.
    Args:
        audio_root: source directory with the wav files
        transcript_root: source directory with the transcript files
            (can be the same as audio_root)
        dst_root: where the processed and segmented files will be stored
        min_slice_duration: minimum number of seconds for a slice of output
        file_count: total number of files processed so far
        keep_low_conf: whether or not to keep low confidence transcriptions
        rem_noises: whether to remove noise symbols
        emojify: whether to convert noise symbols to emoji, lower precedence
    Assumes:
        1. There is exactly one transcripts directory in data_folder
        2. Audio files are all: <audio_root>/audio-wav/fe_03_xxxxx.wav
    """
    transcript_list = glob.glob(os.path.join(transcript_root, "fe_03_p*_tran*", "data", "trans", "*", "*.txt"))
    print("Found {} transcripts.".format(len(transcript_list)))

    count = file_count

    # Grab audio file associated with each transcript, and slice
    for trans_path in tqdm(transcript_list, desc="Matching and segmenting"):
        file_id, _ = os.path.splitext(os.path.basename(trans_path))
        audio_path = os.path.join(audio_root, "audio_wav", file_id + ".wav")

        sample_rate, audio_data = wavfile.read(audio_path)

        # Create a set of segments (a block) for each file
        __process_one_file(
            trans_path,
            sample_rate,
            audio_data,
            file_id,
            dst_root,
            min_slice_duration,
            count,
            keep_low_conf,
            rem_noises,
            emojify,
        )
        count += 1

    return count


def main():
    # Arguments to the script
    audio_root = args.audio_root
    transcript_root = args.transcript_root
    dest_root = args.dest_root

    min_slice_duration = args.min_slice_duration
    keep_low_conf = args.keep_low_conf
    rem_noises = args.remove_noises
    emojify = args.noises_to_emoji

    print(f"Expected number of files to segment: {NUM_FILES}")
    print("With a 80/10/10 split:")
    print(f"Number of training files: {TRAIN_END_IDX}")
    print(f"Number of validation files: {VAL_END_IDX - TRAIN_END_IDX}")
    print(f"Number of test files: {NUM_FILES - VAL_END_IDX}")

    if not os.path.exists(os.path.join(dest_root, 'train/')):
        os.makedirs(os.path.join(dest_root, 'train/'))
        os.makedirs(os.path.join(dest_root, 'val/'))
        os.makedirs(os.path.join(dest_root, 'test/'))
    else:
        # Wipe manifest contents first
        open(os.path.join(dest_root, "manifest_train.json"), 'w').close()
        open(os.path.join(dest_root, "manifest_val.json"), 'w').close()
        open(os.path.join(dest_root, "manifest_test.json"), 'w').close()

    file_count = 0

    for data_set in ['LDC2004S13-Part1', 'LDC2005S13-Part2']:
        print(f"\n\nWorking on dataset: {data_set}")
        file_count = __process_data(
            os.path.join(audio_root, data_set),
            os.path.join(transcript_root, data_set),
            dest_root,
            min_slice_duration,
            file_count,
            keep_low_conf,
            rem_noises,
            emojify,
        )

        print(f"Total file count so far: {file_count}")


if __name__ == "__main__":
    main()
