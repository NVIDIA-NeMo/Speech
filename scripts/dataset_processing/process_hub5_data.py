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

# This script is heavily derived from the Patter HUB5 processing script written
# by Ryan Leary

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import namedtuple
from math import ceil, floor
from operator import attrgetter

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Prepare HUB5 data for training/eval")
parser.add_argument(
    "--data_root",
    default=None,
    type=str,
    required=True,
    help="The path to the root LDC HUB5 dataset directory.",
)
parser.add_argument(
    "--dest_root",
    default=None,
    type=str,
    required=True,
    help="Path to the destination root directory for processed files.",
)

# Optional arguments
parser.add_argument(
    "--min_slice_duration",
    default=10.0,
    type=float,
    help="Minimum audio slice duration after processing.",
)

args = parser.parse_args()

StmUtterance = namedtuple(
    'StmUtterance',
    [
        'filename',
        'channel',
        'speaker_id',
        'begin',
        'end',
        'label',
        'transcript',
    ],
)
STM_LINE_FMT = re.compile(r"^(\w+)\s+(\w+)\s+(\w+)\s+([0-9.]+)\s+([0-9.]+)\s+(<.*>)?\s+(.+)$")

# Transcription errors and their fixes
TRANSCRIPT_BUGS = {"en_4622-B-12079-12187": "KIND OF WEIRD BUT"}


def get_utt_id(segment):
    """
    Gives utterance IDs in a form like: en_4156-a-36558-37113
    """
    return "{}-{}-{}-{}".format(
        segment.filename,
        segment.channel,
        int(segment.begin * 100),
        int(segment.end * 100),
    )


def convert_utterances(sph_path, wav_path):
    """
    Converts a sphere audio file to wav.
    """
    cmd = ["sph2pipe", "-f", "wav", "-p", sph_path, wav_path]
    subprocess.run(cmd)


def create_wavs(data_root, dest_root):
    """
    Converts the English sph files to wav using sph2pipe.
    """
    sph_root = os.path.join(data_root, "hub5e_00", "english")
    sph_list = glob.glob(os.path.join(sph_root, "*.sph"))

    # Iterate over each sphere file and conver to wav
    for sph_path in tqdm(sph_list, desc="Converting to wav", unit="file"):
        sph_name, _ = os.path.splitext(os.path.basename(sph_path))
        wav_path = os.path.join(dest_root, 'full_audio_wav', sph_name + ".wav")
        cmd = ["sph2pipe", "-f", "wav", "-p", sph_path, wav_path]
        subprocess.run(cmd)


def process_transcripts(dataset_root):
    """
    Reads in transcripts for each audio segment and processes them.
    """
    stm_path = os.path.join(
        dataset_root,
        "2000_hub5_eng_eval_tr",
        "reference",
        "hub5e00.english.000405.stm",
    )
    results = []
    chars = set()

    with open(stm_path, "r") as fh:
        for line in fh:
            # lines with ';;' are comments
            if line.startswith(";;"):
                continue

            if "IGNORE_TIME_SEGMENT_" in line:
                continue
            line = line.replace("<B_ASIDE>", "").replace("<E_ASIDE>", "")
            line = line.replace("(%HESITATION)", "UH")
            line = line.replace("-", "")
            line = line.replace("(%UH)", "UH")
            line = line.replace("(%AH)", "UH")
            line = line.replace("(", "").replace(")", "")

            line = line.lower()

            m = STM_LINE_FMT.search(line.strip())
            utt = StmUtterance(*m.groups())

            # Convert begin/end times to float
            utt = utt._replace(begin=float(utt.begin))
            utt = utt._replace(end=float(utt.end))

            # Check for utterance in dict of transcript mistakes
            transcript_update = TRANSCRIPT_BUGS.get(get_utt_id(utt))
            if transcript_update is not None:
                utt = utt._replace(transcript=transcript_update)

            results.append(utt)
            chars.update(list(utt.transcript))
    return results, chars


def write_one_segment(dest_root, speaker_id, count, audio, sr, duration, transcript):
    """
    Writes out one segment of audio, and writes its corresponding transcript
    in the manifest.

    Args:
        dest_root: the path to the output directory root
        speaker_id: ID of the speaker, used in file naming
        count: number of segments from this speaker so far
        audio: the segment's audio data
        sr: sample rate of the audio
        duration: duration of the audio
        transcript: the corresponding transcript
    """
    audio_path = os.path.join(dest_root, "audio", f"{speaker_id}_{count:03}.wav")

    manifest_path = os.path.join(dest_root, "manifest_hub5.json")

    # Write audio
    wavfile.write(audio_path, sr, audio)

    # Write transcript
    transcript = {
        "audio_filepath": audio_path,
        "duration": duration,
        "text": transcript,
    }
    with open(manifest_path, 'a') as f:
        json.dump(transcript, f)
        f.write('\n')


def segment_audio(info_list, dest_root, min_slice_duration):
    """
    Combines audio into >= min_slice_duration segments of the same speaker,
    and writes the combined transcripts into a manifest.

    Args:
        info_list: list of StmUtterance objects with transcript information.
        dest_root: path to output destination
        min_slice_duration: min number of seconds per output audio slice
    """
    info_list = sorted(info_list, key=attrgetter('speaker_id', 'begin'))

    prev_id = None  # For checking audio concatenation
    id_count = 0

    sample_rate, audio_data = None, None
    transcript_buffer = ''
    audio_buffer = []
    buffer_duration = 0.0

    # Iterate through utterances to build segments
    for info in info_list:
        if info.speaker_id != prev_id:
            # Scrap the remainder in the buffers and start next segment
            prev_id = info.speaker_id
            id_count = 0

            sample_rate, audio_data = wavfile.read(os.path.join(dest_root, 'full_audio_wav', info.filename + '.wav'))
            transcript_buffer = ''
            audio_buffer = []
            buffer_duration = 0.0

        # Append utterance info to buffers
        transcript_buffer += info.transcript
        channel = 0 if info.channel.lower() == 'a' else 1
        audio_buffer.append(
            audio_data[
                floor(info.begin * sample_rate) : ceil(info.end * sample_rate),
                channel,
            ]
        )
        buffer_duration += info.end - info.begin

        if buffer_duration < min_slice_duration:
            transcript_buffer += ' '
        else:
            # Write out segment and transcript
            id_count += 1
            write_one_segment(
                dest_root,
                info.speaker_id,
                id_count,
                np.concatenate(audio_buffer, axis=0),
                sample_rate,
                buffer_duration,
                transcript_buffer,
            )

            transcript_buffer = ''
            audio_buffer = []
            buffer_duration = 0.0


def main():
    data_root = args.data_root
    dest_root = args.dest_root

    min_slice_duration = args.min_slice_duration

    if not os.path.exists(os.path.join(dest_root, 'full_audio_wav')):
        os.makedirs(os.path.join(dest_root, 'full_audio_wav'))
    if not os.path.exists(os.path.join(dest_root, 'audio')):
        os.makedirs(os.path.join(dest_root, 'audio'))

    # Create/wipe manifest contents
    open(os.path.join(dest_root, "manifest_hub5.json"), 'w').close()

    # Convert full audio files from .sph to .wav
    create_wavs(data_root, dest_root)

    # Get each audio transcript from transcript file
    info_list, chars = process_transcripts(data_root)

    print("Writing out vocab file", file=sys.stderr)
    with open(os.path.join(dest_root, "vocab.txt"), 'w') as fh:
        for x in sorted(list(chars)):
            fh.write(x + "\n")

    # Segment the audio data
    print("Segmenting audio and writing manifest")
    segment_audio(info_list, dest_root, min_slice_duration)


if __name__ == '__main__':
    main()
