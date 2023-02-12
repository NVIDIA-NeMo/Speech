# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import multiprocessing as mp
from itertools import repeat
from pathlib import Path

import librosa
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.asr.parts.utils.vad_utils import get_frame_labels, load_speech_segments_from_rttm

"""
This script generates a manifest file for synthetic data generated using the NeMo multispeaker speech data simulator.
The audio created from the simulator can be used to train a VAD model using the manifest file contains the following fields:
The manifest file contains the following fields:
    
    audio_filepath (str): Path to audio file.
    offset (float): Offset in seconds for the start of the audio file.
    duration (float): Duration in seconds for the audio file.
    text (str): Transcription of the audio file.
    label (list): List of frame labels for the audio file.
    orig_sample_rate (int): Original sample rate of the audio file.
    vad_frame_unit_secs (float): Duration in seconds for each frame label.

Usage:
    python build_synthetic_vad_manifest.py \
        --input_dir /path/to/synthetic/data \
        --frame_length 0.04 \
        --output_file /path/to/output/manifest.json
"""


def generate_manifest_entry(inputs):
    """
    Generates a manifest entry for a single audio file. 
    This function is parallelized using multiprocessing.Pool.

    Args:
        inputs (tuple): Tuple containing audio file path and frame length in seconds.
            inputs[0]: 
                audio_filepath (str): Path to audio file.
            inputs[1]: 
                vad_frame_unit_secs (float): Duration in seconds for each frame label.

    Returns:
        entry (dict): Dictionary containing manifest entry.
    """
    audio_filepath, vad_frame_unit_secs = inputs
    audio_filepath = Path(audio_filepath)
    y, sr = librosa.load(str(audio_filepath))
    dur = librosa.get_duration(y=y, sr=sr)

    manifest_path = audio_filepath.parent / Path(f"{audio_filepath.stem}.json")
    audio_manifest = read_manifest(manifest_path)
    text = " ".join([x["text"] for x in audio_manifest])

    rttm_path = audio_filepath.parent / Path(f"{audio_filepath.stem}.rttm")
    segments = load_speech_segments_from_rttm(rttm_path)
    labels = get_frame_labels(segments, vad_frame_unit_secs, 0.0, dur)

    entry = {
        "audio_filepath": str(audio_filepath.absolute()),
        "offset": 0.0,
        "duration": dur,
        "text": text,
        "label": labels,
        "orig_sample_rate": sr,
        "vad_frame_unit_secs": vad_frame_unit_secs,
    }
    return entry


def main(args):
    wav_list = list(Path(args.input_dir).glob("*.wav"))
    print(f"Found {len(wav_list)} in directory: {args.input_dir}")

    inputs = zip(wav_list, repeat(args.frame_length))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        manifest_data = list(tqdm(pool.imap(generate_manifest_entry, inputs), total=len(wav_list)))

    write_manifest(args.output_file, manifest_data)
    print(f"Manifest saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", default=None, help="Path to directory containing synthetic data")
    parser.add_argument(
        "-l", "--frame_length", default=0.04, type=float, help="Duration in seconds for each frame label"
    )
    parser.add_argument("-o", "--output_file", default=None, help="Path to output manifest file")

    args = parser.parse_args()
    main(args)
