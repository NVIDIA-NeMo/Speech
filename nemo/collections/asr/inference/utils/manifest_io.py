# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os

import librosa

from nemo.collections.asr.inference.utils.constants import DEFAULT_OUTPUT_DIR_NAME
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.parts.preprocessing.manifest import get_full_path


def make_abs_path(path: str) -> str:
    """
    Make a path absolute
    Args:
        path: (str) Path to the file or folder
    Returns:
        (str) Absolute path
    """
    path = path.strip()
    if not path:
        raise ValueError("Path cannot be empty")
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def get_audio_filepaths(audio_file: str, sort_by_duration: bool = True) -> list[str]:
    """
    Get audio filepaths from a folder or a single audio file
    Args:
        audio_file: (str) Path to the audio file, folder or manifest file
        sort_by_duration: (bool) If True, sort the audio files by duration from shortest to longest
    Returns:
        (list[str]) List of audio filepaths
    """
    audio_file = audio_file.strip()
    audio_file = make_abs_path(audio_file)
    if os.path.isdir(audio_file):
        filepaths = filter(lambda x: x.endswith(".wav"), os.listdir(audio_file))
        filepaths = [os.path.join(audio_file, x) for x in filepaths]
    elif audio_file.endswith(".wav"):
        filepaths = [audio_file]
    elif audio_file.endswith((".json", ".jsonl")):
        manifest = read_manifest(audio_file)
        filepaths = [get_full_path(entry["audio_filepath"], audio_file) for entry in manifest]
    else:
        raise ValueError(f"audio_file `{audio_file}` need to be folder, audio file or manifest file")

    if sort_by_duration:
        durations = [librosa.get_duration(path=audio_filepath) for audio_filepath in filepaths]
        filepaths_with_durations = list(zip(filepaths, durations))
        filepaths_with_durations.sort(key=lambda x: x[1])
        filepaths = [x[0] for x in filepaths_with_durations]
    return filepaths


def get_stem(file_path: str) -> str:
    """
    Get the stem of a file path
    Args:
        file_path: (str) Path to the file
    Returns:
        (str) Filename with extension
    """
    return file_path.split('/')[-1]


def dump_output(output: dict, output_filename: str, output_dir: str | None = None) -> None:
    """
    Dump the transcriptions to a output file
    Args:
        output (dict): Pipeline output, structured as {stream_id: {"text": str, "segments": list}}
        output_filename: (str) Path to the output file
        output_dir: (str | None) Path to the output directory, if None, will write at the same level as the output file
    """
    if output_dir is None:
        # Create default output directory, if not provided
        output_dir = os.path.dirname(output_filename)
        output_dir = os.path.join(output_dir, DEFAULT_OUTPUT_DIR_NAME)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_filename, 'w') as fout:
        for stream_id, data in sorted(output.items(), key=lambda x: x[0]):
            audio_filepath = data["audio_filepath"]
            text = data["text"]
            segments = data["segments"]
            stem = get_stem(audio_filepath)
            stem = os.path.splitext(stem)[0]
            json_filepath = os.path.join(output_dir, f"{stem}.json")
            json_filepath = make_abs_path(json_filepath)
            with open(json_filepath, 'w') as json_fout:
                for segment in segments:
                    json_line = json.dumps(segment.to_dict(), ensure_ascii=False)
                    json_fout.write(f"{json_line}\n")

            item = {"audio_filepath": audio_filepath, "text": text, "json_filepath": json_filepath}
            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()


def calculate_duration(audio_filepaths: list[str]) -> float:
    """
    Calculate the duration of the audio files
    Args:
        audio_filepaths: (list[str]) List of audio filepaths
    Returns:
        (float) Total duration of the audio files
    """
    total_duration = 0
    for audio_filepath in audio_filepaths:
        duration = librosa.get_duration(path=audio_filepath)
        total_duration += duration
    return total_duration
