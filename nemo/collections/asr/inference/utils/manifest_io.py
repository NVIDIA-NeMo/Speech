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


def get_audio_filepaths(audio_file: str, sort_by_duration: bool = True) -> tuple[list[str], list[dict] | None]:
    """
    Get audio filepaths from a folder or a single audio file
    Args:
        audio_file: (str) Path to the audio file, folder or manifest file
        sort_by_duration: (bool) If True, sort the audio files by duration from shortest to longest
    Returns:
        (list[str], list[dict] | None) List of audio filepaths and manifest
    """
    audio_file = audio_file.strip()
    audio_file = make_abs_path(audio_file)
    manifest = None
    if os.path.isdir(audio_file):
        filepaths = filter(lambda x: x.endswith(".wav"), os.listdir(audio_file))
        filepaths = [os.path.join(audio_file, x) for x in filepaths]
    elif audio_file.endswith(".wav"):
        filepaths = [audio_file]
    elif audio_file.endswith((".json", ".jsonl")):
        manifest = read_manifest(audio_file)
        filepaths = [get_full_path(entry["audio_filepath"], audio_file) for entry in manifest]
        # we need to make audio_filepath in the manifest consistent, since this consistency is assumed further
        # in `dump_output` function
        for record, filepath in zip(manifest, filepaths):
            record["audio_filepath"] = filepath
    else:
        raise ValueError(f"audio_file `{audio_file}` need to be folder, audio file or manifest file")

    if sort_by_duration:
        indices = list(range(len(filepaths)))
        durations = [librosa.get_duration(path=filepaths[i]) for i in indices]
        indices_with_durations = list(zip(indices, durations))
        indices_with_durations.sort(key=lambda x: x[1])
        filepaths = [filepaths[i] for i, duration in indices_with_durations]
        if manifest is not None:
            # keep manifest in the same order as filepaths for consistency
            manifest = [manifest[i] for i, duration in indices_with_durations]
    return filepaths, manifest


def get_stem(file_path: str) -> str:
    """
    Get the stem of a file path
    Args:
        file_path: (str) Path to the file
    Returns:
        (str) Filename with extension
    """
    return file_path.split('/')[-1]


def dump_output(
    output: dict, output_filename: str, output_dir: str | None = None, manifest: list[dict] | None = None
) -> None:
    """
    Dump the transcriptions to a output file
    Args:
        output (dict): Pipeline output, structured as {stream_id: {"text": str, "segments": list}}
        output_filename: (str) Path to the output file
        output_dir: (str | None) Path to the output directory, if None, will write at the same level as the output file
        manifest: (list[dict] | None) Original manifest to copy extra fields from
    """
    if output_dir is None:
        # Create default output directory, if not provided
        output_dir = os.path.dirname(output_filename)
        output_dir = os.path.join(output_dir, DEFAULT_OUTPUT_DIR_NAME)

    manifest_index = None
    if manifest is not None:
        manifest_index = {entry["audio_filepath"]: i for i, entry in enumerate(manifest)}

    os.makedirs(output_dir, exist_ok=True)
    with open(output_filename, 'w') as fout:
        for stream_id, data in sorted(output.items(), key=lambda x: x[0]):
            audio_filepath = data["audio_filepath"]
            text = data["text"]
            translation = data["translation"]
            segments = data["segments"]
            stem = get_stem(audio_filepath)
            stem = os.path.splitext(stem)[0]
            json_filepath = os.path.join(output_dir, f"{stem}.json")
            json_filepath = make_abs_path(json_filepath)
            with open(json_filepath, 'w') as json_fout:
                for segment in segments:
                    json_line = json.dumps(segment.to_dict(), ensure_ascii=False)
                    json_fout.write(f"{json_line}\n")

            item = {
                "audio_filepath": audio_filepath,
                "pred_text": text,
                "pred_translation": translation,
                "json_filepath": json_filepath,
            }

            if manifest_index is not None:
                for key in manifest[manifest_index[audio_filepath]]:
                    if key not in item:
                        item[key] = manifest[manifest_index[audio_filepath]][key]

            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()


def calculate_duration(audio_filepaths: list[str]) -> tuple[float, dict[str, float]]:
    """
    Calculate the duration of the audio files
    Args:
        audio_filepaths: (list[str]) List of audio filepaths
    Returns:
        (float) Total duration of the audio files
        (dict[str, float]) Dictionary containing the duration of each audio file
    """
    total_duration = 0
    durations = {}
    for audio_filepath in audio_filepaths:
        duration = librosa.get_duration(path=audio_filepath)
        total_duration += duration
        durations[audio_filepath] = duration
    return total_duration, durations
