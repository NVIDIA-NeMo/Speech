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
import argparse
import glob
import json
import os
from typing import List


from filelock import FileLock

from nemo.utils import logging


def create_transcribed_shard_manifests(prediction_filepaths: List[str]) -> List[str]:
    """
    Creates transcribed shard manifest files by processing predictions and organizing them by shard ID.

    This function reads a `predictions_all.json` file from each given directory, organizes the data by
    shard IDs, and writes the entries to separate shard manifest files. For each shard, the `pred_text`
    field is updated as the main transcription (`text`), and the original transcription (`text`) is
    stored as `orig_text`.

    Args:
        prediction_filepaths (List[str]): A list of file paths to directories containing
            `predictions_all.json` files with prediction data, including shard IDs.

    Returns:
        List[str]: A list of file paths to the combined manifest files (`transcribed_manifest__OP_0..CL_.json`)
        created for each directory.
    """
    all_manifest_filepaths = []
    for prediction_filepath in prediction_filepaths:
        max_shard_id = 0
        shard_data = {}
        full_path = os.path.join(prediction_filepath, "predictions_all.json")
        with open(full_path, 'r') as f:
            for line in f.readlines():
                data_entry = json.loads(line)
                shard_id = data_entry.get("shard_id")
                if max_shard_id < shard_id:
                    max_shard_id = shard_id
                if shard_id not in shard_data:
                    shard_data[shard_id] = []
                shard_data[shard_id].append(data_entry)
        for shard_id, entries in shard_data.items():
            output_filename = os.path.join(prediction_filepath, f"transcribed_manifest_{shard_id}.json")
            with open(output_filename, 'w') as f:
                for data_entry in entries:
                    if data_entry['audio_filepath'].endswith(".wav"):
                        if 'text' in data_entry:
                            data_entry['orig_text'] = data_entry.pop('text')
                        data_entry['text'] = data_entry.pop('pred_text')
                        json.dump(data_entry, f, ensure_ascii=False)
                        f.write("\n")
        shard_manifest_filepath = os.path.join(
            prediction_filepath, f"transcribed_manifest__OP_0..{max_shard_id}_CL_.json"
        )
        all_manifest_filepaths.append(shard_manifest_filepath)
    return all_manifest_filepaths


def create_transcribed_manifests(prediction_filepaths: List[str]) -> List[str]:
    """
    Creates updated transcribed manifest files by processing predictions.

    This function reads prediction files (`predictions_all.json`) from the provided directories,
    updates the transcription data by renaming the `pred_text` field to `text`, and stores the
    original `text` field as `orig_text`. The updated data is written to new transcribed manifest
    files (`transcribed_manifest.json`) in each directory.

    Args:
        prediction_filepaths (List[str]): A list of file paths to directories containing
            prediction files (`predictions_all.json`).

    Returns:
        List[str]: A list of file paths to the newly created transcribed manifest files
        (`transcribed_manifest.json`).
    """
    all_manifest_filepaths = []
    for prediction_filepath in prediction_filepaths:
        prediction_name = os.path.join(prediction_filepath, "predictions_all.json")
        transcripted_name = os.path.join(prediction_filepath, f"transcribed_manifest.json")

        # Open and read the original predictions_all.json file
        with open(transcripted_name, 'w', encoding='utf-8') as f:
            with open(prediction_name, 'r', encoding='utf-8') as pred_f:

                for line in pred_f.readlines():
                    data_entry = json.loads(line)
                    if 'text' in data_entry:
                        data_entry['orig_text'] = data_entry.pop('text')
                    data_entry['text'] = data_entry.pop('pred_text')
                    json.dump(data_entry, f, ensure_ascii=False)
                    f.write("\n")
            # Append the path of the new manifest file to the list
            all_manifest_filepaths.append(transcripted_name)

    return all_manifest_filepaths


def write_sampled_shard_transcriptions(manifest_filepaths: List[str]) -> List[List[str]]:
    """
    Updates transcriptions by merging predicted shard data and transcribed manifest data.
    This function processes prediction and transcribed manifest files, merges them
    by matching the shard_id and audio file paths. For each shard, the corresponding
    data entries are written to a new file.
    Args:
        manifest_filepaths (List[str]): A list of file paths to directories containing
            prediction and transcribed manifest files.
    Returns:
        List[List[str]]: A list of lists containing the file paths to the generated
            transcribed shard manifest files.
    """
    all_manifest_filepaths = []

    # Process each prediction directory
    for prediction_filepath in manifest_filepaths:
        predicted_shard_data = {}
        # Collect entries from prediction files based on shard id
        prediction_path = os.path.join(prediction_filepath, "predictions_all.json")
        with open(prediction_path, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                shard_id = data_entry.get("shard_id")
                audio_filepath = data_entry['audio_filepath']
                predicted_shard_data.setdefault(shard_id, {})[audio_filepath] = data_entry
    max_shard_id = 0
    for full_path in glob.glob(os.path.join(prediction_filepath, f"transcribed_manifest_[0-9]*.json")):
        all_data_entries = []
        with open(full_path, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                shard_id = data_entry.get("shard_id")
                max_shard_id = max(max_shard_id, shard_id)
                all_data_entries.append(data_entry)
        # Write the merged data to a new manifest file keeping new transcriptions
        output_filename = os.path.join(prediction_filepath, f"transcribed_manifest_{shard_id}.json")
        with open(output_filename, 'w') as f:
            for data_entry in all_data_entries:
                audio_filepath = data_entry['audio_filepath']
                # Escape duplicated audio files that end with *dup
                if audio_filepath.endswith(".wav"):
                    if shard_id in predicted_shard_data and audio_filepath in predicted_shard_data[shard_id]:
                        predicted_data_entry = predicted_shard_data[shard_id][audio_filepath]
                        if 'text' in predicted_data_entry:
                            predicted_data_entry['orig_text'] = predicted_data_entry.pop('text')
                        if "pred_text" in predicted_data_entry:
                            predicted_data_entry['text'] = predicted_data_entry.pop('pred_text')
                        json.dump(predicted_data_entry, f, ensure_ascii=False)
                    else:
                        json.dump(data_entry, f, ensure_ascii=False)
                    f.write("\n")

    shard_manifest_filepath = os.path.join(prediction_filepath, f"transcribed_manifest__OP_0..{max_shard_id}_CL_.json")
    all_manifest_filepaths.append([shard_manifest_filepath])

    return all_manifest_filepaths


def write_sampled_transcriptions(manifest_filepaths: List[str]) -> List[str]:
    """
    Updates transcriptions by merging predicted data with transcribed manifest data.

    This function processes prediction and transcribed manifest files within given directories.
    It matches audio file paths to update transcriptions with predictions, ensuring each audio file
    is properly transcribed. The updated data is written to the transcribed manifest file.

    Args:
        manifest_filepaths (List[str]): A list of file paths to directories containing
            the prediction file (`predictions_all.json`) and the transcribed manifest file
            (`transcribed_manifest.json`).

    Returns:
        List[str]: A list of file paths to the updated transcribed manifest files.
    """
    all_manifest_filepaths = []
    for prediction_filepath in manifest_filepaths:
        predicted_data = {}
        prediction_path = os.path.join(prediction_filepath, "predictions_all.json")
        with open(prediction_path, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                path = data_entry['audio_filepath']
                predicted_data[path] = data_entry

        full_path = os.path.join(prediction_filepath, f"transcribed_manifest.json")
        all_data_entries = []
        with open(full_path, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                all_data_entries.append(data_entry)

        output_filename = os.path.join(prediction_filepath, f"transcribed_manifest.json")
        with open(output_filename, 'w') as f:
            for data_entry in all_data_entries:
                audio_filepath = data_entry['audio_filepath']
                if audio_filepath.endswith(".wav"):
                    if audio_filepath in predicted_data:
                        predicted_data_entry = predicted_data[audio_filepath]
                        if 'text' in predicted_data_entry:
                            predicted_data_entry['orig_text'] = predicted_data_entry.pop('text')
                        predicted_data_entry['text'] = predicted_data_entry.pop('pred_text')
                        json.dump(predicted_data_entry, f, ensure_ascii=False)
                        f.write("\n")
                    else:
                        json.dump(data_entry, f, ensure_ascii=False)
                        f.write("\n")
        all_manifest_filepaths.append(output_filename)
    return all_manifest_filepaths


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))  # Default to 0 if not set

    parser = argparse.ArgumentParser(description="Script to create or write transcriptions")
    parser.add_argument("--is_tarred", action="store_true", help="If true, processes tarred manifests")
    parser.add_argument("--full_pass", action="store_true", help="If true, processes full pass manifests")
    parser.add_argument(
        "--prediction_filepaths",
        type=str,
        nargs='+',  # Accepts one or more values as a list
        required=True,
        help="Paths to one or more inference config YAML files.",
    )

    args = parser.parse_args()

    lock_dir = os.path.dirname(args.prediction_filepaths[0])
    lock_file = lock_dir + "/my_script.lock"

    with FileLock(lock_file):
        if rank == 0:
            if args.is_tarred:
                result = (
                    write_sampled_shard_transcriptions(args.prediction_filepaths)
                    if not args.full_pass
                    else create_transcribed_shard_manifests(args.prediction_filepaths)
                )
            else:
                result = (
                    write_sampled_transcriptions(args.prediction_filepaths)
                    if not args.full_pass
                    else create_transcribed_manifests(args.prediction_filepaths)
                )

    # Remove the lock file after the FileLock context is exited
    if os.path.exists(lock_file):
        os.remove(lock_file)
