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

import json
from pathlib import Path


def get_batch_starts_ends(manifest_filepath, batch_size):
    """
    Get the start and end ids of the lines we will use for each 'batch'.
    """

    with open(manifest_filepath, 'r') as f:
        num_lines_in_manifest = sum(1 for _ in f)

    starts = [x for x in range(0, num_lines_in_manifest, batch_size)]
    ends = [x - 1 for x in starts]
    ends.pop(0)
    ends.append(num_lines_in_manifest)

    return starts, ends


def is_entry_in_any_lines(manifest_filepath, entry):
    """
    Returns True if entry is a key in any of the JSON lines in manifest_filepath
    """

    entry_in_manifest = False

    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry in data:
                entry_in_manifest = True

    return entry_in_manifest


def is_entry_in_all_lines(manifest_filepath, entry):
    """
    Returns True is entry is a key in all of the JSON lines in manifest_filepath.
    """
    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry not in data:
                return False

    return True


def get_manifest_lines_batch(manifest_filepath, start, end):
    manifest_lines_batch = []
    with open(manifest_filepath, "r", encoding="utf-8-sig") as f:
        for line_i, line in enumerate(f):
            if line_i >= start and line_i <= end:
                data = json.loads(line)
                if "text" in data:
                    # remove any BOM, any duplicated spaces, convert any
                    # newline chars to spaces
                    data["text"] = data["text"].replace("\ufeff", "")
                    data["text"] = " ".join(data["text"].split())

                    # Replace any horizontal ellipses with 3 separate periods.
                    # The tokenizer will do this anyway. But making this replacement
                    # now helps avoid errors when restoring punctuation when saving
                    # the output files
                    data["text"] = data["text"].replace("\u2026", "...")

                if not Path(data['audio_filepath']).exists():
                    extended_path = Path(Path(manifest_filepath).parent, data['audio_filepath'])
                    if extended_path.exists():
                        data['audio_filepath'] = str(extended_path)
                    else:
                        raise FileNotFoundError(
                            f"Audio file {data['audio_filepath']} not found in {manifest_filepath}"
                        )

                manifest_lines_batch.append(data)

            if line_i == end:
                break
    return manifest_lines_batch
