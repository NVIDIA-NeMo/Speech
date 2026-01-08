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

"""
Example usage:

```bash
python add_eob_labels.py /path/to/manifest/dir
```
where output will be saved in the same directory with `-eob` suffix added to the filename.
"""

import argparse
import json
from pathlib import Path
from string import punctuation

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Add `is_backchannel` labels to manifest files.")
parser.add_argument(
    "input_manifest",
    type=str,
    help="Path to the input manifest file to be cleaned.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Path to the output manifest file after cleaning.",
)
parser.add_argument(
    "-p",
    "--pattern",
    type=str,
    default="*.json",
    help="Pattern to match files in the input directory.",
)


def read_manifest(manifest_path):
    manifest = []
    with open(manifest_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                manifest.append(json.loads(line))
    return manifest


def write_manifest(manifest_path, manifest):
    with open(manifest_path, 'w') as f:
        for item in manifest:
            f.write(json.dumps(item) + '\n')


def clean_text(text):
    text = text.translate(str.maketrans('', '', punctuation)).lower().strip()
    valid_chars = "abcdefghijklmnopqrstuvwxyz'"
    text = ''.join([c for c in text if c in valid_chars or c.isspace() or c == "'"])
    return " ".join(text.split()).strip()


backchannel_phrases = [
    'absolutely',
    'ah',
    'all right',
    'alright',
    'but yeah',
    'definitely',
    'exactly',
    'go ahead',
    'good',
    'great',
    'great thanks',
    'ha ha',
    'hi',
    'i know',
    'i know right',
    'i see',
    'indeed',
    'interesting',
    'mhmm',
    'mhmm mhmm',
    'mhmm right',
    'mhmm yeah',
    'mhmm yes',
    'nice',
    'of course',
    'oh',
    'oh dear',
    'oh man',
    'oh okay',
    'oh wow',
    'oh yes',
    'ok',
    'ok thanks',
    'okay',
    'okay okay',
    'okay thanks',
    'perfect',
    'really',
    'right',
    'right exactly',
    'right right',
    'right yeah',
    'so yeah',
    'sounds good',
    'sure',
    'thank you',
    'thanks',
    "that's awesome",
    'thats right',
    'thats true',
    'true',
    'uh-huh',
    'uh-huh yeah',
    'uhhuh',
    'um-humm',
    'well',
    'what',
    'wow',
    'yeah',
    'yeah i know',
    'yeah i see',
    'yeah mhmm',
    'yeah okay',
    'yeah right',
    'yeah uh-huh',
    'yeah yeah',
    'yep',
    'yes',
    'yes please',
    'yes yes',
    'you know',
    "you're right",
]

backchannel_phrases_nopc = [clean_text(phrase) for phrase in backchannel_phrases]


def check_if_backchannel(text):
    """
    Check if the text is a backchannel phrase.
    """
    # Remove punctuation and convert to lowercase
    text = clean_text(text)
    # Check if the text is in the list of backchannel phrases
    return text in backchannel_phrases_nopc


def add_eob_labels(manifest_path):
    num_eob = 0
    manifest = read_manifest(manifest_path)
    for i, item in enumerate(manifest):
        text = item['text']
        # Check if the text is a backchannel phrase
        is_backchannel = check_if_backchannel(text)
        # Add the EOB label to the text
        if is_backchannel:
            item['is_backchannel'] = True
            num_eob += 1
        else:
            item['is_backchannel'] = False
        manifest[i] = item
    return manifest, num_eob


def main():
    args = parser.parse_args()
    input_manifest = Path(args.input_manifest)

    if input_manifest.is_dir():
        manifest_list = list(input_manifest.glob(args.pattern))
        if not manifest_list:
            raise ValueError(f"No files found in {input_manifest} matching pattern `{args.pattern}`")
    else:
        manifest_list = [input_manifest]

    if args.output is None:
        output_dir = input_manifest if input_manifest.is_dir() else input_manifest.parent
    else:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_num_eob = 0
    print(f"Processing {len(manifest_list)} manifest files...")
    for manifest_path in tqdm(manifest_list, total=len(manifest_list)):
        output_file = output_dir / f"{manifest_path.stem}-eob.json"
        new_manifest, num_eob = add_eob_labels(manifest_path)
        total_num_eob += num_eob
        write_manifest(output_file, new_manifest)
        print(f"Processed {manifest_path} and saved to {output_file}. Number of EOB labels added: {num_eob}")

    print(f"Total number of EOB labels added: {total_num_eob}")


if __name__ == "__main__":
    main()
