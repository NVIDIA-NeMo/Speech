# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os

"""
Create sample JSONL file for functional testing. Each line contains a dictionary
with a single element "text" for storing data.
"""


def create_sample_jsonl(output_file: str, overwrite: bool = False):
    """Create sample JSONL."""
    if os.path.isfile(output_file) and not overwrite:
        print(f"File {output_file} exists and overwrite flag is not set so exiting.")
        return

    texts = [
        "Sample data for functional tests",
        "Once upon a time, in the middle of a dense forest, there was a small house, where lived a pretty little girl "
        "named Little Red Riding Hood.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
        "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
        "nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit "
        "anim id est laborum...",
        "Next please!",
        "¡H E L L O   W O R L D!",
        "Korzystając z okazji chciałbym pozdrowić całą moją rodzinę i przyjaciół",
    ]
    print(f"Writing {len(texts)} line(s) to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode="w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text}, f)
            f.write("\n")
    print("OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create sample JSONL file.")
    parser.add_argument("--output_file", help="Output file name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file if it exists")
    args = parser.parse_args()
    create_sample_jsonl(args.output_file)
