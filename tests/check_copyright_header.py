# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""A script to check that copyright headers exists"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

EXCLUSIONS = ["scripts/get_commonvoice_data.py"]


def get_top_comments(_data):
    # Get all lines where comments should exist
    lines_to_extract = []
    for i, line in enumerate(_data):
        # If empty line, skip
        if line in ["", "\n", "", "\r", "\r\n"]:
            continue
        # If it is a comment line, we should get it
        if line.startswith("#"):
            lines_to_extract.append(i)
        # Assume all copyright headers occur before any import statements not enclosed in a comment block
        elif "import" in line:
            break

    comments = []
    for line in lines_to_extract:
        comments.append(_data[line])

    return comments


def main():
    parser = argparse.ArgumentParser(description="Usage for copyright header insertion script")
    parser.add_argument(
        '--dir',
        help='Path to source files to add copyright header to. Will recurse through subdirectories',
        required=True,
        type=str,
    )
    args = parser.parse_args()

    current_year = int(datetime.today().year)
    starting_year = 2020
    python_header_path = "tests/py_cprheader.txt"
    with open(python_header_path, 'r', encoding='utf-8') as original:
        pyheader = original.read().split("\n")
        pyheader_lines = len(pyheader)

    problematic_files = []

    for filename in Path(args.dir).rglob('*.py'):
        if str(filename) in EXCLUSIONS:
            continue
        with open(str(filename), 'r', encoding='utf-8') as original:
            data = original.readlines()

        data = get_top_comments(data)
        if len(data) < pyheader_lines:
            print(f"{filename} has less header lines than the copyright template")
            problematic_files.append(filename)
            continue

        found = False
        for i, line in enumerate(data):
            if re.search(re.compile("Copyright.*NVIDIA.*", re.IGNORECASE), line):
                # if re.search(re.compile("Copyright.*", re.IGNORECASE), line):
                found = True
                # Check 1st line manually
                year_good = False
                for year in range(starting_year, current_year + 1):
                    year_line = pyheader[0].format(CURRENT_YEAR=year)
                    if year_line in data[i]:
                        year_good = True
                        break
                    year_line_aff = year_line.split(".")
                    year_line_aff = year_line_aff[0] + " & AFFILIATES." + year_line_aff[1]
                    if year_line_aff in data[i]:
                        year_good = True
                        break
                if not year_good:
                    problematic_files.append(filename)
                    print(f"{filename} had an error with the year")
                    break
                while "opyright" in data[i]:
                    i += 1
                for j in range(1, pyheader_lines):
                    if pyheader[j] not in data[i + j - 1]:
                        problematic_files.append(filename)
                        print(f"{filename} missed the line: {pyheader[j]}")
                        break
            if found:
                break
        if not found:
            print(f"{filename} did not match the regex: `Copyright.*NVIDIA.*`")
            problematic_files.append(filename)

    if len(problematic_files) > 0:
        print("check_copyright_headers.py found the following files that might not have a copyright header:")
        for _file in problematic_files:
            print(_file)
        sys.exit(1)


if __name__ == '__main__':
    main()
