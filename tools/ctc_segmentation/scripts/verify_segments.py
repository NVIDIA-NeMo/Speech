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

import argparse
import os
import re
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Compare alignment segments generated with different window sizes")
parser.add_argument(
    "--base_dir",
    default="output",
    type=str,
    required=True,
    help="Path to directory with 'logs' and 'segments' folders generated during the segmentation step",
)


if __name__ == "__main__":
    args = parser.parse_args()
    segments_dir = os.path.join(args.base_dir, "segments")
    if not os.path.exists(segments_dir):
        raise ValueError(f"'segments' directory was not found at {args.base_dir}.")

    all_files = Path(segments_dir).glob("*_segments.txt")
    all_alignment_files = {}
    for file in all_files:
        base_name = re.sub(r"^\d+_", "", file.name)
        if base_name not in all_alignment_files:
            all_alignment_files[base_name] = []
        all_alignment_files[base_name].append(file)

    verified_dir = os.path.join(args.base_dir, "verified_segments")
    os.makedirs(verified_dir, exist_ok=True)

    def readlines(file):
        with open(file, "r") as f:
            lines = f.readlines()
        return lines

    stats = {}
    for part, alignment_files in all_alignment_files.items():
        stats[part] = {}
        num_alignment_files = len(alignment_files)
        all_alignments = []
        for alignment in alignment_files:
            all_alignments.append(readlines(alignment))

        with open(os.path.join(verified_dir, part), "w") as f:
            num_segments = len(all_alignments[0])
            stats[part]["Original number of segments"] = num_segments
            stats[part]["Verified segments"] = 0
            stats[part]["Original Duration, min"] = 0
            stats[part]["Verified Duration, min"] = 0

            for i in range(num_segments):
                line = all_alignments[0][i]
                valid_line = True
                if i == 0:
                    duration = 0
                else:
                    info = line.split("|")[0].split()
                    duration = (float(info[1]) - float(info[0])) / 60
                stats[part]["Original Duration, min"] += duration
                for alignment in all_alignments:
                    if line != alignment[i]:
                        valid_line = False
                if valid_line:
                    f.write(line)
                    stats[part]["Verified segments"] += 1
                    stats[part]["Verified Duration, min"] += duration

    stats = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    stats["Number dropped"] = stats["Original number of segments"] - stats["Verified segments"]
    stats["Duration of dropped, min"] = round(stats["Original Duration, min"] - stats["Verified Duration, min"])
    stats["% dropped, min"] = round(stats["Duration of dropped, min"] / stats["Original number of segments"] * 100)
    stats["Misalignment present"] = stats["Number dropped"] > 0
    stats["Original Duration, min"] = round(stats["Original Duration, min"])
    stats["Verified Duration, min"] = round(stats["Verified Duration, min"])
    stats.loc["Total"] = stats.sum()

    stats_file = os.path.join(args.base_dir, "alignment_summary.csv")
    stats.to_csv(stats_file, index=False)
    print(stats)
    print(f"Alignment summary saved to {stats_file}")
