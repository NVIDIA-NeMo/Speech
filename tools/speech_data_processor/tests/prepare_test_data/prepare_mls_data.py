# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Will take the donwloaded tar file and create a version with only X entries."""

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing MLS test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument("--num_entries", default=200, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for split in ["train", "dev", "test"]:
            os.makedirs(tmpdir_path / split / "audio")
            transcript_path = Path(args.extracted_data_path) / split / "transcripts.txt"
            with open(transcript_path, "rt", encoding="utf8") as fin, open(
                tmpdir_path / split / "transcripts.txt", "wt", encoding="utf8"
            ) as fout:
                for idx, line in enumerate(fin):
                    if idx == args.num_entries:
                        break
                    utt_id = line.split("\t", 1)[0]
                    src_flac_path = os.path.join(
                        args.extracted_data_path, split, "audio", *utt_id.split("_")[:2], utt_id + ".flac"
                    )
                    fout.write(line)
                    tgt_flac_dir = os.path.join(tmpdir_path, split, "audio", *utt_id.split("_")[:2])
                    os.makedirs(tgt_flac_dir, exist_ok=True)
                    shutil.copy(src_flac_path, os.path.join(tgt_flac_dir, utt_id + ".flac"))
        with tarfile.open(os.path.join(args.test_data_folder, "data.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname="data")
