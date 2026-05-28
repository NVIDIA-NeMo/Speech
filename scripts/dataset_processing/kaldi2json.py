# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd

from nemo.utils import logging


def main():
    parser = argparse.ArgumentParser(description="Convert kaldi data folder to manifest.json")
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="data in kaldi format",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="path to store the manifest file",
    )
    parser.add_argument(
        "--with_aux_data",
        default=False,
        action="store_true",
        help="whether to include auxiliary data in the manifest",
    )
    args = parser.parse_args()

    kaldi_folder = args.data_dir
    required_data = {
        "audio_filepath": os.path.join(kaldi_folder, "wav.scp"),
        "duration": os.path.join(kaldi_folder, "segments"),
        "text": os.path.join(kaldi_folder, "text"),
    }
    aux_data = {
        "speaker": os.path.join(kaldi_folder, "utt2spk"),
        "gender": os.path.join(kaldi_folder, "utt2gender"),
    }
    output_names = list(required_data.keys())

    # check if required files exist
    for name, file in required_data.items():
        if not os.path.exists(file):
            raise ValueError(f"{os.path.basename(file)} is not in {kaldi_folder}.")

    # read wav.scp
    wavscp = pd.read_csv(required_data["audio_filepath"], sep=" ", header=None)
    if wavscp.shape[1] > 2:
        logging.warning(
            f"""More than two columns in 'wav.scp': {wavscp.shape[1]}.
            Maybe it contains pipes? Pipe processing can be slow at runtime."""
        )
        wavscp = pd.read_csv(
            required_data["audio_filepath"],
            sep="^([^ ]+) ",
            engine="python",
            header=None,
            usecols=[1, 2],
            names=["wav_label", "audio_filepath"],
        )
    else:
        wavscp = wavscp.rename(columns={0: "wav_label", 1: "audio_filepath"})

    # read text
    text = pd.read_csv(
        required_data["text"],
        sep="^([^ ]+) ",
        engine="python",
        header=None,
        usecols=[1, 2],
        names=["label", "text"],
    )

    # read segments
    segments = pd.read_csv(
        required_data["duration"],
        sep=" ",
        header=None,
        names=["label", "wav_label", "offset", "end"],
    )
    # add offset if needed
    if len(segments.offset) > len(segments.offset[segments.offset == 0.0]):
        logging.info("Adding offset field.")
        output_names.insert(2, "offset")
    segments["duration"] = (segments.end - segments.offset).round(decimals=3)

    # merge data
    wav_segments_text = pd.merge(
        pd.merge(segments, wavscp, how="inner", on="wav_label"),
        text,
        how="inner",
        on="label",
    )

    if args.with_aux_data:
        # check if auxiliary data is present
        for name, aux_file in aux_data.items():
            if os.path.exists(aux_file):
                logging.info(f"Adding info from '{os.path.basename(aux_file)}'.")
                wav_segments_text = pd.merge(
                    wav_segments_text,
                    pd.read_csv(aux_file, sep=" ", header=None, names=["label", name]),
                    how="left",
                    on="label",
                )
                output_names.append(name)
            else:
                logging.info(f"'{os.path.basename(aux_file)}' does not exist. Skipping ...")

    # write data to .json
    entries = wav_segments_text[output_names].to_dict(orient="records")
    with open(args.manifest, "w", encoding="utf-8") as fout:
        for m in entries:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
