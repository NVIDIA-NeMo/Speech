# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# USAGE: python process_aishell2_data.py
#                   --audio_folder=<source data>
#                   --dest_folder=<where to store the results>
import argparse
import json
import os
import subprocess

parser = argparse.ArgumentParser(description="Processing Aishell2 Data")
parser.add_argument("--audio_folder", default=None, type=str, required=True, help="Audio (wav) data directory.")
parser.add_argument("--dest_folder", default=None, type=str, required=True, help="Destination directory.")
args = parser.parse_args()


def __process_data(data_folder: str, dst_folder: str):
    """
    To generate manifest
    Args:
        data_folder: source with wav files
        dst_folder: where manifest files will be stored
    Returns:
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    data_type = ['dev', 'test', 'train']
    for data in data_type:
        dst_file = os.path.join(dst_folder, data + ".json")
        uttrances = []
        wav_dir = os.path.join(data_folder, "wav", data)
        transcript_file = os.path.join(data_folder, "transcript", data, "trans.txt")
        trans_text = {}
        with open(transcript_file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                utterance_id, text = line[0], " ".join(line[1:])
                trans_text[utterance_id] = text.upper()
        session_list = os.listdir(wav_dir)
        for sessions in session_list:
            cur_dir = os.path.join(wav_dir, sessions)
            for wavs in os.listdir(cur_dir):
                audio_id = wavs.strip(".wav")
                audio_filepath = os.path.abspath(os.path.join(cur_dir, wavs))
                duration = subprocess.check_output(["soxi", "-D", audio_filepath])
                duration = float(duration)
                text = trans_text[audio_id]
                uttrances.append(
                    json.dumps(
                        {"audio_filepath": audio_filepath, "duration": duration, "text": text}, ensure_ascii=False
                    )
                )
        with open(dst_file, "w") as f:
            for line in uttrances:
                f.write(line + "\n")


def __get_vocab(data_folder: str, des_dir: str):
    """
    To generate the vocabulary file
    Args:
        data_folder: source with the transcript file
        dst_folder: where the file will be stored
    Returns:
    """
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    trans_file = os.path.join(data_folder, "transcript", "train", "trans.txt")
    vocab_dict = {}
    with open(trans_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            text = " ".join(line[1:])
            for i in text.upper():
                if i in vocab_dict:
                    vocab_dict[i] += 1
                else:
                    vocab_dict[i] = 1
    vocab_dict = sorted(vocab_dict.items(), key=lambda k: k[1], reverse=True)
    vocab = os.path.join(des_dir, "vocab.txt")
    vocab = open(vocab, "w", encoding='utf-8')
    for k in vocab_dict:
        vocab.write(k[0] + "\n")
    vocab.close()


def main():
    source_data = args.audio_folder
    des_dir = args.dest_folder
    print("begin to process data...")
    __process_data(source_data, des_dir)
    __get_vocab(source_data, des_dir)
    print("finish all!")


if __name__ == "__main__":
    main()
