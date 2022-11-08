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
import logging
import os
import random
import re
import subprocess

from nemo.collections.nlp.data.token_classification.token_classification_utils import create_text_and_labels
from nemo.utils import logging

URL = {'tatoeba': 'https://downloads.tatoeba.org/exports/sentences.csv'}


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if not exists.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    """
    source = URL[source]
    if not os.path.exists(destination):
        logging.info(f'Downloading {source} to {destination}')
        subprocess.run(['wget', '-O', destination, source])
    else:
        logging.info(f'{destination} found. Skipping download')


def __process_english_sentences(
    in_file: str, out_file: str, percent_to_cut: float = 0, num_to_combine: int = 1, num_samples: int = -1
):
    """
    Extract English sentences from the Tatoeba dataset.

    Expected in_file format
    that
    contain letters and punctuation marks (,.?).
    Chop and combine sentences.
    Args:
        in_file: local filepath to the tatoeba dataset.
    Format: id [TAB] region_name [TAB] sentence,
    for example: "1276\teng\tLet's try something.\n"
        out_file: local filepath to the clean dataset
        percent_to_cut: Percent of sentences to cut in the middle
            to get examples of incomplete sentences.
            This could be useful since ASR output not always
            represents a complete sentence
        num_to_combine: Number of sentences to combine into
            a single example
        num_samples: Number of samples in the final dataset
    """
    if not os.path.exists(in_file):
        raise FileNotFoundError(f'{in_file} not found.')

    in_file = open(in_file, 'r')
    out_file = open(out_file, 'w')
    lines_to_combine = []
    samples_count = 0

    for line in in_file:
        line = line.split('\t')
        # use only English sentences
        if line[1] == 'eng':
            line = line[2].strip()
            if re.match("^[A-Z][A-Za-z.,'?\s]+$", line):  # nopep8
                # chop some sentences in the middle
                if percent_to_cut > 0:
                    line = line.split()
                    if random.random() < percent_to_cut:
                        line = line[: len(line) // 2]
                    line = ' '.join(line)

                # combine multiple sentences into a single example
                # to make it harder for the model to learn eos punctuation
                if len(lines_to_combine) >= num_to_combine:
                    if samples_count == num_samples:
                        return
                    out_file.write(' '.join(lines_to_combine) + '\n')
                    lines_to_combine = []
                    samples_count += 1
                lines_to_combine.append(line)

    if len(lines_to_combine) > 0 and (samples_count < num_samples or num_samples < 0):
        out_file.write(' '.join(lines_to_combine) + '\n')


def __split_into_train_dev(in_file: str, train_file: str, dev_file: str, percent_dev: float):
    """
    Create train and dev split of the dataset.
    Args:
        in_file: local filepath to the dataset
        train_file: local filepath to the train dataset
        dev_file: local filepath to the dev dataset
        percent_dev: Percent of the sentences in the dev set
    """
    if not os.path.exists(in_file):
        raise FileNotFoundError(f'{in_file} not found.')

    lines = open(in_file, 'r').readlines()
    train_file = open(train_file, 'w')
    dev_file = open(dev_file, 'w')

    dev_size = int(len(lines) * percent_dev)
    train_file.write(' '.join(lines[:-dev_size]))
    dev_file.write(' '.join(lines[-dev_size:]))


def __delete_file(file_to_del: str):
    """
    Deletes the file
    Args:
        file_to_del: local filepath to the file to delete
    """
    if os.path.exists(file_to_del):
        os.remove(file_to_del)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare tatoeba dataset')
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--dataset", default='tatoeba', type=str)
    parser.add_argument("--num_samples", default=-1, type=int, help='-1 to use the whole dataset')
    parser.add_argument("--percent_to_cut", default=0, type=float, help='Percent of sentences to cut in the middle')
    parser.add_argument(
        "--num_lines_to_combine", default=1, type=int, help='Number of lines to combine into single example'
    )
    parser.add_argument("--percent_dev", default=0.2, type=float, help='Size of the dev set, float')
    parser.add_argument("--clean_dir", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if args.dataset != 'tatoeba':
        raise ValueError("Unsupported dataset.")

    logging.info(f'Downloading tatoeba dataset')
    tatoeba_dataset = os.path.join(args.data_dir, 'sentences.csv')
    __maybe_download_file(tatoeba_dataset, args.dataset)

    logging.info(f'Processing English sentences...')
    clean_eng_sentences = os.path.join(args.data_dir, 'clean_eng_sentences.txt')
    __process_english_sentences(
        tatoeba_dataset, clean_eng_sentences, args.percent_to_cut, args.num_lines_to_combine, args.num_samples
    )

    train_file = os.path.join(args.data_dir, 'train.txt')
    dev_file = os.path.join(args.data_dir, 'dev.txt')

    logging.info(
        f'Splitting the {args.dataset} dataset into train and dev sets' + ' and creating labels and text files'
    )
    __split_into_train_dev(clean_eng_sentences, train_file, dev_file, args.percent_dev)

    logging.info(f'Creating text and label files for training')
    create_text_and_labels(args.data_dir, os.path.join(args.data_dir, 'train.txt'))
    create_text_and_labels(args.data_dir, os.path.join(args.data_dir, 'dev.txt'))

    if args.clean_dir:
        logging.info(f'Cleaning up {args.data_dir}')
        __delete_file(clean_eng_sentences)
        __delete_file(tatoeba_dataset)
        __delete_file(train_file)
        __delete_file(dev_file)
    logging.info(f'Processing of the {args.dataset} is complete')
