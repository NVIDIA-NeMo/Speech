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

import os
import pickle
import re
import string
from typing import Dict

from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    fill_class_weights,
    get_freq_weights,
    get_label_stats,
)
from nemo.utils import logging

__all__ = ['get_label_ids', 'create_text_and_labels']


def remove_punctuation(word: str):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation.replace("'", '')
    return re.sub('[' + all_punct_marks + ']', '', word)


def create_text_and_labels(output_dir: str, file_path: str, punct_marks: str = ',.?'):
    """
    Create datasets for training and evaluation.

    Args:
      output_dir: path to the output data directory
      file_path: path to file name
      punct_marks: supported punctuation marks

    The data will be split into 2 files: text.txt and labels.txt. \
    Each line of the text.txt file contains text sequences, where words\
    are separated with spaces. The labels.txt file contains \
    corresponding labels for each word in text.txt, the labels are \
    separated with spaces. Each line of the files should follow the \
    format:  \
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
    """
    if not os.path.exists(file_path):
        raise ValueError(f'{file_path} not found')

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    labels_file = os.path.join(output_dir, 'labels_' + base_name)
    text_file = os.path.join(output_dir, 'text_' + base_name)

    with open(file_path, 'r') as f:
        with open(text_file, 'w') as text_f:
            with open(labels_file, 'w') as labels_f:
                for line in f:
                    line = line.split()
                    text = ''
                    labels = ''
                    for word in line:
                        label = word[-1] if word[-1] in punct_marks else 'O'
                        word = remove_punctuation(word)
                        if len(word) > 0:
                            if word[0].isupper():
                                label += 'U'
                            else:
                                label += 'O'

                            word = word.lower()
                            text += word + ' '
                            labels += label + ' '

                    text_f.write(text.strip() + '\n')
                    labels_f.write(labels.strip() + '\n')

    print(f'{text_file} and {labels_file} created from {file_path}.')


def get_label_ids(
    label_file: str,
    is_training: bool = False,
    pad_label: str = 'O',
    label_ids_dict: Dict[str, int] = None,
    get_weights: bool = True,
    class_labels_file_artifact='label_ids.csv',
):
    """
    Generates str to int labels mapping for training data or checks correctness of the label_ids_dict
    file for non-training files or if label_ids_dict is specified

    Args:
        label_file: the path of the label file to process
        is_training: indicates whether the label_file is used for training
        pad_label: token used for padding
        label_ids_dict: str label name to int ids mapping. Required for non-training data.
            If specified, the check that all labels from label_file are present in label_ids_dict will be performed.
            For training data, if label_ids_dict is None, a new mapping will be generated from label_file.
        get_weights: set to True to calculate class weights, required for Weighted Loss.
        class_labels_file_artifact: name of the file to save in .nemo
    """
    if not os.path.exists(label_file):
        raise ValueError(f'File {label_file} was not found.')

    logging.info(f'Processing {label_file}')
    if not is_training and label_ids_dict is None:
        raise ValueError(
            f'For non training data, label_ids_dict created during preprocessing of the training data '
            f'should be provided'
        )

    # collect all labels from the label_file
    data_dir = os.path.dirname(label_file)
    unique_labels = set(pad_label)
    all_labels = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            all_labels.extend(line)
            unique_labels.update(line)

    # check that all labels from label_file are present in the specified label_ids_dict
    # or generate label_ids_dict from data (for training only)
    if label_ids_dict:
        logging.info(f'Using provided labels mapping {label_ids_dict}')
        for name in unique_labels:
            if name not in label_ids_dict:
                raise ValueError(f'{name} class from {label_file} not found in the provided mapping: {label_ids_dict}')
    else:
        label_ids_dict = {pad_label: 0}
        if pad_label in unique_labels:
            unique_labels.remove(pad_label)
        for label in sorted(unique_labels):
            label_ids_dict[label] = len(label_ids_dict)

    label_ids_filename = os.path.join(data_dir, class_labels_file_artifact)
    if is_training:
        with open(label_ids_filename, 'w') as f:
            labels, _ = zip(*sorted(label_ids_dict.items(), key=lambda x: x[1]))
            f.write('\n'.join(labels))
        logging.info(f'Labels mapping {label_ids_dict} saved to : {label_ids_filename}')

    # calculate label statistics
    base_name = os.path.splitext(os.path.basename(label_file))[0]
    stats_file = os.path.join(data_dir, f'{base_name}_label_stats.tsv')
    if os.path.exists(stats_file) and not is_training and not get_weights:
        logging.info(f'{stats_file} found, skipping stats calculation.')
    else:
        all_labels = [label_ids_dict[label] for label in all_labels]
        logging.info(f'Three most popular labels in {label_file}:')
        total_labels, label_frequencies, max_id = get_label_stats(all_labels, stats_file)
        logging.info(f'Total labels: {total_labels}. Label frequencies - {label_frequencies}')

    if get_weights:
        class_weights_pkl = os.path.join(data_dir, f'{base_name}_weights.p')
        if os.path.exists(class_weights_pkl):
            class_weights = pickle.load(open(class_weights_pkl, 'rb'))
            logging.info(f'Class weights restored from {class_weights_pkl}')
        else:
            class_weights_dict = get_freq_weights(label_frequencies)
            logging.info(f'Class Weights: {class_weights_dict}')
            class_weights = fill_class_weights(class_weights_dict, max_id)

            pickle.dump(class_weights, open(class_weights_pkl, "wb"))
            logging.info(f'Class weights saved to {class_weights_pkl}')
    else:
        class_weights = None

    return label_ids_dict, label_ids_filename, class_weights
