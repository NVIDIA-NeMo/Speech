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

import json
import os
import re
from argparse import ArgumentParser
from multiprocessing import Pool

import tensorflow as tf
from sacremoses import MosesDetokenizer
from tasks_splits_and_features import _TASK_SPLITS_AND_FEATURES_DICT


"""
This script converts the P3 dataset used to train T0 from a tfrecords format to individual JSONL files.

Use instructions:

NOTE: This script requires tensorflow to be installed.

1. Download the P3 dataset by cloning it from Huggingface:
        git clone https://huggingface.co/datasets/bigscience/P3. The raw data should be at P3/data.
2. Run this script: 
    python t0_dataset_preproc.py \
        --p3_dataset_path P3/data \
        --jsonl_output_path P3/data_processed_jsonl
3. The output will be in the jsonl_output_path directory. In the following structure:
    - P3/data_processed_jsonl/train
       - super_glue_cb_does_this_imply.jsonl
       - super_glue_cb_justified_in_saying_score_eval.jsonl
       - .....
    - P3/data_processed_jsonl/val
       - super_glue_cb_does_this_imply.jsonl
       - super_glue_cb_justified_in_saying_score_eval.jsonl
       - .....
4. Each JSONL file is compatible with NeMo's T0JSONLMemMapDataset (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/data/language_modeling/t0_dataset.py)
"""


def _feature_config(shape, dtype):
    if dtype in ("int32", "bool"):
        # int32 and bool are stored as int64 in the tf.train.Example protobuf.
        dtype = "int64"
    if shape and shape[0] is None:
        return tf.io.FixedLenSequenceFeature(shape[1:], dtype, allow_missing=True)
    return tf.io.FixedLenFeature(shape, dtype)


def remove_newline_and_detokenize(x, detokenizer, remove_newlines):
    if remove_newlines:
        x = re.sub(r'\\n+', ' ', x)
        x = re.sub(r'\n+', ' ', x)
        x = re.sub(r'\\r+', ' ', x)
        x = re.sub(r'\r+', ' ', x)
    x = x.strip()
    # NOTE: Moving the detokenizer inside this condition since sacremoses detokenize seems to remove \n as well.
    if remove_newlines:
        x = detokenizer.detokenize([x])
    return x


def write_dataset_to_file(dataset, filename, detokenizer, remove_newlines):
    with open(filename, 'w') as f:
        for item in dataset:
            # NOTE: Although we do `.tolist()` here this is not actually a list. This is just to convert from a numpy to python object so we can check if it is True/False.
            if 'is_correct' in item and item['is_correct'].numpy().tolist() is False:
                print('Skipping example because is_correct is False')
                continue

            item_object = {}
            i = remove_newline_and_detokenize(
                item['inputs_pretokenized'].numpy().decode('utf-8'), detokenizer, remove_newlines
            )
            item_object['input'] = i
            t = remove_newline_and_detokenize(
                item['targets_pretokenized'].numpy().decode('utf-8'), detokenizer, remove_newlines
            )
            item_object['output'] = t
            if 'answer_choices' in item:
                choices = [
                    remove_newline_and_detokenize(x.decode('utf-8'), detokenizer, remove_newlines)
                    for x in item['answer_choices'].numpy().tolist()
                ]
                item_object['choices'] = choices
            f.write(json.dumps(item_object) + '\n')


def write_train_val_test_dataset_to_file(file_name, folder_name, output_folder, detokenizer, split, remove_newlines):
    ds = tf.data.TFRecordDataset(tf.io.gfile.glob([file_name]))
    fdict = _TASK_SPLITS_AND_FEATURES_DICT[folder_name]['features_dict']
    feature_description = {feat: _feature_config(**desc) for feat, desc in fdict.items()}
    ds = ds.map(
        lambda pb: tf.io.parse_single_example(pb, feature_description),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(
        lambda x: {k: tf.cast(v, fdict[k]["dtype"]) for k, v in x.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    write_dataset_to_file(ds, os.path.join(output_folder, split, folder_name + '.jsonl'), detokenizer, remove_newlines)


def process_folder(data_folder, folder_name, output_folder, detokenizer, remove_newlines):
    if not os.path.isdir(os.path.join(data_folder, folder_name)):
        return
    print(f'Processing {folder_name}')
    train_fname = os.path.join(data_folder, folder_name, 'train.tfrecord-00000-of-00001')
    valid_fname = os.path.join(data_folder, folder_name, 'validation.tfrecord-00000-of-00001')
    test_fname = os.path.join(data_folder, folder_name, 'test.tfrecord-00000-of-00001')
    if not os.path.exists(train_fname):
        print(f'Could not find {train_fname}')
        return
    write_train_val_test_dataset_to_file(
        train_fname, folder_name, output_folder, detokenizer, 'train', remove_newlines
    )
    if os.path.exists(valid_fname):
        write_train_val_test_dataset_to_file(
            valid_fname, folder_name, output_folder, detokenizer, 'val', remove_newlines
        )
    if os.path.exists(test_fname):
        write_train_val_test_dataset_to_file(
            test_fname, folder_name, output_folder, detokenizer, 'test', remove_newlines
        )


def process_all_folders(data_folder, output_folder, remove_newlines):
    detokenizer = MosesDetokenizer('en')
    assert os.path.isdir(data_folder)
    if not os.path.exists(output_folder):
        os.system(f'mkdir -p {output_folder}')
    if not os.path.exists(os.path.join(output_folder, 'train')):
        os.system(f'mkdir -p {os.path.join(output_folder, "train")}')
    if not os.path.exists(os.path.join(output_folder, 'val')):
        os.system(f'mkdir -p {os.path.join(output_folder, "val")}')
    if not os.path.exists(os.path.join(output_folder, 'test')):
        os.system(f'mkdir -p {os.path.join(output_folder, "test")}')

    print(f'Found {len(os.listdir(data_folder))} folders to process ...')
    pool_args = []
    for folder_name in os.listdir(data_folder):
        pool_args.append((data_folder, folder_name, output_folder, detokenizer, remove_newlines))
    pool = Pool()
    pool.starmap(process_folder, pool_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--p3_dataset_path",
        type=str,
        required=True,
        help="Path to raw P3 data. Should be a folder containing folders for each task. After cloning the repo this should correspond to P3/data",
    )
    parser.add_argument(
        "--jsonl_output_path",
        type=str,
        required=True,
        help="Path to output folder where JSONL files will be written.",
    )
    parser.add_argument(
        "--remove_newlines", action="store_true", help="Whether to remove newlines from the input and output.",
    )
    args = parser.parse_args()
    process_all_folders(args.p3_dataset_path, args.jsonl_output_path, args.remove_newlines)
