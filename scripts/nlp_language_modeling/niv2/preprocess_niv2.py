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

from sacremoses import MosesDetokenizer

from nemo.collections.common.tokenizers import AutoTokenizer


"""
This script converts the NaturalInstructions v2 dataset into individual JSONL files.

Use instructions:

1. Download the NaturalInstructions dataset by cloning it from allenai:
        git clone https://github.com/allenai/natural-instructions. The raw data should be in the tasks folder.

2. Run this script:
    python preprocess_niv2.py \
        --niv2_dataset_path natural-instructions/tasks  \
        --jsonl_output_path natural-instructions/train_tasks_default_jsonl \
        --splits_file_path natural-instructions/splits/default/train_tasks.txt

3. The output will be in the jsonl_output_path directory.

4. Each JSONL file is compatible with NeMo's T0JSONLMemMapDataset (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/data/language_modeling/t0_dataset.py)
"""


def remove_newline_and_detokenize(x, detokenizer):
    x = re.sub(r'\\n+', ' ', x)
    x = re.sub(r'\n+', ' ', x)
    x = re.sub(r'\\r+', ' ', x)
    x = re.sub(r'\r+', ' ', x)
    x = x.strip()
    x = detokenizer.detokenize([x])
    return x


def detokenize(x, detokenizer):
    x = x.strip()
    # NOTE: Commenting this out since sacremoses seems to remove \n as part of detokenization.
    # x = detokenizer.detokenize([x])
    return x


def is_empty(x, tokenizer):
    return len(tokenizer.text_to_tokens(x.strip())) < 1


def write_dataset_to_file(file_name, output_file_name, detokenizer, tokenizer, idx, total_num_files, remove_newline):
    print(f'Processing file {idx + 1}/{total_num_files} : {file_name} -> {output_file_name}')
    dataset = json.load(open(file_name, 'r'))
    with open(output_file_name, 'w') as f:
        instances = dataset['Instances']
        definitions = dataset['Definition']
        for definition in definitions:
            if is_empty(definition, tokenizer):
                continue
            for instance in instances:
                id = instance['id']
                input = instance['input']
                outputs = instance['output']
                # On rare occasions, the same instance can have multiple outputs. We add all of them as examples.
                if is_empty(input, tokenizer):
                    continue
                for output in outputs:
                    if is_empty(output, tokenizer):
                        continue
                    if remove_newline:
                        prompted_input = definition + ' ' + input
                    else:
                        prompted_input = definition + '\n\n' + input
                    proc_func = remove_newline_and_detokenize if remove_newline else detokenize
                    prompted_input = proc_func(prompted_input, detokenizer)
                    output = proc_func(output, detokenizer)
                    instance_object = {
                        'id': id,
                        'input': prompted_input,
                        'output': output,
                    }
                    f.write(json.dumps(instance_object) + '\n')


def process_folder(data_folder, output_folder, splits_file, remove_newline):
    detokenizer = MosesDetokenizer('en')
    tokenizer = AutoTokenizer("gpt2")
    assert os.path.isdir(data_folder)
    assert os.path.exists(splits_file)
    if not os.path.exists(output_folder):
        os.system(f'mkdir -p {output_folder}')
    if not os.path.exists(os.path.join(output_folder, 'train')):
        os.system(f'mkdir -p {os.path.join(output_folder, "train")}')
    if not os.path.exists(os.path.join(output_folder, 'test')):
        os.system(f'mkdir -p {os.path.join(output_folder, "test")}')

    splits_file_names = [line.strip() + '.json' for line in open(splits_file, 'r')]
    print(f'Found {len(os.listdir(data_folder))} files in the data folder ...')
    print(f'Found {len(splits_file_names)} in the splits in the splits file ...')
    print(f'Processing {len(splits_file_names)}/{len(os.listdir(data_folder))} files ...')
    pool_args = []
    for idx, file_name in enumerate(splits_file_names):
        print(f'Processing file {idx}/{len(splits_file_names)}: {file_name}')
        if not os.path.exists(os.path.join(data_folder, file_name)):
            raise FileNotFoundError(f'Could not find {os.path.join(data_folder, file_name)}')
        if not file_name.endswith('.json'):
            print(f'Skipping {file_name} because it is not a JSON file')
        output_file_name = os.path.join(output_folder, file_name.replace('.json', '.jsonl'))
        pool_args.append(
            (
                os.path.join(data_folder, file_name),
                output_file_name,
                detokenizer,
                tokenizer,
                idx,
                len(splits_file_names),
                remove_newline,
            )
        )

    write_dataset_to_file(
        os.path.join(data_folder, file_name),
        output_file_name,
        detokenizer,
        tokenizer,
        idx,
        len(splits_file_names),
        remove_newline,
    )
    pool = Pool(42)
    pool.starmap(write_dataset_to_file, pool_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--niv2_dataset_path",
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
        "--splits_file_path", type=str, default="default", help="Path to the file that contains splits. ex: ",
    )
    parser.add_argument(
        "--remove_newline", action="store_true", help="Whether to remove newlines from the input and output.",
    )
    args = parser.parse_args()
    process_folder(args.niv2_dataset_path, args.jsonl_output_path, args.splits_file_path, args.remove_newline)
