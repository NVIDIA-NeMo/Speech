# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import pickle
import random
import tarfile
from glob import glob
from typing import List, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization.decoder_dataset import TextNormalizationDecoderDataset
from nemo.utils import logging


"""
The script builds tar files for Tarred TextNormalizationDecoderDataset

See `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization/nn_text_normalization.rst>`
for more details on data format, and en/data_processing.py on how to pre-process the data before tarring.

To run the script, use:

    python create_tarred_dataset.py \
        --input_files = "train_processed/output-00099-of-00100" \
        --input_files = "train_processed/output-00098-of-00100" \
        --lang = "en" \
        --out_dir="TARRED_DATA_OUTPUT_DIR"
        
See the argparse help for more arguments.
"""


def _preprocess_file(input_file: str) -> List[Tuple[List[str]]]:
    """
    Performs initial preprocessing, i.e., urls formatting, removal of "_trans" from Ru set

    Args:
        input_file: path to a file in google TN format

    Returns:
        Processed data. Each element is a Tuple(List[semiotic classes], List[written words], List[spoken words])
    """
    print(f"Reading and running initial pre-processing of {input_file}...")
    cur_split = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # Loop through each line of the file
        cur_classes, cur_tokens, cur_outputs = [], [], []
        for linectx, line in tqdm(enumerate(f)):
            es = line.strip().split('\t')
            if len(es) == 2 and es[0] == '<eos>':
                cur_split.append((cur_classes, cur_tokens, cur_outputs))
                # Reset
                cur_classes, cur_tokens, cur_outputs = [], [], []
                continue
            assert len(es) == 3
            cur_classes.append(es[0])
            cur_tokens.append(es[1])
            cur_outputs.append(es[2])
    return cur_split


def _write_batches_to_tarfiles(
    input_file: str,
    tokenizer: AutoTokenizer,
    tokenizer_name: str,
    mode: str,
    lang: str,
    max_seq_len: int,
    batch_size: int,
    out_dir: str,
    num_batches_per_tarfile: int,
    decoder_data_augmentation: bool = False,
):
    """
    Creates tar files for the input file, i.e.:
        1. Creates a TextNormalizationDecoderDataset from the input file
        2. Constructs batches of size `batch_size`
        3. Saves each created batch to a pickle file and then adds `num_batches_per_tarfile`
            of the pickle files to a tarfile.

    Args:
        input_file: path to cleaned data file. See en/data_processing.py for cleaning.
        tokenizer: tokenizer
        tokenizer_name: the name of the tokenizer, usually corresponds to the pre-trained LM
        mode: model training mode
        max_seq_len: maximum length of the sequence (examples that are longer will be discarded)
        batch_size: batch size
        out_dir: path to output directory
        num_batches_per_tarfile: number of batches saved in each tar file
        decoder_data_augmentation: Set to True to enable data augmentation for the decoder model
        lang: data language
    """

    dataset = TextNormalizationDecoderDataset(
        input_file=input_file,
        raw_instances=_preprocess_file(input_file=input_file),
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        mode=mode,
        max_len=max_seq_len,
        decoder_data_augmentation=decoder_data_augmentation,
        lang=lang,
        use_cache=False,
        max_insts=-1,
        do_tokenize=False,
        initial_shuffle=True,
    )
    dataset.batchify(batch_size)
    file_name = os.path.basename(input_file)
    tar_file_ctr = 0
    tar_file_path = os.path.join(
        out_dir, '%s-batches.%d.%d.%d.tar' % (file_name, batch_size, max_seq_len, tar_file_ctr)
    )
    tar_file_ptr = tarfile.open(tar_file_path, 'w')
    total_batch_ctr = 0
    batch_ctr = 0
    for batch in dataset.batches:
        total_batch_ctr += 1
        batch_ctr += 1
        pickle_file = os.path.join(out_dir, '%s-batch-%d.pkl' % (file_name, total_batch_ctr))

        pickle.dump(batch, open(pickle_file, 'wb'))
        tar_file_ptr.add(pickle_file)
        os.remove(pickle_file)

        if batch_ctr == num_batches_per_tarfile:
            tar_file_ctr += 1
            tar_file_ptr.close()
            tar_file_path = os.path.join(
                out_dir, f'%s-batches.%d.%d.%d.tar' % (file_name, batch_size, max_seq_len, tar_file_ctr)
            )
            tar_file_ptr = tarfile.open(tar_file_path, 'w',)
            batch_ctr = 0

    # return tar files paths that have batches remaining
    remainder_tar_file_path = tar_file_ptr.name
    tar_file_ptr.close()

    return total_batch_ctr, remainder_tar_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='(Inverse) Text Normalization tarred dataset creation')
    parser.add_argument('--transformer_name', type=str, default="t5-small", help='Name of the pretrained LM.')
    parser.add_argument('--mode', type=str, default='tn', choices=constants.MODES, help='(I)TN model training mode.')
    parser.add_argument('--lang', type=str, default='en', choices=constants.SUPPORTED_LANGS, help='language.')
    parser.add_argument(
        '--decoder_data_augmentation',
        action="store_true",
        help='Set to True to use data augmentation for the decoder model.',
    )
    parser.add_argument(
        '-in',
        '--input_files',
        action='append',
        required=True,
        help="Example: -in train_processed/output-00099-of-00100 -in train_processed/output-00098-of-00100",
    )
    parser.add_argument('--out_dir', type=str, required=True, help='Path to store dataloader and tokenizer models.')
    parser.add_argument(
        '--max_seq_length', type=int, default=80, help='Maximum sequence length, longer examples will be discarded.'
    )
    parser.add_argument('--min_seq_length', type=int, default=1, help='Minimum sequence length.')
    parser.add_argument(
        '--num_batches_per_tarfile',
        type=int,
        default=2,
        help='Number batches, i.e., pickle files, included in a single .tar file.',
    )
    parser.add_argument('--n_jobs', type=int, default=-2, help='The maximum number of concurrently running jobs.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size, i.e., number of examples in a single pickle file. This batch size will override the training size.',
    )
    parser.add_argument(
        '--factor', default=8, type=int, help='The final number of tar files will be divisible by the "factor" value'
    )

    args = parser.parse_args()

    # check if tar files exist
    if os.path.exists(args.out_dir):
        tar_files_in_out_dir = glob(f'{args.out_dir}/*.tar')
        if tar_files_in_out_dir:
            raise ValueError(
                f'Tar files detected in {args.out_dir}. Delete the files to re-construct the dataset in the same directory.'
            )
    else:
        os.makedirs(args.out_dir)

    world_size = 1
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_name)

    results_list = Parallel(n_jobs=args.n_jobs)(
        delayed(_write_batches_to_tarfiles)(
            input_file=input_file,
            tokenizer=tokenizer,
            tokenizer_name=args.transformer_name,
            mode=args.mode,
            lang=args.lang,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_length,
            decoder_data_augmentation=args.decoder_data_augmentation,
            out_dir=args.out_dir,
            num_batches_per_tarfile=args.num_batches_per_tarfile,
        )
        for input_file in args.input_files
    )

    total_batches = sum([batch_count for batch_count, _ in results_list])

    # save batches from tar files containing the left over batches (if there's enough batches)
    remainder_tar_file_ctr = 0
    remainder_tar_file_path = os.path.join(
        args.out_dir, f'remainder-batches.tokens.{args.batch_size}.tar_file_{remainder_tar_file_ctr}.tar'
    )
    remainder_tar_file_ptr = tarfile.open(remainder_tar_file_path, 'w')
    batch_in_tar_ctr = 0
    for _, tar_file_path in results_list:
        tar_file_ptr = tarfile.open(tar_file_path, 'r')
        for member in tar_file_ptr.getmembers():
            remainder_tar_file_ptr.addfile(member, tar_file_ptr.extractfile(member.name))
            batch_in_tar_ctr += 1
            if batch_in_tar_ctr == args.num_batches_per_tarfile:
                remainder_tar_file_ctr += 1
                remainder_tar_file_ptr.close()
                remainder_tar_file_path = os.path.join(
                    args.out_dir, f'remainder-batches.tokens.{args.batch_size}.tar_file_{remainder_tar_file_ctr}.tar',
                )
                remainder_tar_file_ptr = tarfile.open(remainder_tar_file_path, 'w',)
                batch_in_tar_ctr = 0
        tar_file_ptr.close()
        os.remove(tar_file_path)

    # log the number of batches remaining as they will be discarded
    num_batches_discarded = len(remainder_tar_file_ptr.getmembers())
    remainder_tar_file_ptr.close()
    os.remove(remainder_tar_file_path)

    tar_file_paths = glob(f'{args.out_dir}/*.tar')
    if args.factor != 1:
        num_tar_files = len(tar_file_paths)
        num_tars_to_drop = num_tar_files % args.factor
        num_batches_discarded += num_tars_to_drop * args.num_batches_per_tarfile

        random.shuffle(tar_file_paths)
        for _ in range(num_tars_to_drop):
            os.remove(tar_file_paths.pop(-1))

    total_batches -= num_batches_discarded
    logging.info(f'Number of batches discarded: {num_batches_discarded}, total batches kept: {total_batches}')

    # dump metadata to json
    metadata = {}
    metadata['num_batches'] = total_batches

    # rename tar files so they can be more easily used with CLI and YAML
    file_name = f'{args.mode}.{args.batch_size}_bs.{args.num_batches_per_tarfile}_b_per_tar.{args.max_seq_length}_len'
    for index, path in enumerate(tar_file_paths):
        os.rename(path, os.path.join(args.out_dir, f'{file_name}.{index}.tar'))

    text_tar_filepaths = f'{file_name}._OP_0..{index}_CL_.tar'
    logging.info(f'Files for brace expansion: "{text_tar_filepaths}"')
    metadata['text_tar_filepaths'] = text_tar_filepaths

    # add tar files to metadata
    tar_file_paths = glob(f'{args.out_dir}/*.tar')
    metadata['tar_files'] = tar_file_paths
    metadata_path = os.path.join(args.out_dir, 'metadata.json')
    json.dump(metadata, open(metadata_path, 'w'))

    num_tar_files = len(tar_file_paths)
    if num_tar_files < world_size:
        raise ValueError(
            (
                f'Number of tar files found: {num_tar_files} is less than world size: {world_size}. '
                f'There should be at least one tar file per GPU (ideally many tar files per GPU). '
                f'This may be due to dataset size. '
                f'Decrease num_batches_per_tarfile or num_tokens_per_batch to increase the number of tarfiles. '
                f'Also using shard_strategy=replicate will use all available tarfiles for every GPU. '
            )
        )
