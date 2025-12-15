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

"""
Execution :

python create_tarred_tokenized_text_lm_dataset.py \
    --text_path=<comma seperated text filepaths> \
    --data_root=<path to output directory> \
    --tokenizer_name="bert-base-cased" \
    --tokenizer_vocab_file=<path to vocab file for tokenizer> \
    --num_shards=64 \
    --chunk_size=8192 \
    --chunk_write_buffer=512 \
    --lower_case \
    --log
"""


import argparse
import glob
import json
import logging
import os
import tarfile

import joblib
import numpy as np
from tqdm import tqdm

from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer

parser = argparse.ArgumentParser(description='Tarred Tokenized dataset for text language modelling')

# Data path arguments
parser.add_argument('--text_path', required=True, default=None, type=str, help='Text paths, seperated by commas')
parser.add_argument('--data_root', required=True, default=None, type=str, help='Output directory')

# General arguments
parser.add_argument(
    '--chunk_write_buffer',
    default=128,
    type=int,
    help='Number of chunks of `chunk_size` to buffer for parallel tokenization and serial write to disk',
)
parser.add_argument('--lower_case', action='store_true', help='Whether to lower case the corpus')
parser.add_argument('--log', action='store_true', help='Whether to print logs to terminal')

# Tokenizer arguments
parser.add_argument('--tokenizer_name', required=False, default=None, type=str, help='Tokenizer name for resolution')
parser.add_argument(
    '--tokenizer_model', required=False, default=None, type=str, help='Path to tokenizer model for sentencepiece'
)
parser.add_argument('--tokenizer_vocab_file', required=False, type=str, default=None, help='Path to a vocab file')
parser.add_argument(
    '--tokenizer_special_tokens', default=None, type=str, nargs='+', help='List of special tokens for the tokenizer'
)

# Tarred dataset arguments
parser.add_argument('--num_shards', default=1, type=int, help='Number of shards for the tarfile')
parser.add_argument('--chunk_size', default=8192, type=int, help='Number of rows of data concatenated into a vector')

parser.set_defaults(log=False, lower_case=False)
args = parser.parse_args()


def __build_dataset_from_text(texts: str, lower_case: bool, chunk_size: int):
    if ',' in texts:
        texts = texts.split(',')
    else:
        texts = [texts]

    num_lines = 0
    text_dataset = []

    for text in texts:
        with open(text, 'r', encoding='utf-8') as in_reader:
            reader = tqdm(iter(lambda: in_reader.readline(), ''), desc="Read 0 lines", unit=' lines')

            for i, line in enumerate(reader):
                # Clean text line
                line = line.replace("\n", "").strip()

                if lower_case:
                    line = line.lower()

                if line:
                    text_dataset.append(line)

                    num_lines += 1
                    if num_lines % 100000 == 0:
                        reader.set_description(f"Read {num_lines} lines")

                    if num_lines % chunk_size == 0:
                        yield text_dataset, num_lines

                        # Empty cache
                        text_dataset = []

            logging.info(f"Finished extracting manifest : {text}")

        logging.info("Finished extracting all manifests ! Number of sentences : {}".format(num_lines))

    if len(text_dataset) != 0:
        yield text_dataset, num_lines


def __tokenize_str(texts, tokenizer):
    tokenized_text = []
    for text in texts:
        tok_text = tokenizer.text_to_ids(text)
        tokenized_text.extend(tok_text)
    return tokenized_text


def __tokenize_text(
    text_paths, tokenizer, tokenized_cachedir, lower_case: bool = False, chunk_size=8192, write_buffer: int = -1
):
    if write_buffer < 1:
        write_buffer = max(os.cpu_count() - write_buffer, 1)

    logging.info(f"Using write chunk buffer of size {write_buffer}")

    if not os.path.exists(tokenized_cachedir):
        os.makedirs(tokenized_cachedir)

    # global parameters
    global_chunk_idx = 0
    chunk_paths = []
    chunk_lens = []

    # buffer parameters
    data_cache = []
    chunk_idx = 0

    text_generator = iter(__build_dataset_from_text(text_paths, lower_case=lower_case, chunk_size=chunk_size))
    global_num_lines = 0
    last_batch = False

    with joblib.Parallel(n_jobs=-2, verbose=10) as parallel:

        while True:
            try:
                data, num_lines = next(text_generator)
                data_cache.append(data)

                global_num_lines = num_lines
            except StopIteration:
                last_batch = True

            # Update counters
            chunk_idx += 1

            if (chunk_idx == write_buffer) or last_batch:
                # write the chunks into disk after parallel tokenization
                tokenized_data_list = parallel(
                    joblib.delayed(__tokenize_str)(chunk, tokenizer) for chunk in data_cache
                )

                # Sequential write cache
                for chunk in tokenized_data_list:
                    fp = os.path.join(tokenized_cachedir, f"chunk_{global_chunk_idx}.npy")
                    chunk = np.asarray(chunk, dtype=np.int64)
                    np.save(fp, chunk, allow_pickle=False)

                    chunk_paths.append(fp)
                    chunk_lens.append(len(chunk))

                    global_chunk_idx += 1

                logging.info(f"Wrote a total of {global_chunk_idx} chunks to file...")

                # reset buffers
                data_cache.clear()
                del data_cache

                data_cache = []
                chunk_idx = 0

                if last_batch:
                    logging.info("Finished tokenizing last chunk")
                    break

    logging.info(
        f"Chunking {global_num_lines} rows into {global_num_lines // chunk_size} tasks (each chunk contains {chunk_size} elements)"
    )
    return chunk_paths, chunk_lens


def __create_chunk(data_root, chunk_path, shard_id, compute_metrics=False):
    """Creates a tarball containing the tokenized text chunks."""
    tar = tarfile.open(os.path.join(data_root, f'text_{shard_id}.tar'), mode='a', encoding='utf-8')

    # We squash the filename since we do not preserve directory structure of tokenized text in the tarball.
    base, ext = os.path.splitext(chunk_path)
    base = base.replace(os.pathsep, '_')
    # Need the following replacement as long as WebDataset splits on first period
    base = base.replace('.', '_')
    squashed_filename = f'{base}{ext}'
    tar.add(chunk_path, arcname=squashed_filename)

    tar.close()

    if compute_metrics:
        data = np.load(chunk_path, allow_pickle=False)
        chunk_len = len(data)
        return (chunk_len,)
    else:
        return None


def __write_tarred_tokenized_text_dataset(data_root, num_shards, chunk_paths, chunk_lens):
    num_chunks = len(chunk_paths)

    if chunk_lens is not None:
        num_text = sum(chunk_lens)
        shard_counts = {chunk_id: chunk_len for chunk_id, chunk_len in enumerate(chunk_lens)}
        compute_metrics = False

    else:
        num_text = 0
        shard_counts = {}
        compute_metrics = True

    for chunk_id, chunk_path in enumerate(tqdm(chunk_paths, desc='Writing chunk ', unit=' chunks')):
        shard_id = chunk_id % num_shards
        metrics = __create_chunk(data_root, chunk_path, shard_id, compute_metrics=compute_metrics)

        if metrics is not None:
            num_text += metrics[0]
            shard_counts[chunk_id] = metrics[0]

    # write metadata
    metadata_path = os.path.join(data_root, 'metadata.json')
    with open(metadata_path, 'w') as f:
        metadata = {'num_chunks': num_chunks, 'num_text': num_text, 'shard_count': shard_counts}
        json.dump(metadata, f, indent=4)
        logging.info("Metadata writen..")


def main():
    text_path = args.text_path
    data_root = args.data_root

    if args.log:
        logging.basicConfig(level=logging.INFO)

    tokenized_cachedir = os.path.join(data_root, '_tokenized_dataset_cachedir')

    if os.path.exists(tokenized_cachedir):
        logging.warning(
            f'Tokenized cache directory {tokenized_cachedir} already potentially contains files.'
            f'In such a case, please be aware that the tarfiles will be **appended** instead of overridden!'
        )

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    chunk_paths = None
    chunk_lens = None

    if os.path.exists(tokenized_cachedir):
        paths = glob.glob(os.path.join(tokenized_cachedir, "*.npy"))
        if len(paths) > 0:
            logging.info("Cached tokenized numpy files found, skipping re-tokenization of dataset")

            chunk_paths = paths
            chunk_lens = None

    if chunk_paths is None:
        if args.tokenizer_name is None:
            raise ValueError("`tokenizer_name` name is required when tokenizing the dataset for the first time.")

        if args.tokenizer_vocab_file is None:
            raise ValueError("`tokenizer_vocab_file` is required when constructing the tokenized dataset")

        tokenizer = get_tokenizer(
            tokenizer_name=args.tokenizer_name,
            tokenizer_model=args.tokenizer_model,
            vocab_file=args.tokenizer_vocab_file,
            special_tokens=args.tokenizer_special_tokens,
        )

        logging.info("Built tokenizer")

        # tokenize text data into sub-words
        chunk_paths, chunk_lens = __tokenize_text(
            text_paths=text_path,
            tokenizer=tokenizer,
            tokenized_cachedir=tokenized_cachedir,
            lower_case=args.lower_case,
            chunk_size=args.chunk_size,
            write_buffer=args.chunk_write_buffer,
        )
        logging.info(f"Tokenized dataset into sub-words and serialized cache at {tokenized_cachedir}")

    # Write tarred dataset
    __write_tarred_tokenized_text_dataset(
        data_root, num_shards=args.num_shards, chunk_paths=chunk_paths, chunk_lens=chunk_lens
    )

    logging.info('Done preparing tokenized dataset!')


if __name__ == "__main__":
    main()
