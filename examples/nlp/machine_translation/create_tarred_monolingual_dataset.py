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

import argparse
import os

from nemo.collections.nlp.data.machine_translation.preproc_mt_data import MTDataPreproc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT dataset pre-processing')
    parser.add_argument(
        '--tokenizer_name', type=str, default='sentencepiece', help='Supports entencepiece and HuggingFace tokenizers',
    )
    parser.add_argument('--tokenizer_model', type=str, default=None, help='Path to tokenizer model')
    parser.add_argument('--bpe_droput', type=float, default=0.0, help='BPE dropout to use')
    parser.add_argument('--clean', action="store_true", help='Whether to clean dataset based on length diff')
    parser.add_argument('--pkl_file_prefix', type=str, default='parallel', help='Prefix for tar and pickle files')
    parser.add_argument('--fname', type=str, required=True, help='Path to monolingual data file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to store dataloader and tokenizer models')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Max Sequence Length')
    parser.add_argument('--min_seq_length', type=int, default=1, help='Min Sequence Length')
    parser.add_argument('--tokens_in_batch', type=int, default=16000, help='# Tokens per batch per GPU')
    parser.add_argument(
        '--lines_per_dataset_fragment',
        type=int,
        default=1000000,
        help='Number of lines to consider for bucketing and padding',
    )
    parser.add_argument(
        '--num_batches_per_tarfile',
        type=int,
        default=1000,
        help='Number of batches (pickle files) within each tarfile',
    )

    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if (args.tokenizer_name == "sentencepiece") and not os.path.exists(args.tokenizer_model):
        assert FileNotFoundError("Could not find tokenizer model %s" % (args.tokenizer))

    tokenizer_model = MTDataPreproc.get_monolingual_tokenizer(
        tokenizer_name=args.tokenizer_name, tokenizer_model=args.tokenizer_model, bpe_dropout=args.bpe_droput
    )

    MTDataPreproc.preprocess_monolingual_dataset(
        clean=args.clean,
        fname=args.fname,
        out_dir=args.out_dir,
        tokenizer=tokenizer_model,
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        tokens_in_batch=args.tokens_in_batch,
        lines_per_dataset_fragment=args.lines_per_dataset_fragment,
        num_batches_per_tarfile=args.num_batches_per_tarfile,
        pkl_file_prefix=args.pkl_file_prefix,
        global_rank=0,
        world_size=1,
    )
