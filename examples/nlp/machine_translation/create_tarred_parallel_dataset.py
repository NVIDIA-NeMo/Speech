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
    parser.add_argument('--shared_tokenizer', action="store_true", help='Whether to share encoder/decoder tokenizers')
    parser.add_argument('--clean', action="store_true", help='Whether to clean dataset based on length diff')
    parser.add_argument('--tar_file_prefix', type=str, default='parallel', help='Prefix for tar files')
    parser.add_argument('--src_fname', type=str, required=True, help='Path to the source file')
    parser.add_argument('--tgt_fname', type=str, required=True, help='Path to the target file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to store dataloader and tokenizer models')
    parser.add_argument('--encoder_model_name', type=str, default=None, help='For use with pretrained encoders')
    parser.add_argument(
        '--decoder_model_name', type=str, default=None, help='For use with pretrained decoders (not yet supported)'
    )
    parser.add_argument(
        '--encoder_tokenizer_model', type=str, default='None', help='Path to pre-trained encoder tokenizer model'
    )
    parser.add_argument(
        '--encoder_tokenizer_name',
        type=str,
        default='sentencepiece',
        help='Encoder BPE Tokenizer Name, Options: [sentencepiece]',
    )
    parser.add_argument('--encoder_tokenizer_vocab_size', type=int, default=32000, help='Encoder Vocab size after BPE')
    parser.add_argument(
        '--encoder_tokenizer_coverage', type=float, default=0.999, help='Encoder Character coverage for BPE'
    )
    parser.add_argument('--encoder_tokenizer_bpe_dropout', type=float, default=0.1, help='Encoder BPE dropout prob')
    parser.add_argument(
        '--encoder_tokenizer_r2l', action="store_true", help='Whether to return encoded sequence from right to left'
    )
    parser.add_argument(
        '--encoder_tokenizer_legacy',
        action="store_true",
        help='Whether to use legacy tokenizer implementation of sentencepiece',
    )
    parser.add_argument(
        '--decoder_tokenizer_model', type=str, default='None', help='Path to pre-trained decoder tokenizer model'
    )
    parser.add_argument(
        '--decoder_tokenizer_name',
        type=str,
        default='sentencepiece',
        help='Encoder BPE Tokenizer Name, Options: [sentencepiece]',
    )
    parser.add_argument('--decoder_tokenizer_vocab_size', type=int, default=32000, help='Encoder Vocab size after BPE')
    parser.add_argument(
        '--decoder_tokenizer_coverage', type=float, default=0.999, help='Encoder Character coverage for BPE'
    )
    parser.add_argument('--decoder_tokenizer_bpe_dropout', type=float, default=0.1, help='Encoder BPE dropout prob')
    parser.add_argument(
        '--decoder_tokenizer_r2l', action="store_true", help='Whether to return encoded sequence from right to left'
    )
    parser.add_argument(
        '--decoder_tokenizer_legacy',
        action="store_true",
        help='Whether to use legacy tokenizer implementation of sentencepiece',
    )
    parser.add_argument('--max_seq_length', type=int, default=512, help='Max Sequence Length')
    parser.add_argument('--min_seq_length', type=int, default=1, help='Min Sequence Length')
    parser.add_argument('--tokens_in_batch', type=int, default=16000, help='# Tokens per batch per GPU')
    parser.add_argument('--coverage', type=float, default=0.999, help='BPE character coverage [0-1]')
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
    parser.add_argument(
        '--n_preproc_jobs', type=int, default=-2, help='Number of processes to use for creating the tarred dataset.',
    )
    parser.add_argument(
        '--byte_fallback',
        action="store_true",
        help='Whether to use byte fallback with sentencepiece for BPE tokenization.',
    )
    parser.add_argument(
        '--split_digits', action="store_true", help='Whether to split digits while tokenizing with sentencepiece.'
    )
    parser.add_argument(
        '--no_split_by_whitespace',
        action="store_true",
        help='If True, this will not respect whitepsaces while learning BPE merges.',
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if (
        args.encoder_tokenizer_model != 'None'
        and args.decoder_tokenizer_model == 'None'
        or args.decoder_tokenizer_model != 'None'
        and args.encoder_tokenizer_model == 'None'
    ):
        if args.shared_tokenizer:
            raise ValueError(
                '''
                If using a pre-trained shared tokenizer,
                both encoder and decoder tokenizers must be the same
                '''
            )
        else:
            raise ValueError('Both encoder and decoder pre-trained tokenizer models must be specified')

    if args.encoder_tokenizer_model == 'None' and args.decoder_tokenizer_model == 'None':
        encoder_tokenizer_model, decoder_tokenizer_model = MTDataPreproc.train_tokenizers(
            out_dir=args.out_dir,
            src_fname=args.src_fname,
            tgt_fname=args.tgt_fname,
            shared_tokenizer=args.shared_tokenizer,
            encoder_tokenizer_name=args.encoder_tokenizer_name,
            encoder_tokenizer_vocab_size=args.encoder_tokenizer_vocab_size,
            encoder_tokenizer_coverage=args.encoder_tokenizer_coverage,
            decoder_tokenizer_name=args.decoder_tokenizer_name,
            decoder_tokenizer_vocab_size=args.decoder_tokenizer_vocab_size,
            decoder_tokenizer_coverage=args.decoder_tokenizer_coverage,
            global_rank=0,
            byte_fallback=args.byte_fallback,
            split_digits=args.split_digits,
            split_by_whitespace=not args.no_split_by_whitespace,
        )
    else:
        encoder_tokenizer_model, decoder_tokenizer_model = args.encoder_tokenizer_model, args.decoder_tokenizer_model

    encoder_tokenizer, decoder_tokenizer = MTDataPreproc.get_enc_dec_tokenizers(
        encoder_tokenizer_name=args.encoder_tokenizer_name,
        encoder_tokenizer_model=encoder_tokenizer_model,
        encoder_bpe_dropout=args.encoder_tokenizer_bpe_dropout,
        encoder_r2l=args.encoder_tokenizer_r2l,
        decoder_tokenizer_name=args.decoder_tokenizer_name,
        decoder_tokenizer_model=decoder_tokenizer_model,
        decoder_bpe_dropout=args.decoder_tokenizer_bpe_dropout,
        decoder_r2l=args.decoder_tokenizer_r2l,
        encoder_tokenizer_legacy=args.encoder_tokenizer_legacy,
        decoder_tokenizer_legacy=args.decoder_tokenizer_legacy,
    )

    _, _ = MTDataPreproc.preprocess_parallel_dataset(
        clean=args.clean,
        src_fname=args.src_fname,
        tgt_fname=args.tgt_fname,
        out_dir=args.out_dir,
        encoder_tokenizer_name=args.encoder_tokenizer_name,
        encoder_model_name=args.encoder_model_name,
        encoder_tokenizer_model=encoder_tokenizer_model,
        encoder_bpe_dropout=args.encoder_tokenizer_bpe_dropout,
        encoder_tokenizer_r2l=args.encoder_tokenizer_r2l,
        decoder_tokenizer_name=args.decoder_tokenizer_name,
        decoder_model_name=args.decoder_model_name,
        decoder_tokenizer_model=decoder_tokenizer_model,
        decoder_tokenizer_r2l=args.decoder_tokenizer_r2l,
        decoder_bpe_dropout=args.decoder_tokenizer_bpe_dropout,
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        tokens_in_batch=args.tokens_in_batch,
        lines_per_dataset_fragment=args.lines_per_dataset_fragment,
        num_batches_per_tarfile=args.num_batches_per_tarfile,
        tar_file_prefix=args.tar_file_prefix,
        global_rank=0,
        world_size=1,
        n_jobs=args.n_preproc_jobs,
        encoder_tokenizer_legacy=args.encoder_tokenizer_legacy,
        decoder_tokenizer_legacy=args.decoder_tokenizer_legacy,
    )
