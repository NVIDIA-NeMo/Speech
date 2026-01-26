# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# flake8: noqa
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301

import argparse
import ast
import math
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from lhotse.cut import Cut
from omegaconf import OmegaConf

import nemo.collections.speechlm2.data.salm_dataset  # noqa
from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig, tokenize, tokenize_with_prompt
from nemo.collections.common.data.lhotse.sampling import (
    MultimodalFixedBucketBatchSizeConstraint2D,
    MultimodalSamplingConstraint,
    TokenCountFilter,
    TokenPerTokenFilter,
)
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.tokenizers import AggregateTokenizer, AutoTokenizer, SentencePieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate token bins for Lhotse dynamic bucketing using a sample of the input dataset. "
        "The dataset is read either from one or more manifest files and supports data weighting. "
        "Unlike estimate_duration_bins.py, this script is intended for text data only. "
        "It supports 2D bucketing. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help='Path to a data input configuration YAML file. '
        'This is the only type of input specification supported for text data.',
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        nargs="+",
        required=True,
        help="Path to one or more SPE tokenizers. More than one means we'll use AggregateTokenizer and --langs argument must also be used. When provided, we'll estimate a 2D distribution for input and output sequence lengths.",
    )
    parser.add_argument(
        "-a", "--langs", nargs="+", help="Language names for each of AggregateTokenizer sub-tokenizers."
    )
    parser.add_argument(
        "-b",
        "--buckets",
        type=int,
        default=30,
        help="The desired number of buckets (dim0 => covers input sequence length / audio duration).",
    )
    parser.add_argument(
        "-s",
        "--sub-buckets",
        type=int,
        default=None,
        help="The desired number of sub-buckets (dim1 => covers output sequence length / num_tokens). "
        "If not provided, we'll only perform 1D bucketing. ",
    )
    parser.add_argument(
        "-n",
        "--num_examples",
        type=int,
        default=-1,
        help="The number of examples (utterances) to estimate the bins. -1 means use all data "
        "(be careful: it could be iterated over infinitely).",
    )
    parser.add_argument(
        "-l",
        "--min_tokens",
        type=float,
        default=-float("inf"),
        help="If specified, we'll filter out examples with less tokens than this number.",
    )
    parser.add_argument(
        "-u",
        "--max_tokens",
        type=float,
        default=float("inf"),
        help="If specified, we'll filter out examples with more tokens than this number.",
    )
    parser.add_argument(
        "--max_tpt",
        type=float,
        default=float("inf"),
        help="If specified, we'll filter out examples with more output tokens per input token than this. ",
    )
    parser.add_argument(
        "-q", "--quiet", type=bool, default=False, help="When specified, only print the estimated duration bins."
    )
    parser.add_argument(
        "-f",
        "--prompt-format",
        type=str,
        help="When specified, we'll use a prompt formatter in addition to the tokenizer for the purpose of estimating token count bins. "
        "This is useful for accurate 2D bucket estimation with models such as EncDecMultiTaskModel (Canary-1B), "
        "or any model where the label sequence consists of a user prompt and a model's response.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt slots provided as a Python list of dicts. It is used together with --prompt-format option."
        "For example, with Canary-1B you may use: [{'role':'user','slots':{'source_lang':'en','target_lang':'en','task':'asr','pnc':'yes'}]",
    )
    parser.add_argument(
        "-m",
        "--measure-total-length",
        type=bool,
        default=False,
        help="When specified, we'll measure the total length (context+answer, i.e. input_ids) instead of context-only length. Total length is more suitable for decoder-only models while context-only length is more suitable for encoder-decoder models.",
    )
    return parser.parse_args()


def estimate_token_buckets(
    cuts: Iterable[Cut],
    num_buckets: int,
    num_subbuckets: int | None,
    quiet: bool,
) -> list[tuple[float, float]]:
    """
    This function is based on lhotse.dataset.sampling.dynamic_bucketing.estimate_duration_buckets.
    It extends it to a 2D bucketing case.
    """
    assert num_buckets > 1
    is_2d = num_subbuckets is not None

    if is_2d:
        constraint = MultimodalFixedBucketBatchSizeConstraint2D([(0.0, 0.0)], [0], measure_total_length=False)
    else:
        constraint = MultimodalSamplingConstraint(measure_total_length=True)

    # Gather the duration and token count statistics for the dataset.
    num_input_tokens = []
    num_output_tokens = []
    for c in cuts:
        ans = constraint.measure_length(c)
        if is_2d:
            itoks, otoks = ans
            num_input_tokens.append(itoks)
            num_output_tokens.append(otoks)
        else:
            num_input_tokens.append(ans)
    num_input_tokens = np.array(num_input_tokens, dtype=np.int32)
    if is_2d:
        num_output_tokens = np.array(num_output_tokens, dtype=np.int32)
        joint = np.rec.fromarrays([num_input_tokens, num_output_tokens])
        joint.sort()
        num_input_tokens = joint.f0
        num_output_tokens = joint.f1
    else:
        num_input_tokens.sort()

    # We are building buckets with equal duration (empirically leads to more even bucket exhaustion over time).
    # We need to determine how much duration to allocate per bucket.
    size_per_bucket = num_input_tokens.sum() / num_buckets

    if not quiet:
        print("Duration distribution:")
        print(pd.Series(num_input_tokens).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    max_input_tokens = num_input_tokens[-1]

    if is_2d:
        tpt = num_output_tokens / num_input_tokens
        if not quiet:
            print("Output tokens per input token distribution:")
            print(pd.Series(tpt).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        max_tpt = tpt.max()
        del tpt

    bins = []
    bin_indexes = [0]
    tot = 0.0

    def _estimate_output_token_buckets(max_bucket_duration):
        # Since this is 2D bucketing, apply the same bin creation logic
        # for the second dimension (i.e. token count) as for the first dimension (duration).
        # That means we aim to have each bucket contain roughly the same number of tokens.
        # Note that this estimation is biased towards more padding if you have
        # a lot of zero-token examples (e.g. non-speech).
        nonlocal bins
        num_tokens_bucket = num_output_tokens[bin_indexes[-1] : binidx]
        num_tokens_bucket.sort()
        tokens_per_subbucket = num_tokens_bucket.sum() / num_subbuckets
        tot_toks = 0
        # Iterate over token counts, and whenever we hit tokens_per_subbucket, create a new 2D bucket bin.
        for num_toks in num_tokens_bucket:
            # Threshold hit: we are creating a new (max_duration, max_num_tokens) bin.
            if tot_toks > tokens_per_subbucket:
                bins.append((max_bucket_duration, num_toks))
                tot_toks = 0
            tot_toks += num_toks
        bins.append((size, math.ceil(size * max_tpt)))

    # Iterate over data, and whenever we hit size_per_bucket, create a new bucket bin.
    for binidx, size in enumerate(num_input_tokens):
        if tot > size_per_bucket:
            # Threshold hit: we are creating a new duration bin (multiplied by number of token bins).
            if is_2d:
                _estimate_output_token_buckets(max_bucket_duration=size)
            else:
                bins.append(size)
            tot = 0.0
        tot += size

    # Estimate an extra 2D bin set for global max duration.
    if num_subbuckets is not None:
        if is_2d:
            _estimate_output_token_buckets(max_bucket_duration=max_input_tokens)
        else:
            bins.append(max_input_tokens)

    return bins


def load_tokenizer(paths: list[str], langs: list[str] = None) -> TokenizerWrapper:
    if len(paths) == 1:
        (p,) = paths
        if Path(p).exists():
            tok = SentencePieceTokenizer(p)
        else:
            # Assume it's HF name
            tok = AutoTokenizer(p, use_fast=True)
    else:
        assert langs is not None and len(paths) == len(
            langs
        ), f"Cannot create AggregateTokenizer; each tokenizer must have assigned a language via --langs option (we got --tokenizers={paths} and --langs={langs})"
        tok = AggregateTokenizer({lang: SentencePieceTokenizer(p) for lang, p in zip(langs, paths)})
    return TokenizerWrapper(tok)


def apply_tokenizer(cut, tokenizer=None, prompt: PromptFormatter = None):
    if prompt is not None:
        cut = tokenize_with_prompt(cut, tokenizer, prompt)
    elif tokenizer is not None:
        cut = tokenize(cut, tokenizer)
    return cut


class RejectionsCounter:
    def __init__(self, predicate: Callable, message: str):
        self.predicate = predicate
        self.message = message
        self.total = 0
        self.rejected = 0

    def __call__(self, example) -> bool:
        ans = self.predicate(example)
        self.total += 1
        if not ans:
            self.rejected += 1
        return ans

    def print_report(self) -> None:
        if self.rejected:
            print(f"{self.message} | Rejected {self.rejected}/{self.total} examples.")


def main():
    args = parse_args()

    if not args.quiet:
        pd.set_option('display.float_format', lambda x: '%.2f' % x)

    tokenizer = None
    prompt = None
    if args.tokenizer is not None:
        tokenizer = load_tokenizer(args.tokenizer, args.langs)
        if args.prompt_format is not None:
            prompt_defaults = None
            if args.prompt is not None:
                prompt_defaults = ast.literal_eval(args.prompt)
            prompt = PromptFormatter.resolve(args.prompt_format)(tokenizer._tokenizer, defaults=prompt_defaults)

    assert args.input.endswith(".yaml")
    config = OmegaConf.merge(
        OmegaConf.structured(LhotseDataLoadingConfig),
        OmegaConf.from_dotlist([f"input_cfg={args.input}", "force_finite=True", "metadata_only=True"]),
    )
    cuts, _ = read_cutset_from_config(config)
    cuts = cuts.map(partial(apply_tokenizer, tokenizer=tokenizer, prompt=prompt), apply_fn=None)
    token_filter = RejectionsCounter(
        TokenCountFilter(args.min_tokens, args.max_tokens, args.measure_total_length), "Token count filtering"
    )
    cuts = cuts.filter(token_filter)
    tpt_filter = RejectionsCounter(TokenPerTokenFilter(-1, args.max_tpt), "Output tokens per input token filtering")
    cuts = cuts.filter(tpt_filter)
    if (N := args.num_examples) > 0:
        cuts = islice(cuts, N)

    token_bins = estimate_token_buckets(
        cuts,
        num_buckets=args.buckets,
        num_subbuckets=args.sub_buckets,
        quiet=args.quiet,
    )
    if args.sub_buckets is not None:
        token_bins = "[" + ','.join(f"[{b:d},{sb:d}]" for b, sb in token_bins) + "]"
    else:
        token_bins = "[" + ','.join(f"{b:d}" for b in token_bins) + "]"
    if args.quiet:
        print(token_bins)
        return
    token_filter.print_report()
    tpt_filter.print_report()
    print("Use the following options in your config:")
    print(f"\tnum_buckets={args.buckets}")
    print(f"\tbucket_duration_bins={token_bins}")


if __name__ == "__main__":
    main()
