# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import ast
import math
import warnings
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from lhotse.cut import Cut
from omegaconf import OmegaConf

from nemo.collections.common.data import apply_prompt_format_fn
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig, tokenize
from nemo.collections.common.data.lhotse.sampling import DurationFilter, FixedBucketBatchSizeConstraint2D
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.tokenizers import (
    AggregateTokenizer,
    CanaryTokenizer,
    SentencePieceTokenizer,
    TokenizerSpec,
)
from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate duration bins for Lhotse dynamic bucketing using a sample of the input dataset. "
        "The dataset is read either from one or more manifest files and supports data weighting. "
        "Unlike estimate_duration_bins.py, this script prepares the setup for 2D bucketing. "
        "This means that each main bucket for audio duration is sub-divided into sub-buckets "
        "for the number of output tokens (supporting BPE and Aggregated tokenizers). "
        "2D bucketing is especially useful for encoder-decoder models where input audio duration is often "
        "not sufficient to stratify the sampling with an optimal GPU utilization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help='Data input. Options: '
        '1) "path.json" - any single NeMo manifest; '
        '2) "[[path1.json],[path2.json],...]" - any collection of NeMo manifests; '
        '3) "[[path1.json,weight1],[path2.json,weight2],...]" - any collection of weighted NeMo manifests; '
        '4) "input_cfg.yaml" - a new option supporting input configs, same as in model training \'input_cfg\' arg; '
        '5) "path/to/shar_data" - a path to Lhotse Shar data directory; '
        '6) "key=val" - in case none of the previous variants cover your case: "key" is the key you\'d use in NeMo training config with its corresponding value ',
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
        default=2,
        help="The desired number of sub-buckets (dim1 => covers output sequence length / num_tokens).",
    )
    parser.add_argument("--text-field", default="text", help="The key in manifests to read transcripts from.")
    parser.add_argument("--lang-field", default="lang", help="The key in manifests to read language from.")
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
        "--min_duration",
        type=float,
        default=-float("inf"),
        help="If specified, we'll filter out utterances shorter than this.",
    )
    parser.add_argument(
        "-u",
        "--max_duration",
        type=float,
        default=float("inf"),
        help="If specified, we'll filter out utterances longer than this.",
    )
    parser.add_argument(
        "--max_tps", type=float, default=None, help="Deprecated. TPS is automatically determined per bucket."
    )
    parser.add_argument(
        "--token_outlier_threshold",
        type=float,
        default=4.0,
        help="The lower this is, the more outliers in transcript token count will be filtered out. "
        "By default allow token counts at 4 sigma away from distribution mean, computed separately for every bucket.",
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
    return parser.parse_args()


def sort_two_arrays(A, B):
    joint = np.rec.fromarrays([A, B])
    joint.sort()
    return joint.f0, joint.f1


def estimate_duration_buckets(
    cuts: Iterable[Cut],
    num_buckets: int,
    num_subbuckets: int,
    max_tps: float,
    max_duration: float,
    token_outlier_threshold: float,
    quiet: bool,
) -> list[tuple[float, float]]:
    """
    This function is based on lhotse.dataset.sampling.dynamic_bucketing.estimate_duration_buckets.
    It extends it to a 2D bucketing case.
    """
    assert num_buckets > 1

    constraint = FixedBucketBatchSizeConstraint2D([(0.0, 0.0)], [0])

    # Gather the duration and token count statistics for the dataset.
    sizes = []
    num_tokens = []
    for c in cuts:
        dur, toks = constraint.measure_length(c)
        sizes.append(dur)
        num_tokens.append(toks)
    sizes = np.array(sizes, dtype=np.float32)
    num_tokens = np.array(num_tokens, dtype=np.int32)
    sizes, num_tokens = sort_two_arrays(sizes, num_tokens)

    # We are building buckets with equal duration (empirically leads to more even bucket exhaustion over time).
    # We need to determine how much duration to allocate per bucket.
    size_per_bucket = sizes.sum() / num_buckets

    if not quiet:
        print("Duration distribution:")
        print(pd.Series(sizes).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]))
    if math.isinf(max_duration):
        max_duration = round(sizes[-1], 3)  # Round to 3 decimal places to be consistent for the output format.

    bins = []
    tps_thresholds = []
    bin_indexes = [0]
    tot = 0.0

    def _estimate_token_buckets(max_bucket_duration, start_idx, end_idx, corr_subbuckets=None):
        # Since this is 2D bucketing, apply the same bin creation logic
        # for the second dimension (i.e. token count) as for the first dimension (duration).
        # That means we aim to have each bucket contain roughly the same number of tokens.
        # Note that this estimation is biased towards more padding if you have
        # a lot of zero-token examples (e.g. non-speech).
        nonlocal bins

        if not corr_subbuckets:
            corr_subbuckets = num_subbuckets

        # Start by discarding outlier examples as defined by token-per-second (TPS) attribute.
        # We empirically determined high TPS examples to cause severe OOMs limiting batch sizes.
        # We cap the TPS for each top-level bucket at 4 standard deviations of TPS.
        # Examples exceeding that TPS value will be discarded during sampling at training time.
        num_tokens_bucket_all = num_tokens[start_idx:end_idx]
        sizes_bucket_all = sizes[start_idx:end_idx]
        non_outlier_indexes = find_non_outliers_z_score(
            num_tokens_bucket_all / sizes_bucket_all, threshold=token_outlier_threshold
        )
        num_tokens_bucket = num_tokens_bucket_all[non_outlier_indexes]
        sizes_bucket = sizes_bucket_all[non_outlier_indexes]
        max_tps_bucket = (num_tokens_bucket / sizes_bucket).max()
        num_tokens_bucket, sizes_bucket = sort_two_arrays(num_tokens_bucket, sizes_bucket)
        if not quiet:
            outlier_tps = np.delete(num_tokens_bucket_all / sizes_bucket_all, non_outlier_indexes)
            print(
                f"[bucket <= {max_bucket_duration:.2f}s] [{num_tokens_bucket.min()} - {num_tokens_bucket.max()}] [approx-max-tps: {max_tps_bucket:.2f}] Discarded {end_idx - start_idx - len(num_tokens_bucket)} max token outliers",
                end=" ",
            )
            if len(outlier_tps) > 0:
                print(f"min-outlier: {outlier_tps.min():.2f}, max-outlier: {outlier_tps.max():.2f}).", end="")
            print()

        tokens_per_subbucket = num_tokens_bucket.sum() / corr_subbuckets
        tot_toks = 0
        # Iterate over token counts, and whenever we hit tokens_per_subbucket, create a new 2D bucket bin.
        for num_toks, size in zip(num_tokens_bucket, sizes_bucket):
            # Threshold hit: we are creating a new (max_duration, max_num_tokens) bin.
            if tot_toks > tokens_per_subbucket:
                bins.append((max_bucket_duration, num_toks))
                tps_thresholds.append(max_tps_bucket)
                tot_toks = 0
            tot_toks += num_toks
        bins.append((max_bucket_duration, num_toks))
        tps_thresholds.append(max_tps_bucket)

    duration_bins = []

    # Iterate over data, and whenever we hit size_per_bucket, register it as a new duration bucket.
    for binidx, size in enumerate(sizes):
        if tot > size_per_bucket:
            size = round(size, 3)  # Round to 3 decimal places to be consistent for the output format.
            duration_bins.append(size)
            bin_indexes.append(binidx)
            tot = 0.0
        tot += size

    if not quiet:
        print(f"Initial duration_bins={duration_bins}")

    skipped_buckets = 1
    start_idx = 0

    # Iterate over newly created duration bins to handle cases where some bins have the same value —
    # this usually happens when the data is skewed.
    # If we detect such bins, we skip estimating token buckets for that particular bin.
    # Instead, we keep track of how many bins got skipped because they had the same duration.
    # Then, when we finally hit a bin with a different duration, we treat all those skipped bins as one "combined" bin.
    # For that combined bin, we create more subbuckets — specifically, the number of skipped bins × `num_subbuckets` (set by the user).
    #
    # Example of durations bins created from skewed duration distribution: [5, 20, 30, 30, 30, 40]
    # Here, we'd end up making token subbuckets for: [5, 20, 40]
    # where [20, 40] bucket will have 4 times more subbuckets (as we combined 4 buckets into 1) than usual bucket in that settings.

    for i, (duration_bin, binidx) in enumerate(zip(duration_bins, bin_indexes[1:])):
        if (i != len(duration_bins) - 1 and duration_bins[i + 1] == duration_bin) or (
            i == len(duration_bins) - 1 and max_duration == duration_bin
        ):
            skipped_buckets += 1
            continue
        _estimate_token_buckets(
            max_bucket_duration=duration_bin,
            start_idx=start_idx,
            end_idx=binidx,
            corr_subbuckets=num_subbuckets * skipped_buckets,
        )
        start_idx = binidx
        skipped_buckets = 1

    # Estimate an extra 2D bin set for global max duration.
    # Also, if the last value in duration_bins is equal to max_duration,
    # we need to make sure we properly handle any previously "skipped" buckets that ended at this max value.
    _estimate_token_buckets(
        max_bucket_duration=max_duration,
        start_idx=start_idx,
        end_idx=len(sizes),
        corr_subbuckets=num_subbuckets * skipped_buckets,
    )
    return bins, tps_thresholds


def find_non_outliers_z_score(data, threshold=4):
    # Note: we don't apply abs() here because we only filter the upper end of the distribution.
    # We don't mind low-token-counts for bucketing purposes.
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.where(z_scores <= threshold)


def load_tokenizer(paths: list[str], langs: list[str] = None, is_canary: bool = True) -> TokenizerSpec:
    if len(paths) == 1:
        tok = SentencePieceTokenizer(paths[0])
    else:
        assert langs is not None and len(paths) == len(
            langs
        ), f"Cannot create AggregateTokenizer; each tokenizer must have assigned a language via --langs option (we got --tokenizers={paths} and --langs={langs})"
        if is_canary:
            tokcls = CanaryTokenizer
        else:
            tokcls = AggregateTokenizer
        tok = tokcls({lang: SentencePieceTokenizer(p) for lang, p in zip(langs, paths)})
    return tok


def apply_tokenizer(cut, tokenizer=None, prompt: PromptFormatter = None):
    if prompt is not None:
        encoded = apply_prompt_format_fn(cut, prompt)
        cut.supervisions[0].tokens = encoded["input_ids"]

    elif tokenizer is not None:
        cut = tokenize(cut, TokenizerWrapper(tokenizer))

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

    if args.max_tps is not None:
        warnings.warn(
            "The option --max_tps has been deprecated in favor of "
            "automatic TPS determination that's variable across buckets."
        )

    tokenizer = None
    prompt = None
    if args.tokenizer is not None:
        tokenizer = load_tokenizer(
            paths=args.tokenizer,
            langs=args.langs,
            is_canary=args.prompt_format is not None and 'canary' in args.prompt_format,
        )
        if args.prompt_format is not None:
            prompt_defaults = None
            if args.prompt is not None:
                prompt_defaults = ast.literal_eval(args.prompt)
            prompt = PromptFormatter.resolve(args.prompt_format)(tokenizer, defaults=prompt_defaults)

    if '=' in args.input:
        inp_arg = args.input
    elif args.input.endswith(".yaml"):
        inp_arg = f"input_cfg={args.input}"
    elif Path(args.input).is_dir():
        inp_arg = f"shar_path={args.input}"
    else:
        inp_arg = f"manifest_filepath={args.input}"
    config = OmegaConf.merge(
        OmegaConf.structured(LhotseDataLoadingConfig),
        OmegaConf.from_dotlist(
            [inp_arg, "metadata_only=true", f"text_field={args.text_field}", f"lang_field={args.lang_field}"]
        ),
    )
    cuts, _ = read_cutset_from_config(config)
    duration_filter = RejectionsCounter(DurationFilter(args.min_duration, args.max_duration), "Duration filtering")
    cuts = cuts.filter(duration_filter)
    cuts = cuts.map(partial(apply_tokenizer, tokenizer=tokenizer, prompt=prompt))
    if (N := args.num_examples) > 0:
        cuts = islice(cuts, N)

    duration_bins, tps_thresholds = estimate_duration_buckets(
        cuts,
        num_buckets=args.buckets,
        num_subbuckets=args.sub_buckets,
        max_duration=args.max_duration,
        max_tps=args.max_tps,
        token_outlier_threshold=args.token_outlier_threshold,
        quiet=args.quiet,
    )
    duration_bins = "[" + ','.join(f"[{b:.3f},{sb:d}]" for b, sb in duration_bins) + "]"
    tps_thresholds = "[" + ",".join(f"{t:.2f}" for t in tps_thresholds) + "]"
    if not args.quiet:
        duration_filter.print_report()
    print("Use the following options in your config:")
    print(f"\tuse_bucketing=1")
    print(f"\tnum_buckets={args.buckets}")
    print(f"\tbucket_duration_bins={duration_bins}")
    print(f"The max_tps setting below is optional, use it if your data has low quality long transcript outliers:")
    print(f"\tmax_tps={tps_thresholds}")


if __name__ == "__main__":
    main()
