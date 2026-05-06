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
"""
Estimate Lhotse dynamic-bucketing bins for SALM-style multimodal training.

This script is the speechlm2 counterpart of:
  * scripts/speech_llm/estimate_token_bins.py        (text-only, 1D/2D)
  * scripts/speech_recognition/estimate_duration_bins_2d.py  (audio, 2D + outlier filtering)

Key properties for the speechlm2 SALM recipe (use_multimodal_sampling=True):

  * Audio cuts and text examples share a single integer-token length axis,
    obtained via ``token_equivalent_duration`` (audio frames cast to tokens).
  * 1D output is a flat integer list ``[i1, ..., iB]``.
  * 2D output is a list of integer pairs ``[[itok_max, otok_max], ...]``
    (input_tokens vs output_tokens), with per-bucket Z-score outlier filtering
    on output-tokens-per-input-token (TPT) and skipped-bucket merging when the
    underlying distribution produces duplicate dim-0 bins.
"""

import argparse
import ast
import math
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from lhotse.cut import Cut
from omegaconf import OmegaConf

import nemo.collections.speechlm2.data.salm_dataset  # noqa: F401  (registers lhotse_as_conversation)
from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig, tokenize, tokenize_with_prompt
from nemo.collections.common.data.lhotse.sampling import (
    DurationFilter,
    MultimodalFixedBucketBatchSizeConstraint2D,
    MultimodalSamplingConstraint,
    TokenCountFilter,
    TokenPerTokenFilter,
)
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.tokenizers import AggregateTokenizer, AutoTokenizer, SentencePieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate Lhotse dynamic-bucketing bins for the speechlm2 SALM recipe. "
        "Supports both 1D (input-token) and 2D ((input_tokens, output_tokens)) bucketing for "
        "mixed audio + text data, using MultimodalSamplingConstraint / "
        "MultimodalFixedBucketBatchSizeConstraint2D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to a data input configuration YAML file with an 'input_cfg' block "
        "(same shape as data.train_ds.input_cfg in a training config).",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        nargs="+",
        required=True,
        help="Path(s) to SPE tokenizer(s) or HuggingFace repo id. More than one path requires --langs "
        "and constructs an AggregateTokenizer.",
    )
    parser.add_argument("-a", "--langs", nargs="+", help="Language names for each AggregateTokenizer sub-tokenizer.")
    parser.add_argument(
        "-b",
        "--buckets",
        type=int,
        default=30,
        help="The desired number of buckets (dim0 => covers input sequence length / audio duration in tokens).",
    )
    parser.add_argument(
        "-s",
        "--sub-buckets",
        type=int,
        default=None,
        help="The desired number of sub-buckets (dim1 => covers output sequence length / num_tokens). "
        "If not provided, we'll only perform 1D bucketing.",
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
        help="If specified, we'll filter out examples with fewer tokens than this number.",
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
        help="If specified, we'll filter out examples with more output tokens per input token than this.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=-float("inf"),
        help="If specified, we'll filter out audio cuts shorter than this many seconds.",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=float("inf"),
        help="If specified, we'll filter out audio cuts longer than this many seconds.",
    )
    parser.add_argument(
        "--token_equivalent_duration",
        type=float,
        default=0.08,
        help="Audio seconds equivalent to one text token; used to convert audio duration to tokens. "
        "Should match the data.train_ds.token_equivalent_duration of the recipe (default 0.08 = "
        "8x40ms encoder frame, matching nvidia/canary-1b-v2).",
    )
    parser.add_argument(
        "--token_outlier_threshold",
        type=float,
        default=6.0,
        help="(2D mode only) Z-score threshold for output-tokens-per-input-token (TPT) outliers; "
        "per top-level bucket, examples > N sigma above the mean are excluded from sub-bucket "
        "estimation. Lower values are more aggressive.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="The supervision/cut field that holds transcripts. Must match the recipe's "
        "data.train_ds.text_field (e.g. 'answer' for the SALM nemotron-nano-v3 recipe), "
        "otherwise lhotse_as_conversation builds turns whose 'message' slot is None and "
        "the prompt formatter crashes.",
    )
    parser.add_argument(
        "--lang-field",
        type=str,
        default="lang",
        help="The supervision/cut field that holds the language code. Must match the recipe's "
        "data.train_ds.lang_field (e.g. 'target_lang' for the SALM nemotron-nano-v3 recipe).",
    )
    parser.add_argument(
        "--audio-locator-tag",
        type=str,
        default=None,
        help="Audio placeholder token. Propagates to datasets in input_cfg (e.g. lhotse_as_conversation), "
        "so AudioTurns get a non-null message slot. Required for any conversation-style input that "
        "interleaves audio with text.",
    )
    parser.add_argument(
        "-q", "--quiet", type=bool, default=False, help="When specified, only print the estimated bins."
    )
    parser.add_argument(
        "-f",
        "--prompt-format",
        type=str,
        help="When specified, use a prompt formatter in addition to the tokenizer. Required for "
        "accurate measurement of decoder-style models like Nemotron Nano v3 / Canary-1B.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt slots provided as a Python list of dicts (used together with --prompt-format). "
        "Example: [{'role':'user','slots':{'source_lang':'en','target_lang':'en','task':'asr','pnc':'yes'}}]",
    )
    parser.add_argument(
        "-m",
        "--measure-total-length",
        type=bool,
        default=False,
        help="If True, measure context+answer length instead of context-only. Set to True for "
        "decoder-only models, False for encoder-decoder.",
    )
    parser.add_argument(
        "--quantize-bins",
        type=str,
        choices=["none", "pow2", "pow2sum"],
        default="none",
        help="Post-quantize the estimated bin caps so they round to model-friendly sizes. "
        "Floor for any non-'none' mode is 2**5 = 32. Modes: "
        "'none' = leave raw integers; "
        "'pow2' = nearest power of 2 (32, 64, 128, 256, ...); "
        "'pow2sum' = nearest single power of 2 OR sum of two distinct powers of 2 with each "
        "exponent >= 5 (e.g. 32, 64, 96, 128, 160, 192, 256, 288, ...). Duplicates produced "
        "by quantization are collapsed.",
    )
    parser.add_argument(
        "--source-config",
        type=str,
        default=None,
        help="If provided together with --output-config, also write a patched copy of this "
        "experiment YAML with data.train_ds.{num_buckets, bucket_duration_bins} updated to "
        "the estimated values (and data.train_ds.bucket_batch_size dropped, since its length "
        "no longer matches and must be re-tuned).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Path to write the patched copy of --source-config to. Required iff --source-config " "is set.",
    )
    return parser.parse_args()


def find_non_outliers_z_score(data, threshold=4.0):
    # Note: we don't apply abs() here because we only filter the upper end of the distribution.
    # We don't mind low ratios for bucketing purposes.
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.where(z_scores <= threshold)


def estimate_token_buckets_1d(
    cuts: Iterable[Cut],
    num_buckets: int,
    token_equivalent_duration: float,
    measure_total_length: bool,
    quiet: bool,
) -> list[int]:
    """1D bucketing: equal-token-mass bins along a single input-length axis.

    Mirrors estimate_duration_buckets in lhotse but operates in token units via
    MultimodalSamplingConstraint, which converts audio cuts to tokens through
    token_equivalent_duration and (optionally) sums context+answer when
    measure_total_length=True.
    """
    assert num_buckets > 1
    constraint = MultimodalSamplingConstraint(
        token_equivalent_duration=token_equivalent_duration,
        measure_total_length=measure_total_length,
    )

    sizes = []
    for c in cuts:
        sizes.append(constraint.measure_length(c))
    sizes = np.array(sizes, dtype=np.int32)
    sizes.sort()

    size_per_bucket = sizes.sum() / num_buckets

    if not quiet:
        print("Input-token distribution:")
        print(pd.Series(sizes).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    bins: list[int] = []
    tot = 0
    for size in sizes:
        if tot > size_per_bucket:
            bins.append(int(size))
            tot = 0
        tot += size
    bins.append(int(sizes[-1]))
    return bins


def estimate_token_buckets_2d(
    cuts: Iterable[Cut],
    num_buckets: int,
    num_subbuckets: int,
    token_equivalent_duration: float,
    measure_total_length: bool,
    token_outlier_threshold: float,
    quiet: bool,
) -> list[tuple[int, int]]:
    """2D bucketing on (input_tokens, output_tokens) with per-bucket TPT outlier filtering.

    Combines:
      * MultimodalFixedBucketBatchSizeConstraint2D from the speech_llm script
        (handles both audio Cuts and text Formattable examples).
      * The outlier filtering and skipped-bucket merging from
        scripts/speech_recognition/estimate_duration_bins_2d.py, adapted from
        seconds-per-token (TPS) to output-tokens-per-input-token (TPT) since
        dim0 is now in tokens, not seconds.
    """
    assert num_buckets > 1
    assert num_subbuckets is not None and num_subbuckets >= 1

    constraint = MultimodalFixedBucketBatchSizeConstraint2D(
        [(0.0, 0.0)],
        [0],
        token_equivalent_duration=token_equivalent_duration,
        measure_total_length=measure_total_length,
    )

    num_input_tokens = []
    num_output_tokens = []
    for c in cuts:
        itoks, otoks = constraint.measure_length(c)
        num_input_tokens.append(itoks)
        num_output_tokens.append(otoks)
    num_input_tokens = np.array(num_input_tokens, dtype=np.int32)
    num_output_tokens = np.array(num_output_tokens, dtype=np.int32)

    # Sort jointly by input length so we can iterate in order and slice the
    # output-token array per top-level bucket.
    joint = np.rec.fromarrays([num_input_tokens, num_output_tokens])
    joint.sort()
    num_input_tokens = joint.f0
    num_output_tokens = joint.f1

    size_per_bucket = num_input_tokens.sum() / num_buckets
    max_input_tokens = int(num_input_tokens[-1])

    if not quiet:
        print("Input-token distribution:")
        print(pd.Series(num_input_tokens).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        tpt_all = num_output_tokens / np.maximum(num_input_tokens, 1)
        print("Output tokens per input token (TPT) distribution:")
        print(pd.Series(tpt_all).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    # First pass: choose dim-0 (input-token) bin edges using equal-mass slicing.
    input_bins: list[int] = []
    bin_indexes: list[int] = [0]
    tot = 0.0
    for binidx, size in enumerate(num_input_tokens):
        if tot > size_per_bucket:
            input_bins.append(int(size))
            bin_indexes.append(binidx)
            tot = 0.0
        tot += size

    if not quiet:
        print(f"Initial input_bins={input_bins}")

    bins: list[tuple[int, int]] = []

    def _estimate_output_token_buckets(max_bucket_input, start_idx, end_idx, corr_subbuckets):
        # Slice this bucket and discard top TPT outliers (Z-score on output/input).
        itoks_bucket_all = num_input_tokens[start_idx:end_idx]
        otoks_bucket_all = num_output_tokens[start_idx:end_idx]
        if len(itoks_bucket_all) == 0:
            return
        tpt_all = otoks_bucket_all / np.maximum(itoks_bucket_all, 1)
        non_outlier_indexes = find_non_outliers_z_score(tpt_all, threshold=token_outlier_threshold)
        otoks_bucket = otoks_bucket_all[non_outlier_indexes]
        itoks_bucket = itoks_bucket_all[non_outlier_indexes]
        if len(otoks_bucket) == 0:
            # Pathological case: all examples in this bucket got flagged. Fall back to the raw slice.
            otoks_bucket = otoks_bucket_all
            itoks_bucket = itoks_bucket_all
        max_tpt_bucket = (otoks_bucket / np.maximum(itoks_bucket, 1)).max()
        # Sort within-bucket by output tokens for sub-bucketing.
        otoks_bucket_sorted = np.sort(otoks_bucket)
        if not quiet:
            outlier_tpt = np.delete(tpt_all, non_outlier_indexes)
            print(
                f"[bucket <= {max_bucket_input} tok] [otoks: {int(otoks_bucket_sorted.min())} - "
                f"{int(otoks_bucket_sorted.max())}] [approx-max-tpt: {max_tpt_bucket:.3f}] "
                f"Discarded {end_idx - start_idx - len(otoks_bucket_sorted)} outliers",
                end=" ",
            )
            if len(outlier_tpt) > 0:
                print(f"(min-outlier: {outlier_tpt.min():.3f}, max-outlier: {outlier_tpt.max():.3f}).", end="")
            print()

        tokens_per_subbucket = otoks_bucket_sorted.sum() / corr_subbuckets
        tot_toks = 0
        for num_toks in otoks_bucket_sorted:
            if tot_toks > tokens_per_subbucket:
                bins.append((max_bucket_input, int(num_toks)))
                tot_toks = 0
            tot_toks += num_toks
        bins.append((max_bucket_input, int(otoks_bucket_sorted[-1])))

    # Second pass: walk the dim-0 bins, merging consecutive bins with identical
    # input-token caps (skewed distributions can produce duplicates) and
    # multiplying their sub-bucket allowance accordingly.
    skipped_buckets = 1
    start_idx = 0
    for i, (input_bin, binidx) in enumerate(zip(input_bins, bin_indexes[1:])):
        is_last = i == len(input_bins) - 1
        if (not is_last and input_bins[i + 1] == input_bin) or (is_last and max_input_tokens == input_bin):
            skipped_buckets += 1
            continue
        _estimate_output_token_buckets(
            max_bucket_input=input_bin,
            start_idx=start_idx,
            end_idx=binidx,
            corr_subbuckets=num_subbuckets * skipped_buckets,
        )
        start_idx = binidx
        skipped_buckets = 1

    # Final bucket carries any remaining skipped sub-buckets up to the global max.
    _estimate_output_token_buckets(
        max_bucket_input=max_input_tokens,
        start_idx=start_idx,
        end_idx=len(num_input_tokens),
        corr_subbuckets=num_subbuckets * skipped_buckets,
    )
    return bins


def load_tokenizer(paths: list[str], langs: list[str] = None) -> TokenizerWrapper:
    if len(paths) == 1:
        (p,) = paths
        if Path(p).exists():
            tok = SentencePieceTokenizer(p)
        else:
            # Assume HuggingFace repo id; trust_remote_code is required for
            # custom tokenizers (e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
            tok = AutoTokenizer(p, use_fast=True, trust_remote_code=True)
    else:
        assert langs is not None and len(paths) == len(langs), (
            f"Cannot create AggregateTokenizer; each tokenizer must have a language assigned via "
            f"--langs (got --tokenizer={paths} and --langs={langs})"
        )
        tok = AggregateTokenizer({lang: SentencePieceTokenizer(p) for lang, p in zip(langs, paths)})
    return TokenizerWrapper(tok)


def apply_tokenizer(cut, tokenizer=None, prompt: PromptFormatter = None):
    if prompt is not None:
        cut = tokenize_with_prompt(cut, tokenizer, prompt)
    elif tokenizer is not None:
        cut = tokenize(cut, tokenizer)
    return cut


_POW2_FLOOR_EXP = 5  # 2**5 = 32


def _quantize_value(v: int, mode: str) -> int:
    """Round a single integer bin cap to the nearest mode-allowed value (>= 32)."""
    if mode == "none":
        return int(v)
    floor = 1 << _POW2_FLOOR_EXP
    v = max(int(v), floor)
    if mode == "pow2":
        log2v = math.log2(v)
        lo = 1 << int(math.floor(log2v))
        hi = 1 << int(math.ceil(log2v))
        return lo if (v - lo) <= (hi - v) else hi
    if mode == "pow2sum":
        # Allowed values: single powers 2^a (a >= 5) and sums 2^a + 2^b (a > b >= 5).
        # Generate candidates densely enough to bracket v.
        max_exp = int(math.ceil(math.log2(2 * v))) + 1
        candidates = set()
        for a in range(_POW2_FLOOR_EXP, max_exp + 1):
            candidates.add(1 << a)
            for b in range(_POW2_FLOOR_EXP, a):
                candidates.add((1 << a) + (1 << b))
        return min(candidates, key=lambda c: (abs(c - v), c))
    raise ValueError(f"Unknown --quantize-bins mode: {mode!r}")


def quantize_bins(bins, mode: str):
    """Quantize each bin cap (or pair) and collapse duplicates while preserving order."""
    if mode == "none":
        return bins
    out = []
    seen = set()
    for b in bins:
        if isinstance(b, (tuple, list)):
            qb = tuple(_quantize_value(x, mode) for x in b)
        else:
            qb = _quantize_value(b, mode)
        if qb not in seen:
            out.append(qb)
            seen.add(qb)
    return out


def maybe_patch_config(source_config, output_config, bins, num_buckets_total):
    """Write a copy of ``source_config`` with the bucket fields updated.

    Drops ``data.train_ds.bucket_batch_size`` since its length must match
    ``num_buckets`` and the new bin layout almost certainly invalidates the old
    batch-size schedule -- the user is expected to re-tune it (e.g. via
    ``scripts/speech_recognition/oomptimizer.py``).
    """
    if source_config is None and output_config is None:
        return
    if source_config is None or output_config is None:
        raise SystemExit("--source-config and --output-config must be provided together")
    with open(source_config) as f:
        cfg = yaml.safe_load(f)
    train_ds = cfg.get("data", {}).get("train_ds") if isinstance(cfg, dict) else None
    if train_ds is None:
        print(f"WARNING: {source_config} has no data.train_ds; skipping patch.")
        return
    # Force list-of-lists for 2D bins so YAML doesn't render Python tuples.
    if bins and isinstance(bins[0], (tuple, list)):
        bins_yaml = [list(b) for b in bins]
    else:
        bins_yaml = list(bins)
    train_ds["num_buckets"] = int(num_buckets_total)
    train_ds["bucket_duration_bins"] = bins_yaml
    train_ds.pop("bucket_batch_size", None)
    with open(output_config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote patched config to {output_config}")
    print("Note: bucket_batch_size was dropped -- re-tune it (e.g. via oomptimizer.py) before training.")


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
        pd.set_option("display.float_format", lambda x: "%.2f" % x)

    tokenizer = None
    prompt = None
    if args.tokenizer is not None:
        tokenizer = load_tokenizer(args.tokenizer, args.langs)
        if args.prompt_format is not None:
            prompt_defaults = ast.literal_eval(args.prompt) if args.prompt is not None else None
            prompt = PromptFormatter.resolve(args.prompt_format)(tokenizer._tokenizer, defaults=prompt_defaults)

    assert args.input.endswith(".yaml"), f"Expected a YAML input config, got: {args.input}"
    dotlist = [
        f"input_cfg={args.input}",
        "force_finite=True",
        "metadata_only=True",
        f"text_field={args.text_field}",
        f"lang_field={args.lang_field}",
        # Propagate to NeMoMultimodalConversation so its total_length / input_length
        # can convert audio turns to token equivalents (otherwise the constraint hits
        # the "token_equivalent_duration must be set" assert in _compute_num_audio_tokens).
        f"token_equivalent_duration={args.token_equivalent_duration}",
    ]
    if args.audio_locator_tag is not None:
        dotlist.append(f"audio_locator_tag={args.audio_locator_tag}")
    config = OmegaConf.merge(
        OmegaConf.structured(LhotseDataLoadingConfig),
        OmegaConf.from_dotlist(dotlist),
    )
    cuts, _ = read_cutset_from_config(config)

    duration_filter = RejectionsCounter(DurationFilter(args.min_duration, args.max_duration), "Duration filtering")
    cuts = cuts.filter(duration_filter)
    cuts = cuts.map(partial(apply_tokenizer, tokenizer=tokenizer, prompt=prompt), apply_fn=None)
    token_filter = RejectionsCounter(
        TokenCountFilter(args.min_tokens, args.max_tokens, args.measure_total_length), "Token count filtering"
    )
    cuts = cuts.filter(token_filter)
    tpt_filter = RejectionsCounter(TokenPerTokenFilter(-1, args.max_tpt), "Output tokens per input token filtering")
    cuts = cuts.filter(tpt_filter)
    if (N := args.num_examples) > 0:
        cuts = islice(cuts, N)

    is_2d = args.sub_buckets is not None
    if is_2d:
        bins = estimate_token_buckets_2d(
            cuts,
            num_buckets=args.buckets,
            num_subbuckets=args.sub_buckets,
            token_equivalent_duration=args.token_equivalent_duration,
            measure_total_length=args.measure_total_length,
            token_outlier_threshold=args.token_outlier_threshold,
            quiet=args.quiet,
        )
    else:
        bins = estimate_token_buckets_1d(
            cuts,
            num_buckets=args.buckets,
            token_equivalent_duration=args.token_equivalent_duration,
            measure_total_length=args.measure_total_length,
            quiet=args.quiet,
        )

    if args.quantize_bins != "none":
        before = len(bins)
        bins = quantize_bins(bins, args.quantize_bins)
        if not args.quiet:
            print(f"Quantization '{args.quantize_bins}': {before} -> {len(bins)} bins after dedupe.")

    if is_2d:
        bins_str = "[" + ",".join(f"[{b:d},{sb:d}]" for b, sb in bins) + "]"
    else:
        bins_str = "[" + ",".join(f"{b:d}" for b in bins) + "]"
    num_buckets_total = len(bins)

    if args.quiet:
        print(bins_str)
        maybe_patch_config(args.source_config, args.output_config, bins, num_buckets_total)
        return

    duration_filter.print_report()
    token_filter.print_report()
    tpt_filter.print_report()
    print("Use the following options in your config:")
    print(f"\tuse_bucketing=1")
    print(f"\tnum_buckets={num_buckets_total}")
    print(f"\tbucket_duration_bins={bins_str}")
    maybe_patch_config(args.source_config, args.output_config, bins, num_buckets_total)


if __name__ == "__main__":
    main()
