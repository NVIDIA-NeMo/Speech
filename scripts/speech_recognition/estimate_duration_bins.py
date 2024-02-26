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
from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate duration bins for Lhotse dynamic bucketing using a sample of the input dataset. "
        "The dataset is read either from one or more manifest files and supports data weighting."
    )
    parser.add_argument(
        "input",
        help='Same input format as in model configs under model.train_ds.manifest_filepath. Options: '
        '1) "path.json"; '
        '2) "[[path1.json],[path2.json],...]"; '
        '3) "[[path1.json,weight1],[path2.json,weight2],...]"',
    )
    parser.add_argument("-b", "--buckets", type=int, default=30, help="The desired number of buckets.")
    parser.add_argument(
        "-n",
        "--num_examples",
        type=int,
        default=-1,
        help="The number of examples (utterances) to estimate the bins. -1 means use all data.",
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
        "-q", "--quiet", type=bool, default=False, help="When specified, only print the estimated duration bins."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.merge(
        OmegaConf.structured(LhotseDataLoadingConfig),
        OmegaConf.from_dotlist([f"manifest_filepath={args.input}", "missing_sampling_rate_ok=true"]),
    )
    cuts, _ = read_cutset_from_config(config)
    min_dur, max_dur = args.min_duration, args.max_duration
    discarded, tot = 0, 0

    def duration_ok(cut) -> bool:
        nonlocal discarded, tot
        ans = min_dur <= cut.duration <= max_dur
        if not ans:
            discarded += 1
        tot += 1
        return ans

    cuts = cuts.filter(duration_ok)
    if (N := args.num_examples) > 0:
        cuts = cuts.subset(first=N)
    duration_bins = estimate_duration_buckets(cuts, num_buckets=args.buckets)
    duration_bins = f"[{','.join(str(round(b, ndigits=5)) for b in duration_bins)}]"
    if args.quiet:
        print(duration_bins)
        return
    if discarded:
        ratio = discarded / tot
        print(f"Note: we discarded {discarded}/{tot} ({ratio:.2%}) utterances due to min/max duration filtering.")
    print("Use the following options in your config:")
    print(f"\tnum_buckets={args.buckets}")
    print(f"\tbucket_duration_bins={duration_bins}")
    print("Computing utterance duration distribution...")
    cuts.describe()  # prints a nice table with duration stats + other info


if __name__ == "__main__":
    main()
