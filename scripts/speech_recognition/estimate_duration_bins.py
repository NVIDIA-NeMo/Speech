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
from itertools import islice
from pathlib import Path

from lhotse.cut import Cut
from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate duration bins for Lhotse dynamic bucketing using a sample of the input dataset. "
        "The dataset is read either from one or more manifest files and supports data weighting.",
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
    parser.add_argument("-b", "--buckets", type=int, default=30, help="The desired number of buckets.")
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
        "-q", "--quiet", type=bool, default=False, help="When specified, only print the estimated duration bins."
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
        OmegaConf.from_dotlist([inp_arg, "metadata_only=true"]),
    )
    cuts, _ = read_cutset_from_config(config)
    min_dur, max_dur = args.min_duration, args.max_duration
    nonaudio, discarded, tot = 0, 0, 0

    def duration_ok(cut) -> bool:
        nonlocal nonaudio, discarded, tot
        tot += 1
        if not isinstance(cut, Cut):
            nonaudio += 1
            return False
        if not (min_dur <= cut.duration <= max_dur):
            discarded += 1
            return False
        return True

    cuts = cuts.filter(duration_ok)
    if (N := args.num_examples) > 0:
        cuts = islice(cuts, N)
    duration_bins = estimate_duration_buckets(cuts, num_buckets=args.buckets)
    duration_bins = f"[{','.join(str(round(b, ndigits=5)) for b in duration_bins)}]"
    if args.quiet:
        print(duration_bins)
        return
    if discarded:
        ratio = discarded / tot
        print(f"Note: we discarded {discarded}/{tot} ({ratio:.2%}) utterances due to min/max duration filtering.")
    if nonaudio:
        print(f"Note: we discarded {nonaudio} non-audio examples found during iteration.")
    print(f"Used {tot - nonaudio - discarded} examples for the estimation.")
    print("Use the following options in your config:")
    print(f"\tnum_buckets={args.buckets}")
    print(f"\tbucket_duration_bins={duration_bins}")


if __name__ == "__main__":
    main()
