# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import numpy as np
import pytest
import torch.utils.data
from lhotse import CutSet, Seconds, SupervisionSegment
from lhotse.dataset import DynamicBucketingSampler
from lhotse.testing.dummies import dummy_cut
from lhotse.testing.random import deterministic_rng

from nemo.collections.common.data.lhotse.dataloader import (
    BucketingFilter,
    FixedBucketBatchSizeConstraint2D,
    get_lhotse_dataloader_from_config,
)


def make_cut(id_: int = 0, duration: Seconds = 1.0, num_tokens: int = 10):
    supervision = SupervisionSegment(f"blah-{id_}", f"blah-{id_}", 0.0, duration, text="a" * num_tokens)
    supervision.tokens = np.zeros((num_tokens,), dtype=np.int32).tolist()
    return dummy_cut(id_, duration=duration, supervisions=[supervision])


@pytest.fixture
def cuts():
    return CutSet(
        [make_cut(i, duration=2.0, num_tokens=4) for i in range(20)]
        + [make_cut(i, duration=2.0, num_tokens=8) for i in range(20)]
        + [make_cut(i, duration=2.0, num_tokens=12) for i in range(20)]
        + [make_cut(i, duration=8.0, num_tokens=8) for i in range(20)]
        + [make_cut(i, duration=8.0, num_tokens=12) for i in range(20)]
        + [make_cut(i, duration=8.0, num_tokens=16) for i in range(20)]
        + [make_cut(i, duration=14.0, num_tokens=12) for i in range(20)]
        + [make_cut(i, duration=14.0, num_tokens=16) for i in range(20)]
        + [make_cut(i, duration=14.0, num_tokens=20) for i in range(20)]
    )


def test_2d_bucketing_expected_bucket_allocation(cuts):

    duration_bins = [
        (5.0, 5),
        (5.0, 11),
        (5.0, 15),
        (7.0, 10),
        (7.0, 13),
        (7.0, 20),
        (8.0, 15),
        (8.0, 17),
        (8.0, 25),
        (15.0, 20),
        (15.0, 29),
        (15.0, 30),
    ]
    batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    sampler = DynamicBucketingSampler(
        cuts.repeat(),
        shuffle=True,
        duration_bins=duration_bins,
        constraint=FixedBucketBatchSizeConstraint2D(
            max_seq_len_buckets=duration_bins,
            batch_sizes=batch_sizes,
            strict_2d=False,
        ),
        buffer_size=1000,
        seed=0,
    )

    for batch_idx, batch in enumerate(sampler):
        # Run for 100 batches and check invariants on each.
        if batch_idx == 100:
            break
        # Note: batch_sizes are indexes into duration_bins when subtracting 1.
        # This way we can determine which bucket the data came from in this test.
        bin_index = len(batch) - 1
        max_duration, max_num_tokens = duration_bins[bin_index]
        for cut in batch:
            # First, check that the sampled examples are indeed below the max duration/num_tokens for its bucket.
            assert cut.duration <= max_duration
            assert len(cut.supervisions[0].tokens) <= max_num_tokens
            # Then, find the previous compatible bucket for each of training example's dimensions,
            # and verify that it was not possible to assign the example to that smaller bucket.
            # We should skip this for bucket_idx==0 (no previous buckets available).
            # Note: max will be an empty sequence in some cases, e.g. when it's the first bucket
            # with a given max_duration, it has the smallest max_num_tokens, leaving previous candidates list
            # for max_num_tokens empty.
            if bin_index > 0:
                # Check for smaller duration buckets (1st dim)
                candidates = [dur for dur, tok in duration_bins[:bin_index] if dur < max_duration]
                if candidates:  # Otherwise, it's already using the smallest bucket.
                    prev_max_duration = max(candidates)
                    assert cut.duration > prev_max_duration
                # Check for smaller token buckets (2nd dim)
                candidates = [
                    tok for dur, tok in duration_bins[:bin_index] if dur == max_duration and tok < max_num_tokens
                ]
                if candidates:  # Otherwise, it's already using the smallest bucket
                    prev_max_num_tokens = max(candidates)
                    assert len(cut.supervisions[0].tokens) > prev_max_num_tokens


@pytest.mark.parametrize(
    ["duration", "num_tokens", "should_keep", "bucket_idx"],
    [
        # Buckets for duration range [0.0-5.0]:
        # * Sweep num_tokens
        (2.0, 0, True, 0),
        (2.0, 5, True, 0),
        (2.0, 10, True, 0),
        (2.0, 11, True, 1),
        (2.0, 20, True, 1),
        (2.0, 21, True, 3),
        (2.0, 30, True, 3),
        (2.0, 31, False, None),
        # * Check the upper bound duration 5.0
        (5.0, 0, True, 0),
        (5.0, 5, True, 0),
        (5.0, 10, True, 0),
        (5.0, 11, True, 1),
        (5.0, 20, True, 1),
        (5.0, 21, True, 3),
        (5.0, 30, True, 3),
        (5.0, 31, False, None),
        # Buckets for duration range [5.0, 10.0]
        # * Sweep num_tokens
        (8.0, 0, True, 2),
        (8.0, 15, True, 2),
        (8.0, 16, True, 3),
        (8.0, 30, True, 3),
        (8.0, 31, False, None),
        # * Check the upper bound duration 10.0
        (10.0, 0, True, 2),
        (10.0, 15, True, 2),
        (10.0, 16, True, 3),
        (10.0, 30, True, 3),
        (10.0, 31, False, None),
        # Durations above max duration
        (20.0, 0, False, None),
        (20.0, 1000, False, None),
    ],
)
def test_2d_bucketing_filter_lenient(duration, num_tokens, should_keep, bucket_idx):
    buckets = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30)]
    batch_sizes = [4, 3, 2, 1]
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=False)
    filter_2d = BucketingFilter(constraint)

    cut = make_cut(duration=duration, num_tokens=num_tokens)
    assert filter_2d(cut) == should_keep
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == bucket_idx


@pytest.mark.parametrize(
    ["duration", "num_tokens", "should_keep", "bucket_idx"],
    [
        # Buckets for duration range [0.0-5.0]:
        # * Sweep num_tokens
        (2.0, 0, True, 0),
        (2.0, 5, True, 0),
        (2.0, 10, True, 0),
        (2.0, 11, True, 1),
        (2.0, 20, True, 1),
        (2.0, 21, False, None),  # <-- strict
        (2.0, 30, False, None),  # <-- strict
        (2.0, 31, False, None),
        # * Check the upper bound duration 5.0
        (5.0, 0, True, 0),
        (5.0, 5, True, 0),
        (5.0, 10, True, 0),
        (5.0, 11, True, 1),
        (5.0, 20, True, 1),
        (5.0, 21, False, None),  # <-- strict
        (5.0, 30, False, None),  # <-- strict
        (5.0, 31, False, None),
        # Buckets for duration range [5.0, 10.0]
        # * Sweep num_tokens
        (8.0, 0, True, 2),
        (8.0, 15, True, 2),
        (8.0, 16, True, 3),
        (8.0, 30, True, 3),
        (8.0, 31, False, None),
        # * Check the upper bound duration 10.0
        (10.0, 0, True, 2),
        (10.0, 15, True, 2),
        (10.0, 16, True, 3),
        (10.0, 30, True, 3),
        (10.0, 31, False, None),
        # Durations above max duration
        (20.0, 0, False, None),
        (20.0, 1000, False, None),
    ],
)
def test_2d_bucketing_filter_strict(duration, num_tokens, should_keep, bucket_idx):
    buckets = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30)]
    batch_sizes = [4, 3, 2, 1]
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=True)
    filter_2d = BucketingFilter(constraint)

    cut = make_cut(duration=duration, num_tokens=num_tokens)
    assert filter_2d(cut) == should_keep
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == bucket_idx


def test_2d_bucketing_filter_strict_max_ratio():
    buckets = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30)]
    max_ratio = [4.0, 4.0, 3.0, 3.0]
    batch_sizes = [4, 3, 2, 1]

    # Without max_ratio it works because both dims fit bucket at idx 1
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=True)
    filter_2d = BucketingFilter(constraint)
    cut = make_cut(duration=2.0, num_tokens=20)
    assert filter_2d(cut) == True
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == 1

    # With max_ratio it's filtered out because 20 / 2.0 = 10.0 but max_ratio is 4.0
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=True, max_ratio=max_ratio)
    filter_2d = BucketingFilter(constraint)
    cut = make_cut(duration=2.0, num_tokens=20)
    assert filter_2d(cut) == False
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == None


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def test_2d_bucketing_strict_mode_flag_works(deterministic_rng, tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet([make_cut(0, duration=1.0, num_tokens=10), make_cut(0, duration=1.0, num_tokens=100)]).to_file(cuts_path)

    # Strict mode enabled
    dloader = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "use_bucketing": True,
            "bucket_duration_bins": [(5.0, 10), (5.0, 20), (10.0, 150), (10.0, 300)],
            "bucket_batch_size": [1, 1, 1, 1],
            "bucketing_2d_strict_mode": True,
        },
        global_rank=0,
        world_size=1,
        dataset=_Identity(),
    )
    batches = [b for b in dloader]
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert len(batches[0][0].supervisions[0].tokens) == 10

    # Strict mode disabled
    dloader = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "use_bucketing": True,
            "bucket_duration_bins": [(5.0, 10), (5.0, 20), (10.0, 150), (10.0, 300)],
            "bucket_batch_size": [1, 1, 1, 1],
            "bucketing_2d_strict_mode": False,
        },
        global_rank=0,
        world_size=1,
        dataset=_Identity(),
    )
    batches = [b for b in dloader]
    assert len(batches) == 2
    assert len(batches[0]) == 1
    assert len(batches[0][0].supervisions[0].tokens) == 100
    assert len(batches[1][0].supervisions[0].tokens) == 10
