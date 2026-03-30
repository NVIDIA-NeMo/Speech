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

import random
import time

import pytest

from nemo.utils.sequence_packing_utils import first_fit


class TestFirstFitBackendConsistency:
    """Verify that the 'naive' and 'segment_tree' backends produce identical results."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "seqlens, pack_size",
        [
            ([], 10),
            ([5], 10),
            ([10], 10),
            ([3, 7], 10),
            ([6, 6], 10),
            ([10, 10, 10], 10),
            ([5, 3, 7, 2, 4], 10),
            ([1, 1, 1, 1, 1], 3),
            ([3, 3, 3, 3], 5),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
            ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 10),
            ([1] * 100, 10),
            ([3, 7, 2, 8, 1, 9, 4, 6, 5, 10], 15),
        ],
        ids=[
            "empty",
            "single",
            "exact_pack_size",
            "two_fit_one_bin",
            "overflow_new_bin",
            "one_per_bin",
            "mixed_small",
            "all_ones",
            "uniform_3",
            "ascending",
            "descending",
            "100_ones",
            "mixed_large_pack",
        ],
    )
    def test_backends_match(self, seqlens, pack_size):
        naive = first_fit(seqlens, pack_size, backend="naive")
        segment_tree = first_fit(seqlens, pack_size, backend="segment_tree")
        assert naive == segment_tree

    @pytest.mark.unit
    def test_backends_match_random_large(self):
        """Compare backends on 5000 random sequences."""
        rng = random.Random(12345)
        seqlens = [rng.randint(1, 500) for _ in range(5000)]
        pack_size = 1024
        naive = first_fit(seqlens, pack_size, backend="naive")
        segment_tree = first_fit(seqlens, pack_size, backend="segment_tree")
        assert naive == segment_tree

    @pytest.mark.unit
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            first_fit([1, 2, 3], 10, backend="invalid")


class TestFirstFitBackendPerformance:
    """Benchmark naive vs segment_tree to confirm the speedup."""

    @pytest.mark.unit
    def test_segment_tree_faster_than_naive(self):
        rng = random.Random(42)
        seqlens = [rng.randint(1, 500) for _ in range(10000)]
        pack_size = 1024

        t0 = time.perf_counter()
        first_fit(seqlens, pack_size, backend="naive")
        naive_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        first_fit(seqlens, pack_size, backend="segment_tree")
        st_time = time.perf_counter() - t0

        speedup = naive_time / st_time
        print(f"\nnaive: {naive_time:.3f}s | segment_tree: {st_time:.3f}s | speedup: {speedup:.1f}x")
        assert speedup > 2, f"Expected significant speedup, got only {speedup:.1f}x"
