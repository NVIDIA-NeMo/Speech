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
"""Test delay pattern application for multi-codebook audio codes."""

import torch

from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder


class TestApplyDelayPattern:
    def test_single_codebook_no_delay(self):
        """Codebook 0 should have no delay (identity)."""
        codes = torch.tensor([[[10, 20, 30, 40, 50]]])  # (1, 1, 5)
        code_lens = torch.tensor([5])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens)
        assert result.shape == (1, 1, 5)
        assert (result[0, 0] == codes[0, 0]).all()

    def test_two_codebooks_delay(self):
        """Codebook 1 should be delayed by 1 frame, padded at start."""
        codes = torch.tensor([[[10, 20, 30, 40],
                               [11, 21, 31, 41]]])  # (1, 2, 4)
        code_lens = torch.tensor([4])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens, pad_value=9999)
        # CB0: [10, 20, 30, 40] (no delay)
        assert result[0, 0].tolist() == [10, 20, 30, 40]
        # CB1: [pad, 11, 21, 31] (delayed by 1, last code 41 drops off)
        assert result[0, 1].tolist() == [9999, 11, 21, 31]

    def test_eight_codebooks_full_delay(self):
        """All 8 codebooks with standard delay pattern."""
        T = 12
        K = 8
        codes = torch.arange(K * T).reshape(1, K, T)
        code_lens = torch.tensor([T])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens, pad_value=-1)
        # Codebook k should start k frames late
        for k in range(K):
            # First k positions should be pad
            assert (result[0, k, :k] == -1).all(), f"CB{k} should have {k} pads at start"
            # Remaining positions should be codes[0, k, 0:T-k]
            assert (result[0, k, k:] == codes[0, k, :T - k]).all()

    def test_delay_pattern_respects_code_lens(self):
        """Positions beyond code_lens should be padded."""
        codes = torch.tensor([[[10, 20, 30, 40, 50],
                               [11, 21, 31, 41, 51]]])  # (1, 2, 5)
        code_lens = torch.tensor([3])  # Only first 3 frames valid
        result = MimiEncoder.apply_delay_pattern(codes, code_lens, pad_value=-1)
        # CB0: [10, 20, 30, -1, -1]
        assert result[0, 0].tolist() == [10, 20, 30, -1, -1]
        # CB1: [-1, 11, 21, -1, -1]
        assert result[0, 1].tolist() == [-1, 11, 21, -1, -1]

    def test_batch_different_lengths(self):
        """Batch with different code_lens should be padded independently."""
        codes = torch.tensor([[[1, 2, 3, 4],
                               [5, 6, 7, 8]],
                              [[10, 20, 30, 40],
                               [50, 60, 70, 80]]])  # (2, 2, 4)
        code_lens = torch.tensor([4, 2])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens, pad_value=-1)
        # Batch 0: full length 4
        assert result[0, 0].tolist() == [1, 2, 3, 4]
        assert result[0, 1].tolist() == [-1, 5, 6, 7]
        # Batch 1: length 2
        assert result[1, 0].tolist() == [10, 20, -1, -1]
        assert result[1, 1].tolist() == [-1, 50, -1, -1]

    def test_delay_pattern_empty_frames(self):
        """Edge case: T < K should still work (all pads for later codebooks)."""
        codes = torch.tensor([[[1, 2],
                               [3, 4],
                               [5, 6]]])  # (1, 3, 2) - only 2 frames
        code_lens = torch.tensor([2])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens, pad_value=-1)
        assert result[0, 0].tolist() == [1, 2]
        assert result[0, 1].tolist() == [-1, 3]
        assert result[0, 2].tolist() == [-1, -1]  # all pad (delay=2 >= T=2)

    def test_default_pad_value(self):
        """Default pad_value should be 2048 (codebook_size)."""
        codes = torch.tensor([[[10, 20],
                               [30, 40]]])  # (1, 2, 2)
        code_lens = torch.tensor([2])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens)
        # CB1 first position should be padded with default 2048
        assert result[0, 1, 0].item() == 2048

    def test_zero_length(self):
        """code_lens=0 should produce all pads."""
        codes = torch.tensor([[[10, 20, 30]]])  # (1, 1, 3)
        code_lens = torch.tensor([0])
        result = MimiEncoder.apply_delay_pattern(codes, code_lens, pad_value=-1)
        assert (result == -1).all()
