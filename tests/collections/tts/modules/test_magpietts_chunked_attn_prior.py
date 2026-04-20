# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Unit tests for MagpieTTSModel._initialize_chunked_attn_prior bounds-check fix.

PR #15499 introduced a bounds check to prevent out-of-bounds access when
prior_weights offsets exceed max_text_len (e.g. during Japanese longform TTS).
"""

import pytest
import torch

from nemo.collections.tts.models.magpietts import ChunkedInferenceConfig, ChunkState


class _StubModel:
    """Minimal stub that exposes only what _initialize_chunked_attn_prior needs."""

    def __init__(self):
        self.chunked_inference_config = ChunkedInferenceConfig()

    @staticmethod
    def _to_int(x):
        return int(x)

    # Bind the real method to this stub class so we can call it without a full model.
    from nemo.collections.tts.models.magpietts import MagpieTTSModel

    _initialize_chunked_attn_prior = MagpieTTSModel._initialize_chunked_attn_prior


def _make_chunk_state(batch_size, previous_attn_len, left_offset=None):
    """Helper: create a ChunkState with previous_attn_len populated (non-empty triggers logic)."""
    state = ChunkState(batch_size=batch_size)
    state.previous_attn_len = list(previous_attn_len)
    state.left_offset = list(left_offset) if left_offset is not None else [0] * batch_size
    return state


class TestInitializeChunkedAttnPrior:
    """Tests for the bounds-check in _initialize_chunked_attn_prior."""

    prior_epsilon = 1e-8
    prior_weights = ChunkedInferenceConfig().prior_weights_init  # (0.5, 1.0, 0.8, 0.2, 0.2)

    def _call(self, chunk_state, current_chunk_len, batch_text_lens, max_text_len, batch_size, use_cfg=False):
        model = _StubModel()
        return model._initialize_chunked_attn_prior(
            chunk_state=chunk_state,
            current_chunk_len=torch.tensor(current_chunk_len, dtype=torch.long),
            batch_text_lens=torch.tensor(batch_text_lens, dtype=torch.long),
            max_text_len=max_text_len,
            batch_size=batch_size,
            use_cfg=use_cfg,
            prior_epsilon=self.prior_epsilon,
            device=torch.device('cpu'),
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normal_case_all_weights_written(self):
        """All 5 prior weights are written when starting point leaves enough room."""
        batch_size = 1
        max_text_len = 10
        batch_text_lens = [8]
        current_chunk_len = [3]
        # current_starting_point = 8 - 3 = 5; offsets 0..4 → indices 5..9, all < 10
        state = _make_chunk_state(batch_size, previous_attn_len=[5])

        result = self._call(state, current_chunk_len, batch_text_lens, max_text_len, batch_size)

        assert result is not None
        assert result.shape == (batch_size, 1, max_text_len)
        starting = batch_text_lens[0] - current_chunk_len[0]  # 5
        for offset, weight in enumerate(self.prior_weights[:5]):
            assert result[0, 0, starting + offset].item() == pytest.approx(
                weight
            ), f"Weight mismatch at offset {offset}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bounds_check_prevents_index_error_near_boundary(self):
        """No IndexError when current_starting_point + some offsets exceed max_text_len."""
        batch_size = 1
        max_text_len = 7
        batch_text_lens = [7]
        current_chunk_len = [2]
        # current_starting_point = 7 - 2 = 5; offsets 0..4 → indices 5,6,7,8,9
        # indices 7,8,9 are >= max_text_len=7 and must be skipped
        state = _make_chunk_state(batch_size, previous_attn_len=[5])

        # Before the fix this would raise an IndexError; now it must succeed silently.
        result = self._call(state, current_chunk_len, batch_text_lens, max_text_len, batch_size)

        assert result is not None
        assert result.shape == (batch_size, 1, max_text_len)
        # Weights at valid indices only
        assert result[0, 0, 5].item() == pytest.approx(self.prior_weights[0])
        assert result[0, 0, 6].item() == pytest.approx(self.prior_weights[1])

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_batch_mixed_boundary_conditions(self):
        """Batch with one item having room for all weights and one near the boundary."""
        batch_size = 2
        max_text_len = 10
        batch_text_lens = [10, 10]
        current_chunk_len = [5, 1]
        # Item 0: starting = 10-5 = 5; offsets 0..4 → indices 5..9, all valid
        # Item 1: starting = 10-1 = 9; offset 0 → idx 9 (valid), offsets 1..4 → OOB
        state = _make_chunk_state(batch_size, previous_attn_len=[5, 9])

        result = self._call(state, current_chunk_len, batch_text_lens, max_text_len, batch_size)

        assert result is not None
        assert result.shape == (batch_size, 1, max_text_len)

        # Item 0: all 5 weights present
        for offset, weight in enumerate(self.prior_weights[:5]):
            assert result[0, 0, 5 + offset].item() == pytest.approx(
                weight
            ), f"Item 0: weight mismatch at offset {offset}"

        # Item 1: only first weight at index 9; offsets 1..4 were out of bounds
        assert result[1, 0, 9].item() == pytest.approx(self.prior_weights[0])
