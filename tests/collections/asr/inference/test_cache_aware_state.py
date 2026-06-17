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

import pytest

from nemo.collections.asr.inference.streaming.state.cache_aware_state import CacheAwareStreamingState

BLANK_ID = 9


class TestCacheAwareStreamingStateLabelBuffer:
    """Tests for the EoU label buffer and its buffer<->global frame mapping."""

    @pytest.mark.unit
    def test_setup_label_buffer(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(5, BLANK_ID)
        assert state.label_buffer == [BLANK_ID] * 5
        assert state.label_buffer_size == 5
        assert state.label_buffer_global_end == 0

    @pytest.mark.unit
    def test_update_label_buffer_slides_and_advances_global_end(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(5, BLANK_ID)

        state.update_label_buffer([1, 2])
        assert state.label_buffer == [BLANK_ID, BLANK_ID, BLANK_ID, 1, 2]
        assert state.label_buffer_global_end == 2

        state.update_label_buffer([3, 4, 5])
        assert state.label_buffer == [1, 2, 3, 4, 5]
        assert state.label_buffer_global_end == 5

    @pytest.mark.unit
    def test_update_label_buffer_longer_than_buffer(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(3, BLANK_ID)
        state.update_label_buffer([1, 2, 3, 4, 5])
        # Only the last `buffer_size` labels are kept, but the global end advances by the full shift.
        assert state.label_buffer == [3, 4, 5]
        assert state.label_buffer_global_end == 5

    @pytest.mark.unit
    def test_update_label_buffer_empty_is_noop(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(3, BLANK_ID)
        state.update_label_buffer([])
        assert state.label_buffer == [BLANK_ID] * 3
        assert state.label_buffer_global_end == 0

    @pytest.mark.unit
    def test_buffer_local_to_global(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(5, BLANK_ID)
        state.update_label_buffer([1, 2, 3, 4, 5])  # global_end = 5
        # left edge = global_end - buffer_size = 0 -> local index maps to itself
        assert state.buffer_local_to_global(0) == 0
        assert state.buffer_local_to_global(4) == 4

        state.update_label_buffer([6, 7])  # global_end = 7, left edge = 2
        assert state.buffer_local_to_global(0) == 2
        assert state.buffer_local_to_global(4) == 6

    @pytest.mark.unit
    def test_get_local_search_start(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(5, BLANK_ID)
        state.update_label_buffer([1, 2, 3, 4, 5, 6, 7])  # global_end = 7, left edge = 2

        # Default search start (global 0) is clamped to the buffer left edge -> local 0.
        assert state.get_local_search_start() == 0

        state.set_eou_search_start(6)
        assert state.get_local_search_start() == 4

        # A search start older than the buffer window clamps to 0.
        state.set_eou_search_start(1)
        assert state.get_local_search_start() == 0


class TestCacheAwareStreamingStateTokenSurvival:
    """Tests for finalizing tokens up to the EoU while keeping the next utterance's tokens."""

    @staticmethod
    def _state_with_tokens():
        state = CacheAwareStreamingState()
        state.tokens = [10, 11, 12, 13]
        state.timesteps = [0, 1, 8, 9]
        state.confidences = [0.1, 0.2, 0.3, 0.4]
        return state

    @pytest.mark.unit
    def test_prepare_finalize_partitions_tokens(self):
        state = self._state_with_tokens()
        # Tokens with timestep < 8 are finalized; the rest survive.
        state.prepare_finalize(resume_global_frame=8)
        assert state.tokens == [10, 11]
        assert state.timesteps == [0, 1]
        assert state.confidences == pytest.approx([0.1, 0.2])

    @pytest.mark.unit
    def test_finalize_cleanup_restore_cycle(self):
        state = self._state_with_tokens()
        state.prepare_finalize(resume_global_frame=8)

        # Simulate the pipeline finalizing the decoded portion.
        state.cleanup_after_eou()
        assert state.tokens == []
        assert state.timesteps == []
        assert state.confidences == []

        # Surviving tokens become the start of the next utterance.
        state.restore_carryover()
        assert state.tokens == [12, 13]
        assert state.timesteps == [8, 9]
        assert state.confidences == pytest.approx([0.3, 0.4])

        # Carryover buffers are cleared after restoring.
        state.prepare_finalize(resume_global_frame=100)
        state.cleanup_after_eou()
        state.restore_carryover()
        assert state.tokens == []

    @pytest.mark.unit
    def test_prepare_finalize_keeps_all_when_resume_before_first_token(self):
        state = self._state_with_tokens()
        # resume frame before every token -> nothing finalized, everything survives.
        state.prepare_finalize(resume_global_frame=0)
        assert state.tokens == []
        state.cleanup_after_eou()
        state.restore_carryover()
        assert state.tokens == [10, 11, 12, 13]
        assert state.timesteps == [0, 1, 8, 9]

    @pytest.mark.unit
    def test_prepare_finalize_keeps_none_when_resume_past_all_tokens(self):
        state = self._state_with_tokens()
        # resume frame after every token -> everything finalized, nothing survives.
        state.prepare_finalize(resume_global_frame=100)
        assert state.tokens == [10, 11, 12, 13]
        state.cleanup_after_eou()
        state.restore_carryover()
        assert state.tokens == []
        assert state.timesteps == []
