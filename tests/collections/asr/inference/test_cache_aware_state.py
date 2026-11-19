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

WORD_START, CONT = 0, 1
IS_WORD_START = {WORD_START: True, CONT: False}


def is_token_start_of_word(token_id: int) -> bool:
    return IS_WORD_START[token_id]


class TestCacheAwareStreamingStateBufferIndexing:

    @pytest.mark.unit
    def test_buffer_local_to_global(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(label_buffer_size=4, blank_id=9)
        state.update_label_buffer([1, 2, 3, 4, 5])  # shift=5 >= buffer size -> label_buffer_global_end=5

        assert state.buffer_local_to_global(0) == 1  # left edge: 5 - 4 + 0
        assert state.buffer_local_to_global(4) == 5  # right edge (one past the buffer)

    @pytest.mark.unit
    def test_get_local_search_start_clamped_to_buffer_window(self):
        state = CacheAwareStreamingState()
        state.setup_label_buffer(label_buffer_size=4, blank_id=9)
        state.update_label_buffer([1, 2, 3, 4, 5])  # left_edge = 5 - 4 = 1

        state.set_eou_search_start(0)  # before the buffer window
        assert state.get_local_search_start() == 0

        state.set_eou_search_start(3)  # inside the window
        assert state.get_local_search_start() == 2  # 3 - left_edge(1)


class TestPrepareFinalizeAndRestoreCarryover:

    @staticmethod
    def _state_with_tokens(tokens, timesteps):
        state = CacheAwareStreamingState()
        state.tokens = list(tokens)
        state.timesteps = list(timesteps)
        state.confidences = [1.0] * len(tokens)
        return state

    @pytest.mark.unit
    def test_split_at_resume_frame(self):
        state = self._state_with_tokens(tokens=[WORD_START, CONT, WORD_START, CONT], timesteps=[0, 1, 5, 6])
        state.prepare_finalize(resume_global_frame=5, is_token_start_of_word=is_token_start_of_word)

        assert state.tokens == [WORD_START, CONT]
        assert state.timesteps == [0, 1]
        assert state._carryover_tokens == [WORD_START, CONT]
        assert state._carryover_timesteps == [5, 6]

    @pytest.mark.unit
    def test_leading_non_word_start_carryover_moves_back_to_finalized(self):
        # A continuation sub-token lands exactly at the resume frame (e.g. late-emitted trailing
        # punctuation sharing the frame with the next word); it must stay with the finalized side.
        state = self._state_with_tokens(tokens=[WORD_START, CONT, CONT, WORD_START], timesteps=[0, 1, 5, 6])
        state.prepare_finalize(resume_global_frame=5, is_token_start_of_word=is_token_start_of_word)

        assert state.tokens == [WORD_START, CONT, CONT]
        assert state.timesteps == [0, 1, 5]
        assert state._carryover_tokens == [WORD_START]
        assert state._carryover_timesteps == [6]

    @pytest.mark.unit
    def test_no_split_without_is_token_start_of_word_predicate(self):
        state = self._state_with_tokens(tokens=[WORD_START, CONT], timesteps=[0, 5])
        state.prepare_finalize(resume_global_frame=5, is_token_start_of_word=None)

        assert state.tokens == [WORD_START]
        assert state._carryover_tokens == [CONT]

    @pytest.mark.unit
    def test_resume_frame_past_all_tokens_keeps_everything_finalized(self):
        state = self._state_with_tokens(tokens=[WORD_START, CONT], timesteps=[0, 1])
        state.prepare_finalize(resume_global_frame=100, is_token_start_of_word=is_token_start_of_word)

        assert state.tokens == [WORD_START, CONT]
        assert state._carryover_tokens == []

    @pytest.mark.unit
    def test_restore_carryover_after_cleanup(self):
        state = self._state_with_tokens(tokens=[WORD_START, CONT, WORD_START], timesteps=[0, 1, 5])
        state.prepare_finalize(resume_global_frame=5, is_token_start_of_word=is_token_start_of_word)

        # Mirrors pipeline usage: finalize (decode from state.tokens), then cleanup, then restore.
        state.cleanup_after_eou()
        assert state.tokens == []

        state.restore_carryover()
        assert state.tokens == [WORD_START]
        assert state.timesteps == [5]
        assert state.confidences == [1.0]

        # Carryover buffers are cleared after restoring, so a second restore is a no-op.
        state.restore_carryover()
        assert state.tokens == []
