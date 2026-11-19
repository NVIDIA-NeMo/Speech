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

import pytest
import torch

from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_ctc_endpointing import CTCGreedyEndpointing
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_rnnt_endpointing import RNNTGreedyEndpointing
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames


class TestGreedyEndpointing:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "inputs, expected",
        [
            ((100, 80), 2),
            ((100, 100), 1),
            ((100, 40), 3),
        ],
    )
    def test_millisecond_to_frames(self, inputs, expected):
        assert millisecond_to_frames(*inputs) == expected

    @pytest.mark.unit
    def test_endpointing_with_negative_stop_history_eou(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(vocabulary=["a", "b", "c"], ms_per_timestep=100, stop_history_eou=-1)
            if isinstance(greedy_endpointing, CTCGreedyEndpointing):
                b = len(greedy_endpointing.greedy_ctc_decoder.vocabulary)
            else:
                b = len(greedy_endpointing.greedy_rnnt_decoder.vocabulary)
            emissions = [0, 1, 2, b, b, b, b, b, b, b, b, b]

            # False case, because stop_history_eou = -1
            assert greedy_endpointing.detect_eou_given_emissions(emissions, 3) == (False, -1)

    @pytest.mark.unit
    def test_endpointing_with_positive_stop_history_eou(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(
                vocabulary=["a", "b", "c"], ms_per_timestep=20, stop_history_eou=100, residue_tokens_at_end=0
            )
            if isinstance(greedy_endpointing, CTCGreedyEndpointing):
                b = len(greedy_endpointing.greedy_ctc_decoder.vocabulary)
            else:
                b = len(greedy_endpointing.greedy_rnnt_decoder.vocabulary)
            emissions = [0, 1, 2, b, b, b, b, b, b, b, b, b]

            for pivot_point in range(len(emissions)):
                eou_detected, eou_detected_at = greedy_endpointing.detect_eou_given_emissions(emissions, pivot_point)
                assert eou_detected == True

    @pytest.mark.unit
    def test_detect_eou_given_timestamps_empty_inputs(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(
                vocabulary=["a", "b", "c"], ms_per_timestep=80, stop_history_eou=100, residue_tokens_at_end=0
            )

            # Test with empty timesteps and tokens
            timesteps = torch.tensor([])
            tokens = torch.tensor([])
            alignment_length = 10

            eou_detected, eou_detected_at = greedy_endpointing.detect_eou_given_timestamps(
                timesteps, tokens, alignment_length
            )
            assert eou_detected == False
            assert eou_detected_at == -1

    @pytest.mark.unit
    def test_detect_eou_given_timestamps_disabled_stop_history(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(
                vocabulary=["a", "b", "c"],
                ms_per_timestep=80,
                stop_history_eou=-1,  # Disabled
                residue_tokens_at_end=0,
            )

            timesteps = torch.tensor([0, 2, 4, 6])
            tokens = torch.tensor([0, 1, 2, 3])
            alignment_length = 10

            eou_detected, eou_detected_at = greedy_endpointing.detect_eou_given_timestamps(
                timesteps, tokens, alignment_length
            )
            assert eou_detected == False
            assert eou_detected_at == -1

    @pytest.mark.unit
    def test_detect_eou_given_timestamps_trailing_silence(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(
                vocabulary=["a", "b", "c"], ms_per_timestep=20, stop_history_eou=80, residue_tokens_at_end=0
            )

            # Last token at position 5, alignment_length is 10
            # Trailing silence = 10 - 4 - 1 = 5 frames > stop_history_eou (4)
            timesteps = torch.tensor([0, 1, 2, 3, 4])
            tokens = torch.tensor([0, 1, 2, 3, 4])
            alignment_length = 10

            eou_detected, eou_detected_at = greedy_endpointing.detect_eou_given_timestamps(
                timesteps, tokens, alignment_length
            )
            assert eou_detected == True
            # eou_detected_at = 4 + 1 + 4//2 = 7
            assert eou_detected_at == 7

    @pytest.mark.unit
    def test_detect_eou_given_timestamps_no_trailing_silence(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(
                vocabulary=["a", "b", "c"], ms_per_timestep=20, stop_history_eou=80, residue_tokens_at_end=0
            )

            # Last token at position 8, alignment_length is 10
            # Trailing silence = 10 - 8 - 1 = 1 frame < stop_history_eou (4)
            timesteps = torch.tensor([0, 1, 2, 3, 8])
            tokens = torch.tensor([0, 1, 2, 3, 4])
            alignment_length = 10

            eou_detected, eou_detected_at = greedy_endpointing.detect_eou_given_timestamps(
                timesteps, tokens, alignment_length
            )
            assert eou_detected == False
            assert eou_detected_at == -1

    @pytest.mark.unit
    def test_detect_eou_given_timestamps_gap_detection(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            greedy_endpointing = endpointing_cls(
                vocabulary=["a", "b", "c"], ms_per_timestep=20, stop_history_eou=80, residue_tokens_at_end=0
            )

            # Large gap between tokens: 8 - 2 - 1 = 5 frames > stop_history_eou (4)
            timesteps = torch.tensor([0, 2, 8, 9])
            tokens = torch.tensor([0, 1, 2, 3])
            alignment_length = 10

            eou_detected, eou_detected_at = greedy_endpointing.detect_eou_given_timestamps(
                timesteps, tokens, alignment_length
            )
            assert eou_detected == True
            # eou_detected_at = 2 + 1 + 4//2 = 5
            assert eou_detected_at == 5

    @pytest.mark.unit
    def test_rnnt_vad_endpointing_disabled(self):
        rnnt_endpointing = RNNTGreedyEndpointing(
            vocabulary=["a", "b", "c"],
            ms_per_timestep=100,
            effective_buffer_size_in_secs=None,  # VAD disabled
            stop_history_eou=100,
        )

        # Test with VAD segments - should raise ValueError since VAD is disabled
        vad_segments = torch.tensor([[0.0, 1.0], [1.5, 2.5]])

        with pytest.raises(
            ValueError, match="Effective buffer size in seconds is required for VAD-based EOU detection"
        ):
            rnnt_endpointing.detect_eou_vad(vad_segments)

    @pytest.mark.unit
    def test_rnnt_vad_endpointing_enabled_no_eou(self):
        rnnt_endpointing = RNNTGreedyEndpointing(
            vocabulary=["a", "b", "c"],
            ms_per_timestep=100,
            effective_buffer_size_in_secs=2.0,  # VAD enabled
            stop_history_eou=100,
        )

        # Test with VAD segments that don't trigger EOU
        vad_segments = torch.tensor([[0.0, 1.45], [1.5, 2.0]])
        eou_detected, eou_detected_at = rnnt_endpointing.detect_eou_vad(vad_segments, stop_history_eou=100)

        assert eou_detected == False
        assert eou_detected_at == -1

    @pytest.mark.unit
    def test_rnnt_vad_endpointing_enabled_with_eou(self):
        rnnt_endpointing = RNNTGreedyEndpointing(
            vocabulary=["a", "b", "c"],
            ms_per_timestep=100,
            effective_buffer_size_in_secs=2.0,  # VAD enabled
            stop_history_eou=100,
        )

        # Test with VAD segments that should trigger EOU
        # Create segments with enough silence to trigger EOU
        vad_segments = torch.tensor([[0.0, 0.5], [1.0, 2.0]])  # Gap of 0.5s between segments
        eou_detected, eou_detected_at = rnnt_endpointing.detect_eou_vad(vad_segments, stop_history_eou=100)

        # This should detect EOU if the silence gap is sufficient
        # The exact behavior depends on the VAD logic implementation
        assert eou_detected == True
        assert eou_detected_at == 5

    @pytest.mark.unit
    def test_rnnt_vad_endpointing_enabled_with_eou_at_end(self):
        rnnt_endpointing = RNNTGreedyEndpointing(
            vocabulary=["a", "b", "c"],
            ms_per_timestep=100,
            effective_buffer_size_in_secs=2.0,  # VAD enabled
            stop_history_eou=100,
        )

        # Test with VAD segments that should trigger EOU
        # Create segments with enough silence to trigger EOU
        vad_segments = torch.tensor([[0.0, 0.5], [1.0, 1.8]])  # Gap of 0.5s between segments
        eou_detected, eou_detected_at = rnnt_endpointing.detect_eou_vad(vad_segments, stop_history_eou=100)

        # This should detect EOU if the silence gap is sufficient
        # The exact behavior depends on the VAD logic implementation
        assert eou_detected == True
        assert eou_detected_at == 18


# Test vocabulary shared by TestDetectEouInBuffer: index 0 is a word-start token (SentencePiece
# leading underscore), index 1 is a word-continuation sub-token, index 2 is punctuation (used as an
# absorb token in most cases). blank_id == len(VOCAB) == 3.
VOCAB = ["▁a", "b", "."]
WORD_START, CONT, PUNCT, BLANK = 0, 1, 2, 3


class TestDetectEouInBuffer:
    """Tests for `GreedyEndpointing.detect_eou_in_buffer`, the cache-aware EoU scan."""

    @staticmethod
    def _make_endpointer(endpointing_cls, **kwargs):
        return endpointing_cls(vocabulary=VOCAB, ms_per_timestep=1, **kwargs)

    @pytest.mark.unit
    def test_mid_buffer_eou_confirmed_by_word_start(self):
        emissions = [WORD_START, BLANK, BLANK, BLANK, WORD_START]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=2, absorb_token_ids={PUNCT})
            assert ep.detect_eou_in_buffer(emissions) == (True, 2, 4)

    @pytest.mark.unit
    def test_mid_word_run_is_rejected(self):
        emissions = [BLANK, BLANK, BLANK, CONT]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=1, absorb_token_ids={PUNCT})
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_end_of_buffer_eou_confirmed_by_large_threshold(self):
        emissions = [WORD_START, BLANK, BLANK, BLANK, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(
                endpointing_cls, stop_history_eou=2, stop_history_eou_end=3, absorb_token_ids={PUNCT}
            )
            assert ep.detect_eou_in_buffer(emissions) == (True, 2, 5)

    @pytest.mark.unit
    def test_end_of_buffer_eou_disabled_by_negative_threshold(self):
        emissions = [WORD_START, BLANK, BLANK, BLANK, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(
                endpointing_cls, stop_history_eou=2, stop_history_eou_end=-1, absorb_token_ids={PUNCT}
            )
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_trailing_punctuation_confirms_eou_even_with_end_disabled(self):
        emissions = [WORD_START, BLANK, BLANK, PUNCT]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(
                endpointing_cls, stop_history_eou=1, stop_history_eou_end=-1, absorb_token_ids={PUNCT}
            )
            assert ep.detect_eou_in_buffer(emissions) == (True, 1, 4)

    @pytest.mark.unit
    def test_silence_after_absorbed_punctuation_is_also_absorbed(self):
        emissions = [WORD_START, BLANK, BLANK, PUNCT, BLANK, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(
                endpointing_cls, stop_history_eou=1, stop_history_eou_end=-1, absorb_token_ids={PUNCT}
            )
            assert ep.detect_eou_in_buffer(emissions) == (True, 1, 6)

    @pytest.mark.unit
    def test_rightmost_qualifying_run_wins(self):
        emissions = [WORD_START, BLANK, BLANK, WORD_START, BLANK, BLANK, WORD_START]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=1, absorb_token_ids={PUNCT})
            assert ep.detect_eou_in_buffer(emissions) == (True, 4, 6)

    @pytest.mark.unit
    def test_search_start_point_shrinks_run_below_threshold(self):
        emissions = [WORD_START, BLANK, BLANK, WORD_START, BLANK, BLANK, WORD_START]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=1, absorb_token_ids={PUNCT})
            assert ep.detect_eou_in_buffer(emissions, search_start_point=5) == (False, -1, -1)

    @pytest.mark.unit
    def test_zero_stop_history_eou_finalizes_whole_buffer(self):
        emissions = [WORD_START, CONT, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=0)
            assert ep.detect_eou_in_buffer(emissions) == (True, 2, 3)

    @pytest.mark.unit
    def test_disabled_stop_history_eou_never_detects(self):
        emissions = [BLANK, BLANK, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=-1)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_empty_buffer_never_detects(self):
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=2)
            assert ep.detect_eou_in_buffer([]) == (False, -1, -1)

    @pytest.mark.unit
    def test_end_threshold_is_clamped_up_to_regular_threshold(self):
        emissions = [WORD_START, BLANK, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            ep = self._make_endpointer(endpointing_cls, absorb_token_ids={PUNCT})
            # A weaker override (1) is clamped up to the regular threshold (3), so a 2-frame trailing
            # run does not qualify even though 2 > 1.
            assert ep.detect_eou_in_buffer(emissions, stop_history_eou=3, stop_history_eou_end=1) == (
                False,
                -1,
                -1,
            )

    @pytest.mark.unit
    def test_end_threshold_falls_back_to_regular_threshold_by_default(self):
        emissions = [WORD_START, BLANK, BLANK, BLANK]
        for endpointing_cls in [CTCGreedyEndpointing, RNNTGreedyEndpointing]:
            # stop_history_eou_end not set at construction -> falls back to stop_history_eou (2).
            ep = self._make_endpointer(endpointing_cls, stop_history_eou=2, absorb_token_ids={PUNCT})
            assert ep.detect_eou_in_buffer(emissions) == (True, 2, 4)

    @pytest.mark.unit
    def test_rnnt_absorbs_trailing_language_tag_like_punctuation(self):
        vocab = ["▁a", "b", ".", "<en-US>"]
        lang_tag, blank = 3, len(vocab)
        emissions = [WORD_START, blank, blank, lang_tag]

        absorbing = RNNTGreedyEndpointing(
            vocabulary=vocab,
            ms_per_timestep=1,
            stop_history_eou=1,
            stop_history_eou_end=-1,
            absorb_token_ids={PUNCT, lang_tag},
        )
        assert absorbing.detect_eou_in_buffer(emissions) == (True, 1, 4)

        # Without the language tag in absorb_token_ids, the same trailing tag is neither a word-start
        # nor absorbable, so it blocks the mid-buffer EoU and (with end-of-buffer disabled) nothing
        # fires.
        not_absorbing = RNNTGreedyEndpointing(
            vocabulary=vocab,
            ms_per_timestep=1,
            stop_history_eou=1,
            stop_history_eou_end=-1,
            absorb_token_ids={PUNCT},
        )
        assert not_absorbing.detect_eou_in_buffer(emissions) == (False, -1, -1)
