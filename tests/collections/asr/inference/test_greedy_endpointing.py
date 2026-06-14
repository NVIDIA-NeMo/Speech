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
import torch

from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_ctc_endpointing import CTCGreedyEndpointing
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_rnnt_endpointing import RNNTGreedyEndpointing
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames

# Vocabulary used by the detect_eou tests below.
# Indices: 0="▁hello" (start of word), 1="world" (mid-word), 2="▁the" (start of word),
#          3="." (punctuation/absorb), 4="," (punctuation/absorb). blank_id = 5.
EOU_VOCAB = ["▁hello", "world", "▁the", ".", ","]
EOU_BLANK = len(EOU_VOCAB)
EOU_ABSORB_IDS = {3, 4}
ENDPOINTING_CLASSES = [CTCGreedyEndpointing, RNNTGreedyEndpointing]


def _make_detect_eou_endpointer(endpointing_cls, ms_per_timestep=20, stop_history_eou=80):
    """Build an endpointer with the shared EoU vocabulary and absorb token ids."""
    return endpointing_cls(
        vocabulary=EOU_VOCAB,
        ms_per_timestep=ms_per_timestep,
        stop_history_eou=stop_history_eou,
        absorb_token_ids=EOU_ABSORB_IDS,
    )


class TestDetectEou:
    """Tests for the full-buffer EoU detection (GreedyEndpointing.detect_eou)."""

    @pytest.mark.unit
    def test_trailing_silence(self):
        # stop_history_eou=80ms, ms_per_timestep=20ms -> threshold = 4 frames.
        # silence run [2..6] has length 5 > 4 and reaches the buffer end.
        b = EOU_BLANK
        emissions = [0, 1, b, b, b, b, b]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            eou, center, resume = ep.detect_eou_in_buffer(emissions)
            assert eou is True
            assert center == 2 + 4 // 2  # silence_start + stop_history // 2
            assert resume == len(emissions)

    @pytest.mark.unit
    def test_start_of_word_after_silence_is_valid(self):
        b = EOU_BLANK
        # token, 5 blanks (idx 1..5), then "▁the" (start of word) at idx 6.
        emissions = [0, b, b, b, b, b, 2, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            eou, center, resume = ep.detect_eou_in_buffer(emissions)
            assert eou is True
            assert center == 1 + 4 // 2
            assert resume == 6

    @pytest.mark.unit
    def test_mid_word_after_silence_is_rejected(self):
        b = EOU_BLANK
        # "world" (mid-word continuation) right after the silence -> not a valid EoU.
        emissions = [0, b, b, b, b, b, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_punctuation_after_silence_is_absorbed(self):
        b = EOU_BLANK
        # "." after the silence is absorbed into the pre-EoU side; resume points past it
        # to the next start-of-word token "▁the".
        emissions = [0, b, b, b, b, b, 3, 2, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            eou, center, resume = ep.detect_eou_in_buffer(emissions)
            assert eou is True
            assert center == 1 + 4 // 2
            assert resume == 7  # punctuation at idx 6 stays with the finalized text

    @pytest.mark.unit
    def test_punctuation_then_mid_word_is_rejected(self):
        b = EOU_BLANK
        emissions = [0, b, b, b, b, b, 3, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_only_punctuation_after_silence_to_end(self):
        b = EOU_BLANK
        emissions = [0, b, b, b, b, b, 3]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            eou, center, resume = ep.detect_eou_in_buffer(emissions)
            assert eou is True
            assert center == 1 + 4 // 2
            assert resume == len(emissions)

    @pytest.mark.unit
    def test_silence_not_exceeding_threshold(self):
        b = EOU_BLANK
        # Exactly 4 silent frames -> not strictly greater than threshold (4) -> no EoU.
        emissions = [0, b, b, b, b, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_search_start_point_skips_earlier_silence(self):
        b = EOU_BLANK
        emissions = [0, b, b, b, b, b, 2, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            # Start searching at the resumed word; the earlier silence is ignored.
            assert ep.detect_eou_in_buffer(emissions, search_start_point=6) == (False, -1, -1)

    @pytest.mark.unit
    def test_disabled_stop_history(self):
        b = EOU_BLANK
        emissions = [0, b, b, b, b, b, b]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls, stop_history_eou=-1)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_zero_stop_history_finalizes_whole_buffer(self):
        b = EOU_BLANK
        emissions = [0, 1, b, 2]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls, stop_history_eou=0)
            eou, center, resume = ep.detect_eou_in_buffer(emissions)
            assert eou is True
            assert center == len(emissions) - 1
            assert resume == len(emissions)

    @pytest.mark.unit
    def test_empty_emissions(self):
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            assert ep.detect_eou_in_buffer([]) == (False, -1, -1)

    @pytest.mark.unit
    def test_out_of_vocab_special_token_after_silence_does_not_crash(self):
        # Some models can emit token ids beyond the base vocabulary (e.g. prompt/special tokens).
        # Such tokens are not word starts, so the EoU is conservatively not validated (and no crash).
        special_token = EOU_BLANK + 1  # id beyond the vocabulary and the blank
        emissions = [0, EOU_BLANK, EOU_BLANK, EOU_BLANK, EOU_BLANK, EOU_BLANK, special_token, 1]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)

    @pytest.mark.unit
    def test_per_request_stop_history_override(self):
        b = EOU_BLANK
        # Default threshold would be huge (800ms -> 40 frames), but per-request override (80ms -> 4)
        # makes the 5-frame trailing silence trigger an EoU.
        emissions = [0, 1, b, b, b, b, b]
        for cls in ENDPOINTING_CLASSES:
            ep = _make_detect_eou_endpointer(cls, stop_history_eou=800)
            assert ep.detect_eou_in_buffer(emissions) == (False, -1, -1)
            eou, center, resume = ep.detect_eou_in_buffer(emissions, stop_history_eou=80)
            assert eou is True
            assert center == 2 + 4 // 2
            assert resume == len(emissions)


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
