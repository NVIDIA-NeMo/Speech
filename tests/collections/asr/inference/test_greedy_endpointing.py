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
