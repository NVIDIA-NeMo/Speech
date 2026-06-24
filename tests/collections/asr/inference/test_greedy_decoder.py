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

from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_ctc_decoder import CTCGreedyDecoder
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import (
    ClippedRNNTGreedyDecoder,
    RNNTGreedyDecoder,
)


class TestCTCGreedyDecoder:

    @pytest.mark.unit
    def test_ctc_greedy_decoder(self):

        vocab = ["a", "b", "c", "d"]
        decoder = CTCGreedyDecoder(vocabulary=vocab)

        assert decoder.blank_id == len(vocab)
        assert decoder.is_token_silent(len(vocab)) == True

        for i in range(len(vocab)):
            assert decoder.is_token_silent(i) == False

        for i in range(len(vocab)):
            assert decoder.is_token_start_of_word(i) == False

        assert decoder.count_silent_tokens([0, 1, 2, 3, 4], 0, 5) == 1
        assert decoder.count_silent_tokens([0, 1, 2, 3, 4], 0, 3) == 0
        assert decoder.first_non_silent_token([1, 2, 3, 4], 0, 5) == 0

        log_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.05], [0.4, 0.3, 0.2, 0.1, 0.05]])
        assert decoder.get_labels(log_probs) == log_probs.argmax(dim=-1).tolist()

    @pytest.mark.unit
    def test_ctc_greedy_decoder_with_previous_token(self):
        vocab = ["a", "b", "c", "d"]
        decoder = CTCGreedyDecoder(vocabulary=vocab)

        log_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.05], [0.1, 0.2, 0.3, 0.4, 0.05], [0.4, 0.3, 0.2, 0.1, 0.05]])
        last_token_id = 3
        output = decoder(log_probs, compute_confidence=False, previous=last_token_id)
        assert output["tokens"] == [0]
        assert output["timesteps"] == [2]

        output = decoder(log_probs, compute_confidence=False, previous=None)
        assert output["tokens"] == [3, 0]
        assert output["timesteps"] == [0, 2]


class TestRNNTGreedyDecoder:

    @pytest.mark.unit
    def test_rnnt_greedy_decoder(self):

        vocab = ["a", "b", "c", "d"]
        decoder = RNNTGreedyDecoder(vocab)

        blank_id = len(vocab)
        assert decoder.blank_id == blank_id
        assert decoder.is_token_silent(blank_id) == True

        for i in range(len(vocab)):
            assert decoder.is_token_silent(i) == False

        for i in range(len(vocab)):
            assert decoder.is_token_start_of_word(i) == False

        assert decoder.count_silent_tokens([0, 1, 2, 3, 4], 0, 5) == 1
        assert decoder.count_silent_tokens([0, 1, 2, 3, 4], 0, 3) == 0
        assert decoder.first_non_silent_token([1, 2, 3, 4], 0, 5) == 0

    @pytest.mark.unit
    def test_call_confidence_passthrough(self):
        """Per-token confidences are propagated to the output, aligned with the decoded tokens."""
        decoder = RNNTGreedyDecoder(["a", "b", "c", "d"])

        output, _, new_offset = decoder(
            global_timestamps=torch.tensor([0, 1, 2, 3]),
            tokens=torch.tensor([0, 1, 2, 3]),
            length=5,
            offset=0,
            confidences=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        )

        assert output["tokens"] == [0, 1, 2, 3]
        assert output["confidences"] == pytest.approx([0.9, 0.8, 0.7, 0.6])
        assert new_offset == 4

    @pytest.mark.unit
    def test_call_confidence_trimmed_by_offset(self):
        """Confidences are trimmed by `offset` together with tokens/timestamps."""
        decoder = RNNTGreedyDecoder(["a", "b", "c", "d"])

        output, _, _ = decoder(
            global_timestamps=torch.tensor([0, 1, 2, 3]),
            tokens=torch.tensor([0, 1, 2, 3]),
            length=5,
            offset=2,
            confidences=[0.9, 0.8, 0.7, 0.6],
        )

        assert output["tokens"] == [2, 3]
        assert output["confidences"] == pytest.approx([0.7, 0.6])

    @pytest.mark.unit
    def test_call_confidence_defaults_to_zero(self):
        """Without confidences, zeros are returned for each decoded token (backward compatible)."""
        decoder = RNNTGreedyDecoder(["a", "b", "c", "d"])

        output, _, _ = decoder(
            global_timestamps=torch.tensor([0, 1, 2, 3]),
            tokens=torch.tensor([0, 1, 2, 3]),
            length=5,
            offset=0,
            confidences=None,
        )

        assert output["tokens"] == [0, 1, 2, 3]
        assert output["confidences"] == [0.0, 0.0, 0.0, 0.0]


class TestClippedRNNTGreedyDecoder:

    @pytest.mark.unit
    def test_confidence_passthrough(self):
        """Per-token confidences are clipped with the same mask as tokens/timesteps."""
        vocab = ["a", "b", "c", "d"]
        decoder = ClippedRNNTGreedyDecoder(vocabulary=vocab, tokens_per_frame=4, endpointer=None)

        tokens = torch.tensor([0, 1, 2, 3])
        timesteps = torch.tensor([0, 1, 2, 3])
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])

        clipped_output, _, is_eou, _, _ = decoder(
            global_timesteps=timesteps,
            tokens=tokens,
            clip_start=0,
            clip_end=4,
            alignment_length=4,
            is_last=True,
            is_start=True,
            confidences=confidences,
        )

        assert is_eou is True
        assert clipped_output["tokens"] == [0, 1, 2, 3]
        assert clipped_output["confidences"] == pytest.approx([0.9, 0.8, 0.7, 0.6])
        assert len(clipped_output["confidences"]) == len(clipped_output["tokens"])

    @pytest.mark.unit
    def test_confidence_clipping_follows_token_mask(self):
        """Confidences outside the clip range are dropped together with their tokens."""
        vocab = ["a", "b", "c", "d"]
        decoder = ClippedRNNTGreedyDecoder(vocabulary=vocab, tokens_per_frame=4, endpointer=None)

        tokens = torch.tensor([0, 1, 2, 3])
        timesteps = torch.tensor([0, 1, 2, 3])
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])

        clipped_output, _, _, _, _ = decoder(
            global_timesteps=timesteps,
            tokens=tokens,
            clip_start=1,
            clip_end=4,
            alignment_length=4,
            is_last=True,
            is_start=True,
            confidences=confidences,
        )

        assert clipped_output["tokens"] == [1, 2, 3]
        assert clipped_output["confidences"] == pytest.approx([0.8, 0.7, 0.6])

    @pytest.mark.unit
    def test_confidence_defaults_to_zero(self):
        """Without confidences, zeros are returned for each clipped token (backward compatible)."""
        vocab = ["a", "b", "c", "d"]
        decoder = ClippedRNNTGreedyDecoder(vocabulary=vocab, tokens_per_frame=4, endpointer=None)

        tokens = torch.tensor([0, 1, 2, 3])
        timesteps = torch.tensor([0, 1, 2, 3])

        clipped_output, _, _, _, _ = decoder(
            global_timesteps=timesteps,
            tokens=tokens,
            clip_start=0,
            clip_end=4,
            alignment_length=4,
            is_last=True,
            is_start=True,
            confidences=None,
        )

        assert clipped_output["tokens"] == [0, 1, 2, 3]
        assert clipped_output["confidences"] == [0.0, 0.0, 0.0, 0.0]
