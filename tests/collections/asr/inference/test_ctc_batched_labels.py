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

import itertools
from functools import partial

import pytest
import torch

from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_ctc_decoder import ClippedCTCGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_ctc_endpointing import CTCGreedyEndpointing
from nemo.collections.asr.parts.utils.asr_confidence_utils import get_confidence_measure_bank

VOCAB = ["▁a", "b", "▁c", "d", "▁e"]  # blank id is 5
BLANK = 5


def _log_probs(labels_per_stream, seed=0):
    """Log probabilities whose argmax is the given label of every frame, with noise on the other classes."""
    torch.manual_seed(seed)
    batch, frames, vocab = len(labels_per_stream), len(labels_per_stream[0]), len(VOCAB) + 1
    logits = torch.randn(batch, frames, vocab)
    for b, labels in enumerate(labels_per_stream):
        for t, label in enumerate(labels):
            logits[b, t, label] += 8.0
    return torch.log_softmax(logits, dim=-1)


def _decoder(with_endpointer=True):
    endpointer = CTCGreedyEndpointing(VOCAB, ms_per_timestep=80, stop_history_eou=160) if with_endpointer else None
    conf_func = partial(get_confidence_measure_bank()["entropy_tsallis_exp"], t=0.5)
    return ClippedCTCGreedyDecoder(VOCAB, tokens_per_frame=3, conf_func=conf_func, endpointer=endpointer)


# stream 0 says "a b", then is silent for 6 frames (end of utterance); stream 1 says "c d e" and keeps going
LABELS = [
    [BLANK, 0, 0, BLANK, 1, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK],
    [2, 2, BLANK, 3, BLANK, 4, 4, BLANK, 2, 2, 3, 3],
    # stream 2 is silent in frames 1-5: a request clipped from frame 4 must not count the silence before its clip
    [0, BLANK, BLANK, BLANK, BLANK, BLANK, 2, 3, 4, 0, 1, 2],
]


@pytest.mark.unit
def test_precomputed_labels_give_the_same_decoding_as_the_per_stream_log_probs():
    log_probs = _log_probs(LABELS)
    decoder = _decoder()
    labels, confidences = decoder.batched_labels_and_confidences(log_probs)
    assert labels == LABELS
    for stream, (clip_start, is_start) in itertools.product(range(3), [(0, True), (4, False)]):
        want = decoder(log_probs[stream], clip_start, 6, is_last=False, is_start=is_start)
        got = decoder(None, clip_start, 6, False, is_start, labels=labels[stream], confidences=confidences[stream])
        assert got[0]["tokens"] == want[0]["tokens"]
        assert got[0]["timesteps"] == want[0]["timesteps"]
        assert got[0]["last_token"] == want[0]["last_token"]
        assert got[0]["confidences"] == pytest.approx(want[0]["confidences"], abs=1e-5)
        assert got[1:] == want[1:]  # tail output, EOU flag, start and end index


@pytest.mark.unit
def test_precomputed_labels_decode_the_expected_tokens_and_end_of_utterance():
    log_probs = _log_probs(LABELS)
    decoder = _decoder()
    labels, confidences = decoder.batched_labels_and_confidences(log_probs)
    quiet = decoder(None, 0, 6, is_last=False, is_start=True, labels=labels[0], confidences=confidences[0])
    assert quiet[0]["tokens"] == [0, 1] and quiet[0]["timesteps"] == [1, 4]
    assert quiet[2] is True  # the stream went silent for longer than the stop history
    talking = decoder(None, 0, 6, is_last=False, is_start=True, labels=labels[1], confidences=confidences[1])
    assert talking[0]["tokens"] == [2, 3, 4] and talking[0]["timesteps"] == [0, 3, 5]
    assert talking[2] is False
    # Frames 3-5 of stream 2 are silent: three frames end the utterance, only two (from the clip start) do not
    for clip_start, is_start, state_idx, want in [(0, True, 0, True), (4, False, 8, False)]:
        mid = decoder(
            None, clip_start, 6, False, is_start, state_end_idx=state_idx, labels=labels[2], confidences=confidences[2]
        )
        assert mid[2] is want


@pytest.mark.unit
def test_precomputed_labels_clip_the_last_request_to_the_whole_buffer():
    log_probs = _log_probs(LABELS)
    decoder = _decoder(with_endpointer=False)
    labels, confidences = decoder.batched_labels_and_confidences(log_probs)
    last = decoder(None, 3, 6, is_last=True, is_start=False, labels=labels[1], confidences=confidences[1])
    # The last request starts at frame 3 and reads to the end of the 12 frame buffer
    assert last[0]["tokens"] == [3, 4, 2, 3] and last[0]["timesteps"] == [3, 5, 8, 10]
    assert last[4] == 12  # end index
