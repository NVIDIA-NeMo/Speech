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

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.inference.model_wrappers.cache_aware_ctc_inference_wrapper import (
    CacheAwareCTCInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines import cache_aware_ctc_pipeline
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame

ENCODER_FRAMES = 6


class _Encoder:
    """Frame t of stream s is 10 * s + t, wherever the stream sits in the batch, and the encoder trims like the real one."""

    def __init__(self, valid_out_len, regular_style):
        self.streaming_cfg = SimpleNamespace(valid_out_len=valid_out_len)
        self.att_context_style = "regular" if regular_style else "chunked_limited"
        self.calls = []

    def cache_aware_stream_step(self, processed_signal, processed_signal_length, keep_all_outputs, **kwargs):
        self.calls.append(keep_all_outputs)
        stream_ids = processed_signal[:, 0, 0]
        encoded = 10 * stream_ids[:, None, None] + torch.arange(ENCODER_FRAMES)[None, None, :]
        encoded = encoded.expand(-1, 2, -1).clone()
        lengths = torch.full((processed_signal.shape[0],), ENCODER_FRAMES)
        valid = self.streaming_cfg.valid_out_len
        if valid > 0 and (not keep_all_outputs or self.att_context_style == "regular"):
            encoded = encoded[:, :, :valid]
            lengths = torch.clamp(lengths, max=valid)
        return encoded, lengths, None, None, None


def _pipeline(monkeypatch, encoder, use_cache=True, return_tail_result=False):
    monkeypatch.setattr(cache_aware_ctc_pipeline, "normalize_log_probs", lambda x: x)
    wrapper = object.__new__(CacheAwareCTCInferenceWrapper)
    wrapper.asr_model = SimpleNamespace(encoder=encoder, decoder=lambda encoder_output: encoder_output.transpose(1, 2))
    wrapper.device = torch.device("cpu")
    wrapper.device_str = "cpu"
    wrapper.compute_dtype = wrapper.cast_dtype = torch.float32
    wrapper.use_amp = False
    wrapper.drop_extra_pre_encoded = 0

    pipeline = object.__new__(CacheAwareCTCPipeline)
    pipeline.asr_model = wrapper
    pipeline.device = torch.device("cpu")
    pipeline.use_cache = use_cache
    pipeline.drop_left_context = None if use_cache else 1
    pipeline.valid_out_len = None if use_cache else 2
    pipeline.return_tail_result = return_tail_result
    pipeline.expected_feature_buffer_len = 4
    pipeline.tokenizer = "tokenizer"
    pipeline.leading_regex_pattern = "pattern"
    pipeline.context_manager = SimpleNamespace(
        get_context=lambda stream_ids: SimpleNamespace(
            cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
        ),
        update_cache=lambda stream_ids, context: None,
        reset_slots=lambda stream_ids, eos_flags: None,
    )
    decoded = []
    pipeline.decode_log_probs = lambda requests, log_probs, tails, ready: decoded.append((requests, log_probs, tails))
    pipeline.text_processor = SimpleNamespace(process=lambda states: None)
    pipeline.get_state = lambda stream_id: stream_id
    pipeline.update_partial_transcript = lambda *args: None
    return pipeline, decoded


def _fbuffer(stream_id, is_last=False):
    return FeatureBuffer(
        features=torch.full((3, 4), float(stream_id)), is_first=False, is_last=is_last, stream_id=stream_id
    )


def _frames(decoded):
    """Per request: its log-prob frames (channel 0, the stream-and-time code) and its tail frames."""
    out = {}
    for requests, log_probs, tails in decoded:
        for i, request in enumerate(requests):
            tail = None if tails is None or tails[i] is None else tails[i][:, 0].tolist()
            out[request.stream_id] = (log_probs[i][:, 0].tolist(), tail)
    return out


def _step(pipeline, fbuffers):
    """transcribe_step_for_feature_buffers up to the decoder, which `_pipeline` records."""
    pipeline.transcribe_step_for_feature_buffers(fbuffers)


# (encoder valid_out_len, regular attention style, use_cache, return_tail_result)
CASES = [
    (3, False, True, False),  # cache on: the encoder alone trims the streams that are not last
    (3, True, True, False),
    (0, False, True, False),  # nothing to trim
    (5, False, False, True),  # cache off: left context dropped, head kept, the rest returned as tail
    (5, False, False, False),
    (4, True, False, True),
    (0, False, False, True),
]


@pytest.mark.unit
@pytest.mark.parametrize("encoder_valid_out_len, regular_style, use_cache, return_tail", CASES)
def test_one_mixed_pass_gives_every_stream_the_frames_it_had_in_a_pass_of_its_own(
    monkeypatch, encoder_valid_out_len, regular_style, use_cache, return_tail
):
    encoder = _Encoder(encoder_valid_out_len, regular_style)
    pipeline, decoded = _pipeline(monkeypatch, encoder, use_cache, return_tail)
    streams = [(3, False), (4, True), (5, False), (6, True)]  # ids differ from the batch position on purpose

    _step(pipeline, [_fbuffer(stream_id, last) for stream_id, last in streams])

    assert encoder.calls == [True] and len(decoded) == 1  # one encoder pass that keeps every output
    together = _frames(decoded)

    alone = {}
    for last in (False, True):
        decoded.clear()
        _step(pipeline, [_fbuffer(stream_id, is_last) for stream_id, is_last in streams if is_last == last])
        alone.update(_frames(decoded))
    assert encoder.calls == [True, False, True]  # the old two passes, each of one kind

    assert together.keys() == alone.keys()
    for stream_id in together:
        assert together[stream_id] == alone[stream_id], stream_id
    # a frame carries the id of its stream, so no frame went to another stream
    assert all(int(v) // 10 == stream_id for stream_id, (head, _) in together.items() for v in head)


@pytest.mark.unit
def test_step_arguments_use_the_scalar_flag_when_every_request_is_of_one_kind(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, _Encoder(3, False))

    assert pipeline.step_arguments([5, 6], [False, False]) == (None, {"keep_all_outputs": False})
    assert pipeline.step_arguments([5, 6], [True, True]) == ([5, 6], {"keep_all_outputs": True})


@pytest.mark.unit
def test_step_arguments_give_a_mixed_batch_a_mask_and_right_paddings_only_for_the_last_requests(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, _Encoder(3, False))
    is_last = [False, True, False, True]

    from_list = pipeline.step_arguments([1, 2, 1, 3], is_last)
    from_tensor = pipeline.step_arguments(torch.tensor([1, 2, 1, 3]), is_last)

    assert from_list == ([0, 2, 0, 3], {"keep_all_outputs_mask": is_last})
    assert isinstance(from_tensor[0], torch.Tensor) and from_tensor[0].tolist() == [0, 2, 0, 3]
    assert from_tensor[1] == {"keep_all_outputs_mask": is_last}


@pytest.mark.unit
def test_frames_reach_the_encoder_as_the_batched_bufferer_tensor_without_a_host_read_of_the_paddings(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, _Encoder(3, False))
    batch = torch.arange(4.0)[:, None, None].expand(4, 3, 4).clone()
    paddings = torch.tensor([1, 2, 1, 3])
    pipeline.bufferer = SimpleNamespace(update=lambda frames: (batch, paddings))
    seen = []
    pipeline.preprocess = lambda features, right_paddings: (
        seen.append((features, right_paddings)) or (torch.zeros(4), torch.zeros(4))
    )
    frames = [Frame(samples=torch.zeros(2), stream_id=i, is_last=i in (1, 3)) for i in range(4)]
    pipeline.asr_model.stream_step = lambda **kwargs: (torch.zeros(4, 2, 3), None, None)

    pipeline.transcribe_step_for_frames(frames)

    features, right_paddings = seen[0]
    assert features is batch  # not split into one tensor per stream and put back together
    assert isinstance(right_paddings, torch.Tensor) and right_paddings.tolist() == [0, 2, 0, 3]


@pytest.mark.unit
def test_preprocess_takes_the_batched_tensor_and_keeps_the_lengths_on_the_device(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, _Encoder(3, False))
    pipeline.expected_feature_buffer_len = 5
    batch = torch.zeros(3, 2, 7)  # two frames too many: trimmed to the expected length

    features, lens = pipeline.preprocess(batch, torch.tensor([0, 2, 1]))

    assert features.shape == (3, 2, 5)
    assert lens.tolist() == [5, 3, 4]


@pytest.mark.unit
def test_a_stream_on_its_last_chunk_has_no_tail_to_decode(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, _Encoder(3, False))
    decoded_tails = []

    class _Greedy:
        def __call__(self, log_probs, compute_confidence, previous):
            decoded_tails.append(tuple(log_probs.shape))
            return {"labels": [], "tokens": [], "timesteps": [], "confidences": []}

    pipeline = object.__new__(CacheAwareCTCPipeline)
    pipeline.blank_id = 0
    pipeline.greedy_ctc_decoder = _Greedy()
    states = {0: SimpleNamespace(label_buffer=[], set_incomplete_segment_tokens=lambda tokens: None)}
    pipeline.get_state = lambda stream_id: states[0]
    pipeline.run_greedy_decoder = lambda state, request, log_probs: False
    requests = [SimpleNamespace(stream_id=0), SimpleNamespace(stream_id=0)]

    pipeline.decode_log_probs(requests, [torch.zeros(2, 3), torch.zeros(2, 3)], [torch.zeros(1, 3), None], set())

    assert decoded_tails == [(1, 3)]  # only the request with a tail
