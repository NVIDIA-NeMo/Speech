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
"""Tests for StreamingSALM inference pipeline (mocked — no real model loading)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.collections.asr.inference.model_wrappers.streaming_salm_inference_wrapper import (
    StreamingSALMInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.streaming_salm_pipeline import StreamingSALMPipeline
from nemo.collections.asr.inference.streaming.state.streaming_salm_state import StreamingSALMStreamingState
from nemo.collections.speechlm2.models.streaming_salm import StreamingState as ModelStreamingState

# The @experimental decorator (wrapt) wraps the class; unwrap it for __new__ usage.
_RawStreamingSALMPipeline = getattr(StreamingSALMPipeline, "__wrapped__", StreamingSALMPipeline)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIMI_FRAME_SAMPLES = 1920  # 24000 * 0.08


def _make_mock_wrapper(device="cpu"):
    """Create a mock StreamingSALMInferenceWrapper with controllable behaviour."""
    wrapper = MagicMock(spec=StreamingSALMInferenceWrapper)
    wrapper.device = torch.device(device)
    wrapper.latency = 1
    wrapper.context = None
    wrapper.sample_rate = 24000
    wrapper.word_separator = " "
    wrapper.blank_token_id = 99999

    # Default encode_audio → returns codes of shape (1, 8, num_frames)
    def _encode(audio, audio_lens):
        num_frames = audio.shape[-1] // MIMI_FRAME_SAMPLES
        codes = torch.zeros(audio.shape[0], 8, num_frames, dtype=torch.long)
        code_lens = torch.full((audio.shape[0],), num_frames, dtype=torch.long)
        return codes, code_lens

    wrapper.encode_audio.side_effect = _encode

    # Default generate_streaming → emits distinct token ids per call
    _call_count = {"n": 0}

    def _gen_streaming(audio_codes, model_state, latency, context):
        _call_count["n"] += 1
        n = _call_count["n"]
        B = 1 if audio_codes is None else audio_codes.shape[0]
        # Return distinct tokens per call so we can verify accumulation
        emitted = [[100 + 2 * n - 1, 100 + 2 * n]] * B if audio_codes is not None else [[200 + n]] * B
        new_state = ModelStreamingState(
            kv_cache=None,
            cache_length=10 * n,
            abs_position=10 * n,
            latency=latency,
            sink_size=5,
            window_size=2048,
            num_processed_frames=n,
            num_emitted_tokens=2 * n,
            last_prediction_was_text=False,
        )
        return emitted, new_state

    wrapper.generate_streaming.side_effect = _gen_streaming

    # ids_to_text
    wrapper.ids_to_text.side_effect = lambda ids: " ".join(str(i) for i in ids)

    # Expose tokenizer proxy
    wrapper.tokenizer = MagicMock()
    wrapper.tokenizer.ids_to_text.side_effect = lambda ids: " ".join(str(i) for i in ids)

    return wrapper


def _make_frame(samples: torch.Tensor, stream_id: int = 0, is_first: bool = False, is_last: bool = False):
    """Create a mock Frame."""
    frame = MagicMock()
    frame.samples = samples
    frame.stream_id = stream_id
    frame.is_first = is_first
    frame.is_last = is_last
    frame.size = samples.shape[0]
    frame.valid_size = samples.shape[0]
    return frame


def _make_pipeline(wrapper=None):
    """Create a pipeline with all necessary attributes set."""
    if wrapper is None:
        wrapper = _make_mock_wrapper()
    pipeline = _RawStreamingSALMPipeline.__new__(_RawStreamingSALMPipeline)
    pipeline.asr_model = wrapper
    pipeline.mimi_frame_samples = MIMI_FRAME_SAMPLES
    pipeline.device = torch.device("cpu")
    pipeline.latency = 1
    pipeline.context = None
    pipeline._state_pool = {}
    return pipeline


# ---------------------------------------------------------------------------
# 1. test_state_initialization — structural test (valid, tests real __init__)
# ---------------------------------------------------------------------------


def test_state_initialization():
    """Verify StreamingSALMStreamingState has correct defaults."""
    state = StreamingSALMStreamingState()
    assert state.model_state is None
    assert state.audio_residual is None
    assert state.audio_residual_len == 0
    # Inherits base class attributes
    assert state.tokens == []
    assert state.final_transcript == ""
    assert state.partial_transcript == ""


# ---------------------------------------------------------------------------
# 2. test_mimi_frame_alignment — tests REAL static method logic
# ---------------------------------------------------------------------------


def test_mimi_frame_alignment():
    """Verify _align_to_mimi_frames correctly splits audio."""
    pipeline = _make_pipeline()

    # 2.5 frames worth of audio
    audio = torch.randn(int(2.5 * MIMI_FRAME_SAMPLES))
    aligned, residual = pipeline._align_to_mimi_frames(audio)

    assert aligned is not None
    assert aligned.shape[0] == 2 * MIMI_FRAME_SAMPLES
    assert residual is not None
    assert residual.shape[0] == int(0.5 * MIMI_FRAME_SAMPLES)

    # Exactly 3 frames → no residual
    audio_exact = torch.randn(3 * MIMI_FRAME_SAMPLES)
    aligned2, residual2 = pipeline._align_to_mimi_frames(audio_exact)
    assert aligned2.shape[0] == 3 * MIMI_FRAME_SAMPLES
    assert residual2 is None

    # Less than 1 frame → no aligned, everything is residual
    audio_tiny = torch.randn(100)
    aligned3, residual3 = pipeline._align_to_mimi_frames(audio_tiny)
    assert aligned3 is None
    assert residual3.shape[0] == 100


# ---------------------------------------------------------------------------
# 3. test_audio_residual_accumulation — tests REAL accumulation logic
# ---------------------------------------------------------------------------


def test_audio_residual_accumulation():
    """Verify leftover samples carry over between steps."""
    pipeline = _make_pipeline()

    state = StreamingSALMStreamingState()
    # Store 500 samples as residual
    state.audio_residual = torch.randn(500)

    # New frame with 1500 samples → total 2000 (> 1920, so 1 complete frame + 80 residual)
    frame = _make_frame(torch.randn(1500))
    audio = pipeline._accumulate_audio(frame, state)
    assert audio.shape[0] == 2000

    aligned, residual = pipeline._align_to_mimi_frames(audio)
    assert aligned.shape[0] == MIMI_FRAME_SAMPLES
    assert residual.shape[0] == 80


# ---------------------------------------------------------------------------
# 4. test_sub_frame_audio_not_encoded — tests REAL branching decision
# ---------------------------------------------------------------------------


def test_sub_frame_audio_not_encoded():
    """Frame too small for one Mimi frame → no encoding, residual stored."""
    wrapper = _make_mock_wrapper()
    pipeline = _make_pipeline(wrapper)

    state = StreamingSALMStreamingState()
    pipeline._state_pool[0] = state

    # Only 500 samples — less than one Mimi frame (1920)
    frame = _make_frame(torch.randn(500), stream_id=0, is_first=True, is_last=False)
    pipeline._process_frame(frame, state)

    # No encoding should have happened
    wrapper.encode_audio.assert_not_called()
    wrapper.generate_streaming.assert_not_called()
    # Residual should be stored
    assert state.audio_residual is not None
    assert state.audio_residual.shape[0] == 500
    assert len(state.tokens) == 0


# ---------------------------------------------------------------------------
# 5. test_tokens_from_generate_streaming_accumulated_correctly
# ---------------------------------------------------------------------------


def test_tokens_from_generate_streaming_accumulated_correctly():
    """Verify exact tokens from generate_streaming appear in state.tokens in order."""
    wrapper = _make_mock_wrapper()
    pipeline = _make_pipeline(wrapper)

    state = StreamingSALMStreamingState()
    pipeline._state_pool[0] = state

    # Two frames, each producing distinct tokens
    frame1 = _make_frame(torch.randn(2 * MIMI_FRAME_SAMPLES), stream_id=0, is_first=True, is_last=False)
    pipeline._process_frame(frame1, state)
    # First call returns [101, 102]
    assert state.tokens == [101, 102]

    frame2 = _make_frame(torch.randn(2 * MIMI_FRAME_SAMPLES), stream_id=0, is_first=False, is_last=False)
    pipeline._process_frame(frame2, state)
    # Second call returns [103, 104]; accumulated
    assert state.tokens == [101, 102, 103, 104]


# ---------------------------------------------------------------------------
# 6. test_is_last_flush_call_and_token_accumulation
# ---------------------------------------------------------------------------


def test_is_last_flush_call_and_token_accumulation():
    """On is_last, generate_streaming(None) is called and flush tokens are accumulated."""
    wrapper = _make_mock_wrapper()
    pipeline = _make_pipeline(wrapper)

    state = StreamingSALMStreamingState()
    pipeline._state_pool[0] = state

    # First frame (not last) — call #1 → tokens [101, 102]
    frame1 = _make_frame(torch.randn(2 * MIMI_FRAME_SAMPLES), stream_id=0, is_first=True, is_last=False)
    pipeline._process_frame(frame1, state)

    # Last frame — call #2 for audio → [103, 104], then call #3 flush (None) → [203]
    frame2 = _make_frame(torch.randn(2 * MIMI_FRAME_SAMPLES), stream_id=0, is_first=False, is_last=True)
    pipeline._process_frame(frame2, state)

    # Verify flush was called with None
    calls = wrapper.generate_streaming.call_args_list
    flush_calls = [c for c in calls if c[0][0] is None]
    assert len(flush_calls) >= 1

    # Verify final transcript includes ALL tokens (audio + flush)
    assert state.final_transcript != ""
    assert state.partial_transcript == ""
    # Tokens: [101, 102] from frame1 + [103, 104] from frame2 audio + [203] from flush
    assert 101 in state.tokens
    assert 103 in state.tokens
    assert 203 in state.tokens


# ---------------------------------------------------------------------------
# 7. test_partial_transcript_reflects_current_tokens
# ---------------------------------------------------------------------------


def test_partial_transcript_reflects_current_tokens():
    """Verify partial_transcript is derived from the actual token list."""
    wrapper = _make_mock_wrapper()
    pipeline = _make_pipeline(wrapper)

    state = StreamingSALMStreamingState()
    pipeline._state_pool[0] = state

    frame = _make_frame(torch.randn(2 * MIMI_FRAME_SAMPLES), stream_id=0, is_first=True, is_last=False)
    pipeline._process_frame(frame, state)

    # ids_to_text was called with the current tokens
    wrapper.ids_to_text.assert_called_once_with([101, 102])
    assert state.partial_transcript == "101 102"


# ---------------------------------------------------------------------------
# 8. test_residual_audio_encoded_on_final_frame
# ---------------------------------------------------------------------------


def test_residual_audio_encoded_on_final_frame():
    """On is_last, leftover residual audio should be padded and encoded."""
    wrapper = _make_mock_wrapper()
    pipeline = _make_pipeline(wrapper)

    state = StreamingSALMStreamingState()
    pipeline._state_pool[0] = state

    # Frame with sub-frame audio → stored as residual, nothing encoded
    frame1 = _make_frame(torch.randn(1000), stream_id=0, is_first=True, is_last=False)
    pipeline._process_frame(frame1, state)
    assert state.audio_residual is not None
    assert wrapper.encode_audio.call_count == 0

    # Final frame with enough audio for 1 frame + some residual
    frame2 = _make_frame(torch.randn(2000), stream_id=0, is_first=False, is_last=True)
    pipeline._process_frame(frame2, state)

    # Total audio = 1000 (residual) + 2000 (new) = 3000
    # Aligned: 1920, residual: 1080
    # On is_last, the 1080-sample residual should also be encoded (padded)
    # So encode_audio should be called at least twice: once for aligned, once for padded residual
    assert wrapper.encode_audio.call_count >= 2


# ---------------------------------------------------------------------------
# 9. test_encode_receives_correct_audio_length
# ---------------------------------------------------------------------------


def test_encode_receives_correct_audio_length():
    """Verify encode_audio receives audio of exactly N*1920 samples."""
    wrapper = _make_mock_wrapper()
    pipeline = _make_pipeline(wrapper)

    state = StreamingSALMStreamingState()
    pipeline._state_pool[0] = state

    # 3.5 Mimi frames worth of audio
    n_samples = int(3.5 * MIMI_FRAME_SAMPLES)
    frame = _make_frame(torch.randn(n_samples), stream_id=0, is_first=True, is_last=False)
    pipeline._process_frame(frame, state)

    # encode_audio should receive exactly 3 * 1920 = 5760 samples
    call_args = wrapper.encode_audio.call_args
    encoded_audio = call_args[0][0]  # first positional arg
    assert encoded_audio.shape[-1] == 3 * MIMI_FRAME_SAMPLES

    # Residual = 0.5 * 1920 = 960 samples
    assert state.audio_residual is not None
    assert state.audio_residual.shape[0] == int(0.5 * MIMI_FRAME_SAMPLES)
