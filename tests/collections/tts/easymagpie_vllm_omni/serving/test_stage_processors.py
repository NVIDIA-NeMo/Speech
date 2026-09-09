# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.stage_processors import talker2code2wav_async_chunk


class _Request:
    external_req_id = "request-0"

    def __init__(self):
        self.finished = False
        self.resumable = True
        self.output_token_ids = []

    def is_finished(self):
        return self.finished


def _manager():
    return SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=2)),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                }
            }
        ),
        code_prompt_token_ids=defaultdict(list),
    )


def _output(value: int):
    return {"audio_codes": torch.tensor([[value, value + 100]], dtype=torch.long)}


def test_async_codec_state_stays_continuous_across_resumable_segments():
    manager = _manager()
    request = _Request()

    # Warm-up is counted over the whole request, including segment boundaries.
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    request.output_token_ids = [0, 0]
    request.finished = True
    warmup_flush = talker2code2wav_async_chunk(manager, _output(2), request, is_finished=True)
    assert warmup_flush.codes.audio.numel() == 0
    manager.code_prompt_token_ids.pop(request.external_req_id, None)

    # The framework buffer is reset per segment, but the processor's request
    # state retains the real acoustic frames and its emission high-water mark.
    request.finished = False
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    request.output_token_ids = [0, 0]
    request.finished = True
    first = talker2code2wav_async_chunk(manager, _output(4), request, is_finished=True)
    torch.testing.assert_close(first.codes.audio, torch.tensor([[3, 103], [4, 104]]))
    assert first.meta.left_context_size == 0
    manager.code_prompt_token_ids.pop(request.external_req_id, None)

    request.finished = False
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(5), request) is None
    request.output_token_ids = [0, 0]
    second = talker2code2wav_async_chunk(manager, _output(6), request)
    torch.testing.assert_close(second.codes.audio, torch.tensor([[5, 105], [6, 106]]))
    assert second.meta.left_context_size == 0

    # Repeated segment flushes at the same length must not duplicate audio.
    request.finished = True
    assert talker2code2wav_async_chunk(manager, None, request, is_finished=True) is None

    # Terminal completion releases the request-persistent state.
    request.resumable = False
    assert talker2code2wav_async_chunk(manager, None, request, is_finished=True) is None
    assert request.external_req_id not in manager._emp_seen_frames
    assert request.external_req_id not in manager._emp_request_speech_delay
    assert request.external_req_id not in manager._emp_emitted_frames
    assert request.external_req_id not in manager._emp_emitted_chunks
    assert request.external_req_id not in manager._emp_frame_buffer_base
    assert request.external_req_id not in manager._emp_frame_buffer


def test_resumable_segment_stop_does_not_flush_partial_codec_chunk():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4})
    request = _Request()

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    request.finished = True
    assert talker2code2wav_async_chunk(manager, _output(2), request, is_finished=True) is None

    request.finished = False
    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    full = talker2code2wav_async_chunk(manager, _output(4), request)
    assert full is not None
    torch.testing.assert_close(full.codes.audio, torch.tensor([[1, 101], [2, 102], [3, 103], [4, 104]]))


def test_async_codec_drops_only_warmup_not_moved_into_prefill():
    manager = _manager()
    manager.config.hf_config = SimpleNamespace(
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )
    manager.connector.config["extra"]["codec_chunk_frames"] = 1
    request = _Request()
    request.additional_information = {"text_prefill_num": 4}

    # The prefill callback carries no generated acoustic frame and must not
    # consume the one remaining warm-up slot.
    assert (
        talker2code2wav_async_chunk(
            manager,
            {"audio_codes": torch.zeros(1, 2, dtype=torch.long)},
            request,
        )
        is None
    )
    assert manager._emp_seen_frames[request.external_req_id] == 0

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    first_audio = talker2code2wav_async_chunk(manager, _output(2), request)
    torch.testing.assert_close(first_audio.codes.audio, torch.tensor([[2, 102]]))


def test_async_codec_forwards_terminal_audio_eos_row():
    manager = _manager()
    manager.config.hf_config = SimpleNamespace(
        streaming_speech_delay=0,
        forced_audio_eos_id=1025,
        codebook_size=1024,
    )
    manager.connector.config["extra"]["codec_chunk_frames"] = 4
    request = _Request()
    request.resumable = False

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    assert talker2code2wav_async_chunk(manager, _output(2), request) is None

    request.finished = True
    terminal = talker2code2wav_async_chunk(
        manager,
        {"audio_codes": torch.tensor([[1025, 777]], dtype=torch.long)},
        request,
        is_finished=True,
    )

    assert bool(terminal.meta.finished)
    torch.testing.assert_close(terminal.codes.audio, torch.tensor([[1, 101], [2, 102], [1025, 777]]))


def test_async_codec_buffer_drops_emitted_rows():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    request = _Request()

    last = None
    for value in range(1, 11):
        last = talker2code2wav_async_chunk(manager, _output(value), request)

    assert last is not None
    torch.testing.assert_close(last.codes.audio, torch.tensor([[9, 109], [10, 110]]))
    assert last.meta.left_context_size == 0
    buffer = manager._emp_frame_buffer[request.external_req_id]
    assert buffer == []
    assert manager._emp_frame_buffer_base[request.external_req_id] == 10
    assert manager._emp_emitted_frames[request.external_req_id] == 10


def test_async_codec_uses_configured_startup_chunk_ramp():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4, "codec_startup_chunk_frames": [1, 2]})
    request = _Request()

    first = talker2code2wav_async_chunk(manager, _output(1), request)
    assert first is not None
    torch.testing.assert_close(first.codes.audio, torch.tensor([[1, 101]]))

    assert talker2code2wav_async_chunk(manager, _output(2), request) is None
    second = talker2code2wav_async_chunk(manager, _output(3), request)
    assert second is not None
    torch.testing.assert_close(second.codes.audio, torch.tensor([[2, 102], [3, 103]]))
    assert second.meta.left_context_size == 0

    for value in range(4, 7):
        assert talker2code2wav_async_chunk(manager, _output(value), request) is None
    steady = talker2code2wav_async_chunk(manager, _output(7), request)
    assert steady is not None
    torch.testing.assert_close(
        steady.codes.audio,
        torch.tensor([[4, 104], [5, 105], [6, 106], [7, 107]]),
    )
    assert steady.meta.left_context_size == 0
    assert manager._emp_emitted_chunks[request.external_req_id] == 3


def test_async_codec_allows_larger_first_chunk_than_steady_state():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4, "codec_startup_chunk_frames": [5]})
    request = _Request()

    for value in range(1, 5):
        assert talker2code2wav_async_chunk(manager, _output(value), request) is None
    first = talker2code2wav_async_chunk(manager, _output(5), request)
    assert first is not None
    torch.testing.assert_close(first.codes.audio, torch.tensor([[1, 101], [2, 102], [3, 103], [4, 104], [5, 105]]))


def test_async_codec_rejects_invalid_startup_chunk_ramp():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4, "codec_startup_chunk_frames": [1, 0]})
    with pytest.raises(ValueError, match="codec_startup_chunk_frames"):
        talker2code2wav_async_chunk(manager, _output(1), _Request())


def test_stateful_codec_emits_only_new_time_major_rows():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 2})
    request = _Request()

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    first = talker2code2wav_async_chunk(manager, _output(2), request)
    assert first.codes.audio.shape == (2, 2)
    torch.testing.assert_close(first.codes.audio, torch.tensor([[1, 101], [2, 102]]))
    assert first.meta.left_context_size == 0
    assert manager._emp_frame_buffer[request.external_req_id] == []

    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    second = talker2code2wav_async_chunk(manager, _output(4), request)
    assert second.codes.audio.shape == (2, 2)
    torch.testing.assert_close(second.codes.audio, torch.tensor([[3, 103], [4, 104]]))
    assert manager._emp_frame_buffer_base[request.external_req_id] == 4
