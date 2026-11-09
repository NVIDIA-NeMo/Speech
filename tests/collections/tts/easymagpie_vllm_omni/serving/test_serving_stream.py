# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls
from easymagpie_vllm_omni.serving_stream import EasyMagpieInputStream, EasyMagpieStreamingSpeechHandler
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer
from vllm.sampling_params import RequestOutputKind, SamplingParams


def test_input_stream_default_pacing_tolerates_loaded_service_queueing():
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=SamplingParams(max_tokens=32),
        text_eos_id=99,
        max_new_tokens=32,
    )

    # The configured watchdog includes both admission queueing and execution.
    assert stream.pace_timeout_s == 67.0


@pytest.mark.asyncio
async def test_input_stream_accepts_small_chunks_while_first_segment_is_queued() -> None:
    """Verify input stream accepts small chunks while first segment is queued."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=SamplingParams(max_tokens=128),
        text_eos_id=999,
        max_new_tokens=128,
        pace_timeout_s=0,
    )
    await stream.put_tokens([1, 2, 3, 4, 5])
    inputs = stream.inputs()
    first = await anext(inputs)
    try:
        # No segment completion or engine consumption while the WebSocket
        # sends more than eight chunks. Receiving must keep draining so the
        # transport can process heartbeat control frames during queue wait.
        for start in range(6, 66, 5):
            await asyncio.wait_for(stream.put_tokens(list(range(start, start + 5))), timeout=0.1)
        await asyncio.wait_for(stream.finish(), timeout=0.1)
        chunks = [first] + [chunk async for chunk in inputs]
        tokens = [token for chunk in chunks for token in chunk.prompt["additional_information"]["text_token"]]
        assert tokens == list(range(1, 66)) + [999]
        offset = 0
        for chunk in chunks:
            info = chunk.prompt["additional_information"]
            if info["text_token"]:
                assert info["text_token_start"] == offset
                offset += len(info["text_token"])
    finally:
        await inputs.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("coalesce", [False, True])
async def test_input_stream_finish_with_full_queue_preserves_tokens_eos_and_tail(coalesce: bool) -> None:
    """Verify input stream finish with full queue preserves tokens EOS and tail.

    Args:
        coalesce: Whether queued input chunks may be combined.
    """
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0], "additional_information": {"use_cfg": True}},
        sampling_params=SamplingParams(max_tokens=32),
        text_eos_id=99,
        max_new_tokens=32,
        pace_timeout_s=0,
        queue_depth=2,
        coalesce_queued_tokens=coalesce,
    )
    await stream.put_tokens([10, 11])
    await stream.put_tokens([12, 13])

    # No consumer can free queue space until finish returns.
    await asyncio.wait_for(stream.finish(), timeout=0.2)
    await asyncio.wait_for(stream.finish(), timeout=0.2)
    chunks = [chunk async for chunk in stream.inputs()]
    tokens = [token for chunk in chunks for token in chunk.prompt["additional_information"]["text_token"]]
    assert tokens == [10, 11, 12, 13, 99]
    offset = 0
    for chunk in chunks:
        info = chunk.prompt["additional_information"]
        if info["text_token"]:
            assert info["text_token_start"] == offset
            offset += len(info["text_token"])
    assert chunks[-1].prompt["additional_information"]["text_token"] == []
    assert sum(chunk.sampling_params.max_tokens for chunk in chunks) == 32
    with pytest.raises(RuntimeError, match="after input.done"):
        await stream.put_tokens([14])


@pytest.mark.asyncio
async def test_input_stream_finish_with_full_queue_after_segment_timeout() -> None:
    """Verify input stream finish with full queue after segment timeout."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=SamplingParams(max_tokens=128),
        text_eos_id=99,
        max_new_tokens=128,
        pace_timeout_s=0.02,
    )
    await stream.put_tokens([1] * 5)
    inputs = stream.inputs()
    try:
        await anext(inputs)
        await stream.put_tokens([2] * 5)
        next_segment = asyncio.create_task(anext(inputs))
        # Let the consumer take the second chunk and wait for Stage 0.
        await asyncio.sleep(0)
        for _ in range(8):
            await stream.put_tokens([3] * 5)
        with pytest.raises(TimeoutError, match="stage-0 segment completion"):
            await next_segment
        # The consumer has exited; finish must not depend on it draining input.
        await asyncio.wait_for(stream.finish(), timeout=0.2)
        assert stream.finished
    finally:
        await inputs.aclose()


@pytest.mark.asyncio
async def test_input_stream_finish_wakes_consumer_waiting_on_empty_queue() -> None:
    """Verify input stream finish wakes consumer waiting on empty queue."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=SamplingParams(max_tokens=32),
        text_eos_id=99,
        max_new_tokens=32,
    )
    inputs = stream.inputs()
    next_input = asyncio.create_task(anext(inputs))
    await asyncio.sleep(0)
    try:
        await stream.finish()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(next_input, timeout=0.2)
    finally:
        next_input.cancel()
        await asyncio.gather(next_input, return_exceptions=True)
        await inputs.aclose()


@pytest.mark.asyncio
async def test_input_stream_rejects_pending_token_overflow_without_blocking_receiver() -> None:
    """Verify input stream rejects pending token overflow without blocking receiver."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=SamplingParams(max_tokens=2048),
        text_eos_id=999,
        max_new_tokens=2048,
        queue_depth=2,
    )
    await stream.put_tokens([1] * 256)
    await stream.put_tokens([2] * 256)
    with pytest.raises(ValueError, match="Pending streaming input"):
        await asyncio.wait_for(stream.put_tokens([3]), timeout=0.1)


@pytest.mark.asyncio
async def test_cfg_input_stream_reserves_one_bound_across_text_eos_and_tail() -> None:
    """Verify CFG input stream reserves one bound across text EOS and tail."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0] * 6, "additional_information": {"speaker_id": "eng", "use_cfg": True}},
        sampling_params=SamplingParams(max_tokens=20),
        text_eos_id=99,
        max_new_tokens=20,
        text_prefill_num=4,
        pace_timeout_s=0,
    )
    await stream.put_tokens([1, 2])  # Buffer until the known phoneme-BOS warmup fits.
    assert not stream._received_first_update
    await stream.put_tokens([3, 4, 5])
    await stream.put_tokens([6, 7])
    await stream.finish()
    chunks = [chunk async for chunk in stream.inputs()]
    assert chunks[0].prompt["additional_information"]["max_stream_tokens"] == 20
    assert chunks[0].prompt["additional_information"]["prefill_text_tokens"] == [1, 2, 3, 4]
    assert chunks[1].prompt["additional_information"] == {"text_token": [6, 7, 99], "text_token_start": 5}
    assert sum(chunk.sampling_params.max_tokens for chunk in chunks) == 20
    assert [chunk.sampling_params.max_tokens for chunk in chunks] == [2, 3, 15]


@pytest.mark.asyncio
async def test_cfg_input_stream_rejects_text_that_exhausts_the_history_bound() -> None:
    """Verify CFG input stream rejects text that exhausts the history bound."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0], "additional_information": {"speaker_id": "eng", "use_cfg": True}},
        sampling_params=SamplingParams(max_tokens=4),
        text_eos_id=99,
        max_new_tokens=4,
        pace_timeout_s=0,
    )
    await stream.put_tokens([1, 2])
    await stream.put_tokens([3])
    await stream.finish()
    with pytest.raises(ValueError, match="max_stream_tokens"):
        _ = [chunk async for chunk in stream.inputs()]


@pytest.mark.asyncio
async def test_cfg_input_stream_waits_for_acoustic_tail_before_final_sentinel() -> None:
    """Verify CFG input stream waits for acoustic tail before final sentinel."""
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0], "additional_information": {"speaker_id": "eng", "use_cfg": True}},
        sampling_params=SamplingParams(max_tokens=20),
        text_eos_id=99,
        max_new_tokens=20,
    )
    await stream.put_tokens([2, 3])
    await stream.finish()
    inputs = stream.inputs()
    boundary = SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[0], finish_reason="length")])
    await anext(inputs)  # initial text
    stream.observe_output(boundary)
    await anext(inputs)  # text EOS
    stream.observe_output(boundary)
    await anext(inputs)  # acoustic tail
    exhausted = asyncio.create_task(anext(inputs, None))
    try:
        await asyncio.sleep(0)
        assert not exhausted.done(), "Final sentinel must wait for acoustic completion"
        stream.observe_output(boundary)
        assert await exhausted is None
    finally:
        exhausted.cancel()
        await asyncio.gather(exhausted, return_exceptions=True)
        await inputs.aclose()


@pytest.mark.asyncio
async def test_input_stream_builds_one_resumable_request_from_token_chunks():
    params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        detokenize=False,
        ignore_eos=True,
        stop_token_ids=[1],
        output_kind=RequestOutputKind.DELTA,
    )
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0, 0], "additional_information": {"speaker_id": "eng"}},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=20,
        pace_timeout_s=0.0,
    )

    await stream.put_tokens([10, 11, 12])
    await stream.put_tokens([13, 14])
    await stream.finish()
    chunks = [chunk async for chunk in stream.inputs()]

    assert chunks[0].prompt["prompt_token_ids"] == [0, 0]
    assert chunks[0].prompt["additional_information"] == {
        "speaker_id": "eng",
        "text_token": [10, 11, 12],
        "text_token_start": 0,
    }
    assert chunks[0].sampling_params.max_tokens == 3
    assert chunks[1].prompt["additional_information"] == {"text_token": [13, 14, 99], "text_token_start": 3}
    assert chunks[1].sampling_params.max_tokens == 3
    assert chunks[2].prompt["additional_information"] == {"text_token": []}
    assert chunks[2].sampling_params.max_tokens == 20


@pytest.mark.asyncio
async def test_input_stream_accumulates_tokens_until_first_prefill_is_complete():
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={
            "prompt_token_ids": [0] * 6,
            "additional_information": {"speaker_id": "eng", "text_prefill_num": 4},
        },
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=32,
        text_prefill_num=4,
        pace_timeout_s=0.0,
    )

    await stream.put_tokens([10, 11, 12])
    await stream.put_tokens([13, 14])
    await stream.finish()
    chunks = [chunk async for chunk in stream.inputs()]

    assert chunks[0].prompt["additional_information"] == {
        "speaker_id": "eng",
        "text_prefill_num": 4,
        "prefill_text_tokens": [10, 11, 12, 13],
        "text_token": [10, 11, 12, 13, 14],
        "text_token_start": 0,
    }
    assert chunks[0].sampling_params.max_tokens == 2


@pytest.mark.asyncio
async def test_http_adapter_folds_four_text_positions_into_prefill():
    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=None))
    adapter._prompt_len = lambda _speaker_id: 2
    adapter._text_stream_metadata = lambda: (99, 4)
    adapter._model_tokenizer = lambda: SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11, 12, 13, 14])
    request = SimpleNamespace(
        input="hello",
        voice="eng",
        ref_audio=None,
        extra_params=None,
    )

    prepared = await adapter.build(
        request,
        sampling_params_list=[],
        has_inline_ref_audio=False,
    )

    prompt = prepared.prompt
    info = prompt["additional_information"]
    assert len(prompt["prompt_token_ids"]) == 6
    assert info["text_tokens"] == [10, 11, 12, 13, 14, 99]
    assert info["prefill_text_tokens"] == [10, 11, 12, 13]
    assert info["text_prefill_num"] == 4


@pytest.mark.asyncio
async def test_http_adapter_builds_zero_shot_prompt_from_reference_audio():
    class FakeServer:
        @staticmethod
        def _validate_ref_audio_format(ref_audio):
            return None

        @staticmethod
        async def _resolve_ref_audio(ref_audio):
            assert ref_audio == "data:audio/wav;base64,AAAA"
            return [0.25, -0.5], 16000

    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=None, server=FakeServer()))
    adapter._arch = lambda: SimpleNamespace(
        num_task_embeddings=1,
        audio_input_token_id=7,
        codec_input_sample_rate=16000,
        ensure_reference_audio_available=lambda: None,
    )
    adapter._text_stream_metadata = lambda: (99, 3)
    adapter._model_tokenizer = lambda: SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11, 12, 13])
    adapter._tokenize = lambda: lambda _text: [20, 21]
    request = SimpleNamespace(
        input="hello",
        voice=None,
        ref_audio="data:audio/wav;base64,AAAA",
        extra_params={"context_text": "[EN]"},
    )

    assert adapter.validate(request) is None
    prepared = await adapter.build(request, sampling_params_list=[], has_inline_ref_audio=True)

    assert prepared.prompt == {
        "prompt_token_ids": [0, 7, 0, 0, 0, 0, 0],
        "multi_modal_data": {"audio": [([0.25, -0.5], 16000)]},
        "additional_information": {
            "context_text": "[EN]",
            "text_tokens": [10, 11, 12, 13, 99],
            "prefill_text_tokens": [10, 11, 12],
            "text_prefill_num": 3,
            "temperature": 0.7,
            "top_k": 80,
        },
    }


@pytest.mark.parametrize(
    ("voice", "ref_audio", "error"),
    [
        ("eng", "data:audio/wav;base64,AAAA", "mutually exclusive"),
        (None, ["data:audio/wav;base64,AAAA"], "single reference"),
    ],
)
def test_http_adapter_rejects_ambiguous_or_multiple_reference_audio(voice, ref_audio, error):
    adapter = _build_adapter_cls()(
        SimpleNamespace(
            engine_client=None,
            server=SimpleNamespace(_validate_ref_audio_format=lambda _ref_audio: None),
        )
    )
    request = SimpleNamespace(input="hello", voice=voice, ref_audio=ref_audio, extra_params=None)

    assert error in adapter.validate(request)


def test_adapter_treats_null_text_eos_as_legacy_vocab_offset():
    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=None))
    adapter._model_config_cache = {
        "text_vocab_size": 100,
        "text_eos_id": None,
        "streaming_phonemes_delay": 3,
        "streaming_speech_delay": 5,
    }

    assert adapter._text_stream_metadata() == (98, 4)


@pytest.mark.asyncio
async def test_input_stream_preserves_one_token_chunks_with_absolute_offsets():
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=32,
        pace_timeout_s=0.0,
        queue_depth=4,
        coalesce_queued_tokens=False,
    )
    await stream.put_tokens([10])
    await stream.put_tokens([11])
    await stream.put_tokens([12])
    await stream.finish()

    chunks = [chunk async for chunk in stream.inputs()]

    assert [chunk.prompt["additional_information"] for chunk in chunks] == [
        {"text_token": [10], "text_token_start": 0},
        {"text_token": [11], "text_token_start": 1},
        {"text_token": [12], "text_token_start": 2},
        {"text_token": [99], "text_token_start": 3},
        {"text_token": []},
    ]


@pytest.mark.asyncio
async def test_input_stream_counts_stage_zero_delta_tokens():
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=32,
        pace_timeout_s=0.01,
    )

    stream.observe_output(SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[1, 1, 1])]))

    assert stream.observed_output_frames == 3


@pytest.mark.asyncio
async def test_input_stream_times_out_before_replacing_segment_conditioning():
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=32,
        pace_timeout_s=0.01,
    )
    await stream.put_tokens([10, 11])
    await stream.put_tokens([12])
    inputs = stream.inputs()

    first = await anext(inputs)
    assert first.sampling_params.max_tokens == 2

    stream.observe_output(SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[1])]))
    with pytest.raises(TimeoutError, match="waiting for stage-0 segment completion"):
        await anext(inputs)


@pytest.mark.asyncio
async def test_input_stream_waits_for_segment_completion_after_all_frames():
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=32,
        pace_timeout_s=1.0,
    )
    await stream.put_tokens([10, 11])
    await stream.put_tokens([12])
    inputs = stream.inputs()
    await anext(inputs)

    stream.observe_output(SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[1, 1], finish_reason=None)]))
    next_input = asyncio.create_task(anext(inputs))
    await asyncio.sleep(0)
    assert not next_input.done()

    stream.observe_output(SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[], finish_reason="length")]))
    assert (await next_input).prompt["additional_information"] == {"text_token": [12], "text_token_start": 2}


@pytest.mark.asyncio
async def test_input_stream_accepts_segment_completion_with_a_dropped_output_frame():
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0]},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=32,
        pace_timeout_s=1.0,
    )
    await stream.put_tokens([10, 11])
    await stream.put_tokens([12])
    inputs = stream.inputs()
    await anext(inputs)

    # Async segment lookahead may discard one in-flight frame. The segment's
    # finish event is authoritative and must allow the next text chunk through.
    stream.observe_output(
        SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[1], finish_reason="length")])
    )

    next_input = await anext(inputs)
    assert next_input.prompt["additional_information"] == {"text_token": [12], "text_token_start": 2}


def test_two_stage_pipeline_exposes_lm_progress_for_stream_pacing():
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE

    lm_stage = EASYMAGPIE_PIPELINE.stages[0]
    assert lm_stage.final_output is True
    assert lm_stage.final_output_type == "latent"
    assert lm_stage.scheduler_cls == "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"


@pytest.mark.asyncio
async def test_handler_tokenizes_ipa_span_split_across_text_events():
    class FakeWebSocket:
        def __init__(self):
            self.received = iter(
                [
                    {"type": "session.config", "voice": "eng", "stream_audio": True, "response_format": "pcm"},
                    {"type": "input.text", "text": "x<bo"},
                    {"type": "input.text", "text": "p>a"},
                    {"type": "input.text", "text": "b<eo"},
                    {"type": "input.text", "text": "p>y"},
                    {"type": "input.done"},
                ]
            )
            self.json_messages = []
            self.audio_chunks = []
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def receive_text(self):
            return json.dumps(next(self.received))

        async def send_json(self, payload):
            self.json_messages.append(payload)

        async def send_bytes(self, payload):
            self.audio_chunks.append(payload)

    class FakeEngine:
        default_sampling_params_list = [
            SamplingParams(max_tokens=20, output_kind=RequestOutputKind.DELTA),
            SamplingParams(max_tokens=20, output_kind=RequestOutputKind.DELTA),
        ]

        def generate(self, *, prompt, **_kwargs):
            async def outputs():
                async for chunk in prompt:
                    count = chunk.sampling_params.max_tokens
                    yield SimpleNamespace(
                        stage_id=0,
                        outputs=[SimpleNamespace(token_ids=[1] * count, finish_reason="length")],
                    )

            return outputs()

        async def abort(self, _request_id):
            raise AssertionError("completed streams must not be aborted")

    class FakeAdapter:
        @staticmethod
        def build_streaming_spec(_request):
            class FakeBaseTokenizer:
                @staticmethod
                def encode(text, add_special_tokens=False):
                    return [10] * len(text)

            class FakePhonemeTokenizer:
                @staticmethod
                def get_vocab():
                    return {f"p{i}": i for i in range(5)}

                @staticmethod
                def encode(text):
                    return SimpleNamespace(ids=[ord(char) - ord("a") for char in text])

            tokenizer = EasyMagpieTextTokenizer(
                FakeBaseTokenizer(),
                text_vocab_size=28,
                phoneme_tokenizer=FakePhonemeTokenizer(),
                text_phoneme_token_offset=20,
                text_phoneme_vocab_size=8,
            )
            return SimpleNamespace(
                prefill_prompt={"prompt_token_ids": [0]},
                tokenizer=tokenizer,
                text_eos_id=9,
                sample_rate=22050,
            )

    class FakeSpeechService:
        _tts_model_type = "easymagpie"
        engine_client = FakeEngine()

        @staticmethod
        def _get_tts_adapter():
            return FakeAdapter()

        @staticmethod
        async def _generate_pcm_chunks(generator, _request_id):
            async for _output in generator:
                yield b"\x01\x02"

    websocket = FakeWebSocket()
    handler = EasyMagpieStreamingSpeechHandler(FakeSpeechService())
    await handler.handle_session(websocket)

    assert websocket.accepted
    assert websocket.audio_chunks
    assert websocket.json_messages[0]["type"] == "audio.start"
    assert websocket.json_messages[-2]["sentence_text"] == "x<bop>ab<eop>y"
    assert websocket.json_messages[-2]["text_tokens"] == 4
    assert websocket.json_messages[-2]["type"] == "audio.done"
    assert websocket.json_messages[-1] == {"type": "session.done", "total_sentences": 1}
