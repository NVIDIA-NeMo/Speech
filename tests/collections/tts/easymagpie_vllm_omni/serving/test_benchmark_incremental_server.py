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

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("websockets")

import benchmark_incremental_server as benchmark  # noqa: E402


class _FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.responses = iter(
            [
                json.dumps({"type": "audio.start", "sample_rate": 22050}),
                b"\x00\x00\x01\x00",
                json.dumps({"type": "audio.done", "error": False}),
                json.dumps({"type": "session.done", "total_sentences": 1}),
            ]
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return next(self.responses)


def test_request_sends_token_chunks_and_collects_streaming_pcm(monkeypatch):
    websocket = _FakeWebSocket()
    monkeypatch.setattr(benchmark.websockets, "connect", lambda *_args, **_kwargs: websocket)

    result = benchmark._do_request(
        {
            "url": "http://localhost:8091",
            "uttid": "test-0",
            "token_ids": [10, 11, 12, 13, 14],
            "tokens_per_chunk": 2,
            "send_delay_s": 0.0,
            "speaker_id": "eng",
            "max_new_tokens": 128,
            "sample_rate": 22050,
            "timeout": 10.0,
            "output_dir": None,
        }
    )

    assert websocket.sent[0]["type"] == "session.config"
    assert websocket.sent[1:4] == [
        {"type": "input.tokens", "tokens": [10, 11]},
        {"type": "input.tokens", "tokens": [12, 13]},
        {"type": "input.tokens", "tokens": [14]},
    ]
    assert websocket.sent[4] == {"type": "input.done"}
    assert result.error is None
    assert result.num_samples == 2
    assert result.num_text_tokens == 5
    assert result.num_input_chunks == 3


def test_make_tasks_preserves_reference_ids_and_avoids_filename_collisions():
    tasks = benchmark._make_tasks(
        [("utt-1", "one", [1]), ("utt-2", "two", [2])],
        2,
        url="http://localhost:8091",
        speaker_id="eng",
        max_new_tokens=128,
        sample_rate=22050,
        timeout=10,
        output_dir="wavs",
        tokens_per_chunk=5,
        send_delay_s=0.0,
    )

    assert {task["uttid"] for task in tasks} == {"utt-1", "utt-2"}


def test_make_tasks_still_supports_more_requests_than_corpus_entries():
    tasks = benchmark._make_tasks(
        [("utt-1", "one", [1])],
        2,
        url="url",
        speaker_id=None,
        max_new_tokens=128,
        sample_rate=22050,
        timeout=10,
        output_dir=None,
        tokens_per_chunk=5,
        send_delay_s=0.0,
    )

    assert len(tasks) == 2
