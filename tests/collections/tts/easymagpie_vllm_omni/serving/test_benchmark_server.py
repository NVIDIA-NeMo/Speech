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


import pytest

pytest.importorskip("numpy")
pytest.importorskip("requests")

import benchmark_server as benchmark  # noqa: E402


class _StreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        assert chunk_size is None
        yield b"\x00"
        yield b"\x40\xff"
        yield b"\x7f"


def test_request_uses_openai_speech_endpoint_and_decodes_streaming_pcm(monkeypatch):
    sent = {}

    def post(url, **kwargs):
        sent["url"] = url
        sent.update(kwargs)
        return _StreamingResponse()

    monkeypatch.setattr(benchmark.requests, "post", post)
    result = benchmark._do_request(
        {
            "url": "http://localhost:8091",
            "uttid": "test",
            "text": "Hello",
            "speaker_id": "eng",
            "max_new_tokens": 128,
            "sample_rate": 22050,
            "timeout": 10,
            "output_dir": None,
        }
    )

    assert sent["url"] == "http://localhost:8091/v1/audio/speech"
    assert sent["json"] == {
        "input": "Hello",
        "voice": "eng",
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "max_new_tokens": 128,
    }
    assert result.error is None
    assert result.num_samples == 2
    assert result.sr == 22050


def test_make_tasks_avoids_output_filename_collisions_when_corpus_is_large_enough():
    tasks = benchmark._make_tasks(
        [("utt-1", "one"), ("utt-2", "two"), ("utt-3", "three")],
        3,
        url="http://localhost:8091",
        speaker_id="eng",
        max_new_tokens=128,
        sample_rate=22050,
        timeout=10,
        output_dir="wavs",
    )

    assert {task["uttid"] for task in tasks} == {"utt-1", "utt-2", "utt-3"}


def test_make_tasks_still_supports_more_requests_than_corpus_entries():
    tasks = benchmark._make_tasks([("utt-1", "one")], 2, "url", None, 128, 22050, 10, None)

    assert len(tasks) == 2


def test_playback_metrics_compare_gap_to_preceding_chunk_duration():
    metrics = benchmark._playback_metrics(
        arrivals=[0.0, 0.10, 0.25],
        durations=[0.08, 0.20, 0.20],
    )

    assert metrics["deadline_misses"] == 1
    assert metrics["underruns"] == 1
    assert metrics["chunk_rtfs"] == pytest.approx([0.8, 4.0 / 3.0])
    assert metrics["headrooms"] == pytest.approx([-0.02, 0.05])
    assert metrics["transitions"] == [
        pytest.approx((0.08, 0.10, True, True)),
        pytest.approx((0.20, 0.15, False, False)),
    ]
