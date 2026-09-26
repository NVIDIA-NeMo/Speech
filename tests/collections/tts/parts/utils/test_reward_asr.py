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

import sys
import textwrap

import pytest
import torch

from nemo.collections.tts.parts.utils.reward_asr import (
    QwenRewardASRBackend,
    RewardASRBackend,
    RewardASRRouter,
)


pytestmark = pytest.mark.unit


class FakeBackend(RewardASRBackend):
    def __init__(self, cfg, device_getter):
        del device_getter
        self.name = cfg["name"]
        self.closed = False

    def transcribe(self, audio_paths, languages):
        return [f"{self.name}:{language}:{audio_path}" for audio_path, language in zip(audio_paths, languages)]

    def close(self):
        self.closed = True


def test_router_preserves_order_across_language_routes():
    router = RewardASRRouter(
        {
            "default_backend": "qwen",
            "language_routes": {"hi": "whisper"},
            "backends": {
                "qwen": {"type": "fake", "name": "qwen"},
                "whisper": {"type": "fake", "name": "whisper"},
            },
        },
        device_getter=lambda: torch.device("cpu"),
        backend_types={"fake": FakeBackend},
    )

    assert router.transcribe(["a.wav", "b.wav", "c.wav"], ["en", "hi", "de"]) == [
        "qwen:en:a.wav",
        "whisper:hi:b.wav",
        "qwen:de:c.wav",
    ]
    router.close()
    assert all(backend.closed for backend in router.backends.values())


def test_qwen_backend_requires_qwen_enabled_runtime(tmp_path):
    with pytest.raises(RuntimeError, match="Qwen-enabled training container"):
        QwenRewardASRBackend(
            {
                "python_executable": str(tmp_path / "missing-python"),
                "worker_script": str(tmp_path / "worker.py"),
            },
            device_getter=lambda: torch.device("cpu"),
        )


def test_qwen_backend_worker_protocol(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import sys

            parser = argparse.ArgumentParser()
            parser.add_argument("--model")
            parser.add_argument("--device")
            parser.add_argument("--batch-size")
            parser.add_argument("--max-new-tokens")
            args = parser.parse_args()
            print(json.dumps({"status": "ready", "model": args.model, "device": args.device}), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                if request["command"] == "shutdown":
                    print(json.dumps({"status": "stopped"}), flush=True)
                    break
                transcripts = [
                    f"{language}:{path}"
                    for path, language in zip(request["audio_paths"], request["languages"])
                ]
                print(json.dumps({"status": "ok", "transcripts": transcripts}), flush=True)
            """
        )
    )
    backend = QwenRewardASRBackend(
        {
            "python_executable": sys.executable,
            "worker_script": str(worker),
            "timeout_seconds": 10,
        },
        device_getter=lambda: torch.device("cpu"),
    )

    assert backend.transcribe(["a.wav", "b.wav"], ["en", "de"]) == ["en:a.wav", "de:b.wav"]
    backend.close()
    assert backend.process is None
