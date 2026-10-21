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
    NemoRewardASRBackend,
    ProcessRewardASRBackend,
    RewardASRBackend,
    RewardASRRouter,
)

pytestmark = pytest.mark.unit


class FakeBackend(RewardASRBackend):
    def __init__(self, cfg, device_getter):
        del device_getter
        self.name = cfg["name"]
        self.closed = False

    def transcribe(self, waveforms, sample_rate, languages):
        assert sample_rate == 24000
        return [f"{self.name}:{language}:{int(wav[0])}" for wav, language in zip(waveforms, languages)]

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

    assert router.transcribe([torch.full((4,), float(i)) for i in (1, 2, 3)], 24000, ["en", "hi", "de"]) == [
        "qwen:en:1",
        "whisper:hi:2",
        "qwen:de:3",
    ]
    router.close()
    assert all(backend.closed for backend in router.backends.values())


def test_qwen_backend_requires_qwen_enabled_runtime(tmp_path):
    with pytest.raises(RuntimeError, match="Qwen-enabled training container"):
        ProcessRewardASRBackend(
            {
                "type": "qwen",
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
            import base64
            import struct
            import json
            import sys

            parser = argparse.ArgumentParser()
            parser.add_argument("--backend")
            parser.add_argument("--model")
            parser.add_argument("--device")
            parser.add_argument("--batch-size")
            parser.add_argument("--max-new-tokens")
            parser.add_argument("--language-map")
            args = parser.parse_args()
            print(json.dumps({"status": "ready", "model": args.model, "device": args.device}), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                if request["command"] == "shutdown":
                    print(json.dumps({"status": "stopped"}), flush=True)
                    break
                assert 'audio_paths' not in request
                transcripts = [
                    f"{language}:{request['sample_rate']}:{len(base64.b64decode(wav))//4}:{struct.unpack('<f', base64.b64decode(wav)[:4])[0]}"
                    for wav, language in zip(request['waveforms'], request['languages'])
                ]
                print(json.dumps({"status": "ok", "transcripts": transcripts}), flush=True)
            """
        )
    )
    backend = ProcessRewardASRBackend(
        {
            "type": "qwen",
            "python_executable": sys.executable,
            "worker_script": str(worker),
            "timeout_seconds": 10,
        },
        device_getter=lambda: torch.device("cpu"),
    )

    waveforms = [torch.full((2400,), 0.5), torch.full((4800,), -0.25)]
    try:
        assert backend.transcribe(waveforms, 24000, ['en', 'de']) == ['en:24000:2400:0.5', 'de:24000:4800:-0.25']
        process = backend.process
    finally:
        backend.close()
    assert backend.process is None
    assert process.stdin.closed and process.stdout.closed


def test_nemo_backend_resamples_and_batches_without_paths():
    from types import SimpleNamespace

    seen = []

    def transcribe(waveforms, **kwargs):
        seen.extend(waveforms)
        assert [wav.numel() for wav in waveforms] == [1600, 3200]
        assert kwargs['override_config'].num_workers == 0
        return ['one', SimpleNamespace(text='two')]

    backend = NemoRewardASRBackend.__new__(NemoRewardASRBackend)
    backend.device_getter = lambda: torch.device('cpu')
    backend.prompted = False
    backend.reset_cuda_graphs_before_transcribe = False
    backend.model = SimpleNamespace(
        cfg=SimpleNamespace(sample_rate=16000),
        to=lambda device: None,
        transcribe=transcribe,
    )
    result = backend.transcribe([torch.ones(2400), torch.ones(4800)], 24000, ['en', 'en'])
    assert result == ['one', 'two']
    assert len(seen) == 2


def test_real_worker_decodes_samples_and_rejects_bad_rates(monkeypatch):
    import base64
    import importlib.util
    import io
    import json
    from pathlib import Path

    import numpy as np

    script = (
        Path(ProcessRewardASRBackend._start.__globals__['__file__']).resolve().parents[5]
        / 'scripts/tts/reward_asr_worker.py'
    )
    spec = importlib.util.spec_from_file_location('test_reward_worker', script)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    responses = []
    seen = []

    def transcribe(waveforms, sample_rate, languages, attention_context):
        assert sample_rate == 24000
        assert languages == ['en']
        assert waveforms[0].dtype == np.float32
        np.testing.assert_array_equal(waveforms[0], [0.5, -0.25])
        seen.append(len(waveforms))
        return ['hello']

    monkeypatch.setattr(worker, 'load_qwen', lambda args: transcribe)
    monkeypatch.setattr(worker, 'emit', responses.append)
    payload = base64.b64encode(np.array([0.5, -0.25], dtype='<f4').tobytes()).decode()
    requests = [
        {'command': 'transcribe', 'waveforms': [payload], 'sample_rate': 24000, 'languages': ['en']},
        {'command': 'transcribe', 'waveforms': [payload], 'sample_rate': 0, 'languages': ['en']},
        {'command': 'shutdown'},
    ]
    monkeypatch.setattr(sys, 'argv', [str(script), '--backend', 'qwen', '--model', 'fake', '--device', 'cpu'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO('\n'.join(json.dumps(request) for request in requests)))
    worker.main()
    assert seen == [1]
    assert [response['status'] for response in responses] == ['ready', 'ok', 'error', 'stopped']
    assert responses[1]['transcripts'] == ['hello']
    assert 'sample_rate' in responses[2]['error']


def test_qwen_worker_passes_waveform_rate_tuples(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    script = (
        Path(ProcessRewardASRBackend._start.__globals__['__file__']).resolve().parents[5]
        / 'scripts/tts/reward_asr_worker.py'
    )
    spec = importlib.util.spec_from_file_location('test_qwen_worker', script)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    seen = []

    def transcribe(audio, language):
        assert language == ['English', 'German']
        assert [rate for _, rate in audio] == [24000, 24000]
        seen.extend(wav.shape[0] for wav, _ in audio)
        return [SimpleNamespace(text='one'), SimpleNamespace(text='two')]

    model = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setitem(
        sys.modules,
        'qwen_asr',
        SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)),
    )
    args = SimpleNamespace(model='fake', device='cpu', batch_size=2, max_new_tokens=10)
    decode = worker.load_qwen(args)
    assert decode([np.ones(100), np.ones(200)], 24000, ['en', 'de'], None) == ['one', 'two']
    assert seen == [100, 200]
