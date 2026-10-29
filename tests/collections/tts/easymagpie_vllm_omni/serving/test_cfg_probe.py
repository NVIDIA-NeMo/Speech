# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Acceptance checks used by the GPU cache-pairing smoke test."""
import asyncio
from types import SimpleNamespace

import probe_cfg_pairing as probe
import pytest


def test_probe_registers_plugin_before_engine_construction(monkeypatch):
    import vllm.plugins
    import vllm_omni

    registered = []
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", lambda: registered.append(True))
    monkeypatch.setattr(probe, "_load_model_meta", lambda *args: {})
    monkeypatch.setattr(probe, "build_prompt", lambda *args: {"additional_information": {}})

    def create_engine(**kwargs):
        assert registered, "custom topology must be registered before AsyncOmni resolves the deployment"
        return SimpleNamespace(shutdown=lambda: None)

    async def run(*args, **kwargs):
        return 1

    monkeypatch.setattr(vllm_omni, "AsyncOmni", create_engine)
    monkeypatch.setattr(probe, "run_probe", run)
    asyncio.run(probe.main(SimpleNamespace(model="model", speaker="eng", text="hello", timeout=1)))


def test_probe_requires_codec_completion_and_closes_request():
    engine = _Engine([_output(stage=0), _output(stage=1, finished=True)])
    assert asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=1)) == 2
    assert engine.aborted == ["public"]
    assert engine.closed


def test_probe_aborts_after_first_output_and_closes_generator():
    engine = _Engine([_output(), _output(stage=1, finished=True)])
    assert asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=1, abort_after_output=True)) == 1
    assert engine.aborted == ["public"]
    assert engine.closed


def test_probe_timeout_aborts_the_request_and_closes_generator():
    class PausedEngine(_Engine):
        async def generate(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
                yield _output()
            finally:
                self.closed = True

    engine = PausedEngine([])
    with pytest.raises(TimeoutError):
        asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=0.01))
    assert engine.aborted == ["public"]
    assert engine.closed


@pytest.mark.parametrize(
    "outputs,message",
    [
        ([{"request_id": "internal"}], "Unexpected physical"),
        ([{"stage": 0, "finished": True}], "without a completed codec"),
        ([{"error": "engine failed"}], "engine failed"),
        ([], "without a completed codec"),
    ],
)
def test_probe_rejects_hidden_output_engine_errors_and_incomplete_runs(outputs, message):
    engine = _Engine([_output(**values) for values in outputs])
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=1))
    assert engine.aborted == ["public"]
    assert engine.closed


class _Engine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.aborted = []
        self.closed = False

    async def generate(self, *args, **kwargs):
        try:
            for output in self.outputs:
                yield output
        finally:
            self.closed = True

    async def abort(self, request_id):
        self.aborted.append(request_id)


def _output(request_id="public", stage=0, finished=False, error=None):
    return SimpleNamespace(request_id=request_id, stage_id=stage, finished=finished, error=error)
