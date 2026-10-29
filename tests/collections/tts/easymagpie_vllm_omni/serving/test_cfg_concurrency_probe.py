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
"""Acceptance and error cleanup of the mixed-request GPU probe."""
import asyncio
from types import SimpleNamespace

import probe_cfg_concurrency as probe
import pytest


@pytest.mark.parametrize('invalid_output', [False, True])
def test_concurrent_probe_submits_mixed_requests_and_cleans_up_on_failure(monkeypatch, tmp_path, invalid_output):
    import vllm.plugins
    import vllm_omni

    registered = []
    monkeypatch.setattr(vllm.plugins, 'load_general_plugins', lambda: registered.append(True))
    monkeypatch.setattr(probe, '_load_model_meta', lambda *args: {})
    monkeypatch.setattr(probe, 'build_prompt', lambda *args: {'additional_information': {}})
    engine = _Engine(invalid_output)

    def create(**kwargs):
        assert registered
        return engine

    monkeypatch.setattr(vllm_omni, 'AsyncOmni', create)
    report_path = tmp_path / 'result.json'
    args = SimpleNamespace(
        model='model',
        speaker='demon',
        text='hello',
        timeout=1,
        cfg_scale=2.5,
        temperature=0.7,
        top_k=80,
        deploy_config='profile',
        output=report_path,
    )
    if invalid_output:
        with pytest.raises(RuntimeError, match='Unexpected physical'):
            asyncio.run(probe.main(args))
        assert not report_path.exists()
    else:
        report = asyncio.run(probe.main(args))
        assert len(report['results']) == 8
        assert sum(r['aborted'] for r in report['results']) == 1
        assert {r['request_id'] for r in report['results']} >= {'reuse-guided', 'reuse-plain'}
        assert report_path.exists()
        assert engine.max_active == 6
        assert sum(info['use_cfg'] for info in engine.received.values()) == 4
        for info in engine.received.values():
            assert (info['cfg_scale'], info['temperature'], info['top_k']) == (2.5, 0.7, 80)
    assert engine.shutdown_called and not engine.active
    assert set(engine.received) == engine.aborted == engine.closed


class _Engine:
    def __init__(self, invalid_output):
        self.invalid_output = invalid_output
        self.active, self.aborted, self.closed = set(), set(), set()
        self.received = {}
        self.max_active = 0
        self.shutdown_called = False

    async def generate(self, prompt, request_id):
        self.active.add(request_id)
        self.received[request_id] = prompt['additional_information']
        self.max_active = max(self.max_active, len(self.active))
        try:
            await asyncio.sleep(0)
            rid = 'private-id' if self.invalid_output else request_id
            yield SimpleNamespace(request_id=rid, stage_id=0, finished=False, error=None)
            await asyncio.sleep(0)
            yield SimpleNamespace(request_id=rid, stage_id=1, finished=True, error=None)
        finally:
            self.active.remove(request_id)
            self.closed.add(request_id)

    async def abort(self, rid):
        self.aborted.add(rid)

    def shutdown(self):
        self.shutdown_called = True
