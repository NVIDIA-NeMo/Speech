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
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import probe_cfg_concurrency as probe
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.mark.parametrize("invalid_output", [False, True])
def test_concurrent_probe_submits_mixed_requests_and_cleans_up_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_output: bool,
) -> None:
    """Verify concurrent probe submits mixed requests and cleans up on failure.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        tmp_path: Temporary directory provided by pytest.
        invalid_output: Whether the fake engine returns an invalid public identity.
    """
    import vllm.plugins
    import vllm_omni

    registered = []
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", lambda: registered.append(True))
    monkeypatch.setattr(probe, "_load_model_meta", lambda *args: {})
    monkeypatch.setattr(probe, "build_prompt", lambda *args: {"additional_information": {}})
    engine = _Engine(invalid_output)

    def create(**kwargs: Any) -> _Engine:
        """Construct the test engine after checking plugin registration.

        Args:
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Returns:
            The new pair with a unique private identity.
        """
        assert registered
        return engine

    monkeypatch.setattr(vllm_omni, "AsyncOmni", create)
    report_path = tmp_path / "result.json"
    args = SimpleNamespace(
        model="model",
        speaker="demon",
        text="hello",
        timeout=1,
        cfg_scale=2.5,
        temperature=0.7,
        top_k=80,
        deploy_config="profile",
        output=report_path,
    )
    if invalid_output:
        with pytest.raises(RuntimeError, match="Unexpected physical"):
            asyncio.run(probe.main(args))
        assert not report_path.exists()
    else:
        report = asyncio.run(probe.main(args))
        assert len(report["results"]) == 8
        assert sum(r["aborted"] for r in report["results"]) == 1
        assert {r["request_id"] for r in report["results"]} >= {"reuse-guided", "reuse-plain"}
        assert report_path.exists()
        assert engine.max_active == 6
        assert sum(info["use_cfg"] for info in engine.received.values()) == 4
        for info in engine.received.values():
            assert (info["cfg_scale"], info["temperature"], info["top_k"]) == (2.5, 0.7, 80)
    assert engine.shutdown_called and not engine.active
    assert set(engine.received) == engine.aborted == engine.closed


class _Engine:
    """Test double for Engine behavior."""

    def __init__(self, invalid_output: bool) -> None:
        """Initialize Engine state.

        Args:
            invalid_output: Whether the fake engine returns an invalid public identity.
        """
        self.invalid_output = invalid_output
        self.active, self.aborted, self.closed = set(), set(), set()
        self.received = {}
        self.max_active = 0
        self.shutdown_called = False

    async def generate(
        self,
        prompt: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[SimpleNamespace]:
        """Yield configured public outputs from the test engine.

        Args:
            prompt: Prepared model prompt.
            request_id: Public request identity.

        Yields:
            Incremental input or public engine output for the current request.
        """
        self.active.add(request_id)
        self.received[request_id] = prompt["additional_information"]
        self.max_active = max(self.max_active, len(self.active))
        try:
            await asyncio.sleep(0)
            rid = "private-id" if self.invalid_output else request_id
            yield SimpleNamespace(request_id=rid, stage_id=0, finished=False, error=None)
            await asyncio.sleep(0)
            yield SimpleNamespace(request_id=rid, stage_id=1, finished=True, error=None)
        finally:
            self.active.remove(request_id)
            self.closed.add(request_id)

    async def abort(self, rid: str) -> None:
        """Record cancellation of the selected test request.

        Args:
            rid: Request identity.
        """
        self.aborted.add(rid)

    def shutdown(self) -> None:
        """Record shutdown of the test engine."""
        self.shutdown_called = True
