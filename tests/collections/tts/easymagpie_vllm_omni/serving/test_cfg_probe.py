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
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import probe_cfg_pairing as probe
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_probe_registers_plugin_before_engine_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify probe registers plugin before engine construction.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
    """
    import vllm.plugins
    import vllm_omni

    registered = []
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", lambda: registered.append(True))
    monkeypatch.setattr(probe, "_load_model_meta", lambda *args: {})
    monkeypatch.setattr(probe, "build_prompt", lambda *args: {"additional_information": {}})

    def create_engine(**kwargs: Any) -> SimpleNamespace:
        """Construct the test engine after checking plugin registration.

        Args:
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Returns:
            The configured fake engine.
        """
        assert registered, "custom topology must be registered before AsyncOmni resolves the deployment"
        return SimpleNamespace(shutdown=lambda: None)

    async def run(
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Run.

        Args:
            *args: Additional positional arguments forwarded to the base API or accepted by the test double.
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Returns:
            Acoustic codes produced with the current graph buffers and sampling settings.
        """
        return 1

    monkeypatch.setattr(vllm_omni, "AsyncOmni", create_engine)
    monkeypatch.setattr(probe, "run_probe", run)
    asyncio.run(probe.main(SimpleNamespace(model="model", speaker="eng", text="hello", timeout=1)))


def test_probe_requires_codec_completion_and_closes_request() -> None:
    """Verify probe requires codec completion and closes request."""
    engine = _Engine([_output(stage=0), _output(stage=1, finished=True)])
    assert asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=1)) == 2
    assert engine.aborted == ["public"]
    assert engine.closed


def test_probe_aborts_after_first_output_and_closes_generator() -> None:
    """Verify probe aborts after first output and closes generator."""
    engine = _Engine([_output(), _output(stage=1, finished=True)])
    assert asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=1, abort_after_output=True)) == 1
    assert engine.aborted == ["public"]
    assert engine.closed


def test_probe_timeout_aborts_the_request_and_closes_generator() -> None:
    """Verify probe timeout aborts the request and closes generator."""

    class PausedEngine(_Engine):
        """Test double for Paused Engine behavior."""

        async def generate(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[SimpleNamespace]:
            """Yield configured public outputs from the test engine.

            Args:
                *args: Additional positional arguments forwarded to the base API or accepted by the test double.
                **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

            Yields:
                Incremental input or public engine output for the current request.
            """
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
def test_probe_rejects_hidden_output_engine_errors_and_incomplete_runs(
    outputs: list[dict[str, Any]],
    message: str,
) -> None:
    """Verify probe rejects hidden output engine errors and incomplete runs.

    Args:
        outputs: Outputs returned by the scheduler or supplied by the test engine.
        message: Message sent over the socket or expected in the failure.
    """
    engine = _Engine([_output(**values) for values in outputs])
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(probe.run_probe(engine, {}, request_id="public", timeout=1))
    assert engine.aborted == ["public"]
    assert engine.closed


class _Engine:
    """Test double for Engine behavior."""

    def __init__(self, outputs: list[SimpleNamespace]) -> None:
        """Initialize Engine state.

        Args:
            outputs: Outputs returned by the scheduler or supplied by the test engine.
        """
        self.outputs = outputs
        self.aborted = []
        self.closed = False

    async def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[SimpleNamespace]:
        """Yield configured public outputs from the test engine.

        Args:
            *args: Additional positional arguments forwarded to the base API or accepted by the test double.
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Yields:
            Incremental input or public engine output for the current request.
        """
        try:
            for output in self.outputs:
                yield output
        finally:
            self.closed = True

    async def abort(self, request_id: str) -> None:
        """Record cancellation of the selected test request.

        Args:
            request_id: Public request identity.
        """
        self.aborted.append(request_id)


def _output(
    request_id: str = "public",
    stage: int = 0,
    finished: bool = False,
    error: str | None = None,
) -> SimpleNamespace:
    """Build an engine output for public identity and completion checks.

    Args:
        request_id: Public request identity.
        stage: Stage producing the configured test output.
        finished: Whether the configured test output is terminal.
        error: Expected error message, or None for success.

    Returns:
        The configured output record for the engine or scheduler test.
    """
    return SimpleNamespace(request_id=request_id, stage_id=stage, finished=finished, error=error)
