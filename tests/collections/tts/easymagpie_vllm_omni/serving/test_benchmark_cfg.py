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

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import benchmark_incremental_server_cfg as incremental
import benchmark_server_cfg as http
import pytest

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator, Sequence
    from types import ModuleType

DEFAULTS = dict(use_cfg=True, cfg_scale=2.5, temperature=0.7, top_k=80)
OVERRIDES = dict(use_cfg=False, cfg_scale=0.0, temperature=0.0, top_k=1)


def _args(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    extra: Sequence[str] = (),
) -> argparse.Namespace:
    """Parse benchmark arguments after replacing the process command line.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        module: Benchmark module under test.
        extra: Additional command-line arguments.

    Returns:
        Parsed benchmark arguments with the requested overrides.
    """
    argv = [module.__name__, "--text-file", "texts.tsv", "-n", "2"]
    if module is incremental:
        argv += ["--model", "model"]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    return module.parse_args()


@pytest.mark.parametrize("module", [http, incremental])
def test_cfg_cli_defaults(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    """Verify CFG cli defaults.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        module: Benchmark module under test.
    """
    args = _args(monkeypatch, module)
    assert http._cfg_params(args) == DEFAULTS
    assert args.concurrency == [1]
    assert args.sample_rate == 32000
    assert args.max_new_tokens == 512


@pytest.mark.parametrize("module", [http, incremental])
@pytest.mark.parametrize("settings", [DEFAULTS, OVERRIDES])
def test_main_propagates_cfg_to_warmup_and_measured_tasks(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    settings: dict[str, Any],
) -> None:
    """Verify main propagates CFG to warmup and measured tasks.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        module: Benchmark module under test.
        settings: Acoustic sampling settings for the logical request.
    """
    extra = [] if settings == DEFAULTS else ["--no-use-cfg", "--cfg-scale", "0", "--temperature", "0", "--top-k", "1"]
    args = _args(monkeypatch, module, extra)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(http, "_load_items", lambda _: [("a", "Hello"), ("b", "World")])
    if module is incremental:
        tokenizer = SimpleNamespace(encode=lambda *_a, **_kw: [10, 11])
        monkeypatch.setattr(incremental.EasyMagpieTextTokenizer, "from_pretrained", lambda _: tokenizer)
    calls = []

    def run_level(
        tasks: list[dict[str, Any]],
        concurrency: int,
        *unused: Any,
    ) -> tuple[list[http.RequestResult], float]:
        """Capture prepared tasks and return a deterministic benchmark duration.

        Args:
            tasks: Prepared benchmark request payloads.
            concurrency: Maximum simultaneous client requests.
            *unused: Additional positional arguments forwarded to the base API or accepted by the test double.

        Returns:
            Prepared values for the caller.
        """
        calls.append(tasks)
        assert concurrency == 1
        assert all(task["extra_params"] == settings for task in tasks)
        return [], 1.0

    monkeypatch.setattr(module, "_run_level", run_level)
    monkeypatch.setattr(module, "_summarize", lambda *_: {})
    monkeypatch.setattr(module, "_print_detailed", lambda *_: None)
    monkeypatch.setattr(module, "_print_summary", lambda *_: None)
    module.main()
    assert [len(tasks) for tasks in calls] == [1, 2]
    assert {task["uttid"] for task in calls[1]} == {"a", "b"}


class _HTTPResponse:
    """Test double for H T T P Response behavior."""

    def __enter__(self) -> _HTTPResponse:
        """Enter the HTTP response test context.

        Returns:
            This HTTP response test double.
        """
        return self

    def __exit__(self, *_: Any) -> bool:
        """Exit the HTTP response test context without suppressing errors.

        Args:
            *_: Additional positional arguments forwarded to the base API or accepted by the test double.

        Returns:
            Whether the requested condition holds.
        """
        return False

    def raise_for_status(self) -> None:
        """Accept the configured successful HTTP test response."""
        pass

    def iter_content(self, **_: Any) -> Iterator[bytes]:
        """Yield deliberately fragmented PCM bytes from the HTTP test response.

        Args:
            **_: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Yields:
            PCM audio fragments.
        """
        yield b"\x00"
        yield b"\x40\xff"
        yield b"\x7f"


@pytest.mark.parametrize("settings", [DEFAULTS, OVERRIDES])
def test_http_transmits_cfg_extra_params_and_decodes_pcm(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, Any],
) -> None:
    """Verify HTTP transmits CFG extra params and decodes PCM.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        settings: Acoustic sampling settings for the logical request.
    """
    sent = {}

    def post(
        url: str,
        **kwargs: Any,
    ) -> _HTTPResponse:
        """Capture the HTTP request and return a successful streaming test response.

        Args:
            url: Serving endpoint URL.
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Returns:
            A successful streaming response with deliberately fragmented PCM.
        """
        assert url == "http://localhost:8091/v1/audio/speech"
        sent.update(kwargs)
        return _HTTPResponse()

    monkeypatch.setattr(http.requests, "post", post)
    task = http._make_tasks([("a", "Hello")], 1, "http://localhost:8091", "demon", 512, 32000, 10, None, **settings)[0]
    result = http._do_request(task)
    assert sent["json"]["extra_params"] == settings
    assert sent["json"]["voice"] == "demon"
    assert sent["json"]["max_new_tokens"] == 512
    assert result.error is None
    assert result.num_samples == 2
    assert result.sr == 32000


class _WebSocket:
    """Test double for Web Socket behavior."""

    def __init__(self) -> None:
        """Initialize WebSocket state."""
        self.sent = []
        self.responses = iter(
            [
                json.dumps(dict(type="audio.start", sample_rate=32000)),
                b"\x00\x40\xff\x7f",
                json.dumps(dict(type="session.done")),
            ]
        )

    async def __aenter__(self) -> _WebSocket:
        """Enter the WebSocket test context.

        Returns:
            This WebSocket test double.
        """
        return self

    async def __aexit__(self, *_: Any) -> bool:
        """Exit the WebSocket test context without suppressing errors.

        Args:
            *_: Additional positional arguments forwarded to the base API or accepted by the test double.

        Returns:
            Whether the requested condition holds.
        """
        return False

    async def send(self, message: str) -> None:
        """Record a WebSocket message sent by the benchmark client.

        Args:
            message: Message sent over the socket or expected in the failure.
        """
        self.sent.append(json.loads(message))

    async def recv(self) -> str | bytes:
        """Return the next configured audio or session event.

        Returns:
            The next JSON event or PCM audio fragment.
        """
        return next(self.responses)


@pytest.mark.parametrize("settings", [DEFAULTS, OVERRIDES])
def test_websocket_transmits_cfg_config_and_preserves_token_chunks(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, Any],
) -> None:
    """Verify WebSocket transmits CFG config and preserves token chunks.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        settings: Acoustic sampling settings for the logical request.
    """
    websocket = _WebSocket()
    monkeypatch.setattr(incremental.websockets, "connect", lambda *_a, **_kw: websocket)
    task = incremental._make_tasks(
        [("a", "Hello", [10, 11, 12])],
        1,
        url="http://localhost:8091",
        speaker_id="demon",
        max_new_tokens=512,
        sample_rate=32000,
        timeout=10,
        output_dir=None,
        tokens_per_chunk=2,
        send_delay_s=0.0,
        **settings,
    )[0]
    result = incremental._do_request(task)
    assert websocket.sent == [
        dict(
            type="session.config",
            voice="demon",
            stream_audio=True,
            response_format="pcm",
            max_new_tokens=512,
            extra_params=settings,
        ),
        dict(type="input.tokens", tokens=[10, 11]),
        dict(type="input.tokens", tokens=[12]),
        dict(type="input.done"),
    ]
    assert result.error is None
    assert result.num_samples == 2
    assert result.num_input_chunks == 2
    assert result.sr == 32000


@pytest.mark.parametrize("module", [http, incremental])
@pytest.mark.parametrize(
    "extra",
    [
        ["--cfg-scale", "nan"],
        ["--cfg-scale", "inf"],
        ["--temperature", "nan"],
        ["--temperature=-inf"],
        ["--top-k", "0"],
        ["--top-k", "-1"],
        ["-n", "0"],
        ["-c", "0"],
    ],
)
def test_invalid_arguments_fail_before_requests(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    extra: Sequence[str],
) -> None:
    """Verify invalid arguments fail before requests.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        module: Benchmark module under test.
        extra: Additional command-line arguments.
    """
    with pytest.raises(SystemExit) as error:
        _args(monkeypatch, module, extra)
    assert error.value.code == 2


def test_cfg_rejects_raw_reference_but_allows_conditional_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CFG rejects raw reference but allows conditional reference.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
    """
    with pytest.raises(SystemExit) as error:
        _args(monkeypatch, http, ["--reference-audio", "reference.wav"])
    assert error.value.code == 2
    args = _args(monkeypatch, http, ["--no-use-cfg", "--reference-audio", "reference.wav"])
    assert args.reference_audio == "reference.wav"
    assert args.use_cfg is False


@pytest.mark.parametrize("module", [http, incremental])
def test_finite_nonpositive_cfg_values_are_supported(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    """Verify finite nonpositive CFG values are supported.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        module: Benchmark module under test.
    """
    args = _args(monkeypatch, module, ["--cfg-scale=-1", "--temperature=-1"])
    assert args.cfg_scale == -1
    assert args.temperature == -1
