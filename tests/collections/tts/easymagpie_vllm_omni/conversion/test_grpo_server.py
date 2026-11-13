# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout transport tests with an injected engine and no serving runtime."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import grpo_server as server  # noqa: E402


def _prompt():
    return {
        "grpo_context_embeddings": torch.zeros(2, 4),
        "grpo_text_embeddings": torch.ones(2, 4),
        "grpo_text_token_ids": [10, 11],
        "temperature": 0.7,
        "top_k": 80,
        "phoneme_sampling_method": "sample",
        "max_audio_steps": 3,
    }


def _service(engine, work_dir):
    arch = SimpleNamespace(
        num_audio_codebooks=2,
        frame_stacking_factor=1,
        phoneme_stacking_factor=1,
        embedding_dim=4,
        streaming_speech_delay=5,
        streaming_phonemes_delay=3,
        text_prefill_num=4,
    )
    return server.RolloutService(
        engine,
        arch,
        work_dir,
        {},
        extract_delta=lambda output: output,
        sampling_params_factory=lambda **kwargs: kwargs,
    )


class _Engine:
    def __init__(self, fail=False):
        self.fail = fail
        self.started = []
        self.closed = set()
        self.aborted = []
        self.rows = torch.tensor(
            [[36, 36, 3, 0, 1, 0, 0, -0.3], [1, 2, 3, 1, 1, -0.5, -0.8, -0.2], [33, 7, 6, 1, 1, -1, -2, -3]],
            dtype=torch.float32,
        )

    async def generate(self, prompt, sampling_params_list, request_id):
        self.started.append(request_id)
        assert prompt["additional_information"]["grpo_rollout"] is True
        assert len(prompt["prompt_token_ids"]) == 6
        assert sampling_params_list[0]["max_tokens"] == 5
        try:
            await asyncio.sleep(0)
            if self.fail and request_id.endswith("-0"):
                raise RuntimeError("generation failed")
            yield self.rows[:2]
            if self.fail:
                await asyncio.Event().wait()
            yield self.rows[2:]
        finally:
            self.closed.add(request_id)

    async def abort(self, request_id):
        self.aborted.append(request_id)


def test_ipc_paths_are_confined_to_existing_files_in_rank_directory(tmp_path):
    work_dir = tmp_path / "rank"
    work_dir.mkdir()
    inside = work_dir / "input.pt"
    inside.write_bytes(b"input")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")

    assert server._confined_path("input.pt", work_dir) == inside
    for path in ("../outside.pt", str(outside), "missing.pt", str(work_dir)):
        with pytest.raises(server.HTTPException) as exc:
            server._confined_path(path, work_dir)
        assert exc.value.status_code == 400


def test_refit_acknowledgments_cover_all_nested_workers():
    payload = {0: [{"version": 4, "loaded_weights": 8}, {"version": 4, "loaded_weights": 12}]}

    assert server._validate_refit_ack(payload, 4) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"todo": "unsupported"},
        {"error": "failed"},
        {"version": 3, "loaded_weights": 2},
        {"version": 4, "loaded_weights": 0},
        {"version": 4, "loaded": 2},
        None,
    ],
)
def test_refit_acknowledgments_fail_closed(payload):
    with pytest.raises(RuntimeError):
        server._validate_refit_ack(payload, 4)


def test_rollout_concatenates_delta_rows_and_preserves_terminal_actions(tmp_path):
    engine = _Engine()
    service = _service(engine, tmp_path)
    service.version = 0
    path = tmp_path / "input.pt"
    torch.save({"version": 0, "prompts": [_prompt(), _prompt()]}, path)

    response = asyncio.run(service.rollout(server.FileRequest(path=str(path), version=0)))

    saved = torch.load(response["path"], weights_only=True)
    assert saved["version"] == 0 and len(saved["rows"]) == 2
    for rows in saved["rows"]:
        torch.testing.assert_close(rows, engine.rows)
        assert rows[-1, :3].tolist() == [33, 7, 6]
    assert len(set(saved["request_ids"])) == 2
    assert engine.closed == set(engine.started)
    assert engine.aborted == []


def test_rollout_failure_aborts_and_drains_the_entire_cohort(tmp_path):
    engine = _Engine(fail=True)
    service = _service(engine, tmp_path)
    service.version = 0
    path = tmp_path / "input.pt"
    torch.save({"version": 0, "prompts": [_prompt(), _prompt()]}, path)

    async def rollout():
        with pytest.raises(server.HTTPException) as exc:
            await asyncio.wait_for(service.rollout(server.FileRequest(path=str(path), version=0)), timeout=2)
        assert exc.value.status_code == 503

    asyncio.run(rollout())

    assert len(engine.started) == 2
    assert set(engine.aborted) == engine.closed == set(engine.started)
    assert service.error is not None
    assert list(tmp_path.glob("rollout_*.pt")) == []


def test_failed_refit_prevents_generating_from_stale_weights(tmp_path):
    class RejectedEngine:
        async def collective_rpc(self, **kwargs):
            return {"todo": "not implemented"}

    service = _service(RejectedEngine(), tmp_path)
    path = tmp_path / "policy.pt"
    torch.save({"decoder.weight": torch.ones(2)}, path)

    async def refit():
        with pytest.raises(server.HTTPException) as exc:
            await service.refit(server.FileRequest(path=str(path), version=1))
        assert exc.value.status_code == 503
        with pytest.raises(server.HTTPException) as exc:
            await service.rollout(server.FileRequest(path=str(path), version=1))
        assert exc.value.status_code == 503

    asyncio.run(refit())

    assert service.version == -1 and service.error is not None


def test_refit_timeout_invalidates_policy_and_cancels_rpc(tmp_path):
    class StalledEngine:
        cancelled = False

        async def collective_rpc(self, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    engine = StalledEngine()
    service = _service(engine, tmp_path)
    service.settings['timeout_seconds'] = 0.01
    path = tmp_path / 'policy.pt'
    torch.save({'decoder.weight': torch.ones(2)}, path)

    async def refit():
        with pytest.raises(server.HTTPException) as exc:
            await service.refit(server.FileRequest(path=str(path), version=1))
        assert exc.value.status_code == 503

    asyncio.run(refit())
    assert engine.cancelled and service.error is not None and service.version == -1


def test_prompt_transport_preserves_bfloat16_embedding_values(tmp_path):
    prompt = _prompt()
    for name in ("grpo_context_embeddings", "grpo_text_embeddings"):
        prompt[name] = torch.randn(2, 4, dtype=torch.bfloat16)
    prepared = _service(_Engine(), tmp_path)._prepare_prompt(prompt)
    for name in ("grpo_context_embeddings", "grpo_text_embeddings"):
        assert prepared[name].dtype == torch.float32
        torch.testing.assert_close(prepared[name], prompt[name].float())
