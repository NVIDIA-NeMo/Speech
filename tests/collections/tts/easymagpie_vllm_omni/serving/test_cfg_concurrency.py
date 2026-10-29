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
"""Mixed requests and atomic cache admission using the real pinned scheduler."""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from easymagpie_vllm_omni.cfg_pairing import is_cfg_hidden_id
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler
from test_cfg_scheduler import _config, _request
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from collections.abc import Collection

    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm_omni.request import OmniRequest


def test_mixed_pair_and_conditional_request_complete_independently() -> None:
    """Verify mixed pair and conditional request complete independently."""
    scheduler = _mixed_scheduler()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_guided("guided"))
    scheduler.add_request(_plain("plain"))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 3
    assert step.num_scheduled_tokens["plain"] == 3
    result = scheduler.update_from_output(step, _batch_output(step, stopped={"plain"}))
    assert {o.request_id for o in result[0].outputs} == {"guided", "plain"}
    assert "plain" not in scheduler.requests
    assert len(scheduler.requests) == 2
    step = scheduler.schedule()
    result = scheduler.update_from_output(step, _batch_output(step, stopped={"guided"}))
    assert {o.request_id for o in result[0].outputs} == {"guided"}
    assert not scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_two_pairs_and_plain_request_have_separate_cache_blocks_and_stop_decisions() -> None:
    """Verify two pairs and plain request have separate cache blocks and stop decisions."""
    scheduler = _mixed_scheduler(max_num_seqs=5)
    scheduler.add_request(_guided("first"))
    scheduler.add_request(_guided("second"))
    scheduler.add_request(_plain("plain"))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 5
    physical = step.scheduled_new_reqs
    for group in range(2):
        blocks = [set(r.block_ids[group]) for r in physical]
        assert all(a.isdisjoint(b) for i, a in enumerate(blocks) for b in blocks[i + 1 :])
    result = scheduler.update_from_output(step, _batch_output(step, stopped={"second"}))
    assert {o.request_id for o in result[0].outputs} == {"first", "second", "plain"}
    assert "second" not in scheduler.requests
    assert "first" in scheduler.requests and "plain" in scheduler.requests
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)


def test_standard_logical_batch_capacity_decodes_all_pairs_and_reuses_freed_slots() -> None:
    """Verify standard logical batch capacity decodes all pairs and reuses freed slots."""
    import yaml
    from conftest import EASYMAGPIE_ROOT

    standard = yaml.safe_load((EASYMAGPIE_ROOT / "deploy/easymagpie.yaml").read_text())["stages"][0]
    logical_capacity = standard["max_num_seqs"]
    scheduler = _mixed_scheduler(
        max_num_seqs=2 * logical_capacity,
        num_blocks=32 * logical_capacity,
        token_budget=2 * standard["max_num_batched_tokens"],
    )
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    for index in range(logical_capacity):
        scheduler.add_request(_guided(f"pair-{index}"))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 * logical_capacity
    scheduler.update_from_output(step, _batch_output(step))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 * logical_capacity
    assert set(step.num_scheduled_tokens.values()) == {1}
    scheduler.update_from_output(step, _batch_output(step, stopped={"pair-0"}))
    assert len(scheduler.cfg_pairs) == logical_capacity - 1
    scheduler.add_request(_guided("replacement"))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 * logical_capacity
    assert "replacement" in step.num_scheduled_tokens
    scheduler.update_from_output(step, _batch_output(step))
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.requests and not scheduler.cfg_pairs
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_sequence_pressure_defers_entire_pair_and_cancellation_allows_reuse() -> None:
    """Verify sequence pressure defers entire pair and cancellation allows reuse."""
    scheduler = _mixed_scheduler(max_num_seqs=2)
    scheduler.add_request(_plain("plain"))
    scheduler.add_request(_guided("waiting"))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"plain": 3}
    assert all(r.num_computed_tokens == 0 for rid, r in scheduler.requests.items() if rid != "plain")
    scheduler.update_from_output(step, _batch_output(step))
    assert scheduler.finish_requests("waiting", RequestStatus.FINISHED_ABORTED) == [("waiting", 0)]
    assert set(scheduler.requests) == {"plain"}
    scheduler.add_request(_guided("replacement"))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"plain": 1}
    scheduler.update_from_output(step, _batch_output(step, stopped={"plain"}))
    step = scheduler.schedule()
    assert "replacement" in step.num_scheduled_tokens and len(step.num_scheduled_tokens) == 2


def test_cache_pressure_queues_pair_without_preempting_running_request() -> None:
    # Ten blocks fit a complete pair's 7-token histories, but cannot hold
    # two complete pairs. Partial reservations must be rolled back.
    """Verify cache pressure queues pair without preempting running request."""
    scheduler = _mixed_scheduler(max_num_seqs=4, num_blocks=10)
    scheduler.add_request(_guided("first"))
    scheduler.add_request(_guided("waiting"))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 and "first" in step.num_scheduled_tokens
    assert "waiting" not in step.num_scheduled_tokens and not step.preempted_req_ids
    scheduler.update_from_output(step, _batch_output(step, stopped={"first"}))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 and "waiting" in step.num_scheduled_tokens
    assert not step.preempted_req_ids


def test_token_budget_defers_whole_pair_during_mixed_prefill() -> None:
    """Verify token budget defers whole pair during mixed prefill."""
    scheduler = _mixed_scheduler(max_num_seqs=3, token_budget=6)
    scheduler.add_request(_plain("plain"))
    scheduler.add_request(_guided("guided"))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"plain": 3}
    scheduler.update_from_output(step, _batch_output(step, stopped={"plain"}))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 and step.total_num_scheduled_tokens == 6


def test_abort_running_pair_preserves_unrelated_request_and_ignores_late_output() -> None:
    """Verify abort running pair preserves unrelated request and ignores late output."""
    scheduler = _mixed_scheduler()
    scheduler.add_request(_guided("guided"))
    scheduler.add_request(_plain("plain"))
    step = scheduler.schedule()
    scheduler.finish_requests("guided", RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(step, _batch_output(step))
    assert set(scheduler.requests) == {"plain"}
    next_step = scheduler.schedule()
    assert next_step.num_scheduled_tokens == {"plain": 1}


def test_infeasible_pair_is_rejected_without_disturbing_existing_request() -> None:
    """Verify infeasible pair is rejected without disturbing existing request."""
    scheduler = _mixed_scheduler(num_blocks=5)
    scheduler.add_request(_plain("plain"))
    step = scheduler.schedule()
    scheduler.update_from_output(step, _batch_output(step))
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="full cache reservation"):
        scheduler.add_request(_guided("too_large"))
    assert set(scheduler.requests) == {"plain"}
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_reservations_leave_engine_watermark_available() -> None:
    """Verify reservations leave engine watermark available."""
    scheduler = _mixed_scheduler(max_num_seqs=4, num_blocks=14)
    scheduler.kv_cache_manager.watermark_blocks = 2
    scheduler.add_request(_guided("first"))
    scheduler.add_request(_guided("waiting"))
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() >= 2
    scheduler.update_from_output(step, _batch_output(step, stopped={"first"}))
    step = scheduler.schedule()
    assert "waiting" in step.num_scheduled_tokens


def test_schedule_error_restores_then_cleans_unadmitted_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify schedule error restores then cleans unadmitted queue.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
    """
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    scheduler = _mixed_scheduler(max_num_seqs=2)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_guided("selected"))
    scheduler.add_request(_plain("waiting"))

    def fail(
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Inject the expected scheduler failure.

        Args:
            *args: Additional positional arguments forwarded to the base API or accepted by the test double.
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Raises:
            RuntimeError: If the asserted test contract is violated.
        """
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(OmniARScheduler, "schedule", fail)
    with pytest.raises(RuntimeError, match="scheduler failed"):
        scheduler.schedule()
    assert not scheduler.requests and not scheduler.waiting and not scheduler.cfg_pairs
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_reservation_error_cleans_selected_and_waiting_units(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify reservation error cleans selected and waiting units.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
    """
    scheduler = _mixed_scheduler(max_num_seqs=4)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_guided("selected"))
    scheduler.add_request(_guided("fails"))
    allocate = scheduler.kv_cache_manager.allocate_slots

    def fail(
        request: OmniRequest,
        *args: Any,
        **kwargs: Any,
    ) -> KVCacheBlocks | None:
        """Inject the expected scheduler failure.

        Args:
            request: Incoming engine request.
            *args: Additional positional arguments forwarded to the base API or accepted by the test double.
            **kwargs: Additional keyword arguments forwarded to the base API or accepted by the test double.

        Returns:
            The wrapped operation's result when no failure is injected.

        Raises:
            RuntimeError: If the asserted test contract is violated.
        """
        if request.request_id == "fails":
            raise RuntimeError("reservation failed")
        return allocate(request, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", fail)
    with pytest.raises(RuntimeError, match="reservation failed"):
        scheduler.schedule()
    assert not scheduler.requests and not scheduler.waiting and not scheduler.cfg_pairs
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_duplicate_request_cannot_free_a_running_history() -> None:
    """Verify duplicate request cannot free a running history."""
    scheduler = _mixed_scheduler()
    scheduler.add_request(_plain("plain"))
    step = scheduler.schedule()
    scheduler.update_from_output(step, _batch_output(step))
    blocks = scheduler.kv_cache_manager.get_blocks("plain")
    with pytest.raises(ValueError, match="Duplicate"):
        scheduler.add_request(_plain("plain"))
    assert scheduler.kv_cache_manager.get_blocks("plain") == blocks


def test_mixed_connector_and_completion_outputs_never_expose_private_branches() -> None:
    """Verify mixed connector and completion outputs never expose private branches."""
    from easymagpie_vllm_omni.cfg_pairing import CFGPairingTransferFilter

    scheduler = _mixed_scheduler(max_num_seqs=5)
    for rid in ("first", "second"):
        scheduler.add_request(_guided(rid))
    scheduler.add_request(_plain("plain"))
    step = scheduler.schedule()
    transfers = []
    scheduler.chunk_transfer_adapter = CFGPairingTransferFilter(
        SimpleNamespace(
            save_async=lambda payload, request, stopped: transfers.append((request.request_id, payload)),
        )
    )
    output = _batch_output(step, stopped={"first", "second", "plain"})
    output.inter_stage_outputs = [{"owner": rid} for rid in output.req_ids]
    result = scheduler.update_from_output(step, output)
    assert sorted(transfers, key=lambda row: row[0]) == [(rid, {"owner": rid}) for rid in ("first", "plain", "second")]
    assert result[0].finished_requests == {"first", "second", "plain"}
    assert not scheduler.cfg_pairs and not scheduler.requests


def test_cache_reservation_covers_decode_to_generation_bound_under_pressure() -> None:
    """Verify cache reservation covers decode to generation bound under pressure."""
    scheduler = _mixed_scheduler(max_num_seqs=4, num_blocks=10)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_guided("first"))
    scheduler.add_request(_guided("waiting"))
    reserved_free = None
    for _ in range(4):
        step = scheduler.schedule()
        assert len(step.num_scheduled_tokens) == 2 and "first" in step.num_scheduled_tokens
        assert not step.preempted_req_ids
        current_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        if reserved_free is None:
            reserved_free = current_free
        # The engine can release obsolete Mamba pages while decoding. It must
        # never consume more than the initial complete-history reservation.
        assert current_free >= reserved_free
        scheduler.update_from_output(step, _batch_output(step))
    assert "first" not in scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2 and "waiting" in step.num_scheduled_tokens


def _guided(rid: str) -> OmniRequest:
    """Build a registered-speaker request with classifier-free guidance enabled.

    Args:
        rid: Request identity.

    Returns:
        A guided known-speaker OmniRequest.
    """
    return _request(
        rid,
        info={
            "use_cfg": True,
            "cfg_scale": 2.5,
            "speaker_id": "eng",
            "text_tokens": [2, 3],
            "temperature": 0.7,
            "top_k": 80,
        },
    )


def _plain(rid: str) -> OmniRequest:
    """Build a conditional registered-speaker request without a CFG partner.

    Args:
        rid: Request identity.

    Returns:
        A conditional known-speaker OmniRequest.
    """
    return _request(rid, info={"speaker_id": "eng", "text_tokens": [2, 3], "temperature": 0.7, "top_k": 80})


def _mixed_scheduler(
    *,
    max_num_seqs: int = 3,
    num_blocks: int = 64,
    token_budget: int = 128,
) -> EasyMagpieCFGScheduler:
    """Construct the real mixed CFG scheduler with a small test cache.

    Args:
        max_num_seqs: Maximum physical sequence slots.
        num_blocks: Number of available cache blocks.
        token_budget: Maximum scheduled tokens per model step.

    Returns:
        The mixed scheduler using the configured test cache and token budget.
    """
    config = _config(num_blocks)
    config.model_config.hf_config.use_codec_latent_audio_embedding = True
    config.model_config.hf_config.codec_num_levels_per_group = [8, 8, 16]
    config.model_config.hf_config.cfg_capability = {"version": 1, "context_cas_disabled": False}
    config.model_config.hf_config.cfg_unk_token_id = 12
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = token_budget
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    mamba = MambaSpec(block_size=4, shapes=((8,),), dtypes=(torch.float32,), mamba_cache_mode="none")
    caches = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["backbone.attn", "predictor.attn"], attention),
            KVCacheGroupSpec(["backbone.mamba"], mamba),
        ],
    )
    return EasyMagpieCFGScheduler(
        vllm_config=config,
        kv_cache_config=caches,
        structured_output_manager=SimpleNamespace(should_advance=lambda request: False),
        block_size=4,
        mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda model: False),
    )


def _batch_output(
    step: SchedulerOutput,
    *,
    stopped: Collection[str] = (),
) -> ModelRunnerOutput:
    """Build sampled outputs for the scheduled physical requests.

    Args:
        step: Scheduled physical requests for the test step.
        stopped: Physical request IDs that stop on this step.

    Returns:
        Model-runner output aligned with the scheduled physical identities.
    """
    ids = list(reversed(step.num_scheduled_tokens))
    # Deliberately contrary private stop values must be overwritten by the scheduler.
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[int(rid in stopped or is_cfg_hidden_id(rid))] for rid in ids],
    )
