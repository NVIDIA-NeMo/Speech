# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in host accounting for complete-text CFG on vLLM/Omni 0.24.

This scheduler alone does not enable or validate asynchronous model execution.
Native engine futures remain the execution queue. Step records below only associate
results and resource fences with the exact SchedulerOutput passed to that queue.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import vllm
import vllm_omni
from easymagpie_vllm_omni.cfg_graph import uses_cfg_decode_graphs
from easymagpie_vllm_omni.cfg_pairing import CFGPairingError, filter_cfg_outputs
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler, _CFGHistoryScheduler, validate_cfg_config
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


class EasyMagpieCFGAsyncScheduler(EasyMagpieCFGScheduler, AsyncScheduler):
    """Atomic CFG admission with native async counters and fenced retirement.

    Selected by the explicit easymagpie_cfg_async topologies. The ordinary CFG
    topology retains synchronous scheduling.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if vllm.__version__ != '0.24.0' or vllm_omni.__version__ != '0.24.0':
            raise RuntimeError('Async CFG accounting requires vLLM/Omni 0.24.0')
        super().__init__(*args, **kwargs)
        self.defer_block_free = True
        self._cfg_steps: deque[_CFGAsyncStep] = deque()
        self._cfg_retiring: dict[str, int] = {}
        self._cfg_failure: Exception | None = None
        self.cfg_confirmed_positions: dict[str, int] = {}

    def add_request(self, request) -> None:
        self._check_health()
        if request.resumable:
            raise ValueError('Async CFG resumable input requires the P6 confirmation gate')
        if request.request_id in self._cfg_retiring:
            raise ValueError('Request identity is retiring; wait for worker cleanup confirmation')
        if request.num_prompt_tokens + request.max_tokens > self.max_model_len:
            raise ValueError('Async CFG full history exceeds max_model_len')
        # Bypass the synchronous resumable/session layer, retaining paired admission.
        return _CFGHistoryScheduler.add_request(self, request)

    def schedule(self, throttle_prefills: bool = False):
        self._check_health()
        if len(self._cfg_steps) >= 2:
            raise CFGPairingError('Async CFG permits at most two unconsumed engine batches')
        waiting = self.waiting
        try:
            self.waiting = self._admit_waiting_units()
            # Omni -> AsyncScheduler -> Scheduler: pair validation is our hook;
            # native progress and placeholder hooks each run exactly once.
            output = OmniARScheduler.schedule(self, throttle_prefills)
            positions = {rid: self.requests[rid].num_computed_tokens for rid in output.num_scheduled_tokens}
            self._cfg_steps.append(
                _CFGAsyncStep(output, self.sched_step_seq, positions, dict(output.num_scheduled_tokens))
            )
            return output
        except Exception as error:
            self._cfg_failure = error
            # No rollback/free of possibly submitted histories on inconsistency.
            raise
        finally:
            if self.waiting is not waiting:
                for request in reversed(list(self.waiting)):
                    waiting.prepend_request(request)
                self.waiting = waiting

    def update_from_output(self, scheduler_output, model_runner_output):
        self._check_health()
        try:
            if not self._cfg_steps or self._cfg_steps[0].output is not scheduler_output:
                raise CFGPairingError('Result does not match the oldest native scheduled batch')
            record = self._cfg_steps[0]
            if record.sequence != self.processed_step_seq + 1:
                raise CFGPairingError("Native completion fence sequence mismatch")
            self._validate_result(scheduler_output, model_runner_output)
            # Preserve the public decision without consuming tokens a second time.
            output = copy.copy(model_runner_output)
            output.sampled_token_ids = [list(row) for row in output.sampled_token_ids]
            for pair in self.cfg_pairs.values():
                if pair.conditional_id in scheduler_output.num_scheduled_tokens:
                    by_id = output.req_id_to_index
                    output.sampled_token_ids[by_id[pair.unconditional_id]] = list(
                        output.sampled_token_ids[by_id[pair.conditional_id]]
                    )
            result = OmniARScheduler.update_from_output(self, scheduler_output, output)
            for pair in list(self.cfg_pairs.values()):
                alive = [rid in self.requests for rid in pair.physical_ids]
                if any(alive) != all(alive):
                    raise CFGPairingError('Async CFG branches finished inconsistently')
                if not any(alive):
                    self.cfg_pairs.pop(pair.conditional_id)
            filter_cfg_outputs(result)
            self._cfg_steps.popleft()
            # Omni bypasses Scheduler.update_from_output, so explicitly maintain
            # its native deferred-free fence, including cleanup-only empty batches.
            self.processed_step_seq = record.sequence
            self._drain_deferred_frees()
            for rid, position in record.positions.items():
                if rid in self.requests:
                    self.cfg_confirmed_positions[rid] = position
            for rid, fence in list(self._cfg_retiring.items()):
                if fence <= self.processed_step_seq:
                    self._cfg_retiring.pop(rid)
                    self.cfg_confirmed_positions.pop(rid, None)
            return result
        except Exception as error:
            self._cfg_failure = error
            raise

    def _validate_config(self, config) -> None:
        if not config.scheduler_config.async_scheduling:
            raise ValueError('Async CFG requires async_scheduling')
        if not config.model_config.enforce_eager and not uses_cfg_decode_graphs(config):
            raise ValueError('Async CFG graphs require the explicit async scheduler and a supported graph policy')
        if config.use_v2_model_runner:
            raise ValueError('Async CFG requires the audited V1 model runner')
        # Apply all permanent CFG exclusions without mutating the live config.
        control = copy.copy(config)
        control.scheduler_config = copy.copy(config.scheduler_config)
        control.scheduler_config.async_scheduling = False
        validate_cfg_config(control)

    def _check_health(self) -> None:
        if self._cfg_failure is not None:
            raise CFGPairingError('Async CFG scheduler is unusable; tear down the executor') from self._cfg_failure

    def _running_token_cost(self) -> int:
        cost = 0
        for request in self.running:
            pending = request.num_output_placeholders
            # Identical eligibility/cap rules to Scheduler.schedule in 0.24.
            if (
                pending > 0
                and request.num_computed_tokens + 2 - pending >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            demand = request.num_tokens_with_spec + pending - request.num_computed_tokens
            cost += max(0, min(demand, self.max_model_len - request.num_computed_tokens - 1))
        return cost

    def _update_after_schedule(self, output) -> None:
        if output.preempted_req_ids:
            raise CFGPairingError('Async CFG does not support preemption')
        for pair in self.cfg_pairs.values():
            pair.validate_step(
                output.num_scheduled_tokens, {rid: self.requests[rid].num_computed_tokens for rid in pair.physical_ids}
            )
            values = [self.requests[rid].num_output_placeholders for rid in pair.physical_ids]
            if values[0] != values[1]:
                raise CFGPairingError('Paired output placeholders disagree')
        if not output.total_num_scheduled_tokens:
            # Base schedule increments only nonempty batches. Empty execution
            # still acknowledges worker cleanup and must have a distinct fence.
            self.sched_step_seq += 1
        for rid, fence in self._cfg_retiring.items():
            if fence == self.sched_step_seq and rid not in output.finished_req_ids:
                raise CFGPairingError("Retirement fence is missing worker cleanup identity")
        AsyncScheduler._update_after_schedule(self, output)

    def _preempt_request(self, request, timestamp) -> None:
        raise CFGPairingError('Async CFG cannot preempt a physical history')

    def _free_request_blocks(self, request) -> None:
        # Next schedule carries finished_req_ids to the worker. FIFO consumption
        # of that cleanup batch fences every earlier reference, including B after
        # terminal A. Reuse native deferred blocks, not a second allocator.
        fence = self.sched_step_seq + 1
        self._cfg_retiring[request.request_id] = fence
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((fence, blocks))

    def _validate_result(self, step, result) -> None:
        ids = result.req_ids
        mapping = result.req_id_to_index
        scheduled = step.num_scheduled_tokens
        if scheduled != self._cfg_steps[0].counts or step.total_num_scheduled_tokens != sum(scheduled.values()):
            raise CFGPairingError("Scheduled batch metadata changed before result application")
        if len(ids) != len(set(ids)) or set(ids) != set(scheduled) or mapping != {rid: i for i, rid in enumerate(ids)}:
            raise CFGPairingError('Malformed sampled request row mapping')
        tokens = result.sampled_token_ids
        if len(tokens) != len(ids) or any(len(row) != 1 for row in tokens):
            raise CFGPairingError('Complete-text async CFG requires one sampled output per scheduled request')
        for rid in scheduled:
            request = self.requests.get(rid)
            expected = sum(rid in record.counts for record in self._cfg_steps)
            if request is not None and request.num_output_placeholders != expected:
                raise CFGPairingError('Output placeholder accounting mismatch')
        for pair in self.cfg_pairs.values():
            selected = [rid in scheduled for rid in pair.physical_ids]
            if any(selected) != all(selected):
                raise CFGPairingError('Incomplete CFG pair result')
            if all(selected) and len({self.requests[rid].num_output_placeholders for rid in pair.physical_ids}) != 1:
                raise CFGPairingError('Paired output placeholder accounting mismatch')


@dataclass(frozen=True)
class _CFGAsyncStep:
    """Result association only; the native engine owns futures and execution."""

    output: Any
    sequence: int
    positions: dict[str, int]
    counts: dict[str, int]
