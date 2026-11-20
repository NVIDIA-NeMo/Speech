# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async host accounting for paired CFG histories on vLLM/Omni 0.24.

Native engine futures remain the execution queue. Step records below only associate
results and resource fences with the exact SchedulerOutput passed to that queue.

Streaming histories keep the native two-step lookahead, like complete-text ones.
The scheduler never submits a step past a text-segment limit, and the only stop it
cannot predict, acoustic EOS, ends the utterance (``stream_policy``). A speculative
successor therefore only ever targets a finished request, whose late result is
discarded. A paused or resumed stream never has an unconfirmed step; that invariant
is checked, not assumed. Public resumable admission still requires the explicit
``EASYMAGPIE_ASYNC_STREAMING`` opt-in pending GPU acceptance.
"""
from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import vllm
import vllm_omni
from easymagpie_vllm_omni.cfg_graph import uses_cfg_decode_graphs
from easymagpie_vllm_omni.cfg_pairing import CFGPairingError, filter_cfg_outputs, is_cfg_hidden_id
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler, _CFGHistoryScheduler, validate_cfg_config
from easymagpie_vllm_omni.config import async_streaming_enabled
from easymagpie_vllm_omni.stream_policy import StreamStop, end_stream_at_terminal_stop
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.serialization import deserialize_additional_information

logger = logging.getLogger(__name__)

# Interval between scheduling summaries in the engine log.
_STATS_LOG_INTERVAL_S = 60.0


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
        self._cfg_applying_step: _CFGAsyncStep | None = None
        self.cfg_step_stats = CFGAsyncStepStats()

    def add_request(self, request) -> None:
        self._check_health()
        streaming = request.resumable or request.request_id in self._cfg_streams
        if streaming and not async_streaming_enabled():
            raise ValueError('Async CFG resumable input remains disabled pending P6 retained-state validation')
        if async_streaming_enabled() and request.request_id not in self.requests and request.prompt_token_ids == [0]:
            # A placeholder prompt is never a fresh synthesis request: this is the tail (update or
            # closing sentinel) of a stream that was aborted or finished, e.g. at acoustic EOS or by a
            # client disconnect racing the input iterator. Raising here is fatal for the engine core.
            logger.info(
                'Dropping late placeholder input for removed stream %s (resumable=%s, retiring=%s, keys=%s)',
                request.request_id,
                request.resumable,
                request.request_id in self._cfg_retiring,
                sorted(deserialize_additional_information(request.additional_information)),
            )
            return
        if request.request_id in self._cfg_retiring:
            raise ValueError('Request identity is retiring; wait for worker cleanup confirmation')
        if request.num_prompt_tokens + request.max_tokens > self.max_model_len:
            raise ValueError('Async CFG full history exceeds max_model_len')
        if streaming:
            # Opt-in: the synchronous session layer validates and queues stream updates.
            return EasyMagpieCFGScheduler.add_request(self, request)
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
            # native progress and placeholder hooks each run exactly once. The
            # native segment-limit skip keeps every running history, streaming or
            # not, from being submitted past its known generation limit.
            output = OmniARScheduler.schedule(self, throttle_prefills)
            positions = {rid: self.requests[rid].num_computed_tokens for rid in output.num_scheduled_tokens}
            self.cfg_step_stats.record_schedule(output, running=len(self.running), in_flight=len(self._cfg_steps))
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
            streams = set(self._cfg_streams)
            # Rows of histories that finished after this step was submitted are
            # speculative work; the native update skips them.
            discarded = sum(rid not in self.requests for rid in record.counts)
            self._cfg_applying_step = record
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
            # Omni arms one async discard at every resumable stop, assuming a
            # speculative successor. A paused stream has none: the native skip
            # never submits past a segment limit, and _handle_stopped_request
            # verified it. Only histories paused by this result carry the marker.
            for rid in record.counts:
                request = self.requests.get(rid)
                if request is not None and request.resumable and request.async_tokens_to_discard:
                    if request.async_tokens_to_discard != 1 or request.num_output_placeholders:
                        raise CFGPairingError('Unexpected streaming completion accounting')
                    request.async_tokens_to_discard = 0
                    self.cfg_step_stats.segment_pauses += not is_cfg_hidden_id(rid)
            self.cfg_step_stats.discarded_rows += discarded
            for client in result.values():
                for public in client.outputs:
                    if public.request_id in streams and public.request_id not in self.requests:
                        public.is_segment_finished = False
            self._prune_streams()
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
            self.cfg_step_stats.maybe_log()
            return result
        except Exception as error:
            self._cfg_failure = error
            raise
        finally:
            self._cfg_applying_step = None

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

    def _pending_stream_steps(self, request, *, exclude_applying: bool = False) -> int:
        # Native resume is part of applying the stopping result, so it may
        # exclude that result. Any other unconfirmed step of a paused or
        # resumed stream would carry stale model state into the next segment.
        if not request.resumable:
            return 0
        return sum(
            request.request_id in record.counts
            for record in self._cfg_steps
            if not exclude_applying or record is not self._cfg_applying_step
        )

    def _can_schedule_history(self, request) -> bool:
        # Consulted only on the waiting/resume admission path, where a history
        # cannot have an unconfirmed step by construction. Running histories,
        # streaming or not, keep their native lookahead.
        return self._pending_stream_steps(request) == 0

    def _handle_stopped_request(self, request) -> bool:
        # The synchronous handler applies the same idempotent policy; apply it
        # here first to account for the reason.
        reason = end_stream_at_terminal_stop(request, eos_ends_stream=True)
        if reason is StreamStop.ACOUSTIC_EOS and not is_cfg_hidden_id(request.request_id):
            self.cfg_step_stats.eos_stream_ends += 1
        finished = super()._handle_stopped_request(request)
        if not finished and (
            self._pending_stream_steps(request, exclude_applying=True) or request.num_output_placeholders
        ):
            raise CFGPairingError('Cannot pause streaming history with an unconfirmed successor step')
        return finished

    def _update_request_as_session(self, session, update) -> None:
        try:
            if self._pending_stream_steps(session, exclude_applying=True) or session.num_output_placeholders:
                raise CFGPairingError('Cannot resume streaming history with an unconfirmed step')
            super()._update_request_as_session(session, update)
        except Exception as error:
            self._cfg_failure = error
            raise

    def _abort_stream_update(self, ids, error) -> None:
        self._cfg_failure = error
        super()._abort_stream_update(ids, error)

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
            if request is not None and request.resumable:
                if request.async_tokens_to_discard:
                    raise CFGPairingError('Streaming history has a stale output marker')
                # The native scheduler never submits past a known segment limit. A
                # successor there would carry its model state into the next segment.
                if expected > 1 and request.num_output_tokens + 1 >= request.max_tokens:
                    raise CFGPairingError('Streaming history has an unconfirmed step past its segment limit')
            if request is not None and request.num_output_placeholders != expected:
                raise CFGPairingError('Output placeholder accounting mismatch')
        for pair in self.cfg_pairs.values():
            selected = [rid in scheduled for rid in pair.physical_ids]
            if any(selected) != all(selected):
                raise CFGPairingError('Incomplete CFG pair result')
            if all(selected) and len({self.requests[rid].num_output_placeholders for rid in pair.physical_ids}) != 1:
                raise CFGPairingError('Paired output placeholder accounting mismatch')


@dataclass
class CFGAsyncStepStats:
    """Cumulative async CFG scheduling counters, logged periodically.

    ``scheduled_rows / running_rows`` near 1 and a high ``overlapped_steps`` share
    mean every running history advances in every batch while the next batch is
    prepared. A drop of either is the signature of histories being held back.

    Attributes:
        steps: Nonempty batches scheduled.
        overlapped_steps: Nonempty batches scheduled while another was unconsumed.
        scheduled_rows: Physical histories scheduled, summed over steps.
        running_rows: Running physical histories at schedule time, summed over steps.
        segment_pauses: Logical streams paused or resumed at a text-segment limit.
        eos_stream_ends: Logical streams ended by acoustic EOS.
        discarded_rows: Speculative result rows for histories that had already finished.
    """

    steps: int = 0
    overlapped_steps: int = 0
    scheduled_rows: int = 0
    running_rows: int = 0
    segment_pauses: int = 0
    eos_stream_ends: int = 0
    discarded_rows: int = 0
    _logged_steps: int = field(default=0, repr=False, compare=False)
    _logged_at: float = field(default_factory=time.monotonic, repr=False, compare=False)

    def record_schedule(self, output: Any, *, running: int, in_flight: int) -> None:
        """Count one native schedule; ``in_flight`` is the number of unconsumed batches before it."""
        if not output.total_num_scheduled_tokens:
            return
        self.steps += 1
        self.overlapped_steps += in_flight > 0
        self.scheduled_rows += len(output.num_scheduled_tokens)
        self.running_rows += running

    def maybe_log(self, interval_s: float = _STATS_LOG_INTERVAL_S) -> None:
        """Log a summary when new steps were scheduled since the last one and the interval elapsed."""
        now = time.monotonic()
        if self.steps == self._logged_steps or now - self._logged_at < interval_s:
            return
        self._logged_steps, self._logged_at = self.steps, now
        logger.info(
            'Async CFG scheduling: %d steps, %.1f%% overlapped, %.3f scheduled/running histories, '
            '%d segment pauses, %d EOS stream ends, %d discarded speculative rows',
            self.steps,
            100.0 * self.overlapped_steps / self.steps,
            self.scheduled_rows / max(1, self.running_rows),
            self.segment_pauses,
            self.eos_stream_ends,
            self.discarded_rows,
        )


@dataclass(frozen=True)
class _CFGAsyncStep:
    """Result association only; the native engine owns futures and execution."""

    output: Any
    sequence: int
    positions: dict[str, int]
    counts: dict[str, int]
