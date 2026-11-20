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
"""Generation-scoped input ownership for vLLM-Omni 0.24.

A native state-map removal or worker abort does not prove that its input producer
has stopped. Keep a separate owner until task exit, including source finalization
and final-sentinel transport. A cleanup timeout poisons the client and retains its
owners; it never grants permission to reuse a physical identity.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps

# Internal bound, deliberately separate from inference and input pacing timeouts.
_PRODUCER_CLEANUP_TIMEOUT = 5.0


def patch_producer_lifecycle() -> None:
    """Install idempotent client-side ownership hooks for the pinned native API."""
    from vllm_omni.engine.messages import ErrorMessage
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    original_add = AsyncOmni._add_streaming_input_request
    if getattr(original_add, '_easymagpie_producer_patched', False):
        return
    original_results = AsyncOmni._process_orchestrator_results
    original_abort = AsyncOmni._abort
    original_summary = AsyncOmni._log_summary_and_cleanup
    original_generate = AsyncOmni.generate

    @wraps(original_add)
    async def add(self, *, request_id, input_stream, **kwargs):
        _check_context(self)
        owners = _owners(self)
        if request_id in owners:
            raise RuntimeError(f'Input producer still owns physical request {request_id}')
        owner = owners[request_id] = _ProducerOwner(self.request_states[request_id], input_source=input_stream)

        async def guarded_input():
            try:
                try:
                    while True:
                        try:
                            chunk = await anext(input_stream)
                        except StopAsyncIteration:
                            if owner.closing:
                                raise asyncio.CancelledError
                            owner.exhausted = True
                            return
                        # Suppressed cancellation may return or yield again.
                        if owner.closing:
                            raise asyncio.CancelledError
                        yield chunk
                finally:
                    await input_stream.aclose()
            except BaseException as error:
                if not isinstance(error, (asyncio.CancelledError, GeneratorExit)):
                    owner.error = error
                # Native sends a normal final sentinel after ordinary source
                # errors. That races the consumer's error-triggered abort and
                # can reach a retiring generation. The done callback reports
                # the saved error; take native's cancellation path to omit it.
                if owner.closing or owner.error is not None:
                    raise asyncio.CancelledError from error
                raise

        guarded = owner.source = guarded_input()
        try:
            owner.task = await original_add(self, request_id=request_id, input_stream=guarded, **kwargs)
        except BaseException:
            await guarded.aclose()
            owners.pop(request_id, None)
            raise

        def producer_done(task):
            if task.cancelled():
                error = None if owner.closing else RuntimeError('Input producer unexpectedly cancelled')
            else:
                error = task.exception()
            error = owner.error or error
            if error is not None:
                # Native final-sentinel exceptions otherwise bypass the output
                # queue, leaving a consumer waiting forever for a terminal output.
                owner.state.queue.put_nowait(ErrorMessage(request_id=request_id, error=str(error), fatal=False))

        owner.task.add_done_callback(producer_done)
        return owner.task

    @wraps(original_results)
    async def results(self, request_id, *args, **kwargs):
        iterator = original_results(self, request_id, *args, **kwargs)
        try:
            async for output in iterator:
                _check_context(self)
                yield output
        finally:
            await iterator.aclose()
        owner = _owners(self).get(request_id)
        if owner is not None:
            # A terminal output can arrive before input exhaustion. Stop a still
            # open source, but let an already exhausted source finish its sentinel.
            await _join(self, owner, cancel=not owner.exhausted)
            if owner.abort_task is not None:
                await _await_cleanup(owner.abort_task)
                raise RuntimeError(f'Request {request_id} aborted')
            if owner.error is not None:
                raise owner.error
            if not owner.task.cancelled() and owner.task.exception() is not None:
                raise owner.task.exception()
        _check_context(self)

    @wraps(original_abort)
    async def abort(self, request_ids):
        owners = _owners(self)
        pending = []
        unmanaged = []
        for rid in request_ids:
            owner = owners.get(rid)
            if owner is None:
                unmanaged.append(rid)
                continue
            if owner.abort_task is None:
                owner.closing = True
                owner.state.queue.put_nowait(ErrorMessage(request_id=rid, error=f'Request {rid} aborted', fatal=False))

                async def retire(rid=rid, owner=owner):
                    await _join(self, owner, cancel=True)
                    owner.transport_task = asyncio.create_task(self.engine.abort_async([rid]))
                    done, _ = await asyncio.wait({owner.transport_task}, timeout=_PRODUCER_CLEANUP_TIMEOUT)
                    if not done:
                        failure = RuntimeError(f'Abort transport did not exit for {rid}')
                        _fail_context(self, failure)
                        raise failure
                    try:
                        owner.transport_task.result()
                    except BaseException as error:
                        _fail_context(self, error)
                        raise
                    if self.request_states.get(rid) is owner.state:
                        self.request_states.pop(rid)
                    owners.pop(rid, None)
                    if owner.error is not None:
                        raise owner.error

                owner.abort_task = asyncio.create_task(retire())
            pending.append(owner.abort_task)
        if unmanaged:
            pending.append(asyncio.create_task(original_abort(self, unmanaged)))
        if pending:
            # Await every owned retirement even when one fails. Cancellation of
            # this caller must not cancel an input finalizer or detach retirement.
            async def complete():
                outcomes = await asyncio.gather(*pending, return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome

            await _await_cleanup(asyncio.create_task(complete()))

    @wraps(original_summary)
    def summary(self, request_id):
        owner = _owners(self).get(request_id)
        if owner is not None and (
            owner.join_task is None or not owner.join_task.done() or owner.join_task.exception() is not None
        ):
            raise RuntimeError(f'Cannot release live input producer {request_id}')
        _check_context(self)
        original_summary(self, request_id)
        _owners(self).pop(request_id, None)

    @wraps(original_generate)
    async def generate(self, *args, **kwargs):
        _check_context(self)
        iterator = original_generate(self, *args, **kwargs)
        try:
            async for output in iterator:
                yield output
        finally:
            # Forward client aclose immediately; an async-for alone leaves the
            # native generator suspended until asynchronous generator GC.
            await iterator.aclose()

    add._easymagpie_producer_patched = True
    AsyncOmni._add_streaming_input_request = add
    AsyncOmni._process_orchestrator_results = results
    AsyncOmni._abort = abort
    AsyncOmni._log_summary_and_cleanup = summary
    AsyncOmni.generate = generate


@dataclass
class _ProducerOwner:
    state: object
    input_source: object
    task: asyncio.Task | None = None
    closing: bool = False
    exhausted: bool = False
    join_task: asyncio.Task | None = None
    abort_task: asyncio.Task | None = None
    transport_task: asyncio.Task | None = None
    error: BaseException | None = None
    source: object = None
    close_task: asyncio.Task | None = None


def _owners(client):
    owners = getattr(client, '_easymagpie_producer_owners', None)
    if owners is None:
        owners = client._easymagpie_producer_owners = {}
    return owners


def _check_context(client):
    failure = getattr(client, '_easymagpie_producer_failure', None)
    if failure is not None:
        raise RuntimeError('Input producer cleanup failed; replace the owning AsyncOmni client') from failure


async def _await_cleanup(task):
    """Join despite repeated caller cancellation, then preserve cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _join(client, owner, *, cancel):
    if cancel:
        owner.closing = True
        # Native generate already cancels on disconnect. A second cancel would
        # interrupt the slow source finalizer we are obligated to join.
        if not owner.task.done() and not owner.task.cancelling():
            owner.task.cancel()
    if owner.join_task is None:

        async def join():
            deadline = asyncio.get_running_loop().time() + _PRODUCER_CLEANUP_TIMEOUT
            for phase in ('producer', 'source finalizer', 'input finalizer'):
                if phase == 'producer':
                    task = owner.task
                else:
                    # Native async-for does not close a suspended source after a
                    # transport error. Task exit alone is insufficient here.
                    source = owner.source if phase == 'source finalizer' else owner.input_source
                    task = owner.close_task = asyncio.create_task(source.aclose())
                remaining = max(0, deadline - asyncio.get_running_loop().time())
                done, _ = await asyncio.wait({task}, timeout=remaining)
                if not done:
                    owner.closing = True
                    failure = RuntimeError(f'Input {phase} did not exit for {owner.state.request_id}')
                    _fail_context(client, failure)
                    raise failure
                if phase != 'producer' and not task.cancelled():
                    try:
                        task.result()
                    except BaseException as error:
                        _fail_context(client, error)
                        raise

        owner.join_task = asyncio.create_task(join())
    await _await_cleanup(owner.join_task)


def _fail_context(client, failure):
    from vllm_omni.engine.messages import ErrorMessage

    client._easymagpie_producer_failure = failure
    # Wake all consumers and fence future admission, retaining unreleased owners.
    # Worker/cache release remains a separate gate.
    states = {id(state): state for state in client.request_states.values()}
    for retained in _owners(client).values():
        retained.closing = True
        states[id(retained.state)] = retained.state
    for state in states.values():
        state.queue.put_nowait(ErrorMessage(request_id=state.request_id, error=str(failure), fatal=True))
