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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from vllm.engine.protocol import StreamingInput
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.outputs import OmniRequestOutput

# Internal bound, deliberately separate from inference and input pacing timeouts.
_PRODUCER_CLEANUP_TIMEOUT = 5.0


def patch_producer_lifecycle() -> None:
    """Install idempotent client-side ownership hooks for the pinned native API."""
    from vllm_omni.engine.messages import ErrorMessage
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    original_add = AsyncOmni._add_streaming_input_request
    if getattr(original_add, "_easymagpie_producer_patched", False):
        return
    original_results = AsyncOmni._process_orchestrator_results
    original_abort = AsyncOmni._abort
    original_summary = AsyncOmni._log_summary_and_cleanup
    original_generate = AsyncOmni.generate

    @wraps(original_add)
    async def add(
        self,
        *,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        **kwargs: Any,
    ) -> asyncio.Task:
        """Register a producer owner and submit a guarded input stream through the native API.

        Args:
            request_id: Physical request ID whose input the producer feeds.
            input_stream: Client input stream to forward.
            **kwargs: Remaining keyword arguments forwarded to the native method.

        Returns:
            Native producer task, with a done callback that reports producer errors to the consumer.

        Raises:
            RuntimeError: If the client is poisoned or a producer still owns ``request_id``.
        """
        _check_context(self)
        owners = _owners(self)
        if request_id in owners:
            raise RuntimeError(f"Input producer still owns physical request {request_id}")
        owner = owners[request_id] = _ProducerOwner(self.request_states[request_id], input_source=input_stream)

        async def guarded_input() -> AsyncGenerator[StreamingInput, None]:
            """Forward input chunks until exhaustion or closing, always closing the client stream.

            Yields:
                Input chunks from the client stream.

            Raises:
                asyncio.CancelledError: If the owner is closing or the source failed, so the native
                    final sentinel is not sent.
            """
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

        def producer_done(task: asyncio.Task) -> None:
            """Queue a non-fatal error message when the producer failed or was unexpectedly cancelled.

            Args:
                task: Finished native producer task.
            """
            if task.cancelled():
                error = None if owner.closing else RuntimeError("Input producer unexpectedly cancelled")
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
    async def results(
        self,
        request_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Yield native results, then join the input producer and surface its failure.

        Args:
            request_id: Physical request ID whose results are read.
            *args: Positional arguments forwarded to the native method.
            **kwargs: Keyword arguments forwarded to the native method.

        Yields:
            Request outputs from the native result stream.

        Raises:
            RuntimeError: If the client is poisoned or the request was aborted.
        """
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
                raise RuntimeError(f"Request {request_id} aborted")
            if owner.error is not None:
                raise owner.error
            if not owner.task.cancelled() and owner.task.exception() is not None:
                raise owner.task.exception()
        _check_context(self)

    @wraps(original_abort)
    async def abort(self, request_ids: list[str]) -> None:
        """Retire owned producers before engine abort and abort unmanaged requests natively.

        Waits for every retirement even when the caller is cancelled.

        Args:
            request_ids: Physical request IDs to abort.
        """
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
                owner.state.queue.put_nowait(ErrorMessage(request_id=rid, error=f"Request {rid} aborted", fatal=False))

                async def retire(
                    rid: str = rid,
                    owner: _ProducerOwner = owner,
                ) -> None:
                    """Join the producer, send the engine abort, and release the request state.

                    Args:
                        rid: Physical request ID to retire.
                        owner: Producer owner of the request.

                    Raises:
                        RuntimeError: If the abort transport does not exit within the cleanup timeout.
                    """
                    await _join(self, owner, cancel=True)
                    owner.transport_task = asyncio.create_task(self.engine.abort_async([rid]))
                    done, _ = await asyncio.wait({owner.transport_task}, timeout=_PRODUCER_CLEANUP_TIMEOUT)
                    if not done:
                        failure = RuntimeError(f"Abort transport did not exit for {rid}")
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
            async def complete() -> None:
                """Wait for all pending retirements, then re-raise the first failure."""
                outcomes = await asyncio.gather(*pending, return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome

            await _await_cleanup(asyncio.create_task(complete()))

    @wraps(original_summary)
    def summary(self, request_id: str) -> None:
        """Release request bookkeeping only after its input producer has been joined.

        Args:
            request_id: Physical request ID to summarize and release.

        Raises:
            RuntimeError: If the producer is still live or the client is poisoned.
        """
        owner = _owners(self).get(request_id)
        if owner is not None and (
            owner.join_task is None or not owner.join_task.done() or owner.join_task.exception() is not None
        ):
            raise RuntimeError(f"Cannot release live input producer {request_id}")
        _check_context(self)
        original_summary(self, request_id)
        _owners(self).pop(request_id, None)

    @wraps(original_generate)
    async def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Forward native generation and close the native generator as soon as the client closes.

        Args:
            *args: Positional arguments forwarded to the native method.
            **kwargs: Keyword arguments forwarded to the native method.

        Yields:
            Request outputs from the native generator.

        Raises:
            RuntimeError: If the client is poisoned by an earlier cleanup failure.
        """
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
    """Ownership record of one physical request's input producer until it has fully exited.

    Attributes:
        state: Native client request state of the physical request.
        input_source: Client input stream passed to the producer.
        task: Native producer task.
        closing: Whether the producer is being stopped deliberately.
        exhausted: Whether the input stream ended normally.
        join_task: Task joining the producer and closing both streams.
        abort_task: Task retiring the request after an abort.
        transport_task: Task sending the engine abort.
        error: Error raised by the input stream.
        source: Guarded stream submitted to the native producer.
        close_task: Task closing the guarded or client stream during a join.
    """

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


def _owners(client: AsyncOmni) -> dict[str, _ProducerOwner]:
    """Return the client's producer owners, creating the mapping on first use.

    Args:
        client: Patched client that holds the owners.

    Returns:
        Producer owners keyed by physical request ID.
    """
    owners = getattr(client, "_easymagpie_producer_owners", None)
    if owners is None:
        owners = client._easymagpie_producer_owners = {}
    return owners


def _check_context(client: AsyncOmni) -> None:
    """Refuse to use a client poisoned by an earlier producer cleanup failure.

    Args:
        client: Patched client to check.

    Raises:
        RuntimeError: If producer cleanup previously failed on this client.
    """
    failure = getattr(client, "_easymagpie_producer_failure", None)
    if failure is not None:
        raise RuntimeError("Input producer cleanup failed; replace the owning AsyncOmni client") from failure


async def _await_cleanup(task: asyncio.Task) -> Any:
    """Join despite repeated caller cancellation, then preserve cancellation.

    Args:
        task: Cleanup task to wait for.

    Returns:
        Result of the task.

    Raises:
        asyncio.CancelledError: If the caller was cancelled while waiting.
    """
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


async def _join(
    client: AsyncOmni,
    owner: _ProducerOwner,
    *,
    cancel: bool,
) -> None:
    """Wait for the producer task, then close the guarded and client streams within one deadline.

    The join runs once per owner; later callers wait for the same task.

    Args:
        client: Patched client that owns the producer.
        owner: Producer owner to join.
        cancel: Whether to cancel the producer task first.

    Raises:
        RuntimeError: If a phase does not exit within the cleanup timeout.
    """
    if cancel:
        owner.closing = True
        # Native generate already cancels on disconnect. A second cancel would
        # interrupt the slow source finalizer we are obligated to join.
        if not owner.task.done() and not owner.task.cancelling():
            owner.task.cancel()
    if owner.join_task is None:

        async def join() -> None:
            """Join the producer, source finalizer, and input finalizer in order, poisoning the client on failure.

            Raises:
                RuntimeError: If a phase does not exit before the deadline.
            """
            deadline = asyncio.get_running_loop().time() + _PRODUCER_CLEANUP_TIMEOUT
            for phase in ("producer", "source finalizer", "input finalizer"):
                if phase == "producer":
                    task = owner.task
                else:
                    # Native async-for does not close a suspended source after a
                    # transport error. Task exit alone is insufficient here.
                    source = owner.source if phase == "source finalizer" else owner.input_source
                    task = owner.close_task = asyncio.create_task(source.aclose())
                remaining = max(0, deadline - asyncio.get_running_loop().time())
                done, _ = await asyncio.wait({task}, timeout=remaining)
                if not done:
                    owner.closing = True
                    failure = RuntimeError(f"Input {phase} did not exit for {owner.state.request_id}")
                    _fail_context(client, failure)
                    raise failure
                if phase != "producer" and not task.cancelled():
                    try:
                        task.result()
                    except BaseException as error:
                        _fail_context(client, error)
                        raise

        owner.join_task = asyncio.create_task(join())
    await _await_cleanup(owner.join_task)


def _fail_context(
    client: AsyncOmni,
    failure: BaseException,
) -> None:
    """Poison the client and wake every consumer with a fatal error, retaining unreleased owners.

    Args:
        client: Patched client to poison.
        failure: Cleanup failure reported to consumers.
    """
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
