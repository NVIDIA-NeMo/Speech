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
"""Exercise actual pinned generate/abort with controlled engine transport; no model."""
from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Awaitable, Iterable, NoReturn

import pytest
import pytest_asyncio
from easymagpie_vllm_omni.producer_lifecycle import patch_producer_lifecycle
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni

if TYPE_CHECKING:
    from vllm_omni.entrypoints.client_request_state import ClientRequestState


class Client:
    """Drive pinned AsyncOmni generate/abort over a recording in-memory engine transport."""

    def __init__(self) -> None:
        """Build the native AsyncOmni stand-in whose engine calls are recorded in `events`."""
        self.events = []
        self.first = asyncio.Event()
        self.releases = []
        self.tasks = []
        self.generators = []
        self.producers = []
        self.params = SamplingParams(temperature=0, max_tokens=4, output_kind=RequestOutputKind.DELTA)
        self.native = SimpleNamespace(
            engine=SimpleNamespace(
                num_stages=1,
                add_request_async=self.add,
                add_streaming_update_async=self.update,
                abort_async=self.abort,
            ),
            model_config=None,
            num_stages=1,
            request_states={},
            log_stats=False,
            prom_metrics=SimpleNamespace(
                request_failed=lambda: None, set_running=lambda v: None, set_waiting=lambda v: None
            ),
            _pause_cond=asyncio.Condition(),
            _paused=False,
            _get_unique_request_id=AsyncOmni._get_unique_request_id,
            _final_output_handler=lambda: None,
            _maybe_expand_sampling_params=lambda p: p,
            resolve_sampling_params_list=lambda p, **kw: p,
            _compute_final_stage_id=lambda m: 0,
            _compute_final_output_stage_ids=lambda m: [0],
            _get_pd_separation_pair=lambda: None,
            _resolve_transfer_replica=lambda *a: None,
            _validate_streaming_input_sampling_params=AsyncOmni._validate_streaming_input_sampling_params,
            _check_engine_output_error=lambda *a: None,
            _process_single_result=lambda message, *a: message.engine_outputs,
        )
        for name in (
            "_add_streaming_input_request",
            "_process_orchestrator_results",
            "_abort_internal_requests",
            "_abort",
            "_fire_failure_counter_if_alive",
            "_log_summary_and_cleanup",
            "_raise_nonfatal_error_message",
        ):
            setattr(self.native, name, MethodType(getattr(AsyncOmni, name), self.native))

    async def add(self, **kw: Any) -> None:
        """Record an engine add-request call and signal the first submission.

        Args:
            **kw: Keyword arguments of the native add-request call.
        """
        self.events.append(("add", kw))
        self.first.set()

    async def update(self, **kw: Any) -> None:
        """Record an engine streaming-update call.

        Args:
            **kw: Keyword arguments of the native streaming-update call.
        """
        self.events.append(("update", kw))

    async def abort(self, ids: Iterable[str]) -> None:
        """Record an engine abort call.

        Args:
            ids: Physical request identities to abort.
        """
        self.events.append(("abort", list(ids)))

    def event(self) -> asyncio.Event:
        """Create a release event that fixture teardown sets.

        Returns:
            A new unset event tracked for teardown.
        """
        event = asyncio.Event()
        self.releases.append(event)
        return event

    async def start(
        self, inputs: AsyncGenerator[StreamingInput, None]
    ) -> tuple[AsyncGenerator[Any, None], asyncio.Task[Any], ClientRequestState]:
        """Start a streaming generation and wait until the engine receives its first request.

        Args:
            inputs: Streaming input source of the generation.

        Returns:
            The public output generator, the task awaiting its first output, and the
            request state of the new physical generation.
        """
        self.first.clear()
        generator = AsyncOmni.generate(self.native, inputs, request_id="external", sampling_params_list=[self.params])
        self.generators.append(generator)
        task = asyncio.create_task(anext(generator))
        self.tasks.append(task)
        await asyncio.wait_for(self.first.wait(), 2)
        state = next(reversed(self.native.request_states.values()))
        self.producers.append(state.input_stream_task)
        return generator, task, state

    async def output(
        self,
        state: ClientRequestState,
        finished: bool = True,
    ) -> None:
        """Queue an engine output message for a request.

        Args:
            state: Request state whose output queue receives the message.
            finished: Whether the output marks the request as finished.
        """
        await state.queue.put(
            OutputMessage(
                request_id=state.request_id,
                stage_id=0,
                engine_outputs=SimpleNamespace(request_id=state.request_id),
                finished=finished,
            )
        )

    def spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        """Schedule an awaitable as a task that fixture teardown cancels.

        Args:
            coro: Awaitable to run.

        Returns:
            The created task.
        """
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task


@pytest_asyncio.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Client]:
    """Provide a client over the patched producer lifecycle and tear down everything it started.

    Args:
        monkeypatch: Fixture used to stub native prompt extraction.

    Yields:
        The controlled client.
    """
    patch_producer_lifecycle()
    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.extract_prompt_components", lambda *a: (None, None, None))
    client = Client()
    yield client
    for event in client.releases:
        event.set()
    for task in client.tasks + client.producers:
        if not task.done():
            task.cancel()
    await asyncio.gather(*client.tasks, *client.producers, return_exceptions=True)
    for generator in client.generators:
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["cancel", "close", "error", "abort", "finish"])
async def test_cleanup_joins_slow_finalizer_before_releasing_generation(
    client: Client,
    trigger: str,
) -> None:
    """Verify each cleanup path waits for a slow input finalizer before releasing the generation.

    Args:
        client: Controlled AsyncOmni client.
        trigger: Cleanup path: cancel, close, error, abort, or finish.
    """
    entered, release = asyncio.Event(), client.event()

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, block, and hold the finalizer until released.

        Yields:
            The first streaming input.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            await asyncio.Event().wait()
        finally:
            entered.set()
            await release.wait()

    generator, task, state = await client.start(inputs())
    if trigger == "cancel":
        task.cancel()
        cleanup = task
    elif trigger == "close":
        await client.output(state, finished=False)
        await task
        cleanup = client.spawn(generator.aclose())
    elif trigger == "error":
        await state.queue.put(ErrorMessage(request_id=state.request_id, error="controlled output failure", fatal=True))
        cleanup = task
    elif trigger == "abort":
        cleanup = client.spawn(AsyncOmni.abort(client.native, "external"))
    else:
        await client.output(state)
        await task
        cleanup = client.spawn(anext(generator))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.sleep(0)
    assert not cleanup.done(), "cleanup returned before the input finalizer"
    assert client.native.request_states[state.request_id] is state
    release.set()
    await asyncio.wait_for(asyncio.wait({cleanup}), 2)
    assert cleanup.done()
    if trigger == "cancel":
        assert cleanup.cancelled()
    elif trigger == "error":
        assert "controlled output failure" in str(cleanup.exception())
    elif trigger == "finish":
        assert isinstance(cleanup.exception(), StopAsyncIteration)
    else:
        assert cleanup.exception() is None
    assert state.input_stream_task.done()
    assert state.request_id not in client.native.request_states
    assert all(kind != "update" for kind, _ in client.events)
    if trigger == "abort":
        await asyncio.wait_for(asyncio.wait({task}), 2)
        assert task.done(), "public abort left the output consumer blocked"
        assert task.exception() is not None
        assert not isinstance(task.exception(), EngineDeadError), "public abort is a request-scoped failure"


@pytest.mark.asyncio
@pytest.mark.parametrize("yield_again", [False, True])
async def test_suppressed_cancellation_cannot_send_after_abort(
    client: Client,
    yield_again: bool,
) -> None:
    """Verify a source that suppresses cancellation cannot send updates after abort.

    Args:
        client: Controlled AsyncOmni client.
        yield_again: Whether the source yields another input after suppressing cancellation.
    """
    entered, release = asyncio.Event(), client.event()

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, then suppress cancellation until released and optionally yield again.

        Yields:
            Streaming inputs of the generation.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            entered.set()
            await release.wait()
            if yield_again:
                yield StreamingInput(prompt={"prompt_token_ids": [8]}, sampling_params=client.params)

    _, task, state = await client.start(inputs())
    abort = client.spawn(AsyncOmni.abort(client.native, "external"))
    await asyncio.wait_for(entered.wait(), 2)
    assert not abort.done()
    assert state.request_id in client.native.request_states
    release.set()
    await asyncio.wait_for(abort, 2)
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done()
    assert task.exception() is not None
    assert [kind for kind, _ in client.events] == ["add", "abort"]


@pytest.mark.asyncio
async def test_repeated_cancellation_and_concurrent_abort_share_retirement(client: Client) -> None:
    """Verify repeated cancellation and a concurrent abort share one retirement without interrupting the finalizer.

    Args:
        client: Controlled AsyncOmni client.
    """
    entered, release = asyncio.Event(), client.event()
    interrupted = False

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, block, and record whether the held finalizer is interrupted.

        Yields:
            The first streaming input.
        """
        nonlocal interrupted
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            await asyncio.Event().wait()
        finally:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted = True
                raise

    _, task, state = await client.start(inputs())
    task.cancel()
    await asyncio.wait_for(entered.wait(), 2)
    abort = client.spawn(AsyncOmni.abort(client.native, "external"))
    await asyncio.sleep(0)
    task.cancel()
    abort.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not abort.done()
    assert not interrupted
    assert client.native.request_states[state.request_id] is state
    release.set()
    await asyncio.wait_for(asyncio.wait({task, abort}), 2)
    assert task.cancelled() and abort.cancelled()
    assert not interrupted
    assert [kind for kind, _ in client.events] == ["add", "abort"]
    assert not client.native.request_states


@pytest.mark.asyncio
async def test_completed_source_waits_for_terminal_transport(client: Client) -> None:
    """Verify completion waits until the terminal non-resumable update has been sent.

    Args:
        client: Controlled AsyncOmni client.
    """
    entered, release = asyncio.Event(), client.event()

    async def update(**kw: Any) -> None:
        """Hold the terminal non-resumable update until released, then record the update.

        Args:
            **kw: Keyword arguments of the native streaming-update call.
        """
        if not kw["resumable"]:
            entered.set()
            await release.wait()
        await client.update(**kw)

    client.native.engine.add_streaming_update_async = update

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield a single streaming input.

        Yields:
            The only streaming input.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)

    generator, task, state = await client.start(inputs())
    await asyncio.wait_for(entered.wait(), 2)
    await client.output(state)
    await task
    completion = client.spawn(anext(generator))
    await asyncio.sleep(0)
    assert not completion.done()
    assert state.request_id in client.native.request_states
    release.set()
    await asyncio.wait_for(asyncio.wait({completion}), 2)
    assert isinstance(completion.exception(), StopAsyncIteration)
    assert state.input_stream_task.done()
    assert not client.native.request_states
    assert [kind for kind, _ in client.events] == ["add", "update"]
    assert not client.events[-1][1]["resumable"]


@pytest.mark.asyncio
async def test_terminal_transport_failure_wakes_consumer(client: Client) -> None:
    """Verify a terminal update failure fails the waiting output consumer and releases the request.

    Args:
        client: Controlled AsyncOmni client.
    """

    async def update(**kw: Any) -> NoReturn:
        """Fail every streaming update.

        Args:
            **kw: Keyword arguments of the native streaming-update call.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("controlled terminal failure")

    client.native.engine.add_streaming_update_async = update

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield a single streaming input.

        Yields:
            The only streaming input.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)

    _, task, state = await client.start(inputs())
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done(), "terminal transport failure left output consumer blocked"
    assert "controlled terminal failure" in str(task.exception())
    assert state.input_stream_task.done()
    assert not client.native.request_states


@pytest.mark.asyncio
async def test_cleanup_timeout_retains_owner_and_rejects_all_new_generation(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a cleanup timeout keeps the producer owner and request state and rejects any new generation.

    Args:
        client: Controlled AsyncOmni client.
        monkeypatch: Fixture used to shorten the producer cleanup timeout.
    """
    monkeypatch.setattr("easymagpie_vllm_omni.producer_lifecycle._PRODUCER_CLEANUP_TIMEOUT", 0.03)
    entered, release = asyncio.Event(), client.event()

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, block, and hold the finalizer until released.

        Yields:
            The first streaming input.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            await asyncio.Event().wait()
        finally:
            entered.set()
            await release.wait()

    _, task, state = await client.start(inputs())
    abort = client.spawn(AsyncOmni.abort(client.native, "external"))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(asyncio.wait({abort, task}), 2)
    assert abort.done() and task.done()
    assert "did not exit" in str(abort.exception())
    assert task.exception() is not None
    assert not state.input_stream_task.done()
    assert client.native.request_states[state.request_id] is state
    assert client.native._easymagpie_producer_owners[state.request_id].task is state.input_stream_task
    with pytest.raises(RuntimeError, match="replace the owning"):
        await anext(
            AsyncOmni.generate(
                client.native, {"prompt_token_ids": [8]}, request_id="fresh", sampling_params_list=[client.params]
            )
        )
    release.set()
    await asyncio.wait_for(state.input_stream_task, 2)
    assert client.native.request_states[state.request_id] is state
    assert [kind for kind, _ in client.events] == ["add"]


@pytest.mark.asyncio
async def test_finalizer_error_cannot_send_terminal_during_abort(client: Client) -> None:
    """Verify a failing source finalizer during abort surfaces its error without sending a terminal update.

    Args:
        client: Controlled AsyncOmni client.
    """

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, block, and raise from the finalizer.

        Yields:
            The first streaming input.

        Raises:
            ValueError: When the source is closed.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            await asyncio.Event().wait()
        finally:
            raise ValueError("controlled finalizer failure")

    _, task, state = await client.start(inputs())
    abort = client.spawn(AsyncOmni.abort(client.native, "external"))
    await asyncio.wait_for(asyncio.wait({abort, task}), 2)
    assert abort.done() and task.done()
    # A finalizer failure must be observable, without issuing an old sentinel.
    assert any("controlled finalizer failure" in str(t.exception()) for t in (abort, task))
    assert state.input_stream_task.done()
    assert [kind for kind, _ in client.events] == ["add", "abort"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_abort_transport_failure_retains_generation_and_poisoned_context(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
) -> None:
    """Verify a failed or timed-out abort transport retains the generation and records a producer failure.

    Args:
        client: Controlled AsyncOmni client.
        monkeypatch: Fixture used to shorten the producer cleanup timeout.
        failed: Whether the abort transport raises instead of blocking past the cleanup timeout.
    """
    monkeypatch.setattr("easymagpie_vllm_omni.producer_lifecycle._PRODUCER_CLEANUP_TIMEOUT", 0.03)
    release = client.event()

    async def abort(ids: Iterable[str]) -> None:
        """Fail, or hold the abort until released and then record it.

        Args:
            ids: Physical request identities to abort.

        Raises:
            RuntimeError: If `failed` is set.
        """
        if failed:
            raise RuntimeError("controlled abort transport failure")
        await release.wait()
        await client.abort(ids)

    client.native.engine.abort_async = abort

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, then block indefinitely.

        Yields:
            The first streaming input.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
        await asyncio.Event().wait()

    _, task, state = await client.start(inputs())
    abort_task = client.spawn(AsyncOmni.abort(client.native, "external"))
    await asyncio.wait_for(asyncio.wait({abort_task, task}), 2)
    assert abort_task.done() and task.done()
    assert abort_task.exception() is not None and task.exception() is not None
    assert state.input_stream_task.done()
    assert client.native.request_states[state.request_id] is state
    assert client.native._easymagpie_producer_failure is not None
    owner = client.native._easymagpie_producer_owners[state.request_id]
    release.set()
    await asyncio.gather(owner.transport_task, return_exceptions=True)
    assert client.native.request_states[state.request_id] is state


@pytest.mark.asyncio
async def test_successful_exhaustion_and_repeated_public_id_keep_distinct_generations(client: Client) -> None:
    """Verify exhausted sources finish cleanly and a reused public id gets a distinct physical generation.

    Args:
        client: Controlled AsyncOmni client.
    """
    physical = []
    for _ in range(2):

        async def inputs() -> AsyncGenerator[StreamingInput, None]:
            """Yield two streaming inputs.

            Yields:
                Streaming inputs of the generation.
            """
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            yield StreamingInput(prompt={"prompt_token_ids": [8]}, sampling_params=client.params)

        generator, task, state = await client.start(inputs())
        physical.append(state.request_id)
        await state.input_stream_task
        await client.output(state)
        await task
        with pytest.raises(StopAsyncIteration):
            await anext(generator)
        assert not client.native.request_states
        assert not client.native._easymagpie_producer_owners
    assert physical[0] != physical[1]
    assert [kind for kind, _ in client.events] == ["add", "update", "update"] * 2
    assert [payload["resumable"] for _, payload in client.events] == [True, True, False] * 2


@pytest.mark.asyncio
async def test_aborting_one_generation_leaves_another_generation_live(client: Client) -> None:
    """Verify aborting one physical generation leaves a concurrent generation running to completion.

    Args:
        client: Controlled AsyncOmni client.
    """

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, then block indefinitely.

        Yields:
            The first streaming input.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
        await asyncio.Event().wait()

    _, first_task, first = await client.start(inputs())
    generator, second_task, second = await client.start(inputs())
    await AsyncOmni._abort_internal_requests(client.native, first.request_id)
    await asyncio.wait_for(asyncio.wait({first_task}), 2)
    assert first_task.done() and first_task.exception() is not None
    assert second.request_id in client.native.request_states
    assert not second_task.done()
    assert not second.input_stream_task.done()
    await client.output(second)
    await second_task
    with pytest.raises(StopAsyncIteration):
        await anext(generator)
    assert not client.native.request_states
    assert not client.native._easymagpie_producer_owners


@pytest.mark.asyncio
async def test_source_error_propagates_and_joins_native_producer(client: Client) -> None:
    """Verify a source error reaches the output consumer after the native producer has exited.

    Args:
        client: Controlled AsyncOmni client.
    """

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, then fail.

        Yields:
            The first streaming input.

        Raises:
            ValueError: After the first input.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
        raise ValueError("controlled source failure")

    _, task, state = await client.start(inputs())
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done()
    assert "controlled source failure" in str(task.exception())
    assert state.input_stream_task.done()
    assert not client.native.request_states
    assert not client.native._easymagpie_producer_owners


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["source", "finalizer"])
async def test_input_failure_aborts_without_sending_normal_final_sentinel(
    client: Client,
    failure_phase: str,
) -> None:
    """Verify an input failure aborts the request without sending the normal final update.

    Args:
        client: Controlled AsyncOmni client.
        failure_phase: Whether the source body ("source") or its finalizer ("finalizer") raises.
    """
    release = client.event()

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, wait for release, then raise from the selected phase.

        Yields:
            The first streaming input.

        Raises:
            ValueError: From the source body or its finalizer, as selected by `failure_phase`.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            await release.wait()
            if failure_phase == "source":
                raise ValueError("controlled input failure")
        finally:
            if failure_phase == "finalizer":
                raise ValueError("controlled input failure")

    _, task, state = await client.start(inputs())
    release.set()
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done()
    assert "controlled input failure" in str(task.exception())
    assert not isinstance(task.exception(), EngineDeadError), "input errors must remain request-scoped"
    assert state.input_stream_task.done()
    assert not client.native.request_states
    assert not client.native._easymagpie_producer_owners
    # A normal final sentinel races the error-triggered abort in the orchestrator
    # and can reach the scheduler after this physical generation starts retiring.
    assert [kind for kind, _ in client.events] == ["add", "abort"]


def test_patch_is_idempotent_and_plugin_installs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify repatching leaves AsyncOmni.generate unchanged and plugin registration installs the patch.

    Args:
        monkeypatch: Fixture used to replace the patch entry point with a recorder.
    """
    from vllm_plugin_easymagpie_omni import _register_streaming_compatibility

    original = AsyncOmni.generate
    patch_producer_lifecycle()
    assert AsyncOmni.generate is original
    calls = []
    monkeypatch.setattr("easymagpie_vllm_omni.producer_lifecycle.patch_producer_lifecycle", lambda: calls.append(True))
    _register_streaming_compatibility()
    assert calls == [True]


@pytest.mark.asyncio
async def test_complete_text_request_preserves_native_completion(client: Client) -> None:
    """Verify a complete prompt finishes through the native path without streaming updates.

    Args:
        client: Controlled AsyncOmni client.
    """
    generator = AsyncOmni.generate(
        client.native, {"prompt_token_ids": [9]}, request_id="complete", sampling_params_list=[client.params]
    )
    client.generators.append(generator)
    task = client.spawn(anext(generator))
    await asyncio.wait_for(client.first.wait(), 2)
    state = next(iter(client.native.request_states.values()))
    await client.output(state)
    assert (await task).request_id == "complete"
    with pytest.raises(StopAsyncIteration):
        await anext(generator)
    assert not client.native.request_states
    assert not client.native._easymagpie_producer_owners
    assert [kind for kind, _ in client.events] == ["add"]


@pytest.mark.asyncio
async def test_readding_owned_physical_generation_is_rejected_without_stopping_owner(client: Client) -> None:
    """Verify re-adding a physical request id that is still owned is rejected without stopping the owner.

    Args:
        client: Controlled AsyncOmni client.
    """

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield one input, then block indefinitely.

        Yields:
            The first streaming input.
        """
        yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
        await asyncio.Event().wait()

    _, task, state = await client.start(inputs())
    fresh = inputs()
    try:
        with pytest.raises(RuntimeError, match="still owns physical request"):
            await AsyncOmni._add_streaming_input_request(
                client.native,
                request_id=state.request_id,
                input_stream=fresh,
                sampling_params_list=[client.params],
                final_stage_id=0,
                final_output_stage_ids=[0],
                arrival_time=0,
            )
    finally:
        await fresh.aclose()
    assert not state.input_stream_task.done()
    assert not task.done()
    await AsyncOmni.abort(client.native, "external")
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done() and task.exception() is not None


@pytest.mark.asyncio
async def test_chunk_transport_error_joins_source_suspended_at_yield(client: Client) -> None:
    """Verify a chunk update failure waits for the source suspended at a yield to finalize.

    Args:
        client: Controlled AsyncOmni client.
    """
    entered, release = asyncio.Event(), client.event()

    async def update(**kw: Any) -> None:
        """Fail resumable chunk updates and record terminal updates.

        Args:
            **kw: Keyword arguments of the native streaming-update call.

        Raises:
            RuntimeError: For a resumable update.
        """
        if kw["resumable"]:
            raise RuntimeError("controlled chunk transport failure")
        await client.update(**kw)

    client.native.engine.add_streaming_update_async = update

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield two inputs and hold the finalizer until released.

        Yields:
            Streaming inputs of the generation.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            yield StreamingInput(prompt={"prompt_token_ids": [8]}, sampling_params=client.params)
        finally:
            entered.set()
            await release.wait()

    _, task, state = await client.start(inputs())
    await asyncio.wait_for(entered.wait(), 2)
    assert not task.done(), "transport failure detached a suspended source finalizer"
    assert state.request_id in client.native.request_states
    release.set()
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done()
    assert "controlled chunk transport failure" in str(task.exception())
    assert not client.native.request_states
    assert not client.native._easymagpie_producer_owners


@pytest.mark.asyncio
async def test_cancel_before_native_producer_start_closes_primed_source(client: Client) -> None:
    """Verify abort closes an already-started source when the producer is cancelled before its first run.

    Args:
        client: Controlled AsyncOmni client.
    """
    entered, release = asyncio.Event(), client.event()

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield two inputs and hold the finalizer until released.

        Yields:
            Streaming inputs of the generation.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            yield StreamingInput(prompt={"prompt_token_ids": [8]}, sampling_params=client.params)
        finally:
            entered.set()
            await release.wait()

    source = inputs()
    await anext(source)
    state = SimpleNamespace(request_id="primed", external_request_id="external", queue=asyncio.Queue())
    client.native.request_states[state.request_id] = state
    try:
        producer = await AsyncOmni._add_streaming_input_request(
            client.native,
            request_id=state.request_id,
            input_stream=source,
            sampling_params_list=[client.params],
            final_stage_id=0,
            final_output_stage_ids=[0],
            arrival_time=0,
        )
        client.producers.append(producer)
        producer.cancel()  # No event-loop turn since task creation.
        abort = client.spawn(AsyncOmni.abort(client.native, "external"))
        await asyncio.wait_for(entered.wait(), 2)
        assert not abort.done()
        assert client.native.request_states[state.request_id] is state
        release.set()
        await asyncio.wait_for(abort, 2)
        assert producer.done()
        assert not client.native.request_states
    finally:
        release.set()
        await source.aclose()


@pytest.mark.asyncio
async def test_source_close_timeout_retains_ownership_after_native_task_exit(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a source close timeout keeps ownership and request state after the native task exits.

    Args:
        client: Controlled AsyncOmni client.
        monkeypatch: Fixture used to shorten the producer cleanup timeout.
    """
    monkeypatch.setattr("easymagpie_vllm_omni.producer_lifecycle._PRODUCER_CLEANUP_TIMEOUT", 0.03)
    entered, release = asyncio.Event(), client.event()

    async def update(**kw: Any) -> None:
        """Fail resumable chunk updates and record terminal updates.

        Args:
            **kw: Keyword arguments of the native streaming-update call.

        Raises:
            RuntimeError: For a resumable update.
        """
        if kw["resumable"]:
            raise RuntimeError("controlled chunk failure before close timeout")
        await client.update(**kw)

    client.native.engine.add_streaming_update_async = update

    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        """Yield two inputs and hold the finalizer until released.

        Yields:
            Streaming inputs of the generation.
        """
        try:
            yield StreamingInput(prompt={"prompt_token_ids": [7]}, sampling_params=client.params)
            yield StreamingInput(prompt={"prompt_token_ids": [8]}, sampling_params=client.params)
        finally:
            entered.set()
            await release.wait()

    _, task, state = await client.start(inputs())
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(asyncio.wait({task}), 2)
    assert task.done()
    assert "source finalizer did not exit" in str(task.exception())
    assert state.input_stream_task.done()
    owner = client.native._easymagpie_producer_owners[state.request_id]
    assert not owner.close_task.done()
    assert client.native.request_states[state.request_id] is state
    assert client.native._easymagpie_producer_failure is not None
    release.set()
    await asyncio.gather(owner.close_task, return_exceptions=True)
    assert client.native.request_states[state.request_id] is state
    assert all(kind != "abort" for kind, _ in client.events)
