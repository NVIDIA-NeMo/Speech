# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Factory helpers for the common RTVI actions used by voice-agent bots.

The actions are parameterized so the same factory works for bots with different
pipeline shapes: pass in whichever aggregators, services, and handlers the bot
actually has. ``None`` entries in ``resettable_services`` are silently skipped.

The reset and update-prompt actions need to queue an ``EndTaskFrame`` onto a
``PipelineTask`` that is typically created *after* the RTVI processor (because
the task needs ``rtvi`` in its observer list). ``TaskRef`` is a tiny holder the
bot sets after constructing the task.
"""

import copy
import dataclasses
import json
from typing import Any, Callable, List, Optional

from loguru import logger
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIProcessor
from pipecat.services.ai_service import AIService


@dataclasses.dataclass
class TaskRef:
    """Mutable handle to a PipelineTask and its running flag.

    Construct early, hand to RTVI action factories, then populate once the task
    exists. ``running`` is flipped by the bot runner during shutdown so handlers
    can avoid queueing frames onto a dead task.
    """

    task: Optional[PipelineTask] = None
    running: bool = False


@dataclasses.dataclass
class SharedStateRef:
    """Mutable handle to the per-scenario ``shared_state`` dict.

    The same dict that's passed to tool constructors is also published here, so
    other RTVI action handlers (specifically ``get_scenario_summary``) can read
    ``shared_state["actions"]`` and ``shared_state["db"]`` without needing tool
    references. ``state`` is reset (re-pointed at a new dict) every time
    ``update_system_prompt`` runs.
    """

    state: dict = dataclasses.field(default_factory=dict)


async def _maybe_end_task(task_ref: TaskRef) -> None:
    if task_ref.running and task_ref.task is not None:
        await task_ref.task.queue_frames([EndTaskFrame()])


def _reset_services(services: List[AIService]) -> None:
    for service in services:
        if service is not None and hasattr(service, "reset"):
            service.reset()


def create_reset_context_action(
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[AIService],
) -> RTVIAction:
    """Build the ``context.reset`` action.

    ``original_messages`` is captured by reference so the action always resets to
    whatever ``update_system_prompt`` last wrote.
    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> bool:
        logger.info("Resetting conversation context...")
        try:
            await _maybe_end_task(task_ref)
            user_aggregator.reset()
            assistant_aggregator.reset()
            user_aggregator.set_messages(copy.deepcopy(original_messages))
            assistant_aggregator.set_messages(copy.deepcopy(original_messages))
            _reset_services(resettable_services)
            logger.info("Conversation context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting context: {e}")
            return False

    return RTVIAction(
        service="context",
        action="reset",
        result="bool",
        arguments=[],
        handler=handler,
    )


def create_update_system_prompt_action(
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[Any],
    *,
    system_role: str,
    system_prompt_suffix: str,
    enable_tool_calling: bool = False,
    llm=None,
    context=None,
    rtvi: Optional[RTVIProcessor] = None,
    tool_factory: Optional[Callable[..., Any]] = None,
    register_schema_tools: Optional[Callable[..., Any]] = None,
    shared_state_ref: Optional[SharedStateRef] = None,
) -> RTVIAction:
    """Build the ``context.update_system_prompt`` action.

    Tool registration is optional. When ``enable_tool_calling`` is True and a
    ``tools`` JSON string is supplied by the caller, ``tool_factory`` is invoked
    per tool to produce schema tools, then ``register_schema_tools`` swaps them
    onto ``llm`` / ``context``. This keeps the factory decoupled from
    evaluation-specific tool registries.

    The action accepts an optional ``shared_state_init`` argument (JSON string)
    used to initialize the per-scenario ``shared_state`` dict before tools are
    instantiated. The bridge populates it from ``Scenario.setup_shared_state``.
    Two supported shapes (both via ``shared_state_init``):
      - **Inline**: ``{"db": {...full content...}, ...}``. Used as-is.
      - **Path-based fallback**: ``{"db_path": "rel/path.json", ...}``. Resolved
        against ``EVAL_DATA_ROOT`` and replaced under the de-suffixed key
        (``db_path`` → ``db``). Missing files raise ``FileNotFoundError`` loudly.

    If ``shared_state_ref`` is provided, the resolved ``shared_state`` is
    published to it so other action handlers (``get_scenario_summary``) can
    read the same dict. Only consumed when tool calling is enabled.
    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> bool:
        try:
            await _maybe_end_task(task_ref)

            new_prompt = arguments.get("prompt", "")
            new_tools_json = arguments.get("tools", "{}")
            if not new_prompt:
                logger.error("No prompt provided in update_system_prompt action")
                return False

            logger.info(f"Updating system prompt to: {new_prompt[:100]}...")

            if arguments.get("add_suffix", True) and system_prompt_suffix:
                new_prompt = f"{new_prompt}\n{system_prompt_suffix}"

            new_messages = [{"role": system_role, "content": new_prompt}]

            original_messages.clear()
            original_messages.extend(new_messages)

            user_aggregator.reset()
            assistant_aggregator.reset()
            user_aggregator.set_messages(copy.deepcopy(new_messages))
            assistant_aggregator.set_messages(copy.deepcopy(new_messages))

            if (
                enable_tool_calling
                and new_tools_json
                and tool_factory is not None
                and register_schema_tools is not None
            ):
                logger.info("Registering new tools...")
                new_tools = json.loads(new_tools_json)

                # Initialize shared_state from the optional shared_state_init
                # payload produced by Scenario.setup_shared_state(). Inline DB
                # content (state["db"]) is the primary path; path-based loading
                # (state["db_path"]) is a fallback for fixtures too large to
                # ship inline.
                shared_state: dict = json.loads(arguments.get("shared_state_init", "{}"))
                if "db_path" in shared_state:
                    # Lazy import to avoid coupling rtvi_actions to evaluation/.
                    from nemo.agents.voice_agent.evaluation import get_eval_data_root

                    db_path = shared_state.pop("db_path")
                    full_path = get_eval_data_root() / db_path
                    if not full_path.exists():
                        raise FileNotFoundError(
                            f"Scenario DB not found at {full_path} (from db_path={db_path!r}). "
                            f"Check EVAL_DATA_ROOT (currently resolves to {get_eval_data_root()})."
                        )
                    shared_state["db"] = json.loads(full_path.read_text())
                    logger.info(f"Loaded scenario DB from {full_path} into shared_state['db']")

                # Publish the dict so sibling action handlers (e.g. get_scenario_summary)
                # can read the same shared_state without needing tool references.
                if shared_state_ref is not None:
                    shared_state_ref.state = shared_state

                new_schema_tools = [
                    tool_factory(tool_name, rtvi=rtvi, shared_state=shared_state, **tool_args)
                    for tool_name, tool_args in new_tools.items()
                ]
                register_schema_tools(
                    llm=llm,
                    context=context,
                    tools=new_schema_tools,
                    cancel_on_interruption=False,
                    keep_existing_tools=False,
                )
            else:
                logger.info(
                    "Tool calling disabled, no tools provided, or tool_factory not configured; skipping tool registration."
                )

            logger.debug(f"user context tools: {user_aggregator._context.tools}")
            logger.debug(f"assistant context tools: {assistant_aggregator._context.tools}")

            _reset_services(resettable_services)

            logger.info("System prompt updated and context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
            return False

    return RTVIAction(
        service="context",
        action="update_system_prompt",
        result="bool",
        arguments=[
            {"name": "prompt", "type": "string", "required": True},
            {"name": "tools", "type": "string", "required": False, "default": "{}"},
            {"name": "add_suffix", "type": "bool", "required": False, "default": True},
            {"name": "shared_state_init", "type": "string", "required": False, "default": "{}"},
        ],
        handler=handler,
    )


def create_get_context_history_action(
    task_ref: TaskRef,
    assistant_aggregator,
) -> RTVIAction:
    """Build the ``context.get_context_history`` action.

    Returns the assistant aggregator's full message list, stringified to match
    the shape evaluation clients expect.
    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        await _maybe_end_task(task_ref)
        try:
            messages = assistant_aggregator._context.get_messages()
            logger.debug(f"Returning context history: {len(messages)} messages")
            return {"context": str(messages)}
        except Exception as e:
            logger.error(f"Error getting context history: {e}")
            return {"context": []}

    return RTVIAction(
        service="context",
        action="get_context_history",
        result="object",
        arguments=[],
        handler=handler,
    )


def create_get_scenario_summary_action(
    shared_state_ref: SharedStateRef,
) -> RTVIAction:
    """Build the ``context.get_scenario_summary`` action.

    Returns ``{"actions": [...], "db": {...}}`` from the per-scenario shared
    state. Auto-aggregating tools (e.g. ``WriteAirlineTool`` subclasses)
    populate ``shared_state["actions"]`` on each successful mutation; the
    fixture-loading flow populates ``shared_state["db"]``. The bridge calls
    this action after ``<exit>`` (or scenario timeout) to retrieve the final
    artifacts without depending on any LLM-callable summary tool.

    Mirrors how ``get_context_history`` is consumed by the bridge.
    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        try:
            actions = shared_state_ref.state.get("actions", [])
            db = shared_state_ref.state.get("db", {})
            logger.debug(f"Returning scenario summary: {len(actions)} action(s), " f"db has {len(db)} top-level keys")
            return {"actions": actions, "db": db}
        except Exception as e:
            logger.error(f"Error getting scenario summary: {e}")
            return {"actions": [], "db": {}}

    return RTVIAction(
        service="context",
        action="get_scenario_summary",
        result="object",
        arguments=[],
        handler=handler,
    )
