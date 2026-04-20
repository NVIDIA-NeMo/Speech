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

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.services.llm_service import FunctionCallParams

from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.evaluation.tools.rtvi_control import SendScenarioSummaryTool
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool


@register_schema_tool_for_eval
class JoinWaitListTool(SendScenarioSummaryTool):
    """Add a customer to the restaurant waitlist. Sends updated waitlist to evaluator via <final_response> tags."""

    def __init__(
        self,
        *,
        rtvi: Optional[RTVIProcessor] = None,
        shared_state: Optional[dict] = None,
        initial_waitlist: str = "[]",
        description: Optional[str] = None,
    ):
        super().__init__(
            description=description
            or "Add a customer to the restaurant waitlist. Requires the customer's name, phone number, and party size.",
            rtvi=rtvi,
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("waitlist", json.loads(initial_waitlist))

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "name": {
                "type": "string",
                "description": "The customer's name.",
            },
            "phone": {
                "type": "string",
                "description": "The customer's phone number.",
            },
            "party_size": {
                "type": "integer",
                "description": "The number of people in the party.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["name", "phone", "party_size"]

    async def _execute(self, params: FunctionCallParams) -> None:
        name = params.arguments.get("name")
        phone = params.arguments.get("phone")
        party_size = params.arguments.get("party_size")
        entry = {"name": name, "phone": phone, "party_size": party_size}
        self.state["waitlist"].append(entry)
        position = len(self.state["waitlist"])
        logger.debug(f"JoinWaitListTool: {name} ({phone}) added at position {position}")

        # Send updated waitlist to evaluator
        summary = json.dumps({"waitlist": self.state["waitlist"], "action": "join", "customer": entry})
        await self.send_scenario_summary(summary)

        await params.result_callback(
            {
                "success": True,
                "message": f"{name} has been added to the waitlist.",
                "position": position,
                "total_in_waitlist": position,
            }
        )


@register_schema_tool_for_eval
class DropWaitListTool(SendScenarioSummaryTool):
    """Remove a customer from the restaurant waitlist. Sends updated waitlist to evaluator via <final_response> tags."""

    def __init__(
        self,
        *,
        rtvi: Optional[RTVIProcessor] = None,
        shared_state: Optional[dict] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            description=description
            or "Remove a customer from the restaurant waitlist by name. Use this when a customer decides to leave the waitlist.",
            rtvi=rtvi,
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("waitlist", [])

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "name": {
                "type": "string",
                "description": "The name of the customer to remove from the waitlist.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["name"]

    async def _execute(self, params: FunctionCallParams) -> None:
        name = params.arguments.get("name")
        waitlist = self.state["waitlist"]
        original_len = len(waitlist)
        self.state["waitlist"] = [entry for entry in waitlist if entry.get("name") != name]
        removed = len(self.state["waitlist"]) < original_len
        logger.debug(f"DropWaitListTool: {name} {'removed' if removed else 'not found'}")

        # Send updated waitlist to evaluator
        summary = json.dumps(
            {
                "waitlist": self.state["waitlist"],
                "action": "drop",
                "customer": {"name": name},
                "removed": removed,
            }
        )
        await self.send_scenario_summary(summary)

        if removed:
            await params.result_callback(
                {
                    "success": True,
                    "message": f"{name} has been removed from the waitlist.",
                    "remaining_in_waitlist": len(self.state["waitlist"]),
                }
            )
        else:
            await params.result_callback(
                {
                    "success": False,
                    "message": f"{name} was not found on the waitlist.",
                    "remaining_in_waitlist": len(self.state["waitlist"]),
                }
            )


@register_schema_tool_for_eval
class GetWaitlistTool(StandardSchemaTool):
    """Check the current waitlist status. Reads from shared state."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(
            description=description
            or "Check the current restaurant waitlist to see who is waiting and their position."
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("waitlist", [])

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    async def _execute(self, params: FunctionCallParams) -> None:
        waitlist = self.state.get("waitlist", [])
        logger.debug(f"GetWaitlistTool: returning {len(waitlist)} entries")
        await params.result_callback(
            {
                "waitlist": waitlist,
                "total_in_waitlist": len(waitlist),
            }
        )
