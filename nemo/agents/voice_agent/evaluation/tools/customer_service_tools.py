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
class LookupAccountTool(StandardSchemaTool):
    """Look up a customer account. Account data lives in shared_state["accounts"]."""

    def __init__(
        self, *, shared_state: Optional[dict] = None, accounts: str = "{}", description: Optional[str] = None
    ):
        super().__init__(
            description=description or "Look up a customer account by account ID to retrieve their details."
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", json.loads(accounts) if isinstance(accounts, str) else accounts)

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID to look up.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        accounts = self.state.get("accounts", {})
        logger.debug(f"LookupAccountTool looking up account: {account_id}")
        if account_id in accounts:
            await params.result_callback(accounts[account_id])
        else:
            await params.result_callback({"error": f"Account '{account_id}' not found."})


@register_schema_tool_for_eval
class CheckOrderStatusTool(StandardSchemaTool):
    """Check the status of a customer order. Order data lives in shared_state["orders"]."""

    def __init__(self, *, shared_state: Optional[dict] = None, orders: str = "{}", description: Optional[str] = None):
        super().__init__(description=description or "Check the status of a customer order by order ID.")
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("orders", json.loads(orders) if isinstance(orders, str) else orders)

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {
                "type": "string",
                "description": "The order ID to check.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id"]

    async def _execute(self, params: FunctionCallParams) -> None:
        order_id = params.arguments.get("order_id")
        orders = self.state.get("orders", {})
        logger.debug(f"CheckOrderStatusTool looking up order: {order_id}")
        if order_id in orders:
            await params.result_callback(orders[order_id])
        else:
            await params.result_callback({"error": f"Order '{order_id}' not found."})


@register_schema_tool_for_eval
class ModifyAccountTool(StandardSchemaTool):
    """Modify a customer account field. Updates shared_state["accounts"] so other tools see the change."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(
            description=description
            or "Modify a customer account by updating a specific field. Use this to change plan, status, or other account details.",
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID to modify.",
            },
            "field": {
                "type": "string",
                "description": "The account field to update, for example: plan, account_status, monthly_rate.",
            },
            "value": {
                "type": "string",
                "description": "The new value for the field.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id", "field", "value"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        field = params.arguments.get("field")
        value = params.arguments.get("value")
        accounts = self.state.get("accounts", {})
        logger.debug(f"ModifyAccountTool: setting {account_id}.{field} = {value}")
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return
        old_value = accounts[account_id].get(field)
        accounts[account_id][field] = value
        await params.result_callback(
            {
                "success": True,
                "account_id": account_id,
                "field": field,
                "old_value": old_value,
                "new_value": value,
                "account": accounts[account_id],
            }
        )


@register_schema_tool_for_eval
class ResolveTicketTool(SendScenarioSummaryTool):
    """Resolve a customer service ticket. Sends resolution + latest account state to evaluator via <final_response>."""

    def __init__(
        self,
        *,
        rtvi: Optional[RTVIProcessor] = None,
        shared_state: Optional[dict] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            description=description
            or "Resolve the customer's issue and log the resolution. Use this after the issue has been fully resolved.",
            rtvi=rtvi,
        )
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer's account ID.",
            },
            "issue_summary": {
                "type": "string",
                "description": "Brief summary of the customer's issue.",
            },
            "resolution_type": {
                "type": "string",
                "description": "Type of resolution applied, for example: refund, replacement, information, escalation, or account_change.",
            },
            "resolution_details": {
                "type": "string",
                "description": "Details of how the issue was resolved.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id", "issue_summary", "resolution_type", "resolution_details"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        accounts = self.state.get("accounts", {})
        account_snapshot = accounts.get(account_id, {})

        resolution = {
            "issue_summary": params.arguments.get("issue_summary"),
            "resolution_type": params.arguments.get("resolution_type"),
            "resolution_details": params.arguments.get("resolution_details"),
            "account_id": account_id,
            "account": account_snapshot,
        }
        logger.debug(f"ResolveTicketTool resolving: {resolution}")
        await self.send_scenario_summary(json.dumps(resolution))
        await params.result_callback({"success": True, "message": "Ticket resolved successfully."})
