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

DEFAULT_RESOLUTION_TYPES: List[str] = [
    "refund",
    "replacement",
    "information",
    "escalation",
    "account_change",
]


def _parse_money(value: str) -> float:
    """Parse a money string like '$1,234.56' or '-$49.99' into a float."""
    s = value.strip().replace(",", "")
    negative = s.startswith("-")
    s = s.lstrip("-+").lstrip("$")
    return -float(s) if negative else float(s)


def _format_money(value: float) -> str:
    """Format a float as a money string, preserving sign: '$1,234.56' or '-$49.99'."""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


@register_schema_tool_for_eval
class LookupAccountTool(StandardSchemaTool):
    """Look up a customer account. Account data (including nested orders) lives in shared_state["accounts"]."""

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
    """Check the status of a customer order. Orders are nested under accounts in shared_state."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or "Check the status of a customer order by account ID and order ID.")
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID the order belongs to.",
            },
            "order_id": {
                "type": "string",
                "description": "The order ID to check.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id", "order_id"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        order_id = params.arguments.get("order_id")
        accounts = self.state.get("accounts", {})
        logger.debug(f"CheckOrderStatusTool looking up {account_id}/{order_id}")
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return
        orders = accounts[account_id].get("orders", {})
        if order_id in orders:
            await params.result_callback(orders[order_id])
        else:
            await params.result_callback({"error": f"Order '{order_id}' not found for account '{account_id}'."})


@register_schema_tool_for_eval
class ProcessRefundTool(StandardSchemaTool):
    """Issue a refund on a customer account: append a negative charge entry and decrement balance."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(
            description=description
            or (
                "Issue a refund on a customer account. Appends a negative charge entry to "
                "recent_charges and reduces the account balance by the refund amount."
            ),
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID to refund.",
            },
            "amount": {
                "type": "number",
                "description": "The refund amount in dollars, for example 49.99.",
            },
            "description": {
                "type": "string",
                "description": "A short description of what is being refunded, for example 'Extended Warranty'.",
            },
            "date": {
                "type": "string",
                "description": "The refund date in YYYY-MM-DD format.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id", "amount", "description", "date"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        amount = float(params.arguments.get("amount"))
        desc = params.arguments.get("description")
        date = params.arguments.get("date")
        accounts = self.state.get("accounts", {})
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return

        account = accounts[account_id]
        charges = account.setdefault("recent_charges", [])
        refund_entry = {
            "description": f"Refund - {desc}",
            "amount": _format_money(-amount),
            "date": date,
        }
        charges.append(refund_entry)

        current_balance = _parse_money(account.get("balance", "$0.00"))
        account["balance"] = _format_money(current_balance - amount)

        logger.debug(f"ProcessRefundTool: refunded ${amount:.2f} to {account_id}")
        await params.result_callback(
            {
                "success": True,
                "account_id": account_id,
                "refund_entry": refund_entry,
                "new_balance": account["balance"],
                "account": account,
            }
        )


@register_schema_tool_for_eval
class StartItemReturnTool(StandardSchemaTool):
    """Initiate a return for a customer order. Sets the order status to 'Return Started'."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(
            description=description
            or (
                "Initiate a return for a customer order. Sets the order status to 'Return Started' "
                "and records the return reason."
            ),
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID the order belongs to.",
            },
            "order_id": {
                "type": "string",
                "description": "The order ID to start a return for.",
            },
            "reason": {
                "type": "string",
                "description": "A short reason for the return, for example 'defective' or 'wrong item'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id", "order_id", "reason"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        order_id = params.arguments.get("order_id")
        reason = params.arguments.get("reason")
        accounts = self.state.get("accounts", {})
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return
        orders = accounts[account_id].get("orders", {})
        if order_id not in orders:
            await params.result_callback({"error": f"Order '{order_id}' not found for account '{account_id}'."})
            return
        orders[order_id]["status"] = "Return Started"
        orders[order_id]["return_reason"] = reason
        logger.debug(f"StartItemReturnTool: return started for {account_id}/{order_id}")
        await params.result_callback(
            {
                "success": True,
                "account_id": account_id,
                "order_id": order_id,
                "order": orders[order_id],
            }
        )


@register_schema_tool_for_eval
class ChangePlanTool(StandardSchemaTool):
    """Change an account's subscription plan. Plan-to-rate mapping is injected at construction."""

    def __init__(
        self,
        *,
        shared_state: Optional[dict] = None,
        plans: str = "{}",
        description: Optional[str] = None,
    ):
        super().__init__(
            description=description
            or (
                "Change a customer's subscription plan. Updates both the plan name and the "
                "monthly rate based on the plan table."
            ),
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})
        self.plans = json.loads(plans) if isinstance(plans, str) else plans

    @property
    def properties(self) -> Dict[str, Any]:
        plan_names = ", ".join(self.plans.keys()) if self.plans else ""
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID to modify.",
            },
            "new_plan": {
                "type": "string",
                "description": (
                    f"The new plan name. Available plans: {plan_names}." if plan_names else "The new plan name."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id", "new_plan"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        new_plan = params.arguments.get("new_plan")
        accounts = self.state.get("accounts", {})
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return
        if new_plan not in self.plans:
            await params.result_callback(
                {"error": f"Plan '{new_plan}' is not available. Available plans: {list(self.plans.keys())}."}
            )
            return
        account = accounts[account_id]
        account["plan"] = new_plan
        account["monthly_rate"] = self.plans[new_plan]
        logger.debug(f"ChangePlanTool: {account_id} -> {new_plan} @ {self.plans[new_plan]}")
        await params.result_callback(
            {
                "success": True,
                "account_id": account_id,
                "new_plan": new_plan,
                "new_monthly_rate": self.plans[new_plan],
                "account": account,
            }
        )


@register_schema_tool_for_eval
class UnlockAccountTool(StandardSchemaTool):
    """Unlock a locked account: set account_status to 'Active' and failed_login_attempts to '0'."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(
            description=description
            or (
                "Unlock a customer account that has been locked due to failed login attempts. "
                "Resets account_status to 'Active' and failed_login_attempts to '0'."
            ),
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID to unlock.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        accounts = self.state.get("accounts", {})
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return
        account = accounts[account_id]
        account["account_status"] = "Active"
        account["failed_login_attempts"] = "0"
        logger.debug(f"UnlockAccountTool: unlocked {account_id}")
        await params.result_callback(
            {
                "success": True,
                "account_id": account_id,
                "account": account,
            }
        )


@register_schema_tool_for_eval
class CancelSubscriptionTool(StandardSchemaTool):
    """Cancel an account's subscription: set plan to 'Canceled' and monthly_rate to '$0.00'."""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(
            description=description
            or (
                "Cancel a customer's subscription. Sets plan to 'Canceled' and monthly_rate to "
                "'$0.00'. Service remains active until the current billing cycle ends."
            ),
        )
        self.state = shared_state if shared_state is not None else {}
        self.state.setdefault("accounts", {})

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "account_id": {
                "type": "string",
                "description": "The customer account ID to cancel.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["account_id"]

    async def _execute(self, params: FunctionCallParams) -> None:
        account_id = params.arguments.get("account_id")
        accounts = self.state.get("accounts", {})
        if account_id not in accounts:
            await params.result_callback({"error": f"Account '{account_id}' not found."})
            return
        account = accounts[account_id]
        account["plan"] = "Canceled"
        account["monthly_rate"] = "$0.00"
        logger.debug(f"CancelSubscriptionTool: canceled {account_id}")
        await params.result_callback(
            {
                "success": True,
                "account_id": account_id,
                "account": account,
            }
        )


@register_schema_tool_for_eval
class ResolveTicketTool(SendScenarioSummaryTool):
    """Resolve a customer service ticket. Sends resolution + latest account snapshot to evaluator."""

    def __init__(
        self,
        *,
        rtvi: Optional[RTVIProcessor] = None,
        shared_state: Optional[dict] = None,
        resolution_types: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            description=description
            or (
                "Resolve the customer's issue and log the resolution. "
                "Use this after the issue has been fully resolved."
            ),
            rtvi=rtvi,
        )
        self.state = shared_state if shared_state is not None else {}
        self.resolution_types = list(resolution_types) if resolution_types else list(DEFAULT_RESOLUTION_TYPES)

    @property
    def properties(self) -> Dict[str, Any]:
        allowed = ", ".join(self.resolution_types)
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
                "enum": self.resolution_types,
                "description": f"Type of resolution applied. Must be exactly one of: {allowed}.",
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
