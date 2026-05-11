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

"""Integration tests for eva_airline Stage A.

Without spinning up an LLM or bot server, exercise:
    1. ``setup_shared_state`` → action-handler-style fixture load → GetReservationTool dispatch
       (success, wrong last name, missing reservation, malformed confirmation).
    2. ``WriteAirlineTool._record_action`` accumulates entries in
       ``shared_state["actions"]`` (the bridge pulls these at end-of-scenario;
       there is no LLM-callable summary tool).
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the repo root to sys.path so we test the working-tree NeMo, not whatever is pip-installed.
nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))

import pytest

from nemo.agents.voice_agent.evaluation import get_eval_data_root
from nemo.agents.voice_agent.evaluation.scenarios import get_eval_scenario
from nemo.agents.voice_agent.evaluation.tools.eva_airline_tools import (
    AIRLINE_ACTION_TYPES,
    GetReservationTool,
    WriteAirlineTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeFunctionCallParams:
    """Minimal stand-in for pipecat's FunctionCallParams for unit testing."""

    def __init__(self, arguments: dict):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, value):
        self.result = value


def _load_fixture_state(eva_id: str) -> dict:
    """Mimic the action handler's db_path → db resolution server-side."""
    path = get_eval_data_root() / "eva_airline_scenarios" / f"{eva_id}.json"
    return {"db": json.loads(path.read_text())}


# ---------------------------------------------------------------------------
# Action-type vocabulary
# ---------------------------------------------------------------------------


def test_action_types_are_one_to_one_with_eva_tool_names():
    """Every AIRLINE_ACTION_TYPE matches an eva tool name verbatim."""
    expected = {
        "rebook_flight",
        "cancel_reservation",
        "process_refund",
        "issue_meal_voucher",
        "issue_hotel_voucher",
        "issue_travel_credit",
        "assign_seat",
        "add_baggage_allowance",
        "add_meal_request",
        "add_to_standby",
        "transfer_to_agent",
    }
    assert set(AIRLINE_ACTION_TYPES) == expected
    # 11 actions = 10 write tools + 1 system tool
    assert len(AIRLINE_ACTION_TYPES) == 11


# ---------------------------------------------------------------------------
# GetReservationTool against the real fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def state_1_1_2():
    """State as the action handler would build it for scenario 1.1.2."""
    return _load_fixture_state("1.1.2")


def test_get_reservation_success(state_1_1_2):
    tool = GetReservationTool(shared_state=state_1_1_2)
    p = _FakeFunctionCallParams({"confirmation_number": "ZK3FFW", "last_name": "Rodriguez"})
    asyncio.run(tool._execute(p))
    assert p.result["status"] == "success"
    res = p.result["reservation"]
    assert res["confirmation_number"] == "ZK3FFW"
    assert res["passengers"][0]["last_name"] == "Rodriguez"


def test_get_reservation_case_insensitive_confirmation(state_1_1_2):
    tool = GetReservationTool(shared_state=state_1_1_2)
    p = _FakeFunctionCallParams({"confirmation_number": "zk3ffw", "last_name": "Rodriguez"})
    asyncio.run(tool._execute(p))
    assert p.result["status"] == "success"


def test_get_reservation_wrong_last_name(state_1_1_2):
    tool = GetReservationTool(shared_state=state_1_1_2)
    p = _FakeFunctionCallParams({"confirmation_number": "ZK3FFW", "last_name": "Smith"})
    asyncio.run(tool._execute(p))
    assert p.result["status"] == "error"
    assert p.result["error_type"] == "authentication_failed"


def test_get_reservation_missing(state_1_1_2):
    tool = GetReservationTool(shared_state=state_1_1_2)
    p = _FakeFunctionCallParams({"confirmation_number": "AAAAAA", "last_name": "Rodriguez"})
    asyncio.run(tool._execute(p))
    assert p.result["status"] == "error"
    assert p.result["error_type"] == "not_found"


def test_get_reservation_malformed_confirmation(state_1_1_2):
    """5-char confirmation fails the Pydantic regex; loud validation error."""
    tool = GetReservationTool(shared_state=state_1_1_2)
    p = _FakeFunctionCallParams({"confirmation_number": "ABC12", "last_name": "Rodriguez"})
    asyncio.run(tool._execute(p))
    assert p.result["status"] == "error"
    assert p.result["error_type"] == "invalid_confirmation_number_format"


def test_get_reservation_db_not_loaded():
    """If shared_state has no db (fixture didn't seed), return a clear error."""
    tool = GetReservationTool(shared_state={})
    p = _FakeFunctionCallParams({"confirmation_number": "ZK3FFW", "last_name": "Rodriguez"})
    asyncio.run(tool._execute(p))
    assert p.result["status"] == "error"
    assert p.result["error_type"] == "db_not_initialized"


# ---------------------------------------------------------------------------
# WriteAirlineTool action recording
# ---------------------------------------------------------------------------


def test_write_airline_tool_records_action():
    """_record_action appends to shared_state['actions']."""

    class _Dummy(WriteAirlineTool):
        def __init__(self, shared_state):
            super().__init__(description="x")
            self.state = shared_state

        @property
        def properties(self):
            return {}

        @property
        def required_properties(self):
            return []

        async def _execute(self, params):
            pass

    state: dict = {}
    tool = _Dummy(state)
    tool._record_action({"action_type": "rebook_flight", "confirmation_number": "ZK3FFW"})
    tool._record_action({"action_type": "issue_meal_voucher", "confirmation_number": "ZK3FFW"})
    assert state["actions"] == [
        {"action_type": "rebook_flight", "confirmation_number": "ZK3FFW"},
        {"action_type": "issue_meal_voucher", "confirmation_number": "ZK3FFW"},
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
