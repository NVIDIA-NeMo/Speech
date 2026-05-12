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

"""Integration tests for eva_airline Stage B.

Without spinning up an LLM or bot server, exercise:
    1. The full happy-path action sequence for ``eva_airline__voluntary_date_change``
       (auth → search → rebook → assign_seat → summary) against the real fixture,
       and confirm the comparator returns success against the scenario's reference.
    2. Cancellation + refund flow.
    3. Voucher issuance amount mapping (delay_over_2_hours → $12, etc.).
    4. Transfer-to-agent terminal action recording.
    5. Every write tool produces an action whose ``action_type`` matches its eva tool name.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Add repo root to sys.path so we test the working-tree NeMo, not whatever is pip-installed.
nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))

import pytest

from nemo.agents.voice_agent.evaluation import get_eval_data_root
from nemo.agents.voice_agent.evaluation.db_hash import compute_db_diff, get_dict_hash
from nemo.agents.voice_agent.evaluation.scenarios import get_eval_scenario, list_eval_scenarios


def _all_eva_airline_scenarios() -> list:
    """All registered eva_airline scenarios, discovered dynamically.

    Each new batch (eva_airline_1x.py, _2x.py, ...) is automatically covered
    by the parametrized tests below as soon as it's registered in
    ``scenarios/data/__init__.py``. No hardcoded list to maintain.
    """
    return sorted(name for name in list_eval_scenarios() if name.startswith("eva_airline__"))


from nemo.agents.voice_agent.evaluation.tools.eva_airline_tools import (
    AIRLINE_ACTION_TYPES,
    AddBaggageAllowanceTool,
    AddMealRequestTool,
    AddToStandbyTool,
    AssignSeatTool,
    CancelReservationTool,
    GetReservationTool,
    IssueHotelVoucherTool,
    IssueMealVoucherTool,
    IssueTravelCreditTool,
    ProcessRefundTool,
    RebookFlightTool,
    SearchRebookingOptionsTool,
    TransferToAgentTool,
)
from nemo.agents.voice_agent.evaluation.utils import check_if_task_success


# ---------------------------------------------------------------------------
# Fakes (mirror those in test_eva_airline_stage_a.py)
# ---------------------------------------------------------------------------


class _FakeFunctionCallParams:
    def __init__(self, arguments: dict):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, value):
        self.result = value


def _load_fixture_state(eva_id: str) -> dict:
    """Build a shared_state dict matching what the action handler initializes
    when ``setup_shared_state`` writes inline DB content."""
    path = get_eval_data_root() / "eva_airline_scenarios" / f"{eva_id}.json"
    return {"db": json.loads(path.read_text())}


def _run(tool, arguments):
    p = _FakeFunctionCallParams(arguments)
    asyncio.run(tool._execute(p))
    return p.result


# ---------------------------------------------------------------------------
# Voluntary date change — full happy-path integration
# ---------------------------------------------------------------------------


def test_voluntary_date_change_happy_path_matches_reference():
    """End-to-end action sequence accumulates actions in shared_state that
    satisfy the scenario's reference_answer via the comparator."""
    state = _load_fixture_state("1.1.2")

    # Auth
    auth_result = _run(
        GetReservationTool(shared_state=state),
        {"confirmation_number": "ZK3FFW", "last_name": "Rodriguez"},
    )
    assert auth_result["status"] == "success"

    # Search the constraint-meeting flight
    search_result = _run(
        SearchRebookingOptionsTool(shared_state=state),
        {
            "origin": "AUS",
            "destination": "LAX",
            "date": "2026-03-25",
            "passenger_count": 1,
            "fare_class": "main_cabin",
        },
    )
    assert search_result["status"] == "success"
    candidates = sorted(
        (o for o in search_result["options"] if o["arrival_time"] <= "16:00"),
        key=lambda o: o["fare"],
    )
    chosen = candidates[0]
    assert chosen["journey_id"] == "FL_SK703_20260325"

    # Rebook
    rebook_result = _run(
        RebookFlightTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK621_20260320",
            "new_journey_id": chosen["journey_id"],
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
    )
    assert rebook_result["status"] == "success"
    assert rebook_result["cost_summary"]["total_collected"] == 115.0

    # Assign window seat
    seat_result = _run(
        AssignSeatTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "passenger_id": "PAX001",
            "journey_id": chosen["journey_id"],
            "seat_preference": "window",
        },
    )
    assert seat_result["status"] == "success"
    assert seat_result["seat_assigned"].endswith("A")  # 'A' is the window letter

    # The bridge will pull these (no LLM-callable summary tool); construct
    # the payload manually as the bridge would.
    payload = {"actions": state["actions"]}
    assert len(payload["actions"]) == 2
    assert payload["actions"][0]["action_type"] == "rebook_flight"
    assert payload["actions"][1]["action_type"] == "assign_seat"

    # Comparator agrees with the scenario's reference_answer
    scenario = get_eval_scenario("eva_airline__voluntary_date_change")
    with (
        tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as rf,
        tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as pf,
    ):
        json.dump(scenario.reference_answer, rf)
        rf.flush()
        json.dump(payload, pf)
        pf.flush()
        assert check_if_task_success(reference=rf.name, prediction=pf.name) is True


def test_voluntary_date_change_scenario_metadata():
    s = get_eval_scenario("eva_airline__voluntary_date_change")
    assert s is not None
    assert s.eva_id == "1.1.2"
    assert s.current_date == "2026-03-17"
    assert s.reference_answer["actions"][0]["action_type"] == "rebook_flight"
    assert s.reference_answer["actions"][1]["action_type"] == "assign_seat"


# ---------------------------------------------------------------------------
# Cancellation + refund flow
# ---------------------------------------------------------------------------


def test_cancel_then_process_refund_records_two_actions():
    """Cancel a non-refundable booking with 24-hour rule (still gets refund), then refund."""
    state = _load_fixture_state("1.1.2")

    # Cancel under 24-hour rule (waives fee, refund eligible)
    cancel = _run(
        CancelReservationTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK621_20260320",
            "cancellation_reason": "24_hour_rule",
        },
    )
    assert cancel["status"] == "success"
    assert cancel["is_refundable"] is True
    refund_eligible = cancel["refund_amount_eligible"]
    assert refund_eligible > 0

    # Refund (fare portion)
    refund = _run(
        ProcessRefundTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "refund_amount": refund_eligible,
            "refund_type": "full_fare",
        },
    )
    assert refund["status"] == "success"
    assert refund["refund_id"].startswith("REF-ZK3FFW-")

    # Both actions recorded
    actions = state["actions"]
    assert len(actions) == 2
    assert actions[0]["action_type"] == "cancel_reservation"
    assert actions[0]["cancellation_reason"] == "24_hour_rule"
    assert actions[1]["action_type"] == "process_refund"
    assert actions[1]["refund_amount"] == refund_eligible


# ---------------------------------------------------------------------------
# Voucher amount mapping — eva-specific business logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "voucher_reason,expected_amount",
    [
        ("delay_over_2_hours", 12),
        ("delay_over_4_hours", 15),
        ("cancellation_wait_same_day", 15),
        ("irrops_overnight", 25),
    ],
)
def test_meal_voucher_amount_mapping(voucher_reason, expected_amount):
    state = _load_fixture_state("1.1.2")
    result = _run(
        IssueMealVoucherTool(shared_state=state),
        {"confirmation_number": "ZK3FFW", "passenger_id": "PAX001", "voucher_reason": voucher_reason},
    )
    assert result["status"] == "success"
    assert result["amount"] == expected_amount
    # Action record carries the same amount
    assert state["actions"][-1]["amount"] == expected_amount
    assert state["actions"][-1]["voucher_reason"] == voucher_reason


def test_hotel_voucher_rejects_more_than_3_nights():
    state = _load_fixture_state("1.1.2")
    result = _run(
        IssueHotelVoucherTool(shared_state=state),
        {"confirmation_number": "ZK3FFW", "passenger_id": "PAX001", "num_nights": 4},
    )
    assert result["status"] == "error"
    assert result["error_type"] == "exceeds_authority"
    # No action recorded on failure
    assert "actions" not in state or not state.get("actions")


# ---------------------------------------------------------------------------
# Transfer-to-agent records a terminal action
# ---------------------------------------------------------------------------


def test_transfer_to_agent_records_action():
    state = _load_fixture_state("1.1.2")
    result = _run(
        TransferToAgentTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "transfer_reason": "passenger_requested",
            "issue_summary": "Customer asked to speak to a supervisor.",
        },
    )
    assert result["status"] == "success"
    assert result["transfer_id"].startswith("TRF-ZK3FFW-")
    assert state["actions"][-1] == {
        "action_type": "transfer_to_agent",
        "confirmation_number": "ZK3FFW",
        "transfer_reason": "passenger_requested",
        "issue_summary": "Customer asked to speak to a supervisor.",
    }


# ---------------------------------------------------------------------------
# All write tools produce an action_type matching their eva tool name (1:1)
# ---------------------------------------------------------------------------


def test_all_write_tool_action_types_present_in_AIRLINE_ACTION_TYPES():
    """If a write tool records an action with an unrecognized type, _record_action warns.

    This is a smoke check that the type strings used in tool implementations
    line up with the locked vocabulary in AIRLINE_ACTION_TYPES.
    """
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


# ---------------------------------------------------------------------------
# Failed validations don't pollute the action list
# ---------------------------------------------------------------------------


def test_failed_validation_does_not_record_action():
    """Pydantic ValidationError → error response, no action appended."""
    state = _load_fixture_state("1.1.2")
    # Malformed journey_id (doesn't match the FL_<flight>_<YYYYMMDD> pattern)
    result = _run(
        CancelReservationTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "not-a-valid-journey-id",
            "cancellation_reason": "voluntary",
        },
    )
    assert result["status"] == "error"
    assert result["error_type"] == "invalid_journey_id_format"
    assert "actions" not in state or not state.get("actions")


def test_db_not_loaded_does_not_record_action():
    """Without shared_state['db'], every tool returns db_not_initialized."""
    tool = RebookFlightTool(shared_state={})
    result = _run(
        tool,
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK621_20260320",
            "new_journey_id": "FL_SK703_20260325",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
    )
    assert result["status"] == "error"
    assert result["error_type"] == "db_not_initialized"


# ---------------------------------------------------------------------------
# Per-tool baggage + meal request quick checks
# ---------------------------------------------------------------------------


def test_add_baggage_allowance_records_action():
    state = _load_fixture_state("1.1.2")
    result = _run(
        AddBaggageAllowanceTool(shared_state=state),
        {"confirmation_number": "ZK3FFW", "journey_id": "FL_SK621_20260320", "num_bags": 2},
    )
    assert result["status"] == "success"
    assert state["actions"][-1]["action_type"] == "add_baggage_allowance"
    assert state["actions"][-1]["num_bags"] == 2


def test_add_meal_request_records_action():
    state = _load_fixture_state("1.1.2")
    result = _run(
        AddMealRequestTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "passenger_id": "PAX001",
            "journey_id": "FL_SK621_20260320",
            "meal_type": "vegetarian",
        },
    )
    assert result["status"] == "success"
    assert state["actions"][-1]["action_type"] == "add_meal_request"
    assert state["actions"][-1]["meal_type"] == "vegetarian"


def test_add_to_standby_validates_passenger_ids():
    state = _load_fixture_state("1.1.2")
    # PAX999 doesn't exist on this reservation
    result = _run(
        AddToStandbyTool(shared_state=state),
        {"confirmation_number": "ZK3FFW", "journey_id": "FL_SK621_20260320", "passenger_ids": ["PAX999"]},
    )
    assert result["status"] == "error"
    assert result["error_type"] == "invalid_passengers"


def test_issue_travel_credit_records_action():
    state = _load_fixture_state("1.1.2")
    result = _run(
        IssueTravelCreditTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "passenger_id": "PAX001",
            "amount": 50.0,
            "credit_reason": "service_recovery",
        },
    )
    assert result["status"] == "success"
    assert state["actions"][-1]["action_type"] == "issue_travel_credit"
    assert state["actions"][-1]["amount"] == 50.0


# ---------------------------------------------------------------------------
# DB-state hash matching (eva-style end-state scoring)
# ---------------------------------------------------------------------------


def test_expected_scenario_db_loads_from_dataset_jsonl():
    """``expected_scenario_db`` cached_property reads from eva_airline_dataset.jsonl."""
    s = get_eval_scenario("eva_airline__voluntary_date_change")
    expected = s.expected_scenario_db
    assert isinstance(expected, dict)
    # Sanity-check the eva-shipped expected state has the tables our tools mutate
    for table in ("_current_date", "reservations", "journeys", "refunds", "travel_credits", "meal_vouchers"):
        assert table in expected, f"expected_scenario_db is missing {table!r}"
    # The expected DB carries the same _current_date as the initial DB
    assert expected["_current_date"] == s.current_date


def test_voluntary_date_change_happy_path_db_state_match():
    """Clean rebook+seat sequence produces a DB that hash-matches eva's expected state.

    Path-independent scoring: this run uses one specific action sequence
    (auth → search → rebook → assign_seat), but any other sequence that
    lands in the same end-state should also match. The hash is the verdict.
    """
    state = _load_fixture_state("1.1.2")

    # Execute the canonical happy path
    _run(GetReservationTool(shared_state=state), {"confirmation_number": "ZK3FFW", "last_name": "Rodriguez"})
    _run(
        RebookFlightTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK621_20260320",
            "new_journey_id": "FL_SK703_20260325",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
    )
    _run(
        AssignSeatTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "passenger_id": "PAX001",
            "journey_id": "FL_SK703_20260325",
            "seat_preference": "window",
        },
    )

    # Hash compare against the eva-shipped ground truth
    scenario = get_eval_scenario("eva_airline__voluntary_date_change")
    expected_hash = get_dict_hash(scenario.expected_scenario_db)
    actual_hash = get_dict_hash(state["db"])
    if expected_hash != actual_hash:
        diff = compute_db_diff(expected_db=scenario.expected_scenario_db, actual_db=state["db"])
        pytest.fail(
            f"DB-state hash mismatch on canonical path. Diff: {json.dumps(diff, indent=2, default=str)[:2000]}"
        )


def test_messy_path_db_state_diverges_from_expected():
    """Rebook → cancel → rebook-again produces extra cancelled bookings; cardinality differs.

    Locks in the property that DB-state matching penalizes side-effect accumulation.
    Three bookings (original cancelled, intermediate cancelled, new confirmed) instead
    of two (original cancelled, new confirmed). Hashes diverge.
    """
    state = _load_fixture_state("1.1.2")
    _run(GetReservationTool(shared_state=state), {"confirmation_number": "ZK3FFW", "last_name": "Rodriguez"})

    # First rebook
    _run(
        RebookFlightTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK621_20260320",
            "new_journey_id": "FL_SK703_20260325",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
    )
    # Cancel that booking (now we have: original cancelled + rebook cancelled)
    _run(
        CancelReservationTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK703_20260325",
            "cancellation_reason": "voluntary",
        },
    )
    # Rebook again (now: original cancelled + intermediate cancelled + new confirmed)
    _run(
        RebookFlightTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "journey_id": "FL_SK703_20260325",
            "new_journey_id": "FL_SK703_20260325",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
    )
    _run(
        AssignSeatTool(shared_state=state),
        {
            "confirmation_number": "ZK3FFW",
            "passenger_id": "PAX001",
            "journey_id": "FL_SK703_20260325",
            "seat_preference": "window",
        },
    )

    scenario = get_eval_scenario("eva_airline__voluntary_date_change")
    expected_hash = get_dict_hash(scenario.expected_scenario_db)
    actual_hash = get_dict_hash(state["db"])
    assert expected_hash != actual_hash, "messy path should diverge from expected, but hashed identical"

    # The diff should specifically show extra/modified bookings on the reservation
    diff = compute_db_diff(expected_db=scenario.expected_scenario_db, actual_db=state["db"])
    assert "reservations" in diff["tables_modified"], f"expected reservations diff, got: {diff}"


# ---------------------------------------------------------------------------
# Coverage: all registered eva_airline scenarios are well-formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", _all_eva_airline_scenarios())
def test_eva_airline_scenarios_register_and_load(scenario_name):
    """Each registered eva_airline scenario instantiates and loads its
    ``expected_scenario_db`` from ``eva_airline_dataset.jsonl``.

    Cheap structural check that catches missing data files, typos in ``eva_id``,
    or scenarios that forgot to set ``eva_id``. Doesn't validate agent behavior —
    just that the scenario is well-formed and the dataset.jsonl entry exists.
    """
    s = get_eval_scenario(scenario_name)
    assert s is not None, f"{scenario_name} is not registered"
    assert s.eva_id, f"{scenario_name}: must declare a non-empty eva_id"
    expected = s.expected_scenario_db
    assert isinstance(expected, dict), f"{scenario_name}: expected_scenario_db is not a dict"
    for table in ("_current_date", "reservations", "journeys"):
        assert table in expected, f"{scenario_name}: expected_db missing {table!r}"


@pytest.mark.parametrize("scenario_name", _all_eva_airline_scenarios())
def test_eva_airline_scenarios_have_spell_out_rule_in_both_prompts(scenario_name):
    """``VOICE_ALPHANUMERIC_RULE`` must land in both agent and user prompts.

    Catches scenarios that overrode user_actions but forgot to include
    ``self.VOICE_ALPHANUMERIC_RULE`` in ``user_actions.guidelines``.
    """
    s = get_eval_scenario(scenario_name)
    agent_prompt = s.get_agent_prompt()
    user_prompt = s.get_user_prompt()
    # The literal example "L, A, X" from VOICE_ALPHANUMERIC_RULE — its presence
    # confirms the rule itself is included (no scenario uses "L, A, X" by accident).
    assert "L, A, X" in agent_prompt, f"{scenario_name}: spell-out rule missing from agent prompt"
    assert "L, A, X" in user_prompt, f"{scenario_name}: spell-out rule missing from user prompt"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
