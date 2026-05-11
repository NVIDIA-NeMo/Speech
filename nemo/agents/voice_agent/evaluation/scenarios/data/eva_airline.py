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
#
# Scenario fixtures (data/eva_airline_scenarios/*.json) are
# adapted from https://github.com/ServiceNow/eva/tree/0.1.3 (MIT-licensed).

# Scenario definitions contain long prose strings (personas, policy bullets);
# wrapping every one hurts readability without improving correctness.
# pylint: disable=line-too-long
# flake8: noqa: E501

import json
from functools import cache, cached_property

from nemo.agents.voice_agent.evaluation import get_eval_data_root
from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Scenario, Task

# ---------------------------------------------------------------------------
# Module-level cached dataset index
# ---------------------------------------------------------------------------


@cache
def _load_eva_airline_dataset_index() -> dict:
    """Index ``eva_airline_dataset.jsonl`` by scenario id, once per process.

    The dataset.jsonl is the per-scenario metadata file shipped by eva alongside
    the scenario fixtures. Each line is a full dataset entry keyed by ``id``
    (e.g. ``"1.1.2"``). Callers pull whichever field they need:
    ``ground_truth.expected_scenario_db`` for DB-state hash matching,
    ``user_goal.decision_tree.must_have_criteria`` for the LLM judge, etc.

    Cached via ``functools.cache`` — reads the file once across all scenario
    instances. The dataset is small (50 lines × ~15KB), and ``EVAL_DATA_ROOT``
    doesn't change within a process run.
    """
    path = get_eval_data_root() / "eva_airline_dataset.jsonl"
    index = {}
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            index[entry["id"]] = entry
    return index


# ---------------------------------------------------------------------------
# Domain base
# ---------------------------------------------------------------------------


class EvaAirlineBaseScenario(Scenario):
    """Base class for airline scenarios ported from eva.

    Subclasses set only ``eva_id`` (e.g. ``"1.1.2"``) — everything else derives:

    - ``current_date`` — read lazily from the bound JSON's ``_current_date``.
    - DB seeding — ``setup_shared_state`` writes ``state["db_path"]`` for the
      action handler to resolve against ``EVAL_DATA_ROOT``.

    Subclasses also declare ``name``, ``user_persona``, ``user_task``,
    ``user_actions``, ``reference_answer`` (a list of expected actions, possibly
    empty for Q&A-only scenarios), and optionally override ``agent_actions`` /
    ``agent_resources`` if the scenario needs domain-specific tweaks.

    The toolset is fixed: every airline scenario gets the full eva 15-tool
    surface plus ``EndConversationTool``. The scenario action list and final
    DB state are pulled by the bridge at end-of-scenario via the
    ``get_scenario_summary`` RTVI action — no LLM-callable summary tool.
    """

    # Subclasses must set ``eva_id``. Default is set so the base class is
    # introspectable; instantiating the base directly will fail at file IO.
    eva_id: str = ""
    max_duration = 900  # 15 minutes default — voice round-trips are ~10× slower than text;
    # observed live runs of voluntary_date_change take 12–14 turns even when the agent
    # operates efficiently. 600s leaves no headroom for the closing protocol.

    # Shared voice-readability rule for both agent and user. Airport codes
    # (LAX, AUS, JFK) and flight numbers (SK703) sound terrible when pronounced
    # as words and round-trip poorly through ASR/TTS. Confirmation numbers
    # need spelling regardless. Use this constant in both agent_actions.guidelines
    # and user_actions.guidelines so the rule stays in sync across scenarios.
    VOICE_ALPHANUMERIC_RULE = (
        "When speaking confirmation numbers, flight numbers, or airport codes, "
        "spell each character one at a time — letters as letters, digits as words. "
        "Examples: 1A2BC4 (spelled out as one, A, two, B, C, four); "
        "SK123 (spelled out as S, K, one, two, three); "
        "LAX (spelled out as L, A, X); "
        "AUS (spelled out as A, U, S). "
        "Never pronounce these identifiers as words. In prompts and guidelines "
        "you'll see codes written as 'CODE (spelled out as letter, letter, digit, ...)' — "
        "the part in parentheses is how to speak it; the part before is the canonical identifier."
    )

    @cached_property
    def _scenario_db(self) -> dict:
        """Load the bound eva scenario JSON. Bridge-side; cached after first read."""
        if not self.eva_id:
            raise ValueError(f"{type(self).__name__} must declare a class attribute eva_id")
        path = get_eval_data_root() / "eva_airline_scenarios" / f"{self.eva_id}.json"
        return json.loads(path.read_text())

    @cached_property
    def current_date(self) -> str:
        """Scenario's ``_current_date`` from the bound JSON. Single source of truth."""
        return self._scenario_db["_current_date"]

    @cached_property
    def expected_scenario_db(self) -> dict:
        """Eva-shipped expected post-run DB state for this scenario.

        Sourced from ``eva_airline_dataset.jsonl``'s ``ground_truth.expected_scenario_db``
        for the matching ``eva_id``. The runner SHA-256-hashes both this and the
        bridge-pulled ``final_scenario_db.json`` to score the scenario on
        end-state correctness (path-independent — any sequence of agent actions
        that lands here passes; see ``evaluation/db_hash.py``).

        Verified on 2026-05-11: a clean run of scenario 1.1.2 produces a DB
        whose canonical hash matches this expected state exactly. Hence we use
        eva's expected_scenario_db as the ground truth for all airline scenarios
        rather than hand-authoring NeMo-specific expected states.

        Raises ``KeyError`` if the eva_id isn't in the dataset (e.g. a scenario
        we authored without a corresponding eva entry).
        """
        return _load_eva_airline_dataset_index()[self.eva_id]["ground_truth"]["expected_scenario_db"]

    def setup_shared_state(self, state: dict, side: str) -> None:
        """Seed the agent side with the scenario DB content (inline, not a path).

        Symmetric with how the bridge pulls the final DB at end-of-scenario:
        full content travels both ways. See plan section 6.5 #8.
        """
        if side == "agent":
            state["db"] = self._scenario_db

    # -- Agent defaults (shared across all airline scenarios) ---------------

    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="customer service agent",
            name="Skye",
            background="You are a voice agent for SkyWay Airlines handling inbound calls for flight changes, rebooking, cancellations, and refunds.",
            personality=(
                "You are calm, professional, and concise. You listen first, confirm critical details "
                "before acting, and explain fees and policies clearly before making any change."
            ),
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal=(
                "Help the caller with their flight change, rebooking, cancellation, or refund request, "
                "applying SkyWay's policies. End the call cleanly with EndConversationTool once the "
                "caller has nothing else to ask."
            ),
            background="You are handling an inbound customer service call for SkyWay Airlines.",
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the caller and ask how you can help.",
                "Authenticate by asking for their confirmation number and last name; call GetReservationTool to load the booking.",
                "Listen to the caller's request and consult the policies in your guidelines.",
                "Use the appropriate tools to fulfill the request — explain any fees or fare differences before confirming.",
                "After the work is done, confirm to the caller in plain language what was changed (or refunded, or vouchered) and ask if there is anything else they need.",
                "Once the caller indicates they have nothing else, exchange goodbyes (e.g., 'Thank you for flying SkyWay, have a great day') and then call EndConversationTool to end the call.",
            ],
            guidelines=[
                f"Today is {self.current_date}.",
                self.VOICE_ALPHANUMERIC_RULE,
                "Do not read internal journey IDs (e.g., FL_SK621_20260320) aloud. Refer to flights by flight number and date instead.",
                "Confirm critical details before executing changes.",
                "Stay concise — this is a phone call, not an email.",
                "Use only the tools provided; do not invent flight numbers, fares, or policies.",
                # Cost math — the agent must translate raw fares into out-of-pocket
                # change cost. Observed failure mode: agent quotes $300 (raw new
                # fare) and tells caller "that's over your $120 budget" when the
                # actual change cost is $115.
                "When discussing rebooking cost with the caller: ALWAYS quote the total out-of-pocket change cost, NEVER the raw new-flight fare. Total change cost = change_fee + max(0, new_fare − old_fare_paid). Example: if the customer paid $260 originally and the new flight's fare is $300 with a $75 change fee, the total they owe is $115, not $300. If the new fare is lower than the old, the fare difference becomes a travel credit — they only pay the change fee.",
                "Voluntary change fees by original fare class: Basic Economy $199 (or $75 for same-day), Main Cabin / Premium Economy $75, Business / First $0. IRROPS-driven changes waive the fee entirely.",
                "If the caller mentions a cost budget (e.g., 'under $120'), evaluate options against the TOTAL CHANGE COST, NOT the raw new-flight fare. A flight whose new-cabin fare is $300 may still fit a $120 budget after subtracting the old fare paid and adding the change fee.",
                # Turn-efficiency hints — voice round-trips are slow; volunteering
                # information the caller is likely to ask next saves 1–2 turns each.
                "When presenting flight options to the caller, ALWAYS include the total change cost for each option upfront (not just the raw fare). Don't make the caller ask a second time for the cost.",
                "Right after a successful rebooking, proactively offer to assign a seat if the caller hasn't requested one yet — e.g., 'Would you like me to assign a seat? Any preference — window, aisle, or middle?' This saves a round of asking and avoids running out of call time.",
                "Do not call EndConversationTool until you have (a) told the caller what was done, (b) asked if there is anything else, and (c) exchanged goodbyes.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        # Full eva 15-tool surface + EndConversationTool. No per-tool kwargs;
        # scenario data flows through shared_state["db"] seeded by setup_shared_state.
        # The action list and final DB state are pulled by the bridge at end of
        # scenario via the get_scenario_summary RTVI action — no LLM-callable
        # summary tool.
        return Resources(
            tools={
                # Read tools (4)
                "GetReservationTool": {},
                "GetFlightStatusTool": {},
                "GetDisruptionInfoTool": {},
                "SearchRebookingOptionsTool": {},
                # Write tools (10)
                "RebookFlightTool": {},
                "CancelReservationTool": {},
                "ProcessRefundTool": {},
                "AssignSeatTool": {},
                "AddBaggageAllowanceTool": {},
                "AddMealRequestTool": {},
                "AddToStandbyTool": {},
                "IssueTravelCreditTool": {},
                "IssueHotelVoucherTool": {},
                "IssueMealVoucherTool": {},
                # System tool (1)
                "TransferToAgentTool": {},
                # Harness tool
                "EndConversationTool": {},
            },
            information=[
                f"Today's date is {self.current_date}.",
            ],
        )

    # -- User defaults (subclasses typically override) ----------------------

    @property
    def user_resources(self) -> Resources:
        return Resources()


# ---------------------------------------------------------------------------
# Voluntary date change — first real seed scenario (eva 1.1.2)
#
# Constraint set: AUS→LAX on 2026-03-25, arrival ≤ 4:00 PM Pacific, total
# rebooking cost ≤ $120, window seat. The scenario DB has only one option
# meeting all four constraints: FL_SK703_20260325 (fare $300, change_fee $75,
# total $115 ≤ $120, arrives 09:25 PT). The agent should rebook to that
# flight and assign a window seat.
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirlineVoluntaryDateChange(EvaAirlineBaseScenario):
    """Voluntary date change with cost cap and window-seat constraint."""

    name = "eva_airline__voluntary_date_change"
    eva_id = "1.1.2"
    description = (
        "Passenger wants to move AUS→LAX from March 20 to March 25, arriving by 4:00 PM Pacific, "
        "for ≤$120 total, keeping a window seat."
    )
    reference_answer = {
        "actions": [
            {
                "action_type": "rebook_flight",
                "confirmation_number": "ZK3FFW",
                "old_journey_id": "FL_SK621_20260320",
                "new_journey_id": "FL_SK703_20260325",
                "rebooking_type": "voluntary",
                "total_collected": 115,
            },
            {
                "action_type": "assign_seat",
                "confirmation_number": "ZK3FFW",
                "passenger_id": "PAX001",
                "seat_preference": "window",
            },
        ]
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Samantha Rodriguez",
            background=(
                "You are Samantha Rodriguez (confirmation ZK3FFW (spelled out as Z, K, three, F, F, W)). You are "
                "currently booked on a flight from Austin AUS (spelled out as A, U, S) to Los Angeles LAX (spelled out as L, A, X) on "
                "March 20, departing at 11:05 AM. Your project deadline moved, so you need to push the "
                "trip to March 25 — but you must arrive in LA no later than 4:00 PM Pacific. You're "
                "price-sensitive: the total cost to change must be $120 or less. You also want to keep "
                "a window seat."
            ),
            personality=(
                "You're direct and to the point — you don't have time for lengthy explanations or "
                "unnecessary back-and-forth. You speak curtly, getting straight to what you need "
                "without much small talk. You'll show mild frustration if things move slowly."
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "Move your March 20 AUS→LAX flight to March 25, arriving by 4:00 PM Pacific, for "
                "$120 or less total, with a window seat assigned. If no option meets all four "
                "criteria after two rounds of search, keep the original booking and end the call."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                # Sequential beats only — one per turn. Conditionals/rules belong in guidelines.
                "Greet the agent and say you need to change your flight to March 25.",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share your trip details when the agent asks (current and desired dates, route, time/cost constraints, seat preference).",
                "When the agent presents flight options, evaluate them and tell the agent which option you choose (or, if none fit your must-haves, follow the failure path in your guidelines).",
                "Once the rebooking is processed and a window seat has been assigned, thank the agent and end the call.",
            ],
            guidelines=[
                # Voice rules.
                self.VOICE_ALPHANUMERIC_RULE,
                # Identity / context — sourced from your persona, applied as needed.
                "Your confirmation number is ZK3FFW (spelled out as Z, K, three, F, F, W). Your last name is Rodriguez.",
                "Your current booking: AUS (spelled out as A, U, S) to LAX (spelled out as L, A, X) on March twenty, departing eleven oh five AM. You want to move it to March twenty-five, arriving by four PM Pacific. You will pay no more than one hundred twenty dollars total to change. You want to keep a window seat.",
                # Decision rules — applied throughout, not on a particular turn.
                "Stick to AUS (spelled out as A, U, S) and LAX (spelled out as L, A, X) — decline any alternative airports.",
                "Decline standby — you only want a confirmed seat.",
                "Reject any option that arrives after four PM Pacific.",
                "Reject any option where the agent cannot guarantee a window seat assignment at booking time.",
                "Before the agent finalizes any rebooking, make sure you know the total all-in cost. If the agent has not stated it, ask once: 'What's the total cost to change, all-in?' Do not re-ask once they've answered for the same option.",
                "If the stated total is over one hundred twenty dollars, decline that option and ask for a different March 25 option that is one hundred twenty dollars or less and arrives by four PM Pacific.",
                "When picking among options that meet all four must-haves, prefer the lowest total cost; on a tie, prefer the earliest arrival.",
                "After the agent confirms the rebooking is processed, ask them to assign a window seat. Do not ask before then.",
                "Failure path: if the agent has searched at least twice and still cannot find a March 25 option meeting all your must-haves, say you'll keep your original flight, thank them, and end the call.",
                "Do not escalate to a live agent. If the agent offers to transfer, decline.",
                "Do not invent new requests beyond moving the flight, capping cost, and keeping a window seat.",
                "End the call with a clear farewell like 'Thanks, that's all. Goodbye.' once the task is complete (or you've decided to keep the original).",
            ],
        )


# ---------------------------------------------------------------------------
# IRROPS rebooking — eva 2.1.1
# Airline canceled SFO→ORD SK302 (mechanical). Same-day rebook + meal voucher.
# Exercises: get_disruption_info, IRROPS fee waiver, issue_meal_voucher.
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirlineIrropsCancellation(EvaAirlineBaseScenario):
    """IRROPS rebooking after airline-canceled flight (mechanical)."""

    name = "eva_airline__irrops_cancellation"
    eva_id = "2.1.1"
    description = (
        "Airline canceled SFO→ORD flight SK302 due to a mechanical issue. "
        "Passenger wants same-day rebooking to arrive in ORD by 8 PM CST, "
        "Main Cabin only, plus a meal voucher for the inconvenience."
    )
    # reference_answer is optional with DB-state matching as the primary signal.
    # IRROPS scenarios have multiple acceptable canonical action sequences
    # (different rebooking flights, optional voucher amount tiers); we rely on
    # expected_scenario_db hash matching for end-state correctness.

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Lucas Rivera",
            background=(
                "You are Lucas Rivera (confirmation FAR0UM (spelled out as F, A, R, zero, U, M)). You were booked on "
                "flight SK302 (spelled out as S, K, three, zero, two) from SFO (spelled out as S, F, O) to ORD (spelled out as O, R, D) on April "
                "fourteenth, originally departing 10:30 AM. The airline canceled the flight due to a "
                "mechanical issue. You need to be rebooked today to arrive in Chicago no later than 8 PM "
                "Central. You prefer a nonstop Main Cabin flight. Aisle seat preferred."
            ),
            personality=(
                "Direct and to the point. Curt, no small talk. You'll show mild frustration if the agent "
                "moves slowly or misses information you've already provided."
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "Get rebooked today on a same-day SFO→ORD Main Cabin flight arriving by 8 PM Central. "
                "Since the cancellation was airline-caused (IRROPS), there should be no fees. Also get a "
                "meal voucher for the inconvenience."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight to Chicago was canceled — can you get me on another flight today?'",
                "Provide your confirmation number and last name when the agent asks.",
                "When asked, share trip details: original flight SK302 (spelled out as S, K, three, zero, two) from SFO (spelled out as S, F, O) to ORD (spelled out as O, R, D) was canceled.",
                "When the agent presents alternative flights, evaluate against your must-haves and pick one that fits.",
                "Once the rebooking is confirmed, ask about a meal voucher for the inconvenience.",
                "Once both the rebooking and meal voucher are confirmed, thank the agent and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is FAR0UM (spelled out as F, A, R, zero, U, M). Last name: Rivera.",
                "Your original booking: flight SK302 (spelled out as S, K, three, zero, two) from SFO (spelled out as S, F, O) to ORD (spelled out as O, R, D) on April fourteenth, departing 10:30 AM in Main Cabin. It was canceled by the airline.",
                "Must-haves: arrive in ORD (spelled out as O, R, D) no later than 8 PM Central today, stay on the SFO (spelled out as S, F, O) to ORD (spelled out as O, R, D) route (no alternate airports), keep Main Cabin (no Basic Economy downgrade), and get a meal voucher.",
                "Since the airline canceled (mechanical issue), this is IRROPS — there should be no change fees or fare differences. If the agent quotes any fee, push back: IRROPS means free rebooking.",
                "Decline any options that arrive after 8 PM Central, change airports, or downgrade your cabin.",
                "Failure path: if no Main Cabin SFO→ORD option today arrives by 8 PM Central after two search rounds, say goodbye and end the call.",
                "Do not escalate to a live agent. If the agent offers to transfer, decline.",
            ],
        )


# ---------------------------------------------------------------------------
# Missed flight standby — eva 3.1.3
# Passenger missed morning flight; wants free standby for 1pm+ departure.
# Exercises: add_to_standby (free), possibly rebook_flight (protective seat).
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirlineMissedFlightStandby(EvaAirlineBaseScenario):
    """Passenger missed morning flight; wants free standby (driving to airport)."""

    name = "eva_airline__missed_flight_standby"
    eva_id = "3.1.3"
    description = (
        "Passenger missed morning ORD→DCA flight. Driving to ORD now (arrives 11 AM). "
        "Wants free standby for a 1 PM-or-later flight arriving by 6 PM Eastern — $0 cost."
    )

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Justin Sanders",
            background=(
                "You are Justin Sanders (confirmation EPXYEK (spelled out as E, P, X, Y, E, K)). You missed your 10 AM "
                "flight from ORD (spelled out as O, R, D) to DCA (spelled out as D, C, A) this morning. You're driving to Chicago "
                "O'Hare now and will arrive around 11 AM, so you need a departure at 1 PM or later (you "
                "need time to clear security). You want $0 additional cost — no change fees, no fare "
                "differences. Standby for free is acceptable; paying a change fee is not."
            ),
            personality=(
                "Direct, curt, no small talk. Pressed for time and a little stressed about missing the flight. "
                "Shows mild frustration at delays in the call."
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "Get on a free standby for an ORD (spelled out as O, R, D) to DCA (spelled out as D, C, A) flight today at 1 PM or later, "
                "arriving in DCA by 6 PM Eastern. Total additional cost must be $0."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I missed my flight this morning — can you get me on the cheapest option to D, C, A today?'",
                "Provide your confirmation number and last name when the agent asks.",
                "Explain that you missed the morning flight and need a departure at 1 PM or later (you're driving to ORD and won't be through security until then).",
                "When the agent presents options, prefer free standby. Reject any option with a change fee or fare difference.",
                "Once the standby placement (and any protective backup booking) is confirmed at $0 cost, thank the agent and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is EPXYEK (spelled out as E, P, X, Y, E, K). Last name: Sanders.",
                "Your original booking: ORD (spelled out as O, R, D) to DCA (spelled out as D, C, A) on June eleventh, departing 10 AM. You missed it (passenger fault, not airline).",
                "Must-haves: total additional cost is $0; arrive in DCA (spelled out as D, C, A) by 6 PM Eastern today; route stays ORD (spelled out as O, R, D) to DCA (spelled out as D, C, A); departure at 1 PM or later (security clearance time).",
                "Standby is free and acceptable. Confirmed rebooking with any fee or fare difference is NOT acceptable.",
                "If the agent offers a paid rebooking, push back: 'I want a free option — standby is fine.'",
                "Failure path: if after two attempts the agent can't offer a $0 option arriving by 6 PM ET, say goodbye and end the call.",
                "Do not escalate to a live agent. If the agent offers to transfer, decline.",
            ],
        )


# ---------------------------------------------------------------------------
# Voluntary cancellation + full refund — eva 5.1.1
# Passenger wants full refund to original credit card, not travel credit.
# Exercises: cancel_reservation, process_refund (cash, not credit).
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirlineCancellationRefund(EvaAirlineBaseScenario):
    """Voluntary cancellation with full refund to the original payment method."""

    name = "eva_airline__cancellation_refund"
    eva_id = "5.1.1"
    description = (
        "Passenger wants to cancel DCA→LAX trip (confirmation 8JVSDF) and get a "
        "full cash refund back to the original credit card (NOT a travel credit)."
    )

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Mark Lewis",
            background=(
                "You are Mark Lewis (confirmation 8JVSDF (spelled out as eight, J, V, S, D, F)). You have a DCA (spelled out as D, C, A) "
                "to LAX (spelled out as L, A, X) booking departing May twentieth at 9:10 AM. Your plans changed and you "
                "no longer need this trip. You want to cancel the entire booking and get a full refund "
                "back to the original credit card — not a travel credit."
            ),
            personality=(
                "Direct, curt, no small talk. Clear about what you want. Polite but firm — "
                "if the agent tries to push a travel credit, you'll insist on cash refund."
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "Cancel the entire booking under confirmation 8JVSDF (spelled out as eight, J, V, S, D, F) — all segments — "
                "and get a full refund processed back to the original credit card. Not a travel credit."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to cancel my flight.'",
                "Provide your confirmation number and last name when the agent asks.",
                "Explain that you want to cancel the entire booking and get a full refund back to the original credit card (not a travel credit).",
                "Confirm the cancellation when the agent presents the details.",
                "Once the agent has confirmed cancellation AND processed the full credit-card refund with a reference, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 8JVSDF (spelled out as eight, J, V, S, D, F). Last name: Lewis.",
                "Must-haves: the entire booking is canceled (all segments), a full cash refund is processed back to the original credit card (NOT travel credit), and the agent provides a concrete cancellation/refund reference.",
                "If the agent only offers a travel credit, push back firmly: 'I want cash refund to my credit card, not a travel credit.'",
                "Failure path: if after two clear restatements the agent still can't process the cancellation + full credit-card refund, say goodbye and end the call.",
                "Do not escalate to a live agent. If the agent offers to transfer, decline.",
            ],
        )


# ---------------------------------------------------------------------------
# Edge case: fabricated disruption — eva 7.2.1
# Passenger claims flight was canceled; it actually departed on time.
# Tests whether the agent verifies via get_flight_status before granting
# IRROPS benefits. Acceptable outcomes: paid rebook, refund per policy, or end.
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirlineFabricatedDisruption(EvaAirlineBaseScenario):
    """Edge case: passenger claims a flight was canceled when it departed on time.

    The agent should verify flight status before granting any IRROPS entitlements
    (fee waivers, free rebooking). Acceptable resolutions: paid rebook per voluntary
    policy, refund per policy, or graceful end-of-call. This scenario tests the
    agent's discipline around verifying claims rather than just accommodating them.
    """

    name = "eva_airline__escalation_edge_case"
    eva_id = "7.2.1"
    description = (
        "Passenger claims SK445 (DFW→LAX, 2026-04-15 8 AM) was canceled and demands "
        "free rebooking. The flight actually departed on time. Tests whether the agent "
        "verifies via get_flight_status before granting IRROPS benefits."
    )

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Mitchell Barnes",
            background=(
                "You are Mitchell Barnes (confirmation BZIW48 (spelled out as B, Z, I, W, four, eight)). You are calling "
                "claiming your morning flight SK445 (spelled out as S, K, four, four, five) from DFW (spelled out as D, F, W) to "
                "LAX (spelled out as L, A, X) was canceled. You want a free rebooking to a later flight today. "
                "(In the scenario's reality the flight actually departed on time — but you believe / claim "
                "it was canceled and you want the airline to accommodate you for free.)"
            ),
            personality=(
                "Direct, curt, slightly entitled. You believe the airline owes you a free rebook because "
                "(you claim) the cancellation was their fault."
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "Get rebooked to a later DFW (spelled out as D, F, W) to LAX (spelled out as L, A, X) flight today at no additional cost, "
                "claiming the airline canceled your morning flight."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight this morning was cancelled and I need to be rebooked.'",
                "Provide your confirmation number and last name when the agent asks.",
                "Insist that flight SK445 (spelled out as S, K, four, four, five) today was canceled and you should be rebooked for free.",
                "If the agent verifies the flight actually departed on time and offers paid options instead, evaluate them but reiterate your claim once.",
                "Accept whatever resolution the agent offers (paid rebooking, refund, etc.), OR if no acceptable option emerges after two attempts, say goodbye and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is BZIW48 (spelled out as B, Z, I, W, four, eight). Last name: Barnes.",
                "Your claimed booking: flight SK445 (spelled out as S, K, four, four, five) from DFW (spelled out as D, F, W) to LAX (spelled out as L, A, X) on April fifteenth, departing 8 AM. You claim it was canceled.",
                "Insist on free rebooking (claim IRROPS entitlements). Do not pay any fee or fare difference if you can avoid it.",
                "If the agent verifies via flight status that the flight actually departed on time and explains there's no IRROPS waiver, you may: (a) accept a paid rebooking under voluntary policy, (b) accept a cancellation/refund per policy, or (c) say goodbye and end the call.",
                "Failure path: if after two attempts the agent offers neither a rebooking nor a refund with a concrete confirmation, say goodbye and end the call.",
                "Do not escalate to a live agent. If the agent offers to transfer, decline.",
            ],
        )
