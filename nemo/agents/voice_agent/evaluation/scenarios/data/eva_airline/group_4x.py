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
# Scenario fixtures and prose adapted from https://github.com/ServiceNow/eva/tree/0.1.3 (MIT-licensed).
# Auto-scaffolded from eva_airline_dataset.jsonl entries via nemo_experiments/generate_eva_airline_scaffolds.py.

# pylint: disable=line-too-long
# flake8: noqa: E501

from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Task
from nemo.agents.voice_agent.evaluation.scenarios.data.eva_airline import EvaAirlineBaseScenario


# ---------------------------------------------------------------------------
# eva 4.1.1: You want to move your existing ORD to LAX trip from the 6:00 PM flight to a confirmed seat on the 2:
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline411(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.1.1. Review prose before shipping."""

    name = "eva_airline__4_1_1"
    eva_id = "4.1.1"
    description = "You want to move your existing ORD to LAX trip from the 6:00 PM flight to a confirmed seat on the 2:00 PM flight today, with no same-day change fee charged because of your Gold status, and you want an"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Steven Nelson",
            background=(
                "You want to move your existing ORD to LAX trip from the 6:00 PM flight to a confirmed seat on the 2:00 PM flight today, with no same-day change fee charged because of your Gold status, and you want an aisle seat if available."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to move your existing ORD to LAX trip from the 6:00 PM flight to a confirmed seat on the 2:00 PM flight today, with no same-day change fee charged because of your Gold status, and you want an aisle seat if available."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Can you move me to the earlier flight today?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is I810KI (spelled out as I, eight, one, zero, K, I). Your last name is Nelson. Your first name is Steven.",
                "Your booking: ORD (spelled out as O, R, D) to LAX (spelled out as L, A, X) on 2026-05-14 departing at 18:00.",
                "Must-have: You are rebooked onto a confirmed seat on the 2:00 PM flight today (2026-05-14) from ORD to LAX (not standby).",
                "Must-have: The agent confirms the same-day confirmed change fee is $0 (no $75 fee charged).",
                "Must-have: The agent confirms you have an aisle seat assigned on the new 2:00 PM flight.",
                "When the agent asks for verification, provide your confirmation number and last name exactly as listed in information_required.",
                "When the agent summarizes your request or asks what you want, restate clearly: you need a confirmed seat on the 2:00 PM ORD→LAX flight today and you want an aisle seat.",
                "When the agent presents flight options, only consider options that are exactly ORD→LAX on 2026-05-14 departing at 2:00 PM with a confirmed seat. Immediately reject any option that is not 2:00 PM, is standby-only, changes airports, or changes the travel date.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on ORD to LAX.",
                "Edge case: If the agent offers standby as the way to get on the 2:00 PM flight, decline and insist on a confirmed seat only.",
                "Failure path: If the agent cannot place you on a confirmed seat on the 2:00 PM ORD→LAX flight today with a $0 same-day change fee and an aisle seat after two clear attempts to find and confirm it, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 4.1.2: You want to change your same-day flight from the 9:00 AM SEA to DFW departure to the 3:00 PM departu
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline412(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.1.2. Review prose before shipping."""

    name = "eva_airline__4_1_2"
    eva_id = "4.1.2"
    description = "You want to change your same-day flight from the 9:00 AM SEA to DFW departure to the 3:00 PM departure, as long as you can get a confirmed seat and the total change cost stays under $80."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Brian Hill",
            background=(
                "You want to change your same-day flight from the 9:00 AM SEA to DFW departure to the 3:00 PM departure, as long as you can get a confirmed seat and the total change cost stays under $80."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change your same-day flight from the 9:00 AM SEA to DFW departure to the 3:00 PM departure, as long as you can get a confirmed seat and the total change cost stays under $80."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to change my flight to a later one today.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is JL42CX (spelled out as J, L, four, two, C, X). Your last name is Hill.",
                "Your booking: SEA (spelled out as S, E, A) to DFW (spelled out as D, F, W) on 2026-08-06 departing at 09:00.",
                "Must-have: You must be moved onto a same-day 3:00 PM flight from SEA to DFW with a confirmed seat (not standby).",
                "Must-have: The total out-of-pocket cost the agent quotes for the same-day change (all fees plus any fare difference) must be $80 or less.",
                "If the agent asks for verification details, provide the confirmation code and last name exactly as given in information_required, then wait for the agent to read back the correct reservation (SEA→DFW, 9:00 AM) and confirm it is yours.",
                "When the agent presents any change option(s), evaluate each against the must-have criteria first: it must be SEA→DFW at 3:00 PM today with a confirmed seat, and the total cost must be $80 or less. Reject any option that is not the 3:00 PM flight, is standby, changes airports, or costs more than $80.",
                "If the agent offers the 3:00 PM SEA→DFW confirmed-seat option for $80 or less AND also offers a window seat, accept immediately and clearly tell the agent to book it.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on SEA to DFW only.",
                "Edge case: If the agent suggests standby instead of a confirmed seat, decline and restate that you need a confirmed seat on the 3:00 PM flight.",
                "Failure path: If the agent cannot offer a confirmed-seat 3:00 PM SEA→DFW flight today for a total cost of $80 or less after you have clearly stated those requirements once, say you will keep your original flight for now, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 4.1.3: You want to change your same-day flight from the 4:00 PM departure to the 1:00 PM departure, with no
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline413(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.1.3. Review prose before shipping."""

    name = "eva_airline__4_1_3"
    eva_id = "4.1.3"
    description = "You want to change your same-day flight from the 4:00 PM departure to the 1:00 PM departure, with no same-day change fee, and you want your seat assignment carried over."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Patrick Young",
            background=(
                "You want to change your same-day flight from the 4:00 PM departure to the 1:00 PM departure, with no same-day change fee, and you want your seat assignment carried over."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change your same-day flight from the 4:00 PM departure to the 1:00 PM departure, with no same-day change fee, and you want your seat assignment carried over."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Can you move me to an earlier flight today?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is AZ3UM9 (spelled out as A, Z, three, U, M, nine). Your last name is Young. Your first name is Patrick.",
                "Your booking: DEN (spelled out as D, E, N) to SEA (spelled out as S, E, A) on 2026-06-23 departing at 16:00.",
                "Must-have: You are rebooked onto the 1:00 PM departure today (2026-06-23) for the same origin and destination as currently booked.",
                "Must-have: You are not charged any same-day change fee (total change fee charged must be $0).",
                "Must-have: A seat assignment is confirmed on the new 1:00 PM flight (the agent explicitly confirms you have a seat assigned on the new flight).",
                "If the agent asks for verification details, provide the confirmation code AZ3UM9 and your last name Young, and answer any basic verification questions succinctly.",
                "When the agent asks what you want to change, state clearly that you want to move from the 4:00 PM flight to the 1:00 PM flight today (2026-06-23).",
                "When the agent presents flight options, evaluate them against the must-have criteria only.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on keeping the same origin and destination as your current booking.",
                "Edge case: If the agent suggests standby instead of a confirmed seat on the 1:00 PM flight, decline and restate that you need a confirmed seat on the 1:00 PM departure.",
                "Failure path: If the agent cannot rebook you to the 1:00 PM flight today with a confirmed seat and a $0 change fee after two clear attempts to restate your requirements, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 4.1.5: You want to change to the earliest possible same-day flight home that departs in about 45 minutes, a
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline415(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.1.5. Review prose before shipping."""

    name = "eva_airline__4_1_5"
    eva_id = "4.1.5"
    description = "You want to change to the earliest possible same-day flight home that departs in about 45 minutes, and you need the change completed fast. You also want to keep any same-day change fee under $80."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Joseph Scott",
            background=(
                "You want to change to the earliest possible same-day flight home that departs in about 45 minutes, and you need the change completed fast. You also want to keep any same-day change fee under $80."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change to the earliest possible same-day flight home that departs in about 45 minutes, and you need the change completed fast. You also want to keep any same-day change fee under $80."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, can you move me to the next flight home as soon as possible?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 240QJE (spelled out as two, four, zero, Q, J, E). Your last name is Scott. Your first name is Joseph.",
                "Your booking: BOS (spelled out as B, O, S) to RDU (spelled out as R, D, U) on 2026-04-21 departing at 18:10.",
                "Must-have: You are rebooked onto a flight that departs in approximately 45 minutes from the time of the call (current time 2026-04-21 14:15 ET), i.e., a departure time around 15:00 ET the same day.",
                "Must-have: Any stated same-day change fee charged by the airline is under $80 (fare difference can be additional; this criterion applies only to the change fee itself).",
                "If the agent asks for authentication details, provide the confirmation code and last name exactly as given in information_required, then wait for the agent to read back the reservation and confirm it is yours.",
                "When the agent presents one or more same-day flight options, evaluate each option using ONLY the must-have criteria (departure time around 15:00 ET today and change fee under $80).",
                "If multiple options meet all must-have criteria, choose the option with the earliest departure time. If there is a tie, choose the option with the lower total additional cost (change fee + fare difference).",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on keeping the same origin and destination as your current booking.",
                "Edge case: If the agent suggests standby instead of a confirmed seat, decline and ask for a confirmed seat option that still meets the timing requirement.",
                "Failure path: If the agent cannot offer any flight departing around 15:00 ET today after two rounds of searching/presenting alternatives, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 4.2.1: You want to get added to the standby list for the 12:00 PM flight instead of waiting for your confir
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline421(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.2.1. Review prose before shipping."""

    name = "eva_airline__4_2_1"
    eva_id = "4.2.1"
    description = "You want to get added to the standby list for the 12:00 PM flight instead of waiting for your confirmed 5:00 PM flight, making sure standby is free and your original 5:00 PM booking stays protected if"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Michelle Baker",
            background=(
                "You want to get added to the standby list for the 12:00 PM flight instead of waiting for your confirmed 5:00 PM flight, making sure standby is free and your original 5:00 PM booking stays protected if you don’t clear standby."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to get added to the standby list for the 12:00 PM flight instead of waiting for your confirmed 5:00 PM flight, making sure standby is free and your original 5:00 PM booking stays protected if you don’t clear standby."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Can you put me on standby for the noon flight?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is W19LAE (spelled out as W, one, nine, L, A, E). Your last name is Baker.",
                "Your booking: SFO (spelled out as S, F, O) to LAX (spelled out as L, A, X) on 2026-07-08 departing at 17:00.",
                "Must-have: You are added to the standby list for the 12:00 PM flight (same route as your current trip) and the agent explicitly confirms you are now on the standby list.",
                "Must-have: The agent explicitly confirms there is no charge for being added to standby (total additional cost is $0).",
                "Must-have: The agent explicitly confirms your original confirmed 5:00 PM flight remains protected/kept as-is if you do not clear standby.",
                "When the agent asks for booking details to locate your reservation, provide your confirmation code and last name exactly as given in information_required.",
                "When the agent asks which earlier flight you want standby for, specify: the 12:00 PM flight (noon) today; do not ask for any other time unless the agent says the 12:00 PM standby list is unavailable.",
                "If the agent can add you to standby for the 12:00 PM flight with $0 cost and keep your 5:00 PM flight protected, ask them to do it now and stay on the line until they confirm it has been completed.",
                "Edge case: If the agent asks if you want to switch (rebook) to the 12:00 PM flight instead of standby, decline and restate you only want to be added to standby while keeping the 5:00 PM flight as backup.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on keeping your original airports.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Failure path: If the agent cannot add you to standby for the 12:00 PM flight while keeping your original 5:00 PM flight protected and with $0 additional cost after you have clearly restated those needs one time, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 4.2.4: You want to keep your confirmed 5:00 PM flight as a backup, but also get added to standby for both t
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline424(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.2.4. Review prose before shipping."""

    name = "eva_airline__4_2_4"
    eva_id = "4.2.4"
    description = "You want to keep your confirmed 5:00 PM flight as a backup, but also get added to standby for both the 11:00 AM and 1:00 PM flights so you have the best chance of getting out earlier."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Timothy Robinson",
            background=(
                "You want to keep your confirmed 5:00 PM flight as a backup, but also get added to standby for both the 11:00 AM and 1:00 PM flights so you have the best chance of getting out earlier."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to keep your confirmed 5:00 PM flight as a backup, but also get added to standby for both the 11:00 AM and 1:00 PM flights so you have the best chance of getting out earlier."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Can you add me to standby for an earlier flight today?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is CR27HC (spelled out as C, R, two, seven, H, C). Your last name is Robinson. Your first name is Timothy.",
                "Your booking: SFO (spelled out as S, F, O) to LAX (spelled out as L, A, X) on 2026-11-19 departing at 17:00.",
                "Must-have: You are added to the standby list for the 11:00 AM departure (same origin and destination as your current booking).",
                "Must-have: You are added to the standby list for the 1:00 PM departure (same origin and destination as your current booking).",
                "Must-have: Your currently confirmed 5:00 PM flight remains confirmed and protected as a fallback (it is not canceled and not replaced by standby-only).",
                "If the agent asks for identification details to locate your booking, provide your confirmation code and last name exactly as given, then wait for the agent to confirm they found your reservation.",
                "After the reservation is found, clearly state you want to be on standby for BOTH the 11:00 AM and 1:00 PM flights, while keeping your confirmed 5:00 PM flight as your backup.",
                "When the agent describes what they can do, evaluate it against all must-have criteria: you must be on standby for both earlier flights and still have the 5:00 PM confirmed as fallback.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on your original airports.",
                "Edge case: If the agent asks you to choose only one standby flight, insist you want both 11:00 AM and 1:00 PM; if they still refuse after a second attempt, follow the failure_condition.",
                "Failure path: If after two clear attempts the agent cannot add you to standby for both the 11:00 AM and 1:00 PM flights while keeping your 5:00 PM flight confirmed, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 4.2.5: You want to be added to standby for the 10:00 AM flight and have the agent confirm your Gold-elite p
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline425(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 4.2.5. Review prose before shipping."""

    name = "eva_airline__4_2_5"
    eva_id = "4.2.5"
    description = "You want to be added to standby for the 10:00 AM flight and have the agent confirm your Gold-elite priority standby placement, while keeping your original confirmed flight as a backup."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Rebecca Walker",
            background=(
                "You want to be added to standby for the 10:00 AM flight and have the agent confirm your Gold-elite priority standby placement, while keeping your original confirmed flight as a backup."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to be added to standby for the 10:00 AM flight and have the agent confirm your Gold-elite priority standby placement, while keeping your original confirmed flight as a backup."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Can you put me on standby for the 10:00 AM flight?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is NTJBNE (spelled out as N, T, J, B, N, E). Your last name is Walker. Your first name is Rebecca.",
                "Your booking: ATL (spelled out as A, T, L) to DCA (spelled out as D, C, A) on 2026-06-02 departing at 13:30.",
                "Must-have: You are successfully added to standby specifically for the 10:00 AM flight, and the agent confirms your Gold-elite priority standby list position as #2 (position 2 on the list).",
                "Must-have: The agent confirms your seat assignment and any checked bags from your current booking will be transferred/ready to transfer if you clear the standby flight (i.e., you will not lose your seat/bag arrangements by being on standby).",
                "Must-have: The agent confirms your original booked flight remains confirmed and protected as a fallback if you do not clear standby.",
                "After the agent authenticates you, if the agent asks which flight you mean, specify: the 10:00 AM flight on your same route/date as your current booking; do not introduce any other times.",
                "If the agent can add you to standby for the 10:00 AM flight, do not debate policies; proceed by asking the agent to confirm (a) your standby list position number and (b) that your original confirmed booking stays protected.",
                "If the agent confirms standby was added but does NOT confirm your standby list position, ask exactly once: 'What number am I on the standby list?' and wait for a numeric position.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on your original airports.",
                "Edge case: If the agent asks for payment information or tries to charge a fee for being added to standby, say you only want to be placed on standby (not to buy a new ticket) and ask them to proceed without charging; if they insist a charge is required to do anything, follow the failure condition.",
                "Failure path: If the agent cannot add you to standby for the 10:00 AM flight or cannot confirm both (1) your standby list position number and (2) that your original booking remains protected after two clear attempts to get those confirmations, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )
