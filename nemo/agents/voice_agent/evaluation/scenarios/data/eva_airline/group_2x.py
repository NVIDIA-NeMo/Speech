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
# eva 2.1.2: You want help after your ATL to SEA flight was canceled: get rebooked onto the first available fligh
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline212(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.1.2. Review prose before shipping."""

    name = "eva_airline__2_1_2"
    eva_id = "2.1.2"
    description = "You want help after your ATL to SEA flight was canceled: get rebooked onto the first available flight tomorrow with a confirmed seat, and get both an overnight hotel voucher and a $25 meal voucher for"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Ava Murphy",
            background=(
                "You want help after your ATL to SEA flight was canceled: get rebooked onto the first available flight tomorrow with a confirmed seat, and get both an overnight hotel voucher and a $25 meal voucher for the overnight disruption."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want help after your ATL to SEA flight was canceled: get rebooked onto the first available flight tomorrow with a confirmed seat, and get both an overnight hotel voucher and a $25 meal voucher for the overnight disruption."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight got canceled—can you get me rebooked and help with a hotel?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is PP248Z (spelled out as P, P, two, four, eight, Z). Your last name is Murphy. Your first name is Ava.",
                "Your booking: ATL (spelled out as A, T, L) to SEA (spelled out as S, E, A) on 2026-06-07 departing at 18:45 (flight SK518 (spelled out as S, K, five, one, eight)).",
                "Must-have: You are rebooked onto the first available flight tomorrow morning (2026-06-08) from ATL to SEA with a confirmed seat (not standby).",
                "Must-have: You receive a hotel voucher for 1 night for the overnight delay/disruption.",
                "Must-have: You receive a $25 meal voucher for the overnight delay/disruption.",
                "If the agent asks to look up your booking, provide your confirmation number and last name exactly as requested.",
                "When the agent presents rebooking options, evaluate each option against the must-have criteria first: it must be for 2026-06-08 ATL→SEA and must be a confirmed seat (not standby). If none meet this, tell the agent you need a confirmed seat ATL→SEA on 2026-06-08 and ask them to keep searching.",
                "If the agent offers the next available flight tomorrow at 7:15 AM (or any other tomorrow option) that meets the must-haves, then check the nice-to-have arrival time (before 2:00 PM PT).",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on ATL to SEA only.",
                "Edge case: If the agent suggests standby instead of a confirmed seat, decline standby and ask for confirmed-seat options only.",
                "Failure path: If, after 2 rounds of searching/clarifying, the agent cannot book any confirmed-seat ATL→SEA flight on 2026-06-08 OR cannot provide either the 1-night hotel voucher or the $25 meal voucher, say you will figure it out yourself, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.1.6: You want to cancel your canceled LAX to Seattle trip and get a full refund back to your original pay
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline216(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.1.6. Review prose before shipping."""

    name = "eva_airline__2_1_6"
    eva_id = "2.1.6"
    description = "You want to cancel your canceled LAX to Seattle trip and get a full refund back to your original payment method because you no longer need to travel."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Zoe Brown",
            background=(
                "You want to cancel your canceled LAX to Seattle trip and get a full refund back to your original payment method because you no longer need to travel."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel your canceled LAX to Seattle trip and get a full refund back to your original payment method because you no longer need to travel."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight got canceled and I need a refund.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is YLCNSG (spelled out as Y, L, C, N, S, G). Your last name is Brown. Your first name is Zoe.",
                "Your booking: LAX (spelled out as L, A, X) to SEA (spelled out as S, E, A) on 2026-09-10 departing at 16:10 (flight SK490 (spelled out as S, K, four, nine, zero)).",
                "Must-have: The agent confirms a full cash refund is processed back to the original payment method (not a travel credit or voucher).",
                "Must-have: The agent confirms the refund includes all paid ancillary fees on the booking (for example, any seat fees and checked bag fees).",
                "Must-have: The agent provides a concrete refund confirmation/reference (e.g., a refund confirmation number or clearly states the refund has been processed for confirmation code YLCNSG).",
                "When the agent asks for identification details, provide the confirmation code YLCNSG and last name Brown.",
                "If the agent offers rebooking options, decline rebooking and restate that you do not want to travel anymore because the event was canceled, and you want a full refund back to the original payment method.",
                "If the agent offers a travel credit or voucher instead of a cash refund, reject it and ask them to process a refund back to the original payment method due to the flight being canceled.",
                "Edge case: If the agent asks if you still want to travel on different dates or times, say no and repeat that you only want a refund to the original payment method.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked (LAX and SEA), decline and restate that you are not traveling and only want a refund.",
                "Edge case: If the agent suggests standby, decline and restate that you are not traveling and only want a refund.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Failure path: If the agent will not process a cash refund to the original payment method after you clearly decline rebooking/credit and restate your request two times, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.2.2: You want to get rebooked from your heavily delayed JFK to LAX flight onto an option that still gets
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline222(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.2.2. Review prose before shipping."""

    name = "eva_airline__2_2_2"
    eva_id = "2.2.2"
    description = "You want to get rebooked from your heavily delayed JFK to LAX flight onto an option that still gets you into LAX by 11:00 PM Pacific tonight, and you also want the $15 meal voucher you’re owed for the"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Hannah Foster",
            background=(
                "You want to get rebooked from your heavily delayed JFK to LAX flight onto an option that still gets you into LAX by 11:00 PM Pacific tonight, and you also want the $15 meal voucher you’re owed for the long delay."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to get rebooked from your heavily delayed JFK to LAX flight onto an option that still gets you into LAX by 11:00 PM Pacific tonight, and you also want the $15 meal voucher you’re owed for the long delay."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight is delayed and I need to switch to a different flight.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your last name is Foster. Your first name is Hannah.",
                "Your booking: JFK (spelled out as J, F, K) to LAX (spelled out as L, A, X) on 2026-08-08 departing at 18:30.",
                "Must-have: You are rebooked from JFK to LAX for travel today (2026-08-08) on an itinerary that arrives in LAX no later than 11:00 PM Pacific.",
                "Must-have: You receive a meal voucher issued for this disruption in the amount consistent with a 5+ hour delay (i.e., a $15 meal voucher), and the agent provides a voucher code or other concrete issuance reference.",
                "If the agent asks to look up the reservation, provide the confirmation code and last name exactly as given, then wait for the agent to read back the correct trip details before discussing preferences.",
                "When the agent presents rebooking options, evaluate each option against the must-have criteria first: (a) arrives LAX by 11:00 PM Pacific today and (b) you will receive a $15 meal voucher with a code/reference.",
                "If at least one option meets the must-have criteria and is also direct, choose the direct option that departs sooner (earlier departure time). Accept it when the agent asks for confirmation.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on JFK to LAX only.",
                "Edge case: If the agent suggests standby instead of a confirmed rebooking, decline standby and ask for confirmed rebooking options that meet the must-have arrival time.",
                "Failure path: If, after two total searches/attempts, the agent cannot offer any rebooking that arrives LAX by 11:00 PM Pacific today OR the agent refuses/does not issue a meal voucher with a code/reference, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.2.4: You want to confirm the details of your delayed BOS to DFW flight, get the $12 meal voucher you’re e
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline224(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.2.4. Review prose before shipping."""

    name = "eva_airline__2_2_4"
    eva_id = "2.2.4"
    description = "You want to confirm the details of your delayed BOS to DFW flight, get the $12 meal voucher you’re entitled to, and keep your original booking unchanged so you can wait for the same flight."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Natalie Cruz",
            background=(
                "You want to confirm the details of your delayed BOS to DFW flight, get the $12 meal voucher you’re entitled to, and keep your original booking unchanged so you can wait for the same flight."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to confirm the details of your delayed BOS to DFW flight, get the $12 meal voucher you’re entitled to, and keep your original booking unchanged so you can wait for the same flight."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I’m calling about my delayed flight—can you tell me the updated details?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is N53W23 (spelled out as N, five, three, W, two, three). Your last name is Cruz. Your first name is Natalie.",
                "Your booking: BOS (spelled out as B, O, S) to DFW (spelled out as D, F, W) on 2026-06-30 departing at 14:30.",
                "Must-have: Your existing flight booking remains unchanged (no rebooking, no cancellation, and no change to origin/destination: BOS → DFW).",
                "Must-have: You receive a meal voucher in the amount of $12, and the agent provides a voucher code or other concrete issuance confirmation.",
                "Must-have: You receive the updated departure time and current gate information for flight SK610 on 2026-06-30 (if gate is not assigned yet, the agent must explicitly say it is not assigned and tell you where/when to check for updates).",
                "After the agent authenticates you, clearly state you are planning to wait for your original flight and you only need the updated departure time/gate and any assistance available due to the delay.",
                "If the agent offers rebooking options, decline them once and restate that you want to keep the original flight unchanged and just want the updated departure/gate information and the meal voucher.",
                "If the agent provides updated departure time/gate info but does not issue a $12 meal voucher, ask one time: \"Can you issue the meal voucher I’m eligible for because of the delay?\"",
                "Edge case: If the agent asks for your confirmation number and last name, provide: N53W23 and Cruz.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on your original airports (BOS to DFW).",
                "Edge case: If the agent suggests standby, decline and say you will wait for your original confirmed flight instead.",
                "Edge case: If the agent offers a refund or cancellation, decline and restate you are not cancelling and want to keep the booking unchanged.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Failure path: If after two clear requests the agent cannot provide an updated departure time/gate status and cannot issue (or cannot confirm issuance of) a $12 meal voucher while also keeping your booking unchanged, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.2.5: You want to get a meal voucher because your MIA to JFK flight is delayed, and you want to understand
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline225(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.2.5. Review prose before shipping."""

    name = "eva_airline__2_2_5"
    eva_id = "2.2.5"
    description = "You want to get a meal voucher because your MIA to JFK flight is delayed, and you want to understand where you can use it."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Lisa Peterson",
            background=(
                "You want to get a meal voucher because your MIA to JFK flight is delayed, and you want to understand where you can use it."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to get a meal voucher because your MIA to JFK flight is delayed, and you want to understand where you can use it."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight is delayed—can I get a meal voucher?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is NHNRTO (spelled out as N, H, N, R, T, O). Your last name is Peterson.",
                "Your booking: MIA (spelled out as M, I, A) to JFK (spelled out as J, F, K) on 2026-04-25 departing at 14:30 (flight SK255 (spelled out as S, K, two, five, five)).",
                "Must-have: Receive a meal voucher issued for the delay on flight SK255 MIA→JFK (the agent must confirm it has been issued, not just that you are eligible).",
                "Must-have: The agent confirms the voucher is valid for use at airport terminal restaurants (i.e., you can use it in the terminal).",
                "If the agent asks to verify your booking, provide your confirmation code and last name exactly as requested.",
                "If the agent says you are eligible for a meal voucher, ask what the voucher amount will be and where it can be used before accepting anything.",
                "If the agent offers a $15 voucher and confirms it can be used at terminal restaurants, accept and do not negotiate further.",
                "Edge case: If the agent asks if you still want to stay on the delayed flight, say yes, you are staying on the same flight.",
                "Edge case: If the agent offers rebooking, standby, refunds, or cancellation, decline and repeat that you are only calling about a meal voucher for the delay.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Failure path: If the agent cannot (after two total tries) issue any meal voucher or cannot provide any voucher code/identifier showing it was issued, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.3.2: You want to keep your trip after the airline moved your flight to a 12:30 PM departure, and you want
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline232(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.3.2. Review prose before shipping."""

    name = "eva_airline__2_3_2"
    eva_id = "2.3.2"
    description = "You want to keep your trip after the airline moved your flight to a 12:30 PM departure, and you want a $12 meal voucher for the schedule disruption while making sure your seat and any checked bags sta"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Rachel Bennett",
            background=(
                "You want to keep your trip after the airline moved your flight to a 12:30 PM departure, and you want a $12 meal voucher for the schedule disruption while making sure your seat and any checked bags stay confirmed on the updated flight."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to keep your trip after the airline moved your flight to a 12:30 PM departure, and you want a $12 meal voucher for the schedule disruption while making sure your seat and any checked bags stay confirmed on the updated flight."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi—my flight time changed and I need help confirming everything.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 7MMHTS (spelled out as seven, M, M, H, T, S). Your last name is Bennett. Your first name is Rachel.",
                "Your booking: ORD (spelled out as O, R, D) to LGA (spelled out as L, G, A) on 2026-09-06 departing at 12:30.",
                "Must-have: Your booking remains confirmed on the updated departure time of 12:30 PM (you are not moved to a different airport).",
                "Must-have: You receive a meal voucher in the amount of $12 for the disruption, and the agent provides a voucher code or other specific voucher reference as proof it was issued.",
                "Must-have: The agent confirms your ancillaries are still in place on the updated flight: your seat assignment is confirmed and your checked baggage (if any) remains attached to the trip.",
                "After the agent authenticates you, state that you are okay with the new 12:30 PM departure and you want to keep the booking, but you are requesting a $12 meal voucher due to the schedule disruption.",
                "If the agent offers alternative flights, decline them and repeat that you want to keep the 12:30 PM flight unless the agent says your current booking cannot remain confirmed; only consider alternatives if the agent explicitly says they cannot keep you confirmed on 12:30 PM.",
                "If alternatives must be considered (only if the 12:30 PM cannot stay confirmed), choose the option that keeps the same origin and destination airports and has the earliest departure time on the original travel date; if multiple options have the same departure time, choose the one with the fewest stops.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on your original airports.",
                "Edge case: If the agent suggests standby instead of a confirmed booking, decline standby and insist on staying confirmed on the 12:30 PM flight (or a confirmed alternative only if 12:30 PM cannot be kept).",
                "Failure path: If, after two clear attempts, the agent cannot keep your booking confirmed at 12:30 PM OR cannot issue a $12 meal voucher with a voucher code/reference OR cannot confirm your seat and baggage are attached to the updated flight, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.3.4: You want to cancel your trip because the airline moved your flight much later, and you want a full r
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline234(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.3.4. Review prose before shipping."""

    name = "eva_airline__2_3_4"
    eva_id = "2.3.4"
    description = "You want to cancel your trip because the airline moved your flight much later, and you want a full refund back to your original payment method, including any seat and bag fees."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Ashley Howard",
            background=(
                "You want to cancel your trip because the airline moved your flight much later, and you want a full refund back to your original payment method, including any seat and bag fees."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel your trip because the airline moved your flight much later, and you want a full refund back to your original payment method, including any seat and bag fees."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, my flight time got changed a lot and I need to cancel and get a refund.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is DX8W4I (spelled out as D, X, eight, W, four, I). Your last name is Howard. Your first name is Ashley.",
                "Your booking: SFO (spelled out as S, F, O) to SEA (spelled out as S, E, A) on 2026-08-25 departing at 22:30.",
                "Must-have: You receive a full cash refund back to the original payment method for the entire impacted trip because the new schedule no longer works.",
                "Must-have: The refund explicitly includes all ancillary fees you paid on this booking (at minimum: any paid seat selection and any checked-bag fees).",
                "Must-have: The agent confirms the refund has already been processed (not just promised) and provides a concrete confirmation/reference of the completed refund action (e.g., refund confirmation/reference number or a statement that the refund is processed for confirmation code DX8W4I with the total refunded amount).",
                "After the agent finds your booking and explains the schedule change, clearly state that the new later flight no longer works and you want to cancel and get a full refund back to the original payment method.",
                "If the agent offers rebooking options, evaluate them against your needs: if any option restores an arrival time close enough that your original reason for travel still works, ask for the earliest available option on the same route and confirm it is free of change fees due to the schedule change. If none work, explicitly decline rebooking and restate you want a full refund instead.",
                "If the agent offers anything other than a cash refund to the original payment method (for example, travel credit), reject it once and restate you require a full refund back to the original payment method because the airline changed the schedule significantly.",
                "Edge case: If the agent asks for your confirmation number and last name, provide DX8W4I and Howard exactly.",
                "Edge case: If the agent suggests travel credit instead of a refund, decline once and restate you need the refund back to the original payment method due to the schedule change; if they still cannot do it, follow the failure_condition.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on keeping the original airports.",
                "Edge case: If the agent suggests standby as a solution, decline and restate you want a refund.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Failure path: If the agent refuses or is unable to process a full refund to the original payment method (including seat and bag fees) after you have clearly requested it and asked them once to re-check options, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.4.1: You need to get rebooked after your DEN to JFK flight was canceled, making sure you still arrive in
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline241(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.4.1. Review prose before shipping."""

    name = "eva_airline__2_4_1"
    eva_id = "2.4.1"
    description = "You need to get rebooked after your DEN to JFK flight was canceled, making sure you still arrive in New York by 10:00 PM Eastern today, stay in Main Cabin or better, and receive the $15 meal voucher y"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Nathan Whitfield",
            background=(
                "You need to get rebooked after your DEN to JFK flight was canceled, making sure you still arrive in New York by 10:00 PM Eastern today, stay in Main Cabin or better, and receive the $15 meal voucher you’re owed for the disruption."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You need to get rebooked after your DEN to JFK flight was canceled, making sure you still arrive in New York by 10:00 PM Eastern today, stay in Main Cabin or better, and receive the $15 meal voucher you’re owed for the disruption."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight just got canceled after we had to turn back—can you get me rebooked?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 5KR950 (spelled out as five, K, R, nine, five, zero). Your last name is Whitfield. Your first name is Nathan.",
                "Your booking: DEN (spelled out as D, E, N) to JFK (spelled out as J, F, K) on 2026-06-14 departing at 10:15.",
                "Must-have: You must be rebooked to travel from DEN to JFK (no airport changes) on 2026-06-14 with an arrival time no later than 10:00 PM ET.",
                "Must-have: Your rebooked itinerary must be in Main Cabin or a better cabin class (no downgrade below Main Cabin).",
                "Must-have: You must receive a meal voucher worth $15, and the agent must provide a voucher confirmation/code or other concrete issuance proof during the call.",
                "Must-have: The agent must confirm the rebooking is completed (not just proposed) by providing a confirmed new itinerary (flight number(s) and times) and stating it is ticketed/confirmed under your booking.",
                "After the agent authenticates you, briefly explain that you were already airborne and the flight returned to DEN and was canceled, and that you need to get to JFK by 10:00 PM ET today.",
                "When the agent presents rebooking options, evaluate each option against the must-have criteria first (DEN→JFK, arrive by 10:00 PM ET on 2026-06-14, Main Cabin or better). Discard any option that violates any must-have criterion.",
                "If the agent offers at least one option that meets all must-have criteria and is nonstop, select the nonstop option even if it departs later than connecting options, as long as it still arrives by 10:00 PM ET.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on DEN to JFK only.",
                "Edge case: If the agent offers standby instead of a confirmed seat, decline and ask for a confirmed rebooking option that meets the must-have criteria.",
                "Failure path: If, after the agent has searched and presented alternatives at least two separate times, they cannot offer any DEN→JFK itinerary on 2026-06-14 that arrives by 10:00 PM ET in Main Cabin or better, say you can’t proceed and say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 2.4.2: You want to confirm what is happening with your MSP to LAX flight after it returned to the gate, mak
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline242(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 2.4.2. Review prose before shipping."""

    name = "eva_airline__2_4_2"
    eva_id = "2.4.2"
    description = "You want to confirm what is happening with your MSP to LAX flight after it returned to the gate, make sure you are staying on the same re-departing flight, and receive the correct meal voucher for the"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Clara Johansson",
            background=(
                "You want to confirm what is happening with your MSP to LAX flight after it returned to the gate, make sure you are staying on the same re-departing flight, and receive the correct meal voucher for the long delay."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to confirm what is happening with your MSP to LAX flight after it returned to the gate, make sure you are staying on the same re-departing flight, and receive the correct meal voucher for the long delay."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi—can you tell me what’s going on with my flight that came back to Minneapolis? We took off at the scheduled time around 12:30, and then had to turn around'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is KUT629 (spelled out as K, U, T, six, two, nine). Your last name is Johansson. Your first name is Clara.",
                "Your booking: MSP (spelled out as M, S, P) to LAX (spelled out as L, A, X) on 2026-05-02 departing at 12:30.",
                "Must-have: You remain booked on flight SK418 from MSP to LAX re-departing today (2026-05-02) rather than being moved to a different flight.",
                "Must-have: The agent tells you the updated re-departure time for SK418 and provides the updated estimated arrival time into LAX",
                "Must-have: The agent confirms your gate information for SK418 (either the current gate number if available, or explicitly states they have checked and there is no gate update available yet).",
                "Must-have: The agent issues a meal voucher for the disruption and provides a voucher code or other concrete issuance confirmation, and the voucher value is $15.",
                "Must-have: The agent confirms your seat assignment remains unchanged on the re-departure (i.e., you keep the same seat you already had on SK418).",
                "If the agent asks for verification details, provide your confirmation code and last name exactly as given in information_required, and answer any follow-up identity questions briefly.",
                "When the agent explains the situation or presents any options (stay on the flight, rebook, refund, etc.), evaluate them against all must-have criteria.",
                "If the agent confirms you are staying on SK418 and provides the updated departure/arrival time but has not yet issued the meal voucher, ask once: \"Can you also issue the meal voucher for this delay and give me the voucher code?\"",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on MSP to LAX only.",
                "Edge case: If the agent suggests standby as a solution, decline and repeat that you want to stay confirmed on the SK418 re-departure.",
                "Failure path: If the agent cannot confirm you are still on SK418 and cannot provide any clear flight status/update after 3 attempts to clarify (including asking for your confirmation code and last name), say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )
