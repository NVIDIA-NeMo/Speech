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
# eva 6.1.1: You want to get rebooked to Seattle on the first available flight tomorrow with a confirmed seat, an
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline611(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 6.1.1. Review prose before shipping."""

    name = "eva_airline__6_1_1"
    eva_id = "6.1.1"
    description = "You want to get rebooked to Seattle on the first available flight tomorrow with a confirmed seat, and you want a hotel voucher for tonight since your flight was canceled."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Gregory Walker",
            background=(
                "You want to get rebooked to Seattle on the first available flight tomorrow with a confirmed seat, and you want a hotel voucher for tonight since your flight was canceled."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to get rebooked to Seattle on the first available flight tomorrow with a confirmed seat, and you want a hotel voucher for tonight since your flight was canceled."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight to Seattle got canceled and I need help getting rebooked.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is SOCATW (spelled out as S, O, C, A, T, W). Your last name is Walker.",
                "Your booking: OAK (spelled out as O, A, K) to SEA (spelled out as S, E, A) on 2026-07-21 departing at 16:10.",
                "Must-have: You are rebooked with a CONFIRMED seat on the first available flight tomorrow (2026-07-22) departing at 8:15 AM to SEA (Seattle).",
                "Must-have: You receive a hotel voucher for exactly 1 night for tonight due to the cancellation, and the agent confirms it has been issued (not just promised) with a voucher/reference provided.",
                "Must-have: Receive a meal voucher code from the agent",
                "When the agent asks for verification details, provide the confirmation code and last name exactly as listed in information_required. Do not provide extra details unless asked.",
                "When the agent presents rebooking options, evaluate each option against the must-have criteria first: it must be for 2026-07-22, must depart at 8:15 AM, must arrive in SEA, and must be a confirmed seat. If any option fails any must-have item, reject it and restate that you need the first available confirmed seat tomorrow at 8:15 AM to SEA and a 1-night hotel voucher.",
                "If the agent offers an option that meets all must-have criteria but does not meet nice-to-haves (e.g., not direct, or cannot get you to SEA today), ask exactly ONE time: 'Is there any way to get to Seattle today, or a direct flight tomorrow instead?'",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on flying to SEA (Seattle) as originally planned.",
                "Edge case: If the agent offers standby instead of a confirmed seat, decline and restate that you need a confirmed seat.",
                "Failure path: If the agent cannot rebook you to a confirmed seat on the 2026-07-22 8:15 AM flight to SEA after you clearly restate that requirement one time, or if the agent refuses or is unable to issue a 1-night hotel voucher after rebooking, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 6.1.4: You're supposed to fly tomorrow but you want to get rebooked today to the Orange County area, and if
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline614(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 6.1.4. Review prose before shipping."""

    name = "eva_airline__6_1_4"
    eva_id = "6.1.4"
    description = "You're supposed to fly tomorrow but you want to get rebooked today to the Orange County area, and if there are no seats to Santa Ana (SNA), you’re willing to fly into a nearby airport that’s within ab"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Donna Young",
            background=(
                "You're supposed to fly tomorrow but you want to get rebooked today to the Orange County area, and if there are no seats to Santa Ana (SNA), you’re willing to fly into a nearby airport that’s within about a 45-minute drive."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You're supposed to fly tomorrow but you want to get rebooked today to the Orange County area, and if there are no seats to Santa Ana (SNA), you’re willing to fly into a nearby airport that’s within about a 45-minute drive."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I need help rebooking my flight for today.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is R0SDRU (spelled out as R, zero, S, D, R, U). Your last name is Young. Your first name is Donna.",
                "Your booking: SEA (spelled out as S, E, A) to SNA (spelled out as S, N, A) on 2026-06-17 departing at 14:10.",
                "Must-have: You must be rebooked onto a flight departing on 2026-06-16 that arrives in the Orange County area today (arrives either SNA, LGB, or LAX on 2026-06-16).",
                "Must-have: If the destination is not SNA, it must be an alternate airport within about a 45-minute drive of SNA; you will accept only LGB or LAX as alternates.",
                "After the agent authenticates you and presents rebooking options, evaluate each option: (a) does it arrive on 2026-06-16, and (b) is the arrival airport SNA OR (if not SNA) LGB/LAX only.",
                "If the agent offers any option to SNA arriving on 2026-06-16, accept the SNA option that arrives earliest (if multiple, pick the earliest arrival time).",
                "If the agent offers no SNA options but offers LGB and/or LAX arriving on 2026-06-16, ask exactly one time: \"Can you double-check if anything to SNA opens up today, even later?\"",
                "Edge case: If the agent asks for your confirmation number and last name, provide exactly: confirmation number R0SDRU and last name Young.",
                "Edge case: If the agent suggests flying to airports other than SNA, LGB, or LAX (for example BUR, ONT, SAN), decline and restate that you can only do SNA, or alternates LGB/LAX.",
                "Edge case: Do not accept travel on a different date than 2026-06-16 under any circumstances.",
                "Edge case: If the agent offers standby instead of a confirmed seat, decline standby and ask for confirmed-seat options only.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Failure path: If the agent cannot provide any rebooking option that arrives on 2026-06-16 to SNA, LGB, or LAX after two clear search attempts (the initial search and one additional search you request), say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 6.3.1: You want to get help with your cancelled DCA to LAX trip because you were rebooked into an unaccepta
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline631(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 6.3.1. Review prose before shipping."""

    name = "eva_airline__6_3_1"
    eva_id = "6.3.1"
    description = "You want to get help with your cancelled DCA to LAX trip because you were rebooked into an unacceptable itinerary, and you want to be transferred to a live agent or supervisor who can handle it."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Sharon Scott",
            background=(
                "You want to get help with your cancelled DCA to LAX trip because you were rebooked into an unacceptable itinerary, and you want to be transferred to a live agent or supervisor who can handle it."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to get help with your cancelled DCA to LAX trip because you were rebooked into an unacceptable itinerary, and you want to be transferred to a live agent or supervisor who can handle it."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight got canceled and the rebooking they gave me isn’t acceptable—can you help?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is E66N08 (spelled out as E, six, six, N, zero, eight). Your last name is Scott. Your first name is Sharon.",
                "Your booking: DCA (spelled out as D, C, A) to LAX (spelled out as L, A, X) on 2026-05-10 departing at 09:10.",
                "Must-have: You are transferred to a live human agent or supervisor during this contact (not just given advice to call back).",
                "Must-have: Before the transfer happens, the agent clearly summarizes the issue as: your Business Class direct flight DCA→LAX was cancelled, you were rebooked onto DCA→DFW→LAX with a long layover and Economy seating, and no better same-day alternatives were found after searching.",
                "Must-have: Receive a meal voucher code from the agent",
                "If the agent asks for identification details to access the booking, provide the confirmation code and last name exactly as given in information_required, and answer any basic verification questions briefly.",
                "Once the agent acknowledges the cancellation/rebooking situation, state clearly that the DCA→DFW→LAX rebooking with a 6-hour layover and Economy seats is not acceptable because you paid for Business and a nonstop flight.",
                "If the agent presents any new flight options, evaluate them: (a) they do NOT replace your must-have criteria (you still require a live-agent/supervisor transfer), and (b) you prefer options that meet the nice-to-haves (Business and/or nonstop).",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than DCA and LAX, decline and insist on DCA to LAX only.",
                "Edge case: If the agent offers standby as a solution, decline standby and repeat that you want a supervisor/live agent because the current rebooking is unacceptable.",
                "Failure path: If the agent refuses to transfer you to a live agent/supervisor or cannot complete the transfer after two clear requests, say you will call back and end the call.",
                "Escalation: If the agent cannot offer a Business Class or nonstop alternative after one request for better options, or if you feel the agent is not addressing the downgrade/layover problem, you should explicitly request a supervisor or live human agent immediately. If you already requested it once and the agent does not proceed, request it one more time and then follow the failure_condition.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 6.3.4: You want the airline to fix the situation caused by your delayed flight by getting you rebooked on t
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline634(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 6.3.4. Review prose before shipping."""

    name = "eva_airline__6_3_4"
    eva_id = "6.3.4"
    description = "You want the airline to fix the situation caused by your delayed flight by getting you rebooked on the next available flight at no extra cost and receiving a $15 meal voucher, and you also want to see"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Dennis Baker",
            background=(
                "You want the airline to fix the situation caused by your delayed flight by getting you rebooked on the next available flight at no extra cost and receiving a $15 meal voucher, and you also want to see if they can offer any extra compensation for the trouble."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want the airline to fix the situation caused by your delayed flight by getting you rebooked on the next available flight at no extra cost and receiving a $15 meal voucher, and you also want to see if they can offer any extra compensation for the trouble."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight is delayed four hours and I need help.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is A83QV2 (spelled out as A, eight, three, Q, V, two). Your last name is Baker. Your first name is Dennis.",
                "Your booking: SFO (spelled out as S, F, O) to SEA (spelled out as S, E, A) on 2026-06-05 departing at 15:00 (flight SK745 (spelled out as S, K, seven, four, five)).",
                "Must-have: Receive a meal voucher worth exactly $15 for the delay, with a voucher code or other concrete issuance confirmation provided during the call",
                "Must-have: Be rebooked onto the next available flight option the agent can book for your same trip (same origin and destination as originally booked) with no additional charges, and the agent must confirm the rebooking is completed with updated flight details and a confirmation/reference that the change is finalized",
                "If the agent asks for verification details, provide your confirmation code and last name exactly as given, and answer any basic questions about which flight you are calling about (SK745) and the issue (about a 4-hour delay).",
                "Early in the call (right after stating the delay problem), state you are very frustrated and explicitly mention that you are considering posting about the experience on social media unless this is handled properly.",
                "When the agent presents solutions or options, evaluate them against the must-have criteria first: you need BOTH a $15 meal voucher with a code AND a completed fee-free rebooking onto the next available flight the agent can book for the same route.",
                "Edge case: If the agent asks if there is anything else they can help with after the must-have criteria are satisfied, say 'No, that’s all' and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on keeping your original origin and destination airports.",
                "Edge case: If the agent offers standby instead of a confirmed rebooking, decline standby and ask for a confirmed seat on the next available flight instead.",
                "Failure path: If after two clear attempts the agent cannot provide BOTH a $15 meal voucher with a code AND a completed no-cost rebooking on the same origin/destination, and they also do not transfer you to a supervisor when you request it, say goodbye and end the call.",
                "Escalation: If the agent refuses or cannot issue the $15 meal voucher, or cannot complete a no-cost rebooking after a second attempt to search alternatives, ask once to be transferred to a supervisor/live agent for additional help and compensation options; if transferred, stop and end the call once the transfer is initiated.",
            ],
        )
