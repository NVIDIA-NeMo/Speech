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
# eva 3.1.5: You need to recover your itinerary after missing your ATL to ORD flight this morning and still arriv
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline315(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 3.1.5. Review prose before shipping."""

    name = "eva_airline__3_1_5"
    eva_id = "3.1.5"
    description = "You need to recover your itinerary after missing your ATL to ORD flight this morning and still arrive in Minneapolis (MSP) by 6:00 PM Central Time today, with no more than one connection."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Jason Price",
            background=(
                "You need to recover your itinerary after missing your ATL to ORD flight this morning and still arrive in Minneapolis (MSP) by 6:00 PM Central Time today, with no more than one connection."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You need to recover your itinerary after missing your ATL to ORD flight this morning and still arrive in Minneapolis (MSP) by 6:00 PM Central Time today, with no more than one connection."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I missed my flight this morning and I need to get rebooked to Minneapolis today.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is M9M6FJ (spelled out as M, nine, M, six, F, J). Your last name is Price. Your first name is Jason.",
                "Your booking: ATL (spelled out as A, T, L) to MSP (spelled out as M, S, P) on 2026-09-08 departing at 08:00.",
                "Must-have: You must arrive in MSP by 6:00 PM Central Time on 2026-09-08.",
                "Must-have: The rebooked route must have no more than 1 connection (0 or 1 stop total).",
                "Must-have: You must stay in main cabin.",
                "If the agent asks for verification details, provide the confirmation code and last name exactly as given in information_required, then answer any follow-up identity questions briefly.",
                "When the agent asks what happened, state you missed the 8:00 AM ATL→ORD flight due to traffic and you are trying to save the ORD→MSP connection and still reach MSP by 6:00 PM CT today.",
                "When the agent presents any rebooking options, evaluate EACH option using these rules in order: (1) reject any option that arrives after 6:00 PM CT today; (2) reject any option with 2 or more connections;  (3) reject any option that isn't in main cabin, (4) among remaining options, prefer any option with total added cost under $200; (5) if more than one option meets the must-haves, choose the one with the earliest arrival time; if arrival times tie, choose the lowest added cost.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, accept alternate airports only if the destination is still MSP; do not accept changing the destination away from MSP.",
                "Edge case: Do not accept standby-only solutions; you will only accept a confirmed rebooked itinerary that meets the must-have criteria.",
                "Edge case: Do not accept any other fare class option besides main cabin",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Failure path: If, after two complete search attempts by the agent, the agent cannot offer any main cabin rebooking option that arrives in MSP by 6:00 PM CT today with no more than 1 connection, say you understand, ask them to stop, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 3.3.4: You want to rebook your flight to a date about three weeks from now (after you renew your passport)
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline334(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 3.3.4. Review prose before shipping."""

    name = "eva_airline__3_3_4"
    eva_id = "3.3.4"
    description = "You want to rebook your flight to a date about three weeks from now (after you renew your passport) and keep the total extra cost under $200."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Sarah Adams",
            background=(
                "You want to rebook your flight to a date about three weeks from now (after you renew your passport) and keep the total extra cost under $200."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to rebook your flight to a date about three weeks from now (after you renew your passport) and keep the total extra cost under $200."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to rebook my flight because I was denied boarding over my passport expiration date.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 1QTFVX (spelled out as one, Q, T, F, V, X). Your last name is Adams. Your first name is Sarah.",
                "Your booking: JFK (spelled out as J, F, K) to LHR (spelled out as L, H, R) on 2026-12-05 departing at 19:30.",
                "Must-have: The new departure date must be on or after 2026-12-26 (about 3 weeks from today, after passport renewal).",
                "Must-have: The total additional amount you personally have to pay to rebook must be less than $200 USD.",
                "If the agent asks for your booking details to look it up, provide your confirmation number and last name exactly as given, then wait for the agent to describe the current itinerary before discussing new dates or costs.",
                "When the agent asks what you want, state clearly: you need to move the trip to a date on or after 2026-12-26 because your passport renewal will take time, and you want the total extra cost to stay under $200.",
                "When the agent presents rebooking options, evaluate each option against the must-have criteria first: (1) date is on/after 2026-12-26, and (2) your out-of-pocket cost is under $200 total. Disregard any option that fails either must-have.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on keeping the original origin and destination airports.",
                "Edge case: If the agent suggests standby as a solution, decline and ask for a confirmed seat on a flight on/after 2026-12-26 under $200 extra instead.",
                "Failure path: If the agent cannot offer any rebooking option departing on or after 2026-12-26 with a total additional cost under $200 after two distinct searches/attempts, say you will handle it later, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )
