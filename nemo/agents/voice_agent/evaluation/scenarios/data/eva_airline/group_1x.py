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
# Auto-scaffolded from eva_airline_dataset.jsonl entries in the 1.x sub-flow (voluntary changes).
# See nemo_experiments/generate_eva_airline_scaffolds.py for the generator.

# pylint: disable=line-too-long
# flake8: noqa: E501

from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Task
from nemo.agents.voice_agent.evaluation.scenarios.data.eva_airline import EvaAirlineBaseScenario


# ---------------------------------------------------------------------------
# eva 1.1.3: You want to move only your outbound flight from Chicago (ORD) to Miami (MIA) to June 3, while keepin
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline113(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.1.3. Review prose before shipping."""

    name = "eva_airline__1_1_3"
    eva_id = "1.1.3"
    description = "You want to move only your outbound flight from Chicago (ORD) to Miami (MIA) to June 3, while keeping your return flight on June 12 exactly the same."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="David Okonkwo",
            background=(
                "You want to move only your outbound flight from Chicago (ORD) to Miami (MIA) to June 3, while keeping your return flight on June 12 exactly the same."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to move only your outbound flight from Chicago (ORD) to Miami (MIA) to June 3, while keeping your return flight on June 12 exactly the same."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I need to change my flight to an earlier date.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is IM2XU4 (spelled out as I, M, two, X, U, four). Your last name is Okonkwo. Your first name is David.",
                "Your booking: ORD (spelled out as O, R, D) to MIA (spelled out as M, I, A) on 2026-06-05 departing at 13:10.",
                "Must-have: The return flight must remain unchanged on June 12 (no date change, no time change, and not rebooked onto a different return itinerary).",
                "Must-have: The outbound flight must be changed to June 3 (ORD to MIA).",
                "Must-have: The June 3 outbound departure time must be after 12:00 PM (noon) Chicago time (CST/CDT as applicable).",
                "Must-have: Airports must stay ORD (origin) and MIA (destination) for the outbound.",
                "If the agent asks for authentication details, provide the confirmation code IM2XU4 and last name Okonkwo. If the agent reads back your itinerary, confirm they have the correct reservation and immediately clarify you want to change ONLY the outbound to June 3 and keep the June 12 return exactly as-is.",
                "When the agent presents outbound options for June 3, evaluate each option against the must-have criteria first: (a) outbound date June 3, (b) ORD to MIA, (c) departs after 12:00 PM Chicago time, and (d) agent confirms the June 12 return remains unchanged. Discard any option that fails any must-have criterion.",
                "If at least one option meets all must-have criteria AND the total added cost is under $100, accept the option with the lowest total added cost. If there is a tie, accept the one with the earliest departure time after 12:00 PM.",
                "Edge case: If the agent suggests changing the return flight (date or time) to make the outbound change work, decline and restate that the June 12 return must stay exactly the same.",
                "Edge case: If the agent suggests alternate airports (anything other than ORD for departure or MIA for arrival), decline and insist on ORD to MIA only.",
                "Edge case: If the agent offers a flight on a different date than June 3 for the outbound, decline.",
                "Edge case: If the agent offers a June 3 outbound that departs at or before 12:00 PM Chicago time, decline.",
                "Edge case: If the agent proposes standby instead of a confirmed seat, decline and request confirmed options only.",
                "Failure path: If the agent cannot provide any June 3 outbound option departing after 12:00 PM from ORD to MIA while keeping the June 12 return unchanged after two clear search attempts, say you will call back later, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.1.4: You want to keep your outbound flight on August 14 as-is, but change only your return flight from Au
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline114(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.1.4. Review prose before shipping."""

    name = "eva_airline__1_1_4"
    eva_id = "1.1.4"
    description = "You want to keep your outbound flight on August 14 as-is, but change only your return flight from August 20 to a return on August 23, departing after 2:00 PM Eastern, and you want the total extra cost"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Emily Johansson",
            background=(
                "You want to keep your outbound flight on August 14 as-is, but change only your return flight from August 20 to a return on August 23, departing after 2:00 PM Eastern, and you want the total extra cost to be under $100."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to keep your outbound flight on August 14 as-is, but change only your return flight from August 20 to a return on August 23, departing after 2:00 PM Eastern, and you want the total extra cost to be under $100."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to change my return flight date.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is KOLTSF (spelled out as K, O, L, T, S, F). Your last name is Johansson. Your first name is Emily.",
                "Your booking: SEA (spelled out as S, E, A) to BOS (spelled out as B, O, S) on 2026-08-14 departing at 08:10.",
                "Must-have: Your outbound flight on 2026-08-14 must remain unchanged (same date and still confirmed).",
                "Must-have: Your return must be rebooked to 2026-08-23 departing after 2:00 PM Eastern Time.",
                "Must-have: The total additional cost you pay for the change (all fees and fare difference combined) must be under $100.",
                "After the agent asks for details, provide your confirmation code and last name, then state clearly: keep the outbound on 2026-08-14 unchanged and change only the return to 2026-08-23 after 2:00 PM ET, with total added cost under $100.",
                "When the agent presents rebooking options, evaluate each option against all must-have criteria (outbound unchanged, return date/time requirement, and total added cost under $100).",
                "If the agent offers at least one option that meets all must-have criteria, choose the option with the lowest total added cost; if there is a tie, choose the earliest departure time after 2:00 PM ET.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on SEA to BOS and BOS to SEA only.",
                "Edge case: If the agent suggests standby, decline and ask for confirmed-seat rebooking options only.",
                "Failure path: If the agent cannot provide any return-only rebooking option on 2026-08-23 departing after 2:00 PM ET with total added cost under $100 after two total option rounds (the initial set plus one additional search you requested), say you will call back later, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.1.5: You want to change your round-trip flights from Boston to Denver so the outbound moves to November 3
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline115(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.1.5. Review prose before shipping."""

    name = "eva_airline__1_1_5"
    eva_id = "1.1.5"
    description = "You want to change your round-trip flights from Boston to Denver so the outbound moves to November 3 and the return moves to November 8, while keeping the total extra cost under $250, making sure you "

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="James Patel",
            background=(
                "You want to change your round-trip flights from Boston to Denver so the outbound moves to November 3 and the return moves to November 8, while keeping the total extra cost under $250, making sure you stay in main cabin for both flights, and making sure you get back to Boston by 8:00 PM Eastern on the return."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change your round-trip flights from Boston to Denver so the outbound moves to November 3 and the return moves to November 8, while keeping the total extra cost under $250, making sure you stay in main cabin for both flights, and making sure you get back to Boston by 8:00 PM Eastern on the return."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I need to change my round-trip flight dates.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is YTM924 (spelled out as Y, T, M, nine, two, four). Your last name is Patel. Your first name is James.",
                "Your booking: BOS (spelled out as B, O, S) to DEN (spelled out as D, E, N) on 2026-11-01 departing at 08:10.",
                "Must-have: Your new outbound flight date must be 2026-11-03 from BOS to DEN and your new return flight date must be 2026-11-08 from DEN to BOS (do not accept any other dates).",
                "Must-have: Both your new flights must be for main cabin (do not accept any other fare class/cabin)",
                "Must-have: The total additional amount you pay for changing BOTH flights combined (all change fees plus any fare difference) must be $250 or less.",
                "Must-have: On the return (DEN to BOS) on 2026-11-08, the scheduled arrival into BOS must be no later than 8:00 PM EST.",
                "After the agent authenticates you, when they ask what you want: state clearly that you want to move the outbound to 2026-11-03 and the return to 2026-11-08, and that you need the total added cost for both changes to stay at $250 or less, and the return must arrive BOS by 8:00 PM EST.",
                "When the agent presents flight options, evaluate each complete proposed solution as a pair (outbound option + return option) against all must-have criteria: correct dates (11/03 and 11/08), return arrival by 8:00 PM EST, main cabin fare class, and total added cost for both changes combined <= $250.",
                "If the agent presents at least one pair of options that meets ALL must-haves AND also has an outbound departure before 9:00 AM EST, accept the pair that has the lowest total added cost. If there is a tie in total added cost, accept the one with the earliest outbound departure time.",
                "Edge case: If the agent asks for your confirmation number and last name, provide confirmation code YTM924 and last name Patel exactly.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked (anything other than BOS and DEN), decline and insist on BOS->DEN and DEN->BOS only.",
                "Edge case: If the agent suggests standby as the solution, decline standby and ask for confirmed seats only.",
                "Edge case: If the agent offers options that require splitting the party across different flights, decline and restate you need your itinerary kept together (you are traveling as one passenger).",
                "Edge case: If the agent asks if you want to change only one direction, say no—you want to change both outbound and return dates.",
                "Failure path: If after 2 complete re-search attempts the agent cannot offer any rebooking pair that meets all must-have criteria (exact dates 2026-11-03 and 2026-11-08, return arrives BOS by 8:00 PM EST, main cabin, and total added cost <= $250), say you will call back later, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.2.1: You want to move your LAX to SFO flight today from the late afternoon to an earlier direct flight th
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline121(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.2.1. Review prose before shipping."""

    name = "eva_airline__1_2_1"
    eva_id = "1.2.1"
    description = "You want to move your LAX to SFO flight today from the late afternoon to an earlier direct flight that leaves before 2:00 PM, as long as the same-day change fee stays under $80."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Kenji Thompson",
            background=(
                "You want to move your LAX to SFO flight today from the late afternoon to an earlier direct flight that leaves before 2:00 PM, as long as the same-day change fee stays under $80."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to move your LAX to SFO flight today from the late afternoon to an earlier direct flight that leaves before 2:00 PM, as long as the same-day change fee stays under $80."
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
                "Your confirmation number is 6VORJU (spelled out as six, V, O, R, J, U). Your last name is Thompson. Your first name is Kenji.",
                "Your booking: LAX (spelled out as L, A, X) to SFO (spelled out as S, F, O) on 2026-06-18 departing at 17:30.",
                "Must-have: New departure time is today (2026-06-18) and departs LAX before 2:00 PM Pacific.",
                "Must-have: Same-day change fee is under $80 total (acceptable: $0 to $79.99).",
                "Must-have: It is a direct flight from LAX to SFO (no connections and no airport changes).",
                "If the agent asks for verification details, provide your confirmation code and last name exactly as given in information_required, then wait for the agent to read back your reservation and confirm it is yours; if they read back a different name or itinerary, correct them and re-provide the details.",
                "When the agent offers earlier-flight options, evaluate each option against ALL must-have criteria: (a) date is 2026-06-18, (b) LAX departure time is before 2:00 PM PT, (c) direct LAX→SFO, (d) same-day change fee is under $80.",
                "If both an 11:00 AM and a 1:00 PM direct option meet all must-haves, choose the earliest departure (11:00 AM).",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked, decline and insist on LAX to SFO only.",
                "Edge case: If the agent suggests standby instead of a confirmed earlier flight, decline standby and ask for a confirmed seat on an earlier direct flight before 2:00 PM.",
                "Failure path: If the agent cannot provide any direct LAX→SFO option departing before 2:00 PM PT today with a same-day change fee under $80 after one additional search/attempt, say you will keep your original flight and say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.2.2: You need to move your DCA→ATL flight to a later departure today because you’re running late, while s
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline122(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.2.2. Review prose before shipping."""

    name = "eva_airline__1_2_2"
    eva_id = "1.2.2"
    description = "You need to move your DCA→ATL flight to a later departure today because you’re running late, while still getting into Atlanta by 5:00 PM EST today and keeping any extra cost under $100."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Maria Martinez",
            background=(
                "You need to move your DCA→ATL flight to a later departure today because you’re running late, while still getting into Atlanta by 5:00 PM EST today and keeping any extra cost under $100."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You need to move your DCA→ATL flight to a later departure today because you’re running late, while still getting into Atlanta by 5:00 PM EST today and keeping any extra cost under $100."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I’m going to miss my flight this morning—can you move me to a later one today?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 70RDH8 (spelled out as seven, zero, R, D, H, eight). Your last name is Martinez.",
                "Your booking: DCA (spelled out as D, C, A) to ATL (spelled out as A, T, L) on 2026-05-05 departing at 07:00.",
                "Must-have: New itinerary must arrive in ATL no later than 5:00 PM EST on 2026-05-05.",
                "Must-have: Any additional out-of-pocket cost charged today (change fee plus any fare difference) must be less than or equal to $100 total.",
                "Must-have: Must be in main cabin",
                "After the agent authenticates you, immediately explain the key constraint: you overslept and you need a later flight today from DCA to ATL, and you must arrive by 5:00 PM today.",
                "When the agent presents same-day flight options (expected: 11 AM, 2 PM, 5 PM), evaluate each option against must-have criteria first: (1) arrival time in ATL is by 5:00 PM today, and (2) total additional cost is $100 or less. (3) Main cabin fare class",
                "If multiple options meet both must-have criteria, choose the option with the lowest additional cost; if there is a tie in cost, choose the earliest departure time among the tied options.",
                "Edge case: If the agent suggests flying from or to a different airport than DCA and ATL, decline and insist on DCA→ATL only.",
                "Edge case: If the agent suggests standby instead of a confirmed seat, decline and ask for a confirmed later flight today instead.",
                "Edge case: If the agent offers you a basic economy fare decline and insist on staying in main cabin",
                "Edge case: If the agent offers an option that arrives after 5:00 PM EST today, reject it and restate that you must arrive by 5:00 PM.",
                "Edge case: If the agent asks whether you accept fees or fare differences above $100, clearly say no and restate your $100 maximum.",
                "Failure path: If the agent cannot provide any later main cabin flight today that arrives in ATL by 5:00 PM AND keeps the total additional cost at $100 or less after one additional attempt to re-check options, say you can’t proceed, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.2.3: You want to change your booked JFK to LAX red-eye to a nonstop daytime flight on the same travel dat
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline123(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.2.3. Review prose before shipping."""

    name = "eva_airline__1_2_3"
    eva_id = "1.2.3"
    description = "You want to change your booked JFK to LAX red-eye to a nonstop daytime flight on the same travel date, ideally departing between 8:00 AM and 3:00 PM Eastern."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="William Kim",
            background=(
                "You want to change your booked JFK to LAX red-eye to a nonstop daytime flight on the same travel date, ideally departing between 8:00 AM and 3:00 PM Eastern."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change your booked JFK to LAX red-eye to a nonstop daytime flight on the same travel date, ideally departing between 8:00 AM and 3:00 PM Eastern."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to change my flight to an earlier daytime departure.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is XXF6OH (spelled out as X, X, F, six, O, H). Your last name is Kim. Your first name is William.",
                "Your booking: JFK (spelled out as J, F, K) to LAX (spelled out as L, A, X) on 2026-09-14 departing at 23:45.",
                "Must-have: New flight must depart on the same travel date as your currently booked JFK→LAX trip.",
                "Must-have: New flight must be a daytime departure between 8:00 AM and 3:00 PM Eastern Time.",
                "Must-have: New itinerary must remain JFK (origin) to LAX (destination) with no connections (nonstop/direct).",
                "If the agent asks for verification details, provide your confirmation code and last name exactly as given in information_required, then wait for the agent to confirm they found your reservation before discussing times or price.",
                "Once the agent presents one or more rebooking options, evaluate each option using this exact order: (1) nonstop JFK→LAX, (2) same travel date, (3) departure time between 8:00 AM and 3:00 PM ET, (4) lowest total added cost.",
                "Immediately reject any option that is not nonstop, is not JFK→LAX, is on a different date, or departs outside 8:00 AM–3:00 PM ET; tell the agent you need a nonstop JFK to LAX flight in that time window and ask them to look again.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked (anything other than JFK→LAX), decline and insist on JFK to LAX only.",
                "Edge case: If the agent suggests a connecting itinerary, decline and restate that you only want a nonstop/direct flight.",
                "Edge case: If the agent offers standby instead of a confirmed seat, decline and ask for confirmed rebooking options only.",
                "Edge case: If the agent asks you to confirm a change that departs outside 8:00 AM–3:00 PM ET, decline and restate your time window.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Failure path: If the agent cannot provide any nonstop JFK→LAX option on the same travel date departing between 8:00 AM and 3:00 PM ET after two separate searches/attempts, say you will keep your current booking for now, say goodbye, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.3.1: You want to change your existing flight so you fly from San Francisco to Detroit on the same travel
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline131(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.3.1. Review prose before shipping."""

    name = "eva_airline__1_3_1"
    eva_id = "1.3.1"
    description = "You want to change your existing flight so you fly from San Francisco to Detroit on the same travel date, arriving by 6:00 PM Eastern, with no more than one connection, and keeping any extra cost unde"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Sophia Andersen",
            background=(
                "You want to change your existing flight so you fly from San Francisco to Detroit on the same travel date, arriving by 6:00 PM Eastern, with no more than one connection, and keeping any extra cost under $200."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change your existing flight so you fly from San Francisco to Detroit on the same travel date, arriving by 6:00 PM Eastern, with no more than one connection, and keeping any extra cost under $200."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I need to change my flight destination to Detroit.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is MLATG2 (spelled out as M, L, A, T, G, two). Your last name is Andersen. Your first name is Sophia.",
                "Your booking: SFO (spelled out as S, F, O) to ORD (spelled out as O, R, D) on 2026-08-20 departing at 08:10.",
                "Must-have: New itinerary must arrive in Detroit (DTW) by 6:00 PM Eastern Time on the original travel date.",
                "Must-have: Total additional amount you must pay to rebook (all-in) must be under $200 USD.",
                "Must-have: Itinerary must have no more than 1 connection (0 or 1 stop).",
                "Must-have: Origin airport must remain SFO and destination airport must be DTW (do not accept nearby/alternate airports).",
                "If the agent asks to look up your reservation, provide the confirmation code and last name exactly as listed in information_required.",
                "After the agent confirms they found the correct reservation, explain that your conference moved and you need to fly to DTW instead of ORD, and restate the must-have criteria (arrive by 6:00 PM ET on the original travel date, under $200 extra, no more than 1 connection, SFO→DTW only).",
                "When the agent presents one or more DTW options, evaluate each option against ALL must-have criteria. Ignore any option that fails even one must-have criterion.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests flying from or to a different airport than originally booked (including OAK/SJC instead of SFO or any airport other than DTW), decline and insist on SFO to DTW only.",
                "Edge case: If the agent suggests arriving after 6:00 PM ET or traveling on a different date, decline and restate that you must arrive by 6:00 PM ET on the original travel date.",
                "Failure path: If the agent cannot find any SFO→DTW itinerary on the original travel date that arrives by 6:00 PM ET with no more than 1 connection and under $200 extra after two clear search attempts (or says they cannot make destination changes), say you will handle it later, thank them, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 1.3.2: You want to change your existing flight so you depart from Newark (EWR) instead of JFK while still f
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline132(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 1.3.2. Review prose before shipping."""

    name = "eva_airline__1_3_2"
    eva_id = "1.3.2"
    description = "You want to change your existing flight so you depart from Newark (EWR) instead of JFK while still flying to Los Angeles (LAX), staying in Main Cabin, and leaving around the same time as your original"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Alexander Volkov",
            background=(
                "You want to change your existing flight so you depart from Newark (EWR) instead of JFK while still flying to Los Angeles (LAX), staying in Main Cabin, and leaving around the same time as your original departure."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to change your existing flight so you depart from Newark (EWR) instead of JFK while still flying to Los Angeles (LAX), staying in Main Cabin, and leaving around the same time as your original departure."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I need to change my flight to leave from Newark instead of JFK.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is 2DS6M0 (spelled out as two, D, S, six, M, zero). Your last name is Volkov. Your first name is Alexander.",
                "Your booking: JFK (spelled out as J, F, K) to LAX (spelled out as L, A, X) on 2026-07-22 departing at 10:30.",
                "Must-have: New departure airport must be EWR (Newark).",
                "Must-have: Destination must remain LAX (Los Angeles).",
                "Must-have: New flight must depart within 2 hours of your original JFK departure time (you will ask the agent what your original departure time is if they haven’t stated it).",
                "Must-have: Cabin must remain Main Cabin (no downgrade to Basic Economy and no forced upgrade to a different cabin).",
                "If the agent asks to look up your booking, provide your confirmation number and last name exactly as given in information_required, and confirm you want to switch the origin from JFK to EWR while keeping LAX as the destination.",
                "If the agent has not told you your original JFK departure time yet, ask: \"What time was my original flight departing?\" and use that time as the reference point for the 'within 2 hours' must-have criterion.",
                "When the agent presents one or more EWR→LAX options, evaluate each option against all must-have criteria first (EWR origin, LAX destination, Main Cabin, and departure within 2 hours of the original departure time). Discard any option that fails any must-have criterion.",
                "Edge case: If the agent suggests keeping JFK as the origin or switching to any origin other than EWR, decline and restate that you must depart from EWR.",
                "Edge case: If the agent suggests changing the destination from LAX, decline and restate that LAX must remain the destination.",
                "Edge case: If the agent offers Basic Economy, decline and restate you need Main Cabin.",
                "Edge case: If the agent offers a flight outside the 'within 2 hours of the original departure time' window, decline and ask for options within the window (up to one additional search attempt total as described in failure_condition).",
                "Edge case: If the agent offers standby instead of a confirmed seat, decline standby and ask for confirmed options only.",
                "Failure path: If, after two search attempts, the agent cannot offer any rebooking option that departs from EWR to LAX within 2 hours of the original departure time in Main Cabin, say you will keep your current booking unchanged for now, thank them, and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )
