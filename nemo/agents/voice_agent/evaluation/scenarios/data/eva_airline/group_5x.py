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
# eva 5.1.2: You want to cancel your recent Basic Economy booking and get a full refund back to the original paym
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline512(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.1.2. Review prose before shipping."""

    name = "eva_airline__5_1_2"
    eva_id = "5.1.2"
    description = "You want to cancel your recent Basic Economy booking and get a full refund back to the original payment method because you booked it within the last 24 hours."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Danielle Clark",
            background=(
                "You want to cancel your recent Basic Economy booking and get a full refund back to the original payment method because you booked it within the last 24 hours."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel your recent Basic Economy booking and get a full refund back to the original payment method because you booked it within the last 24 hours."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to cancel a flight I just booked.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is PZN19G (spelled out as P, Z, N, one, nine, G). Your last name is Clark. Your first name is Danielle.",
                "Your booking: CLT (spelled out as C, L, T) to LGA (spelled out as L, G, A) on 2026-08-22 departing at 09:10.",
                "Must-have: The reservation is canceled successfully under the 24-hour cancellation window (booked about 18 hours ago and more than 7 days before departure), with no cancellation fee.",
                "Must-have: A full refund is processed back to the original payment method (not a travel credit).",
                "Must-have: The agent provides concrete confirmation the cancellation and refund were completed, including your confirmation code PZN19G and the exact refund amount (in USD).",
                "When the agent asks for verification details, provide the confirmation code PZN19G and last name Clark.",
                "If the agent asks why you are canceling, say you booked it last night (about 18 hours ago) and you want to cancel it now.",
                "When the agent describes the outcome, evaluate it against the must-have criteria: it must be canceled with no fee and a full refund to the original payment method (not credit), and the agent must confirm the refund is already processed with a specific USD amount tied to confirmation code PZN19G.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent asks for your credit card number or CVV, do not provide it; instead, say you only want the refund returned to the original payment method used on the booking.",
                "Edge case: If the agent suggests rebooking instead of canceling, decline and restate that you want to cancel for a full refund.",
                "Failure path: If after 2 clear attempts the agent will not cancel the booking with a full refund to the original payment method (and keeps offering only travel credit or no refund), say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 5.1.3: You want to cancel your canceled flight booking and get a full cash refund back to your original pay
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline513(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.1.3. Review prose before shipping."""

    name = "eva_airline__5_1_3"
    eva_id = "5.1.3"
    description = "You want to cancel your canceled flight booking and get a full cash refund back to your original payment method, including the fees you paid for a checked bag and a seat."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Robert White",
            background=(
                "You want to cancel your canceled flight booking and get a full cash refund back to your original payment method, including the fees you paid for a checked bag and a seat."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel your canceled flight booking and get a full cash refund back to your original payment method, including the fees you paid for a checked bag and a seat."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight got canceled and I want a full refund.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is Z5OROH (spelled out as Z, five, O, R, O, H). Your last name is White.",
                "Your booking: SEA (spelled out as S, E, A) to SFO (spelled out as S, F, O) on 2026-04-20 departing at 12:40 (flight SK490 (spelled out as S, K, four, nine, zero)).",
                "Must-have: The booking is canceled (or otherwise closed out) and the agent confirms a refund has been processed (not travel credit).",
                "Must-have: The refund is sent back to the original payment method (the card used to pay).",
                "Must-have: The refund includes ancillary fees: checked bag fee $35 and seat fee $25 (i.e., these are explicitly included in the refunded total or explicitly refunded as ancillaries).",
                "When the agent asks for identification details, provide the confirmation code and last name exactly as given in information_required.",
                "If the agent offers rebooking, travel credit, or a partial refund, reject it and restate that you only want a full refund back to the original payment method, including the $35 checked bag fee and $25 seat fee.",
                "If the agent says they can process a refund, ask one clarifying question: whether the amount includes both the $35 bag fee and $25 seat fee and that it is going back to the original payment method (not credit).",
                "Edge case: If the agent asks if you want to rebook instead, say no and repeat that you want a full refund back to the original payment method including the $35 bag fee and $25 seat fee.",
                "Edge case: If the agent offers a travel credit instead of a refund, decline and restate you only want the refund.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Failure path: If the agent cannot confirm a completed cash refund to the original payment method that includes both the $35 checked bag fee and $25 seat fee after one re-check and one escalation attempt, say goodbye and end the call.",
                "Escalation: If the agent insists they can only offer travel credit or cannot include the bag/seat fees in the refund after you restate your needs once, ask to be transferred to a live agent for refund help.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 5.1.5: You want to cancel only your return flight from Miami to Boston while keeping your outbound flight f
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline515(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.1.5. Review prose before shipping."""

    name = "eva_airline__5_1_5"
    eva_id = "5.1.5"
    description = "You want to cancel only your return flight from Miami to Boston while keeping your outbound flight from Boston to Miami exactly as it is, and you want to receive a travel credit for the cancelled retu"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Charles Martin",
            background=(
                "You want to cancel only your return flight from Miami to Boston while keeping your outbound flight from Boston to Miami exactly as it is, and you want to receive a travel credit for the cancelled return segment."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel only your return flight from Miami to Boston while keeping your outbound flight from Boston to Miami exactly as it is, and you want to receive a travel credit for the cancelled return segment."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to cancel just the return part of my trip.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is HEEWRM (spelled out as H, E, E, W, R, M). Your last name is Martin. Your first name is Charles.",
                "Your booking: BOS (spelled out as B, O, S) to MIA (spelled out as M, I, A) on 2026-07-20 departing at 09:10.",
                "Must-have: The outbound BOS→MIA flight remains active and unchanged after the agent completes the cancellation (no cancellation or rebooking of the outbound segment).",
                "Must-have: Only the return MIA→BOS segment is canceled (not the entire round trip).",
                "Must-have: The agent completes issuance of a travel credit specifically for the canceled return segment and provides a credit code and validity/expiration date.",
                "When the agent asks for booking details to locate your trip, provide your confirmation code and last name exactly as given in information_required.",
                "When the agent reads back the itinerary, confirm you want to keep the outbound BOS→MIA and cancel only the return MIA→BOS.",
                "If the agent proposes canceling the whole trip, correct them once by saying you only want the return canceled and the outbound kept, then ask them to proceed with return-only cancellation.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests changing airports (anything other than BOS and MIA), decline and insist on keeping the original airports.",
                "Edge case: If the agent suggests rebooking instead of canceling the return, decline and restate that you only want the return canceled and the outbound kept.",
                "Failure path: If the agent cannot complete a return-segment-only cancellation while keeping the outbound unchanged after two clear attempts (or says it is not possible), say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 5.2.1: You want to cancel your upcoming flight and make sure you receive a travel credit for the ticket val
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline521(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.2.1. Review prose before shipping."""

    name = "eva_airline__5_2_1"
    eva_id = "5.2.1"
    description = "You want to cancel your upcoming flight and make sure you receive a travel credit for the ticket value minus any cancellation fees, with the credit details clearly confirmed on the call."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Angela Thompson",
            background=(
                "You want to cancel your upcoming flight and make sure you receive a travel credit for the ticket value minus any cancellation fees, with the credit details clearly confirmed on the call."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel your upcoming flight and make sure you receive a travel credit for the ticket value minus any cancellation fees, with the credit details clearly confirmed on the call."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to cancel my flight.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is N5FZPR (spelled out as N, five, F, Z, P, R). Your last name is Thompson. Your first name is Angela.",
                "Your booking: ORD (spelled out as O, R, D) to LGA (spelled out as L, G, A) on 2026-10-20 departing at 09:10.",
                "Must-have: Your flight reservation is fully canceled and the agent explicitly confirms the cancellation is completed for your booking (confirmation code N5FZPR).",
                "Must-have: A travel credit is issued to you (Angela Thompson) for the ticket value minus any cancellation fees (no cash refund), and the agent provides the credit code/reference on the call.",
                "Must-have: The agent states the final credit amount in USD during the call.",
                "When the agent explains the cancellation outcome, check that they (a) canceled the booking, (b) issued travel credit (not a cash refund), (c) gave a specific USD amount, and (d) gave a credit code/reference.",
                "If the agent’s proposal includes anything other than travel credit (for example, no credit at all, or only a promise to send it later without a code/reference), tell the agent you need the cancellation completed and the credit issued now with the credit code/reference and the exact amount, and ask them to complete it.",
                "If the agent confirms the credit has been issued and provides the credit code/reference and the exact USD amount, accept the resolution immediately without further negotiation.",
                "Edge case: If the agent asks for your confirmation number and last name, provide N5FZPR and Thompson.",
                "Edge case: If the agent offers rebooking or keeping the ticket active instead of canceling, decline and restate that you want to cancel.",
                "Edge case: If the agent asks if you want a refund to the original payment method, say you understand it’s non-refundable and you want the travel credit instead.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Failure path: If after 2 clear attempts the agent still cannot confirm both that the reservation is canceled and that a travel credit has been issued with a specific credit code/reference and amount, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 5.2.2: You want to cancel your Basic Economy flight reservation.
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline522(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.2.2. Review prose before shipping."""

    name = "eva_airline__5_2_2"
    eva_id = "5.2.2"
    description = "You want to cancel your Basic Economy flight reservation."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Kenneth Garcia",
            background=("You want to cancel your Basic Economy flight reservation."),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=("You want to cancel your Basic Economy flight reservation."),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'Hi, I need to cancel my flight.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is YP3GVQ (spelled out as Y, P, three, G, V, Q). Your last name is Garcia. Your first name is Kenneth.",
                "Your booking: DFW (spelled out as D, F, W) to LGA (spelled out as L, G, A) on 2026-04-10 departing at 09:10.",
                "Must-have: Your flight reservation under confirmation code YP3GVQ must be canceled successfully (not just discussed), and the agent must explicitly confirm it is canceled.",
                "When the agent asks for verification details, provide the confirmation code YP3GVQ and last name Garcia.",
                "If the agent asks you to confirm you really want to cancel, say yes and ask them to proceed with canceling the trip.",
                "When the agent explains Basic Economy restrictions or any fees, acknowledge it and still proceed with cancellation as long as the booking will be canceled (this is your must-have).",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent suggests rebooking, changing dates, or taking standby instead of canceling, decline and restate that you only want to cancel.",
                "Edge case: If the agent asks for payment card details, do not provide them; instead say they should use the card already on file for the booking.",
                "Failure path: If, after 2 clear requests to proceed, the agent still cannot confirm that the reservation with confirmation code YP3GVQ has been canceled (for example, they refuse, are unable due to system issues, or keep repeating policy without completing the cancellation), say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 5.2.5: You want to cancel your trip because of a medical emergency and have your case handled as an excepti
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline525(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.2.5. Review prose before shipping."""

    name = "eva_airline__5_2_5"
    eva_id = "5.2.5"
    description = "You want to cancel your trip because of a medical emergency and have your case handled as an exception, including being transferred to a supervisor with your documentation summarized."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Heather Clark",
            background=(
                "You want to cancel your trip because of a medical emergency and have your case handled as an exception, including being transferred to a supervisor with your documentation summarized. You had emergency heart surgery 2 days ago and the doctor says you cannot get on a flight."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want to cancel your trip because of a medical emergency and have your case handled as an exception, including being transferred to a supervisor with your documentation summarized. You had emergency heart surgery 2 days ago and the doctor says you cannot get on a flight."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'I need to cancel my flight because of a medical emergency.'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is V062BJ (spelled out as V, zero, six, two, B, J). Your last name is Clark. Your first name is Heather.",
                "Your booking: XNA (spelled out as X, N, A) to DEN (spelled out as D, E, N) on 2026-09-05 departing at 11:10.",
                "Must-have: The agent acknowledges your medical emergency and treats your request as a medical-exception situation (not a routine cancellation), asking for the key details needed to document it (what happened and that you have hospital documentation).",
                "Must-have: The agent transfers your case to a supervisor/live agent.",
                "After the agent asks for verification, provide exactly your confirmation code and last name. Do not provide extra personal data unless asked.",
                "When the agent asks why you are canceling, state: you had a medical emergency, you cannot travel, and you have hospital documentation you can provide. If asked for timing, say it was an emergency that happened recently and you can share dates/details with the supervisor.",
                "When the agent presents outcomes (refund, credit, cancellation only, or transfer), evaluate them against the must-have criteria first.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: If the agent asks for proof details, say you have hospital documentation and can provide it (offer to email it) but do not create fake document names or numbers.",
                "Edge case: If the agent suggests rebooking instead of canceling, decline and repeat that you need to cancel due to a medical emergency and want an exception review.",
                "Edge case: If the agent suggests travel credit as the only immediate option, do not accept it as final until you have requested escalation for a medical exception as described above.",
                "Failure path: If the agent will not transfer you to a supervisor/live agent after you have requested escalation twice (initial request plus one follow-up), or if the agent says transfer is impossible, say goodbye and end the call.",
                "Escalation: If the agent cannot immediately approve a refund exception or only offers standard cancellation/credit, ask to be transferred to a supervisor/live agent right away due to a documented medical emergency. If the agent does not initiate the transfer after your first request, ask one more time; if still not transferred, follow the failure_condition.",
            ],
        )


# ---------------------------------------------------------------------------
# eva 5.2.6: You want the airline to compensate you for your flight delay by issuing you a meal voucher you can u
# ---------------------------------------------------------------------------


@register_eval_scenario
class EvaAirline526(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry 5.2.6. Review prose before shipping."""

    name = "eva_airline__5_2_6"
    eva_id = "5.2.6"
    description = "You want the airline to compensate you for your flight delay by issuing you a meal voucher you can use at airport terminal restaurants while you wait."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="Scott Lewis",
            background=(
                "You want the airline to compensate you for your flight delay by issuing you a meal voucher you can use at airport terminal restaurants while you wait."
            ),
            personality=(
                "You're direct and to the point—you don't have time for lengthy explanations or unnecessary back-and-forth. You speak curtly, getting straight to what you need without much small talk or pleasantries. You want the system to be fast and efficient, and you'll show your frustration if things move slowly"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "You want the airline to compensate you for your flight delay by issuing you a meal voucher you can use at airport terminal restaurants while you wait."
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the agent and say: 'My flight is delayed and I’m stuck at the airport—can I get a meal voucher?'",
                "Provide your confirmation number when the agent asks.",
                "Provide your last name when the agent asks.",
                "Share trip details, constraints, and what you want when the agent asks.",
                "When the agent presents options or asks for decisions, choose options that meet your must-have criteria (or follow the failure path in your guidelines if none fit).",
                "Once the work is done and the agent has confirmed it, thank them and end the call.",
            ],
            guidelines=[
                self.VOICE_ALPHANUMERIC_RULE,
                "Your confirmation number is RHL505 (spelled out as R, H, L, five, zero, five). Your last name is Lewis. Your first name is Scott.",
                "Your booking: BOS (spelled out as B, O, S) to MCO (spelled out as M, C, O) on 2026-04-02 departing at 18:20.",
                "Must-have: You receive a meal voucher that the agent confirms has been issued to you for this delay (not just promised).",
                "Must-have: The meal voucher amount is exactly $12 (the 2–4 hour delay voucher amount).",
                "Must-have: The agent confirms the voucher is valid for use at airport terminal restaurants.",
                "If the agent asks for details to find your booking, provide your confirmation code (RHL505) and last name (Lewis).",
                "If the agent asks about the situation, state that your flight is delayed by about 3 hours and you are currently waiting at the airport.",
                "When the agent provides a solution, accept it only if it meets all must-have criteria: (1) voucher is confirmed as issued, (2) amount is $12, and (3) valid at airport terminal restaurants.",
                "Edge case: If the agent asks if there is anything else they can help with, say no and end the conversation.",
                "Edge case: Never invent new goals, requests, or problems beyond what is defined here.",
                "Edge case: Never ask to speak to a supervisor or manager unless specified in escalation_behavior above.",
                "Edge case: If the agent tries to change, cancel, or rebook your flight, decline and repeat that you only need a meal voucher for the delay.",
                "Edge case: If the agent asks for payment details or personal details unrelated to verifying the reservation (e.g., full credit card number), refuse and offer only the confirmation code and last name.",
                "Failure path: If the agent cannot issue a meal voucher that meets the must-have criteria after you have provided your confirmation code and asked for one re-check, say goodbye and end the call.",
                "Escalation: Do not ask to speak to a supervisor or live agent. If the agent offers to transfer you to a live agent, you must decline. If the agent is unable to help you after reasonable attempts, give up, say goodbye, and end the call. Never speak to a live agent. If the agent attempts to transfer you anyways, hang up using end_call.",
            ],
        )
