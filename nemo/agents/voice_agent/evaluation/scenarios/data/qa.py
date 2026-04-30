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

# Scenario definitions contain long prose strings (personas, instructions);
# wrapping every one hurts readability without improving correctness.
# pylint: disable=line-too-long
# flake8: noqa: E501

from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Scenario, Task


class QABaseScenario(Scenario):
    """
    Base class for QA evaluation scenarios.
    Provides sensible defaults for all 8 required properties.
    Not registered as an eval scenario itself.
    """

    max_duration = 60
    ignore_capitalization = True
    ignore_punctuation = True
    clean_text = True

    # Subclasses must override these
    _user_question: str = ""
    _user_name: str = "Alex"
    _user_background: str = "You are a curious person who wants to ask a question to an AI agent."
    _user_personality: str = "You are curious and communicative, with a friendly demeanor."

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name=self._user_name,
            background=self._user_background,
            personality=self._user_personality,
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Ask a question to the AI agent and wait for the answer.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                f"Ask the question: '{self._user_question}'",
            ],
            guidelines=[
                "Only ask the designated question to the agent, do not ask any other questions.",
                "If your question is answered, and the agent is asking if you have any other questions, say 'No, "
                "that's all I have.'",
                "Say 'Thank you for your answer, goodbye.' after the agent has answered your question.",
            ],
        )

    @property
    def user_resources(self) -> Resources:
        return Resources()

    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="helpful AI assistant",
            name="Lisa",
            background="You are a helpful AI assistant who can answer questions.",
            personality="You are friendly and helpful to the user. You are always concise and to the point.",
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal=(
                "Answer the questions from user, save the answer to the question using the `SaveQuestionAnswerTool` "
                "tool, and end the conversation with the `EndConversationTool` tool."
            ),
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Hello, I'm Lisa, what can I help you with?', say it only once at the "
                "beginning of the conversation.",
                "Answer a question from the user.",
                "Use the `SaveQuestionAnswerTool` tool to log your answer to the user's question.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no "
                "more questions.",
            ],
            guidelines=[
                "Always answer the questions from the user.",
                "After answering a question, use the `SaveQuestionAnswerTool` tool to log the question and your "
                "answer.",
                "When the user says goodbye or has no more questions, use the `EndConversationTool` tool to end the "
                "conversation.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "SaveQuestionAnswerTool": {},
                "EndConversationTool": {},
            },
        )


# ---------------------------------------------------------------------------
# 1. Geography: Capital of France
# ---------------------------------------------------------------------------
@register_eval_scenario
class QACapitalFrance(QABaseScenario):
    """QA scenario asking about the capital of France."""

    name = "qa__capital_france"
    description = "QA scenario: What is the capital of France?"
    reference_answer = {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
    }

    _user_question = "What is the capital of France?"
    _user_name = "John"
    _user_background = "You are a geography enthusiast who enjoys learning about world capitals."
    _user_personality = "You are curious and communicative, with a friendly demeanor and prompt decision-making."


# ---------------------------------------------------------------------------
# 2. Math: Basic arithmetic
# ---------------------------------------------------------------------------
@register_eval_scenario
class QAMathAddition(QABaseScenario):
    """QA scenario asking a basic arithmetic question."""

    name = "qa__math_addition"
    description = "QA scenario: What is 15 plus 27?"
    reference_answer = {
        "question": "What is 15 plus 27?",
        "answer": "The answer is 42.",
    }

    _user_question = "What is 15 plus 27?"
    _user_name = "Maria"
    _user_background = "You are a student who is practicing basic math."
    _user_personality = "You are eager to learn and straightforward in your communication."


# ---------------------------------------------------------------------------
# 3. Science: Speed of light
# ---------------------------------------------------------------------------
@register_eval_scenario
class QASpeedOfLight(QABaseScenario):
    """QA scenario asking about the speed of light."""

    name = "qa__speed_of_light"
    description = "QA scenario: What is the speed of light?"
    reference_answer = {
        "question": "What is the speed of light?",
        "answer": "The speed of light is about 300,000 kilometers per second.",
    }

    _user_question = "What is the speed of light?"
    _user_name = "David"
    _user_background = "You are a physics student curious about fundamental constants of the universe."
    _user_personality = "You are analytical and precise, preferring clear and factual answers."


# ---------------------------------------------------------------------------
# 4. History: First moon landing
# ---------------------------------------------------------------------------
@register_eval_scenario
class QAMoonLanding(QABaseScenario):
    """QA scenario asking about the first moon landing."""

    name = "qa__moon_landing"
    description = "QA scenario: When was the first moon landing?"
    reference_answer = {
        "question": "When was the first moon landing?",
        "answer": "The first moon landing was in 1969.",
    }

    _user_question = "When was the first moon landing?"
    _user_name = "Sarah"
    _user_background = "You are a history buff who enjoys learning about major milestones in human achievement."
    _user_personality = "You are enthusiastic and love discussing historical events. You communicate clearly."


# ---------------------------------------------------------------------------
# 5. Literature: Author of Romeo and Juliet
# ---------------------------------------------------------------------------
@register_eval_scenario
class QARomeoAndJuliet(QABaseScenario):
    """QA scenario asking who wrote Romeo and Juliet."""

    name = "qa__romeo_and_juliet"
    description = "QA scenario: Who wrote Romeo and Juliet?"
    reference_answer = {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "Romeo and Juliet was written by William Shakespeare.",
    }

    _user_question = "Who wrote Romeo and Juliet?"
    _user_name = "Emily"
    _user_background = "You are a literature student who is studying classic plays."
    _user_personality = "You are thoughtful and articulate, with a love for storytelling and the arts."


# ---------------------------------------------------------------------------
# 6. Weather: San Francisco (uses GetCityWeatherTool)
# ---------------------------------------------------------------------------
@register_eval_scenario
class QAWeatherSanFrancisco(QABaseScenario):
    """QA scenario asking about the weather in San Francisco using GetCityWeatherTool."""

    name = "qa__weather_san_francisco"
    description = "QA scenario: What is the weather in San Francisco?"
    reference_answer = {
        "question": "What is the weather in San Francisco?",
        "answer": "The weather in San Francisco is sunny with a temperature of 20 degrees Celsius and a low UV index.",
    }

    _user_question = "What is the weather in San Francisco?"
    _user_name = "Carlos"
    _user_background = "You are planning a trip to San Francisco and want to know the weather."
    _user_personality = "You are practical and organized, always planning ahead for your travels."

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "SaveQuestionAnswerTool": {},
                "EndConversationTool": {},
                "GetCityWeatherTool": {},
            },
        )


# ---------------------------------------------------------------------------
# 7. Weather: New York (uses GetCityWeatherTool)
# ---------------------------------------------------------------------------
@register_eval_scenario
class QAWeatherNewYork(QABaseScenario):
    """QA scenario asking about the weather in New York using GetCityWeatherTool."""

    name = "qa__weather_new_york"
    description = "QA scenario: What is the weather in New York?"
    reference_answer = {
        "question": "What is the weather in New York?",
        "answer": "The weather in New York is sunny with a temperature of 20 degrees Celsius and a low UV index.",
    }

    _user_question = "What is the weather in New York?"
    _user_name = "Priya"
    _user_background = "You are a travel blogger researching weather conditions in major US cities."
    _user_personality = (
        "You are outgoing and detail-oriented, always looking for accurate information to share with your readers."
    )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "SaveQuestionAnswerTool": {},
                "EndConversationTool": {},
                "GetCityWeatherTool": {},
            },
        )


# ---------------------------------------------------------------------------
# 8. Math word problem
# ---------------------------------------------------------------------------
@register_eval_scenario
class QAMathWordProblem(QABaseScenario):
    """QA scenario with a math word problem."""

    name = "qa__math_word_problem"
    description = "QA scenario: If you have 12 apples and give away 5, how many do you have left?"
    reference_answer = {
        "question": "If you have 12 apples and give away 5, how many do you have left?",
        "answer": "You have 7 apples left because 12 minus 5 equals 7.",
    }

    _user_question = "If you have 12 apples and give away 5, how many do you have left?"
    _user_name = "Tom"
    _user_background = "You are a parent helping your child with homework and want to verify the answer."
    _user_personality = "You are patient and methodical, preferring step-by-step explanations."


# ---------------------------------------------------------------------------
# 9. General knowledge: Largest ocean
# ---------------------------------------------------------------------------
@register_eval_scenario
class QALargestOcean(QABaseScenario):
    """QA scenario asking about the largest ocean."""

    name = "qa__largest_ocean"
    description = "QA scenario: What is the largest ocean on Earth?"
    reference_answer = {
        "question": "What is the largest ocean on Earth?",
        "answer": "The largest ocean on Earth is the Pacific Ocean.",
    }

    _user_question = "What is the largest ocean on Earth?"
    _user_name = "Yuki"
    _user_background = "You are an environmental science student studying oceanography."
    _user_personality = (
        "You are inquisitive and environmentally conscious, with a calm and respectful communication style."
    )


# ---------------------------------------------------------------------------
# 10. Technology: Inventor of the telephone
# ---------------------------------------------------------------------------
@register_eval_scenario
class QATelephoneInventor(QABaseScenario):
    """QA scenario asking who invented the telephone."""

    name = "qa__telephone_inventor"
    description = "QA scenario: Who invented the telephone?"
    reference_answer = {
        "question": "Who invented the telephone?",
        "answer": "The telephone was invented by Alexander Graham Bell.",
    }

    _user_question = "Who invented the telephone?"
    _user_name = "Kevin"
    _user_background = "You are a technology enthusiast who is fascinated by the history of inventions."
    _user_personality = "You are energetic and talkative, always excited to learn about how things were created."
