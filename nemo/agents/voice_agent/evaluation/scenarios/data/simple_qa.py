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

import json
from typing import Optional

from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Scenario, Task
from nemo.agents.voice_agent.utils.audio import NoiseConfig


@register_eval_scenario
class SimpleQA(Scenario):
    """
    Simple QA scenario.
    """

    name = "simple_qa_1"
    description = "Simple QA example scenario"
    reference_answer = {
        "question": "What is the answer to life, the universe, and everything?",
        "answer": "The answer is 42.",
    }
    max_duration = 90

    ignore_capitalization = True
    ignore_punctuation = True
    clean_text = True

    # User section
    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="John",
            background="You are a curious human who wants to ask a question to an AI agent.",
            personality="You are communicative and positive, with clear needs, friendly demeanor, and prompt decision-making.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Ask a question to the AI agent and wait for the answer.",
            background="You are reading a book about some science fiction story.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the question: 'What is the answer to life, the universe, and everything?'",
            ],
            guidelines=[
                "Only ask the designated question to the agent, do not ask any other questions.",
                "If your question is answered, and the agent is asking if you have any other questions, say 'No, that's all I have.'",
                "Say 'Thank you for your answer, goodbye.' after the agent has answered your question.",
            ],
        )

    @property
    def user_resources(self) -> Resources:
        return Resources(
            information=[
                "The book you are reading is called 'The Hitchhiker's Guide to the Galaxy'.",
            ],
        )

    # Agent section
    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="helpful AI agent",
            name="Lisa",
            background="You are a helpful AI agent who can answer questions.",
            personality="You are friendly and helpful to the user. You are always concise and to the point.",
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal="Answer the questions from user, save the answer to the question to the `SaveQuestionAnswerTool` tool, and end the conversation with the `EndConversationTool` tool.",
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Hello, I'm Lisa, what can I help you with?', say it only once at the beginning of the conversation.",
                "Answer a question from the user",
                "Use the `SaveQuestionAnswerTool` tool to log your answer to the user's question.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
            guidelines=[
                "Always answer the questions from the user",
                "After answering a question, ask the user if they have any other questions.",
                "After you have answered a question, use the `SaveQuestionAnswerTool` tool to log your answer to the user's question.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "EndConversationTool": {},
                "SaveQuestionAnswerTool": {},
            },
            information=[
                "The user is reading a book called 'The Hitchhiker's Guide to the Galaxy'.",
            ],
        )


@register_eval_scenario
class SimpleQA2(SimpleQA):
    """
    Simple QA scenario with the answer to the question 'What is 1 plus 1?' as 2.
    """

    name = "simple_qa_2"
    description = "Simple QA example scenario with the answer to the question 'What is 1 plus 1?'."
    reference_answer = {"question": "What is 1 plus 1?", "answer": "The answer is 2."}

    ignore_capitalization = True
    ignore_punctuation = True
    clean_text = True

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="John",
            background="You are a curious human who wants to ask a question to an AI agent.",
            personality="You are communicative and positive, with clear needs, friendly demeanor, and prompt decision-making.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Ask a question to the AI agent and wait for the answer.",
            background="You are reading a book about some science fiction story.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the question: 'What is the result of 1 plus 1?'",
            ],
            guidelines=[
                "Only ask the designated question to the agent, do not ask any other questions.",
                "If your question is answered, and the agent is asking if you have any other questions, say 'No, that's all I have.'",
                "Say 'Thank you for your answer, goodbye.' after the agent has answered your question.",
            ],
        )

    @property
    def user_resources(self) -> Resources:
        return Resources()

    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="helpful AI agent",
            name="Lisa",
            background="You are a helpful AI agent who can answer questions.",
            personality="You are friendly and helpful to the user. You are always concise and to the point.",
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "EndConversationTool": {},
                "SaveQuestionAnswerTool": {},
            },
        )


@register_eval_scenario
class SimpleQA3(SimpleQA2):
    name = "simple_qa_3"
    description = "Simple QA example scenario."
    reference_answer = {
        "question": "What is the weather in San Francisco?",
        "answer": "The weather in San Francisco is sunny with a temperature of 20 degrees Celsius and a low UV index.",
    }

    ignore_capitalization = True
    ignore_punctuation = True
    clean_text = True

    def get_user_prompt(self) -> str:
        return (
            """You are a friendly human user named Bob, and you are testing a voice assistant. Speak exactly as the following sentences one by one, wait for response before saying the next sentence:
             1. "Hi I'm Bob, what is weather in San Francisco?" 
             2. "Thank you for your answers, goodbye".\n
        """
            + self.general_prompt
        )

    def get_agent_prompt(self) -> str:
        return (
            """
        You are a helpful AI agent named Lisa. 
        Start by greeting the user with 'Hi, I'm Lisa, your helpful AI assistant. How can I help you today?'.
        Then answer the questions from the user one by one. 
        After you have answered a question, use the `SaveQuestionAnswerTool` tool to log your answer to the user's question.
        Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions. 
        """
            + self.general_prompt
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
