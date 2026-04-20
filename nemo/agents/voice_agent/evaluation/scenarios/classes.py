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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nemo.agents.voice_agent.utils.audio import NoiseConfig

GENERAL_PROMPT = "Keep your responses concise and conversational since they will be spoken aloud. Avoid special characters. Use only simple, plain text sentences. Always punctuate your responses using standard sentence punctuation: commas, periods, question marks, exclamation points, etc. Always spell out numbers as words. Avoid using emojis."


@dataclass
class Persona:
    """
    Persona configuration for the scenario.

    Attributes:
        role: The role of the persona, e.g., "human user" or "helpful AI agent".
        name: The name of the persona, e.g. "Bob", "Lisa", "Charlie", etc.
              The name and role will be combined to a sentence like "You are a {role} named {name}." in the system prompt.
        background: The background of the persona, e.g.,
            - For user: "You are a student who is studying at the university. You like to play basketball in your free time"
            - For agent: "You are a helpful AI agent who can help the user with their questions and tasks."
        personality: Detailed description on the personality of the persona.
            For example:
            - For user:
              - "You are determined and straightforward, but sometime you make mistakes."
              - "You are Passive in communication, unclearneeds, repeatedly seeks confirmation, and slow in decision-making."
            - For agent:
              - "You have a great sense of humor while being helpful and friendly to the user. Your responses are concise and conversational."
              - "You are friendly and helpful to the user. You can guide the user to finish their task when they show hesitation."
        language: The language used by the persona, e.g. "English", "Chinese", "Spanish", etc. Only used for TTS generation. If provided, the prompt will have additional information about the language.
        accent: The accent of the persona if any, e.g. "American", "British", "Australian", etc. Only used for TTS generation. If provided, the prompt will have additional information about the accent.
    """

    role: str
    name: str
    background: str
    personality: str
    language: Optional[str] = None
    accent: Optional[str] = None

    def to_prompt_section(self) -> str:
        lines = [f"You are a {self.role} named {self.name}."]
        general_prompt = f"You need to stick to your designated role and complete your task by following the information below. {GENERAL_PROMPT}"
        if self.background:
            lines.append(self.background)
        if self.personality:
            lines.append(self.personality)
        if self.language and self.accent:
            lines.append(f"You speak {self.language} with a {self.accent} accent.")
        elif self.language:
            lines.append(f"You speak {self.language}.")
        elif self.accent:
            lines.append(f"You speak with a {self.accent} accent.")
        lines.append(general_prompt)
        return "\n".join(lines)


@dataclass
class Resources:
    """
    Resources configuration for the scenario.

    Attributes:
        tools: A dictionary of available tools, where the key is the tool name and the value is a dictionary of tool arguments to be passed to the tool constructor.
        documents: A dictionary of available documents, where the key is the document name and the value is a file path. The file can be read by using a `read_file` tool.
        information: A list of additional information strings. For example, the agent will have some FAQs or other information that is relevant to the scenario.
    """

    tools: Dict[str, Dict[str, str]] = field(default_factory=dict)
    documents: Dict[str, str] = field(default_factory=dict)
    information: List[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        sections = ["# Resources"]
        if self.documents:
            doc_list = "\n".join(f"- {name}: {path}" for name, path in self.documents.items())
            sections.append(
                f"## Available Documents\nYou can read the following documents by using tools:\n{doc_list}"
            )
        if self.information:
            info_list = "\n".join(f"- {info}" for info in self.information)
            sections.append(
                f"## Additional Information\nYou can use the following information for reference:\n{info_list}"
            )
        return "\n\n".join(sections)

    def to_tools_json_string(self) -> str:
        """
        Get the tools for the scenario in a json string.
        """
        return json.dumps(self.tools) if self.tools else "{}"


@dataclass
class Task:
    """
    Task configuration for the scenario.

    Attributes:
        goal: The goal of the task for user/agent. For example:
            - For user: "Order a chicken sandwich and a side salad"
            - For agent: "Help the user to order food at the restaurant."
        background: The background of the task for user/agent. For example:
            - For user: "You are hungry and just arrived at a pizza restaurant. "
            - For agent: "You are a restaurant assistant who wants to help the user to order food at the restaurant."
    """

    goal: str
    background: str = field(default="")

    def to_prompt_section(self) -> str:
        prompt = "# Task\n\n"
        if self.background:
            prompt += self.background + "\n"
        prompt += f"Your goal is to: {self.goal}"
        return prompt


@dataclass
class Actions:
    """
    Actions configuration for the scenario.

    Attributes:
        instructions: An itemized list of instructions for the user/agent must follow step by step in order to complete the task.
        For example, for a task of ordering a pizza:
            - For user: [
                            "Ask the agent for the available pizza options",
                            "Order a pepperoni pizza and ask for the prize",
                            "Ask the agent if extra cheese is available and add it if available",
                            "Finish the order and ask for the prize",
                        ]
            - For agent: [
                            "Greet the user by saying 'welcome to the pizza restaurant! How can I help you today?'",
                            "Ask the user for what they would like to order and help them make the order",
                            "Summarize the order and confirm with the user if the order is correct",
                            "Ask the user for their name and associate it with the order",
                            "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully",
                            "Thank the user for their order and say goodbye",
                        ]

        guidelines: An itemized list of guidelines that the user/agent must comply with. For example:
            - For user: [
                            ""
                        ]
            - For agent: [
                            "Do not make up any items not on the menu",
                            "Always use the `PlaceOrderTool` tool to place the order",
                            "Always confirm with the user if the order is correct before placing the order",
                        ]
    """

    instructions: List[str] = field(default_factory=list)
    guidelines: List[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        sections = ["# Actions"]
        if self.instructions:
            header = "You must follow the following instructions step by step in the given order to complete the task, do not perform multiple instructions in a single turn:\n"
            numbered = "\n".join(f"Step {i+1}: {inst}" for i, inst in enumerate(self.instructions))
            sections.append(f"## Instructions\n{header}{numbered}")
        if self.guidelines:
            header = "You must always comply with the following guidelines during the task:\n"
            bulleted = "\n".join(f"- {r}" for r in self.guidelines)
            sections.append(f"## Guidelines\n{header}{bulleted}")
        return "\n\n".join(sections)


class Scenario:
    """Base class for all evaluation scenarios."""

    def __init__(
        self,
        *,
        noise_config: Optional[NoiseConfig] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_duration: Optional[int] = None,
        reference_answer: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        ignore_capitalization: Optional[bool] = False,
        ignore_punctuation: Optional[bool] = False,
        clean_text: Optional[bool] = False,
    ):
        """
        Initialize the scenario.

        Args:
            rtvi: The RTVI processor to use for sending messages to the evaluator.
            noise_config: The noise configuration to use for the scenario.
            name: The name of the scenario.
            description: The description of the scenario.
            max_duration: The max duration of the scenario in seconds.
            reference_answer: The reference answer for the scenario, must be able to be converted to a json string.
            ignore_capitalization: Whether to ignore capitalization when comparing the reference answer and the final agent response.
            ignore_punctuation: Whether to ignore punctuation when comparing the reference answer and the final agent response.
            clean_text: Whether to clean the text when comparing the reference answer and the final agent response.
        """
        if not hasattr(self, "name"):
            self.name = name
        self.noise_config = noise_config
        if not hasattr(self, "description"):
            self.description = description
        if not hasattr(self, "max_duration"):
            self.max_duration = max_duration
        if not hasattr(self, "general_prompt"):
            self.general_prompt = GENERAL_PROMPT
        if not hasattr(self, "reference_answer"):
            self.reference_answer = reference_answer
        if not hasattr(self, "reference_file"):
            self.reference_file = "reference_answer.json"
        if not hasattr(self, "ignore_capitalization"):
            self.ignore_capitalization = ignore_capitalization
        if not hasattr(self, "ignore_punctuation"):
            self.ignore_punctuation = ignore_punctuation
        if not hasattr(self, "clean_text"):
            self.clean_text = clean_text

    def get_user_tools(self) -> str:
        """
        Get the tools for the user in a json string.
        The json string should be in the following format:
        ```
        {
            "tool_name_1": {
                "arg1_name": "value1",
                "arg2_name": "value2",
            },
            "tool_name_2": {
                "arg1_name": "value1",
                "arg2_name": "value2",
            },
            ...
        }
        ```
        """
        return self.user_resources.to_tools_json_string()

    def get_agent_tools(self) -> str:
        """
        Get the tools for the agent in a json string.

        The json string should be in the following format:
        ```
        {
            "tool_name_1": {
                "arg1_name": "value1",
                "arg2_name": "value2",
            },
            "tool_name_2": {
                "arg1_name": "value1",
                "arg2_name": "value2",
            },
            ...
        }
        ```
        """
        return self.agent_resources.to_tools_json_string()

    def get_user_prompt(self) -> str:
        """Get the user prompt for the scenario."""
        sections = []
        sections.append(self.user_persona.to_prompt_section())
        sections.append(self.user_task.to_prompt_section())
        sections.append(self.user_actions.to_prompt_section())
        resources_section = self.user_resources.to_prompt_section()
        if resources_section:
            sections.append(resources_section)
        prompt = "\n\n".join(s for s in sections if s)
        return prompt

    def get_agent_prompt(self) -> str:
        """Get the agent prompt for the scenario."""
        sections = []
        sections.append(self.agent_persona.to_prompt_section())
        sections.append(self.agent_task.to_prompt_section())
        sections.append(self.agent_actions.to_prompt_section())
        resources_section = self.agent_resources.to_prompt_section()
        if resources_section:
            sections.append(resources_section)
        prompt = "\n\n".join(s for s in sections if s)
        return prompt

    @property
    def user_task(self) -> Task:
        raise NotImplementedError("Subclasses must implement this method to return the user task.")

    @property
    def agent_task(self) -> Task:
        raise NotImplementedError("Subclasses must implement this method to return the agent task.")

    @property
    def user_resources(self) -> Resources:
        raise NotImplementedError("Subclasses must implement this method to return the user resources.")

    @property
    def agent_resources(self) -> Resources:
        raise NotImplementedError("Subclasses must implement this method to return the agent resources.")

    @property
    def user_actions(self) -> Actions:
        raise NotImplementedError("Subclasses must implement this method to return the user actions.")

    @property
    def agent_actions(self) -> Actions:
        raise NotImplementedError("Subclasses must implement this method to return the agent actions.")

    @property
    def user_persona(self) -> Persona:
        raise NotImplementedError("Subclasses must implement this method to return the user persona.")

    @property
    def agent_persona(self) -> Persona:
        raise NotImplementedError("Subclasses must implement this method to return the agent persona.")

    def save(self, output_dir: str):
        """Save the scenario to a file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save user prompt and tools
        user_prompt = self.get_user_prompt()
        user_tools = self.get_user_tools()
        agent_prompt = self.get_agent_prompt()
        agent_tools = self.get_agent_tools()

        with open(output_dir / "user_prompt.txt", "w") as f:
            f.write(user_prompt)
        with open(output_dir / "user_tools.json", "w") as f:
            json.dump(user_tools, f, indent=4)
        with open(output_dir / "agent_prompt.txt", "w") as f:
            f.write(agent_prompt)
        with open(output_dir / "agent_tools.json", "w") as f:
            json.dump(agent_tools, f, indent=4)

        # save metadata
        metadata = {
            "name": self.name,
            "description": self.description,
            "max_duration": self.max_duration,
            "noise_config": self.noise_config.to_dict() if self.noise_config else None,
            "ignore_capitalization": self.ignore_capitalization,
            "ignore_punctuation": self.ignore_punctuation,
            "clean_text": self.clean_text,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # save reference answer
        if self.reference_answer:
            # if the reference answer is a string, convert it to a dictionary
            if isinstance(self.reference_answer, str):
                reference_answer = {"message": self.reference_answer}
            else:
                reference_answer = self.reference_answer
            with open(output_dir / self.reference_file, "w") as f:
                json.dump(reference_answer, f, indent=4)
