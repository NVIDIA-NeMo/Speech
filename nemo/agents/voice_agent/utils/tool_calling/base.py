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

from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService


class StandardSchemaTool:
    """
    Base class for all standard tools with FunctionSchema.
    """

    def __init__(self, *, description: str, name: Optional[str] = None):
        self.name = name if name is not None else self.__class__.__name__
        self.description = description
        if not self.name:
            raise ValueError(f"Name is required for tool {self.__class__}")
        if not self.description:
            raise ValueError(f"Description is required for tool {self.__class__}")

    @property
    def schema(self) -> FunctionSchema:
        """
        Return the FunctionSchema for the tool. Refer to
        https://docs.pipecat.ai/guides/learn/function-calling#using-the-standard-schema-recommended
        for more details.

        An example of the FunctionSchema:
        ```
        schema = FunctionSchema(
            name="get_current_weather",
            description="Get the current weather in a location",
            properties={
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            required=["location", "format"]
        )
        ```
        """
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=self.properties,
            required=self.required_properties,
        )

    async def __call__(self, params: FunctionCallParams) -> None:
        """
        The actual tool calling logic, push back the results to the LLM.
        """
        try:
            results = await self._execute(params)
        except Exception as e:
            logger.error(f"Error in tool calling: {e}")
            await params.result_callback({"error": str(e)})
            return
        await params.result_callback(results)

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.

        An example of the properties:
        ```
        properties = {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
            },
        }
        ```
        """
        raise NotImplementedError("Subclasses must implement this method to return the properties for the tool.")

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.

        An example of the required properties:
        ```
        required_properties = ["location", "format"]
        ```
        """
        raise NotImplementedError(
            "Subclasses must implement this method to return the required properties for the tool."
        )

    async def _execute(self, params: FunctionCallParams) -> Dict[str, Any]:
        """
        The actual tool execution logic.

        An example of get_current_weather tool where it returns the weather information as a dictionary:
        ```
        results = {
            "location": "San Francisco, CA",
            "format": "celsius",
        }
        ```
        """
        raise NotImplementedError("Subclasses must implement this method to implement the tool logic.")


def register_schema_tools_to_llm(
    llm: OpenAILLMService,
    context: OpenAILLMContext,
    tools: List[StandardSchemaTool],
    cancel_on_interruption: bool = True,
    keep_existing_tools: bool = True,
) -> None:
    """
    Register standard schema tools to the LLM.
    Args:
        llm: The LLM service to use.
        context: The LLM context to use.
        tools: The list of tools to register.
        cancel_on_interruption: Whether to cancel the LLM call on interruption.
        keep_existing_tools: Whether to keep the existing tools in the context.
    """
    all_schemas = []
    for tool in tools:
        if not isinstance(tool, StandardSchemaTool):
            logger.warning(f"Tool {tool.__class__.__name__} is not a `StandardSchemaTool`, skipping...")
            continue
        all_schemas.append(tool.schema)
        logger.info(f"Registering standard schematool `{tool.name}` with schema properties: {tool.schema.properties}")
        llm.register_function(
            function_name=tool.name,
            handler=tool,
            cancel_on_interruption=cancel_on_interruption,
        )
    if keep_existing_tools:
        existing_tools = context.tools
        if not isinstance(existing_tools, list):
            existing_tools = []
        all_schemas.extend(existing_tools)
    tools_schema = ToolsSchema(standard_tools=all_schemas)
    context.set_tools(tools_schema)
