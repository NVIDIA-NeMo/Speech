# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class ToolCallingMixin:
    """
    A mixin class for tool calling.
    Subclasses must implement the `setup_tool_calling` method to register all available tools
    using `self.register_direct_function()`. Then the `__init__` method of the subclass should
    call the `setup_tool_calling` method to register the tools.
    """

    def setup_tool_calling(self):
        """
        Setup the tool calling mixin by registering all available tools using self.register_direct_function().
        """
        raise NotImplementedError(
            "Subclasses must implement this method to register all available functions "
            "using self.register_direct_function()"
        )

    def register_direct_function(self, function_name: str, function: DirectFunction):
        """
        Register a direct function to be called by the LLM.

        Args:
            function_name: The name of the function to register.
            function: The direct function to register.
        """
        if not hasattr(self, "direct_functions"):
            self.direct_functions = {}
        logger.info(
            f"[{self.__class__.__name__}] Registering direct function name {function_name} to "
            f"{function.__module__ + '.' + function.__qualname__}"
        )
        self.direct_functions[function_name] = function

    @property
    def available_tools(self) -> dict[str, DirectFunction]:
        """
        Return a dictionary of available tools, where the key is the tool name and the value is the direct function.
        """
        tools = {}
        for function_name, function in self.direct_functions.items():
            tools[function_name] = function
        return tools


def register_direct_tools_to_llm(
    *,
    llm: OpenAILLMService,
    context: OpenAILLMContext,
    tool_mixins: list[ToolCallingMixin] = [],
    tools: list[DirectFunction] = [],
    cancel_on_interruption: bool = True,
) -> None:
    """
    Register direct tools to the LLM.
    Args:
        llm: The LLM service to use.
        context: The LLM context to use.
        tools: The list of tools (instances of either `DirectFunction` or `ToolCallingMixin`) to use.
    """
    all_tools = []
    for tool in tool_mixins:
        if not isinstance(tool, ToolCallingMixin):
            logger.warning(f"Tool {tool.__class__.__name__} is not a ToolCallingMixin, skipping.")
            continue
        for function_name, function in tool.available_tools.items():
            logger.info(f"Registering direct function {function_name} from {tool.__class__.__name__}")
            all_tools.append(function)

    for tool in tools:
        logger.info(f"Registering direct function: {tool.__module__ + '.' + tool.__qualname__}")
        all_tools.append(tool)

    if not all_tools:
        logger.warning("No direct tools provided.")
        return
    else:
        logger.info(f"Registering {len(all_tools)} direct tools to the LLM.")

    tools_schema = ToolsSchema(standard_tools=all_tools)
    context.set_tools(tools_schema)

    for tool in all_tools:
        llm.register_direct_function(tool, cancel_on_interruption=cancel_on_interruption)
