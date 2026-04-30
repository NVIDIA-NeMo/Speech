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

# Scenario definitions contain long prose strings (personas, instructions, account data);
# wrapping every one hurts readability without improving correctness.
# pylint: disable=line-too-long
# flake8: noqa: E501
from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessage, RTVITextMessageData
from pipecat.services.llm_service import FunctionCallParams

from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool

FINAL_RESPONSE_START_TAG = "<final_response>"
FINAL_RESPONSE_END_TAG = "</final_response>"
EXIT_MESSAGE_START_TAG = "<exit>"
EXIT_MESSAGE_END_TAG = "</exit>"


@register_schema_tool_for_eval
class SendRTVIMessageTool(StandardSchemaTool):
    """
    Send a scenario finished message to the evaluator.
    """

    DESCRIPTION: str = """
        Send a message to the orchestrator.
        """

    def __init__(self, *, description: Optional[str] = None, rtvi: Optional[RTVIProcessor] = None):
        if description is None:
            description = self.DESCRIPTION
        if rtvi is None:
            rtvi = RTVIProcessor()
        super().__init__(description=description)
        self._rtvi = rtvi

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "message": {
                "type": "string",
                "description": "The message to be sent in the required format.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["message"]

    async def send_rtvi_message(self, message: str) -> None:
        """
        Send a message.

        Args:
            message: The message to be sent.
        """
        message = RTVIServerMessage(data=RTVITextMessageData(text=message))
        await self._rtvi.push_transport_message(message, exclude_none=True)

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Send a message.

        Args:
            params: The function call parameters.
        """
        message = params.arguments.get("message")
        await self.send_rtvi_message(message)
        await params.result_callback({"success": True, "message": "message sent to the RTVIclient."})


@register_schema_tool_for_eval
class SendScenarioSummaryTool(SendRTVIMessageTool):
    """
    Send a "Scnario Summary" message after the user has no more requests
    and the agent has answered all the user's questions The input message should contain all required information
    in the required format.
    """

    def __init__(self, *, rtvi: Optional[RTVIProcessor] = None, description: Optional[str] = None):
        if description is None:
            description = """
            Send a "Task Summary" message to summarize how the agent has helped the user to finish the task.
            """
        super().__init__(description=description, rtvi=rtvi)

    async def send_scenario_summary(self, message: str) -> None:
        """
        Send a "Scnario Summary" message.

        Args:
            message: The message to be sent.
        """
        message = f"{FINAL_RESPONSE_START_TAG}{message}{FINAL_RESPONSE_END_TAG}"
        logger.debug(f"Sending scenario summary message: {message}")
        await self.send_rtvi_message(message)

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Send a "Scnario Summary" message to the client, which
        should contain all required information for the evaluation.
        """
        message = params.arguments.get("message")
        await self.send_scenario_summary(message)
        await params.result_callback({"success": True, "message": "Scenario summary message sent."})


@register_schema_tool_for_eval
class SendExitMessageTool(SendRTVIMessageTool):
    """
    Send an "Exit" message to indicate that the scenario is finished.
    """

    def __init__(self, rtvi: RTVIProcessor, description: Optional[str] = None):
        if description is None:
            description = (
                'Send an "Exit" message to the orchestrator to indicate that the task is finished, '
                "and it's safe to stop the pipeline. This tool should only be used when the user "
                "has no more requests and the agent has answered all the user's questions."
            )
        super().__init__(description=description, rtvi=rtvi)

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    async def send_exit_message(self, message: str = "The task is finished.") -> None:
        """
        Send an "Exit" message.

        Args:
            message: The message to be sent.
        """
        message = f"{EXIT_MESSAGE_START_TAG}{message}{EXIT_MESSAGE_END_TAG}"
        logger.debug(f"Sending exit message: {message}")
        await self.send_rtvi_message(message)

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Send an "Exit" message.

        Args:
            params: The function call parameters.
        """
        message = "The task is finished."
        await self.send_exit_message(message)
        await params.result_callback({"success": True, "message": "Exit message sent."})
