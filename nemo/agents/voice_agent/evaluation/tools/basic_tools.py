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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.services.llm_service import FunctionCallParams

from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.evaluation.tools.rtvi_control import SendExitMessageTool, SendScenarioSummaryTool
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool


@register_schema_tool_for_eval
class GetCityWeatherTool(StandardSchemaTool):
    """
    Get the weather of a city.
    """

    DESCRIPTION: str = """
        Get the weather of a city. You need to provide the city name to get the weather.
        """

    def __init__(self, *, description: Optional[str] = None):
        if description is None:
            description = self.DESCRIPTION
        super().__init__(description=description)

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "city_name": {
                "type": "string",
                "description": "The name of the city to get the weather of.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["city_name"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Get the weather of a city.
        """
        city_name = params.arguments.get("city_name")
        logger.debug(f"Getting weather of {city_name}")
        results = {
            "city": city_name,
            "weather": "sunny",
            "temperature": "20 degrees Celsius",
            "uv_index": "low",
        }
        await params.result_callback(results)


@register_schema_tool_for_eval
class ReadFileTool(StandardSchemaTool):
    """
    Read a file.
    """

    DESCRIPTION: str = """
        Read a file from the file system. You need to provide the file path to read the file.
        """

    def __init__(self):
        super().__init__(description=self.DESCRIPTION)

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "file_path": {
                "type": "string",
                "description": "The path of the file to read.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["file_path"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Read a file from the file system.
        """
        file_path = params.arguments.get("file_path")
        logger.debug(f"Reading file from {file_path}")
        try:
            with Path(file_path).open("r") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            await params.result_callback({"error": str(e)})
            return

        logger.debug(f"Loaded file {file_path} with content: `{content}`")
        results = {
            "file_path": file_path,
            "content": content,
        }
        await params.result_callback(results)


@register_schema_tool_for_eval
class PlaceOrderTool(SendScenarioSummaryTool):
    """
    Place order tool.
    """

    def __init__(
        self, *, rtvi: Optional[RTVIProcessor] = None, auto_validate: bool = False, valid_item_names: List[str] = None
    ):
        """
        Args:
            rtvi: The RTVI processor to use for sending messages to the evaluator.
            auto_validate: Whether to automatically validate the order items and total price.
        """
        super().__init__(description="Place an order for the customer", rtvi=rtvi)
        self.required_item_keys = ["name", "unit_price", "quantity"]
        self.auto_validate = auto_validate
        self.valid_item_names = valid_item_names

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "items": {
                "type": "list",
                "description": "A list of items to be ordered, each item is a dictionary with the following keys: name, unit_price, and quantity. For example, [{'name': 'xxx', 'unit_price': '1.00', 'quantity': '1'}, {'name': 'yyy', 'unit_price': '2.00', 'quantity': '2'}].",
            },
            "customer_name": {
                "type": "string",
                "description": "The name of the customer.",
            },
            "customer_phone": {
                "type": "string",
                "description": "The phone number of the customer.",
            },
            "total_price": {
                "type": "float",
                "description": "The total price of the order.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["items", "customer_name", "total_price"]

    def _validate_item(self, item: Dict[str, Any]) -> None:
        """
        Validate an item in the order.

        Args:
            item: The item to be validated.
        """
        if not self.auto_validate:
            return
        for key in self.required_item_keys:
            if key not in item:
                raise ValueError(
                    f"Each item in the `items` parameter must have a `{key}` key, but the item is: {item}."
                )
        if self.valid_item_names and item["name"] not in self.valid_item_names:
            raise ValueError(
                f"The item {item['name']} is not on the menu. All valid item names are: {self.valid_item_names}."
            )
        if float(item["unit_price"]) < 0:
            raise ValueError(
                f"The unit price of the item {item['name']} is negative. The unit price must be non-negative."
            )
        if float(item["quantity"]) < 0:
            raise ValueError(
                f"The quantity of the item {item['name']} is negative. The quantity must be non-negative."
            )

    async def _execute(self, params: FunctionCallParams) -> None:
        items = params.arguments.get("items")
        customer_name = params.arguments.get("customer_name")
        customer_phone = params.arguments.get("customer_phone", None)
        total_price = params.arguments.get("total_price")

        if customer_name is None:
            raise ValueError("The `customer_name` parameter is required.")
        if total_price is None:
            raise ValueError("The `total_price` parameter is required.")
        if items is None:
            raise ValueError("The `items` parameter is required.")

        # check order items and calculate the total price
        calculated_total_price = 0.0
        for item in items:
            self._validate_item(item)
            calculated_total_price += float(item["unit_price"]) * float(item["quantity"])
        if self.auto_validate and not np.isclose(calculated_total_price, float(total_price)):
            raise ValueError(
                f"The total price is incorrect. The calculated total price is {calculated_total_price} but the expected total price is {total_price}."
            )

        order_details = {
            "items": items,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "total_price": total_price,
        }

        order_details = json.dumps(order_details)
        # send the scenario summary message to the RTVI client
        await self.send_scenario_summary(order_details)
        results = {
            "success": True,
            "message": f"The order has been placed successfully for the customer {customer_name} with the total price {total_price}.",
            "order_details": order_details,
        }
        await params.result_callback(results)


@register_schema_tool_for_eval
class SaveQuestionAnswerTool(SendScenarioSummaryTool):
    """
    Send an answer to the user.
    """

    def __init__(self, *, rtvi: Optional[RTVIProcessor] = None, description: Optional[str] = None):
        if description is None:
            description = "Save a question and answer pair to the conversation history for future reference."
        super().__init__(description=description, rtvi=rtvi)

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "question": {
                "type": "string",
                "description": "The question to that user asked.",
            },
            "answer": {
                "type": "string",
                "description": "The agent's answer to the question.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["question", "answer"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Send an answer to the user.
        """
        question = params.arguments.get("question")
        answer = params.arguments.get("answer")
        message = {
            "question": question,
            "answer": answer,
        }
        message = json.dumps(message)
        await self.send_scenario_summary(message)
        await params.result_callback({"success": True, "message": "Question and answer logged."})


@register_schema_tool_for_eval
class EndConversationTool(SendExitMessageTool):
    """
    End the conversation with the user.
    """

    def __init__(self, *, rtvi: Optional[RTVIProcessor] = None, description: Optional[str] = None):
        if description is None:
            description = "End the conversation with the user."
        super().__init__(description=description, rtvi=rtvi)
