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
from typing import Dict, Any, List, Optional
import numpy as np
from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Persona, Actions, Resources, Task, Scenario
from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.evaluation.tools.rtvi_control import SendScenarioSummaryTool
from nemo.agents.voice_agent.utils.audio import NoiseConfig
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.services.llm_service import FunctionCallParams

FASTBITE_MENU_ITEMS = [
    "Classic Cheeseburger", 
    "Crispy Chicken Sandwich", 
    "Veggie Wrap", 
    "Fish Sandwich", 
    "Cheeseburger Combo", 
    "Chicken Sandwich Combo", 
    "Veggie Wrap Combo", 
    "French Fries", 
    "Chicken Nuggets", 
    "Side Salad", 
    "Fountain Soda", 
    "Iced Tea", 
    "Lemonade", 
    "Bottled Water",
]
FASTBITE_MENU = f"""
## Fast Bites Lunch Menu

Burgers and Sandwiches:
1. Classic Cheeseburger 
   - price: $6.99
   - Juicy beef patty, cheddar cheese, pickles, ketchup & mustard on a toasted bun.
2. Crispy Chicken Sandwich 
   - price: $6.49
   - Fried chicken filet, lettuce, mayo, and pickles on a brioche bun.
3. Veggie Wrap 
   - price: $5.49
   - Grilled vegetables, hummus, lettuce, and tomato in a spinach wrap.
4. Fish Sandwich 
   - price: $5.99
   - Fried fish filet, lettuce, mayo, and pickles on a brioche bun.

Combo Deals (includes small fries and fountain soda)
4. Cheeseburger Combo 
   - price: $8.99
5. Chicken Sandwich Combo 
   - price: $8.49
6. Veggie Wrap Combo 
   - price: $7.49
7. Fish Sandwich Combo 
   - price: $7.99

Sides:
7. French Fries
 - Small - $1.49
 - Medium - $1.89
 - Large - $2.29
8. Chicken Nuggets
 - 4 pieces - $2.29
 - 8 pieces -  $3.99
 - 12 pieces -  $5.99
9. Side Salad -  $2.99

Drinks:
10. Fountain Soda 
   - price: $1.99
   - choices: Coke, Diet Coke, Sprite, Fanta
11. Iced Tea
   - price: $2.29
12. Lemonade
   - price: $2.29
13. Bottled Water
   - price: $1.49


All valid item names are: {FASTBITE_MENU_ITEMS}.
"""


@register_schema_tool_for_eval
class PlaceOrderTool(SendScenarioSummaryTool):
    """
    Place order tool.
    """
    def __init__(self, *, rtvi: Optional[RTVIProcessor] = None, auto_validate: bool = False):
        """
        Args:
            rtvi: The RTVI processor to use for sending messages to the evaluator.
            auto_validate: Whether to automatically validate the order items and total price.
        """
        super().__init__(description="Place an order for the customer", rtvi=rtvi)
        self.required_item_keys = ["name", "unit_price", "quantity"]
        self.auto_validate = auto_validate
    
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
                raise ValueError(f"Each item in the `items` parameter must have a `{key}` key, but the item is: {item}.")
        if item["name"] not in FASTBITE_MENU_ITEMS:
            raise ValueError(f"The item {item['name']} is not on the menu. All valid item names are: {FASTBITE_MENU_ITEMS}.")
        if float(item["unit_price"]) < 0:
            raise ValueError(f"The unit price of the item {item['name']} is negative. The unit price must be non-negative.")
        if float(item["quantity"]) < 0:
            raise ValueError(f"The quantity of the item {item['name']} is negative. The quantity must be non-negative.")

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
            raise ValueError(f"The total price is incorrect. The calculated total price is {calculated_total_price} but the expected total price is {total_price}.")

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


@register_eval_scenario
class FastBiteScenario(Scenario):
    """
    FastBite scenario.
    """
    name = "fastbite"
    description = "FastBite example scenario"
    reference_answer = {
        "items": [{"name": "chicken sandwich", "unit_price": "6.49", "quantity": "1"}, {"name": "side salad", "unit_price": "2.99", "quantity": "1"}],
        "customer_name": "John",
        "customer_phone": "123-456-7890",
        "total_price": "12.98",
    }
    expected_format = {
        "items": [{"name": "???", "unit_price": "???", "quantity": "???"}, {"name": "???", "unit_price": "???", "quantity": "???"}, {"name": "???", "unit_price": "???", "quantity": "???"}],
        "customer_name": "???",
        "customer_phone": "???",
        "total_price": "???",
    }

    def __init__(self, *, rtvi: RTVIProcessor, noise_config: Optional[NoiseConfig] = None):
        super().__init__(
            rtvi=rtvi,
            noise_config=noise_config or NoiseConfig(random_white_noise=True, white_noise_db=-40.0),
            name=self.name,
            description=self.description,
        )

    # User section
    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="John",
            background="You work at NVIDIA as a software engineer. Your phone number is 123-456-7890.",
            personality="You are communicative and positive, with clear needs, friendly demeanor, and prompt decision-making.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a chicken sandwich and a side salad",
            background="You are hungry and just arrived at a restaurant called FastBites.",
            reference=json.dumps(self.reference_answer),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent for the available options for sandwiches",
                "Order a chicken sandwich",
                "Add a side salad to the order",
                "Finish the order and ask for the prize",
            ],
            guidelines=[
                "If asked about whether to get a combo deal, decline it",
                "Do not order any items other than a chicken sandwich and a side salad",
            ],
        )

    @property
    def user_resources(self) -> Resources:
        return Resources(
            information=["The restaurant is called FastBites and it is famous for its sandwiches."],
        )

    # Agent section
    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="helpful AI agent",
            name="Lisa",
            background="You are a helpful AI agent who serves as a restaurant assistant at FastBites to help customers order food from the lunch menu.",
            personality="You are friendly and helpful to the user. You can guide the user to finish their task when they show hesitation. You are always concise and to the point.",
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal="Help the user to order food at the restaurant.",
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'welcome to FastBites! I'm Lisa, what can I help you with?'",
                "Ask the user for what they would like to order and help them make the order",
                "Summarize the order and confirm with the user if the order is correct",
                "Ask the user for their name and associate it with the order",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully",
                "Thank the user for their order and say goodbye, and use the `SendExitMessageTool` tool to send the `order_details` returned by the `PlaceOrderTool` tool",
            ],
            guidelines=[
                "Do not make up any items not on the menu",
                "If the customer ask for a sandwich or burger, always ask if they want to make it into a combo deal",
                "Always use the `PlaceOrderTool` tool to place the order",
                "Always confirm with the user if the order is correct before placing the order with the `PlaceOrderTool` tool",
                "At the end, always use the `SendExitMessageTool` tool to send the `order_details` returned by the `PlaceOrderTool` tool",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "SendExitMessageTool": {},
                "PlaceOrderTool": {"auto_validate": "False"},  # Don't let agent correct itself if the order is incorrect
            },
            information=[
                f"The menu of the restaurant is:\n{FASTBITE_MENU}",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items."
                f"The expected order format is: {self.expected_format}"
            ],
        )