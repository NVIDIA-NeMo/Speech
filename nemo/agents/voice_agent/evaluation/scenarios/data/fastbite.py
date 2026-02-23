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

from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Scenario, Task
from nemo.agents.voice_agent.utils.audio import NoiseConfig

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


@register_eval_scenario
class FastBiteScenario(Scenario):
    """
    FastBite scenario.
    """

    name = "fastbite"
    description = "FastBite example scenario"
    reference_answer = {
        "items": [
            {"name": "chicken sandwich", "unit_price": "6.49", "quantity": "1"},
            {"name": "side salad", "unit_price": "2.99", "quantity": "1"},
        ],
        "customer_name": "John",
        "customer_phone": "123-456-7890",
        "total_price": "12.98",
    }
    expected_format = {
        "items": [
            {"name": "???", "unit_price": "???", "quantity": "???"},
            {"name": "???", "unit_price": "???", "quantity": "???"},
            {"name": "???", "unit_price": "???", "quantity": "???"},
        ],
        "customer_name": "???",
        "customer_phone": "???",
        "total_price": "???",
    }

    noise_config = NoiseConfig(random_white_noise=True, white_noise_db=-40.0)

    max_duration = 180

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
            goal="Order a chicken sandwich and a side salad.",
            background="You are hungry and just arrived at a restaurant called FastBites.",
            reference=json.dumps(self.reference_answer),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent for the available options for sandwiches.",
                "Order a chicken sandwich.",
                "Add a side salad to the order.",
                "Finish the order and ask for the price.",
            ],
            guidelines=[
                "If asked about whether to get a combo deal, say 'No, I don't want a combo deal.'",
                "Do not order any items other than one chicken sandwich and one side salad.",
                "Provide your name and/or phone number when asked for them.",
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
                "Greet the user by saying 'Welcome to FastBites! I'm Lisa, what can I help you with?'.",
                "Ask the user for what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and associate it with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu",
                "If the customer ask for a sandwich or burger, always ask if they want to make it into a combo deal.",
                "Always use the `PlaceOrderTool` tool to place the order.",
                "Before placing the order, ask for the user's name and associate it with the order.",
                "Always confirm with the user if the order is correct before placing the order with the `PlaceOrderTool` tool.",
                "After you have successfully placed the order, use the `EndConversationTool` tool to end the conversation.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "EndConversationTool": {},
                "PlaceOrderTool": {
                    "auto_validate": "False"
                },  # Don't let agent correct itself if the order is incorrect
            },
            information=[
                f"The menu of the restaurant is:\n{FASTBITE_MENU}",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items."
                f"The expected order format is: {self.expected_format}.",
            ],
        )
