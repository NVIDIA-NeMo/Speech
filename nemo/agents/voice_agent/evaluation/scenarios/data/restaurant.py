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

# ---------------------------------------------------------------------------
# Menu constants
# ---------------------------------------------------------------------------

PIZZA_PALACE_MENU_ITEMS = [
    "Pepperoni Pizza",
    "Margherita Pizza",
    "BBQ Chicken Pizza",
    "Veggie Supreme Pizza",
    "Hawaiian Pizza",
    "Extra Cheese",
    "Garlic Bread",
    "Caesar Salad",
    "Fountain Soda",
    "Iced Tea",
    "Bottled Water",
]
PIZZA_PALACE_MENU = f"""
## Pizza Palace Menu

Pizzas (12-inch):
1. Pepperoni Pizza
   - price: $9.99
   - Classic pepperoni with mozzarella cheese on hand-tossed dough.
2. Margherita Pizza
   - price: $8.99
   - Fresh mozzarella, tomatoes, and basil on a thin crust.
3. BBQ Chicken Pizza
   - price: $11.99
   - Grilled chicken, red onion, cilantro, and tangy BBQ sauce.
4. Veggie Supreme Pizza
   - price: $10.49
   - Bell peppers, mushrooms, olives, onions, and spinach.
5. Hawaiian Pizza
   - price: $10.49
   - Ham, pineapple, and mozzarella cheese.

Add-Ons:
6. Extra Cheese
   - price: $1.50
   - Add extra mozzarella to any pizza.

Sides:
7. Garlic Bread
   - price: $3.49
   - Toasted garlic bread with butter and herbs.
8. Caesar Salad
   - price: $4.99
   - Romaine lettuce, croutons, parmesan, and Caesar dressing.

Drinks:
9. Fountain Soda
   - price: $1.99
   - choices: Coke, Diet Coke, Sprite, Fanta
10. Iced Tea
    - price: $2.29
11. Bottled Water
    - price: $1.49

All valid item names are: {PIZZA_PALACE_MENU_ITEMS}.
"""

BURGER_BARN_MENU_ITEMS = [
    "Classic Burger",
    "Bacon Cheeseburger",
    "Mushroom Swiss Burger",
    "Spicy Jalapeno Burger",
    "Veggie Burger",
    "Classic Burger Combo",
    "Bacon Cheeseburger Combo",
    "Onion Rings",
    "French Fries",
    "Milkshake",
    "Fountain Soda",
    "Lemonade",
]
BURGER_BARN_MENU = f"""
## Burger Barn Menu

Burgers:
1. Classic Burger
   - price: $7.49
   - Beef patty, lettuce, tomato, onion, pickles, and house sauce on a sesame bun.
2. Bacon Cheeseburger
   - price: $8.99
   - Beef patty, crispy bacon, cheddar cheese, lettuce, and mayo.
3. Mushroom Swiss Burger
   - price: $9.49
   - Beef patty, sauteed mushrooms, Swiss cheese, and garlic aioli.
4. Spicy Jalapeno Burger
   - price: $8.99
   - Beef patty, jalapenos, pepper jack cheese, chipotle mayo.
5. Veggie Burger
   - price: $7.99
   - Plant-based patty, lettuce, tomato, avocado, and vegan mayo.

Combo Deals (includes medium fries and fountain soda):
6. Classic Burger Combo
   - price: $10.49
7. Bacon Cheeseburger Combo
   - price: $11.99

Sides:
8. Onion Rings
   - price: $3.49
   - Crispy battered onion rings.
9. French Fries
   - price: $2.49
   - Golden crispy fries.

Drinks:
10. Milkshake
    - price: $4.99
    - choices: Chocolate, Vanilla, Strawberry
11. Fountain Soda
    - price: $1.99
    - choices: Coke, Diet Coke, Sprite, Fanta
12. Lemonade
    - price: $2.29

All valid item names are: {BURGER_BARN_MENU_ITEMS}.
"""

DELI_DELIGHTS_MENU_ITEMS = [
    "Turkey Club Sandwich",
    "Italian Sub",
    "Grilled Cheese",
    "BLT Sandwich",
    "Tuna Melt",
    "Chicken Noodle Soup",
    "Tomato Soup",
    "Potato Chips",
    "Pickle Spear",
    "Fresh Squeezed OJ",
    "Coffee",
    "Bottled Water",
]
DELI_DELIGHTS_MENU = f"""
## Deli Delights Menu

Sandwiches:
1. Turkey Club Sandwich
   - price: $8.49
   - Sliced turkey, bacon, lettuce, tomato, and mayo on toasted sourdough.
2. Italian Sub
   - price: $9.49
   - Salami, ham, provolone, lettuce, tomato, onion, and Italian dressing on a hoagie roll.
3. Grilled Cheese
   - price: $5.99
   - Cheddar and Swiss cheese grilled on sourdough bread.
4. BLT Sandwich
   - price: $6.99
   - Crispy bacon, lettuce, and tomato with mayo on white bread.
5. Tuna Melt
   - price: $7.49
   - House-made tuna salad with melted cheddar on rye bread.

Soups:
6. Chicken Noodle Soup
   - price: $4.49
   - Hearty chicken soup with egg noodles and vegetables.
7. Tomato Soup
   - price: $3.99
   - Creamy tomato basil soup.

Sides:
8. Potato Chips
   - price: $1.49
   - Kettle-cooked potato chips.
9. Pickle Spear
   - price: $0.99
   - Crunchy dill pickle spear.

Drinks:
10. Fresh Squeezed OJ
    - price: $3.49
    - Freshly squeezed orange juice.
11. Coffee
    - price: $2.49
    - Freshly brewed drip coffee.
12. Bottled Water
    - price: $1.49

All valid item names are: {DELI_DELIGHTS_MENU_ITEMS}.
"""

# ---------------------------------------------------------------------------
# Expected order format (shared across all restaurant scenarios)
# ---------------------------------------------------------------------------

EXPECTED_ORDER_FORMAT = {
    "items": [
        {"name": "???", "unit_price": "???", "quantity": "???"},
    ],
    "customer_name": "???",
    "customer_phone": "???",
    "total_price": "???",
}


# ---------------------------------------------------------------------------
# Base class (NOT registered)
# ---------------------------------------------------------------------------


class RestaurantBaseScenario(Scenario):
    """
    Base class for all restaurant evaluation scenarios.
    Provides sensible defaults for agent persona, agent task, and user resources
    so that concrete subclasses only need to override what differs.
    """

    max_duration = 180
    ignore_capitalization = True
    ignore_punctuation = True
    clean_text = False
    noise_config = NoiseConfig(random_white_noise=True, white_noise_db=-20.0)

    # -- Agent defaults (shared across all restaurant scenarios) -------------

    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="helpful AI agent",
            name="Lisa",
            background="You are a helpful AI restaurant assistant who helps customers place food orders.",
            personality="You are friendly and helpful to the user. You can guide the user to finish their task when they show hesitation. You are always concise and to the point.",
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal="Help the user to order food at the restaurant.",
        )

    @property
    def user_resources(self) -> Resources:
        return Resources()


# ---------------------------------------------------------------------------
# Concrete registered scenarios
# ---------------------------------------------------------------------------


@register_eval_scenario
class PizzaPepperoni(RestaurantBaseScenario):
    """Order a pepperoni pizza with extra cheese at Pizza Palace."""

    name = "restaurant__pizza_pepperoni"
    description = "Order a pepperoni pizza with extra cheese at Pizza Palace"
    reference_answer = {
        "items": [
            {"name": "Pepperoni Pizza", "unit_price": "9.99", "quantity": "1"},
            {"name": "Extra Cheese", "unit_price": "1.50", "quantity": "1"},
        ],
        "customer_name": "Charlie",
        "customer_phone": "314-527-8960",
        "total_price": "11.49",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Charlie",
            background="You work as a graphic designer at a tech startup. Your phone number is 314-527-8960.",
            personality="You are communicative and positive, with clear needs, friendly demeanor, and prompt decision-making.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a pepperoni pizza with extra cheese.",
            background="You are hungry after work and just walked into Pizza Palace.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what pizza options are available.",
                "Order one pepperoni pizza.",
                "Ask if you can add extra cheese, and add it to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than one pepperoni pizza and one extra cheese.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Pizza Palace! I'm Lisa, how can I help you today?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": PIZZA_PALACE_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class PizzaVeggieCombo(RestaurantBaseScenario):
    """Order a veggie supreme pizza with garlic bread and an iced tea at Pizza Palace."""

    name = "restaurant__pizza_veggie_combo"
    description = "Order a veggie supreme pizza, garlic bread, and iced tea at Pizza Palace"
    reference_answer = {
        "items": [
            {"name": "Veggie Supreme Pizza", "unit_price": "10.49", "quantity": "1"},
            {"name": "Garlic Bread", "unit_price": "3.49", "quantity": "1"},
            {"name": "Iced Tea", "unit_price": "2.29", "quantity": "1"},
        ],
        "customer_name": "Diana",
        "customer_phone": "629-381-4075",
        "total_price": "16.27",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Diana",
            background="You are a yoga instructor who prefers vegetarian food. Your phone number is 629-381-4075.",
            personality="You are calm, health-conscious, and polite. You know exactly what you want and communicate it clearly.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a veggie supreme pizza, a side of garlic bread, and an iced tea.",
            background="You just finished teaching a class and stopped by Pizza Palace for dinner.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask what vegetarian pizzas are available.",
                "Order one veggie supreme pizza.",
                "Add a garlic bread and an iced tea to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than one veggie supreme pizza, one garlic bread, and one iced tea.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Pizza Palace! I'm Lisa, how can I help you today?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": PIZZA_PALACE_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class PizzaPartyOrder(RestaurantBaseScenario):
    """Order two BBQ chicken pizzas, a caesar salad, and two fountain sodas at Pizza Palace."""

    name = "restaurant__pizza_party_order"
    description = "Large party order with two BBQ chicken pizzas, a caesar salad, and two sodas at Pizza Palace"
    reference_answer = {
        "items": [
            {"name": "BBQ Chicken Pizza", "unit_price": "11.99", "quantity": "2"},
            {"name": "Caesar Salad", "unit_price": "4.99", "quantity": "1"},
            {"name": "Fountain Soda", "unit_price": "1.99", "quantity": "2"},
        ],
        "customer_name": "Marcus",
        "customer_phone": "847-502-9163",
        "total_price": "32.95",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Marcus",
            background="You are a college student ordering pizza for a small study group. Your phone number is 847-502-9163.",
            personality="You are upbeat and social, a bit chatty but decisive when it comes to food choices.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order two BBQ chicken pizzas, one caesar salad, and two fountain sodas.",
            background="You are picking up food for your study group at Pizza Palace.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what pizzas are on the menu.",
                "Order two BBQ chicken pizzas.",
                "Add one caesar salad and two fountain sodas to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than two BBQ chicken pizzas, one caesar salad, and two fountain sodas.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Pizza Palace! I'm Lisa, how can I help you today?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": PIZZA_PALACE_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class BurgerClassic(RestaurantBaseScenario):
    """Order a classic burger with onion rings at Burger Barn."""

    name = "restaurant__burger_classic"
    description = "Order a classic burger and onion rings at Burger Barn"
    reference_answer = {
        "items": [
            {"name": "Classic Burger", "unit_price": "7.49", "quantity": "1"},
            {"name": "Onion Rings", "unit_price": "3.49", "quantity": "1"},
        ],
        "customer_name": "Ethan",
        "customer_phone": "736-290-5814",
        "total_price": "10.98",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Ethan",
            background="You are a high school teacher on your lunch break. Your phone number is 736-290-5814.",
            personality="You are straightforward and efficient. You prefer to keep things simple and don't waste time.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a classic burger and a side of onion rings.",
            background="You stopped by Burger Barn for a quick lunch during your break.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what burgers are available.",
                "Order one classic burger.",
                "Add one order of onion rings.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "If asked about whether to get a combo deal, say 'No thanks, just the burger and onion rings.'",
                "Do not order any items other than one classic burger and one onion rings.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Burger Barn! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Ask the user if they would like to upgrade to a combo deal.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "When a customer orders a burger, ask if they want a combo deal.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": BURGER_BARN_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class BurgerBaconCombo(RestaurantBaseScenario):
    """Order a bacon cheeseburger combo and a milkshake at Burger Barn."""

    name = "restaurant__burger_bacon_combo"
    description = "Order a bacon cheeseburger combo and a chocolate milkshake at Burger Barn"
    reference_answer = {
        "items": [
            {"name": "Bacon Cheeseburger Combo", "unit_price": "11.99", "quantity": "1"},
            {"name": "Milkshake", "unit_price": "4.99", "quantity": "1"},
        ],
        "customer_name": "Sophia",
        "customer_phone": "918-374-6205",
        "total_price": "16.98",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Sophia",
            background="You are a nurse who just finished a long shift at the hospital. Your phone number is 918-374-6205.",
            personality="You are warm and friendly, but tired. You want a filling meal and are happy to go for a combo deal to save time.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a bacon cheeseburger combo and a chocolate milkshake.",
            background="You are craving a hearty burger after a twelve-hour shift and stopped at Burger Barn.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what combo deals are available.",
                "Order one bacon cheeseburger combo.",
                "Add a chocolate milkshake to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than one bacon cheeseburger combo and one milkshake.",
                "When asked about milkshake flavor, say chocolate.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Burger Barn! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "If the user orders a milkshake, ask which flavor they would like.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": BURGER_BARN_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class BurgerSpicyFeast(RestaurantBaseScenario):
    """Order a spicy jalapeno burger, french fries, a lemonade, and onion rings at Burger Barn."""

    name = "restaurant__burger_spicy_feast"
    description = "Order a spicy jalapeno burger with fries, onion rings, and lemonade at Burger Barn"
    reference_answer = {
        "items": [
            {"name": "Spicy Jalapeno Burger", "unit_price": "8.99", "quantity": "1"},
            {"name": "French Fries", "unit_price": "2.49", "quantity": "1"},
            {"name": "Onion Rings", "unit_price": "3.49", "quantity": "1"},
            {"name": "Lemonade", "unit_price": "2.29", "quantity": "1"},
        ],
        "customer_name": "Jake",
        "customer_phone": "462-819-3057",
        "total_price": "17.26",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Jake",
            background="You are a firefighter who loves spicy food. Your phone number is 462-819-3057.",
            personality="You are bold and adventurous with food. You like to order a lot and enjoy trying spicy options.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a spicy jalapeno burger, french fries, onion rings, and a lemonade.",
            background="You just got off a shift and are very hungry. You stopped by Burger Barn for a big meal.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what burgers they have.",
                "Order a spicy jalapeno burger.",
                "Add french fries and onion rings as sides.",
                "Add a lemonade to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "If asked about whether to get a combo deal, say 'No, I want to order the items separately.'",
                "Do not order any items other than one spicy jalapeno burger, one french fries, one onion rings, and one lemonade.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Burger Barn! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Ask the user if they would like to upgrade to a combo deal.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "When a customer orders a burger, ask if they want a combo deal.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": BURGER_BARN_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class DeliTurkeyClub(RestaurantBaseScenario):
    """Order a turkey club sandwich and a coffee at Deli Delights."""

    name = "restaurant__deli_turkey_club"
    description = "Order a turkey club sandwich and coffee at Deli Delights"
    reference_answer = {
        "items": [
            {"name": "Turkey Club Sandwich", "unit_price": "8.49", "quantity": "1"},
            {"name": "Coffee", "unit_price": "2.49", "quantity": "1"},
        ],
        "customer_name": "Rachel",
        "customer_phone": "580-261-4937",
        "total_price": "10.98",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Rachel",
            background="You are a freelance writer working from a nearby cafe. Your phone number is 580-261-4937.",
            personality="You are quiet and thoughtful. You prefer simple, quality meals and communicate concisely.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a turkey club sandwich and a coffee.",
            background="You are taking a lunch break from writing and walked over to Deli Delights.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what sandwiches they have.",
                "Order one turkey club sandwich.",
                "Add a coffee to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than one turkey club sandwich and one coffee.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Deli Delights! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": DELI_DELIGHTS_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class DeliItalianSubLunch(RestaurantBaseScenario):
    """Order an Italian sub, tomato soup, potato chips, and fresh squeezed OJ at Deli Delights."""

    name = "restaurant__deli_italian_sub_lunch"
    description = "Order an Italian sub, tomato soup, chips, and OJ at Deli Delights"
    reference_answer = {
        "items": [
            {"name": "Italian Sub", "unit_price": "9.49", "quantity": "1"},
            {"name": "Tomato Soup", "unit_price": "3.99", "quantity": "1"},
            {"name": "Potato Chips", "unit_price": "1.49", "quantity": "1"},
            {"name": "Fresh Squeezed OJ", "unit_price": "3.49", "quantity": "1"},
        ],
        "customer_name": "Tony",
        "customer_phone": "273-940-8162",
        "total_price": "18.46",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Tony",
            background="You are a construction worker on your lunch break. Your phone number is 273-940-8162.",
            personality="You are direct and no-nonsense. You know exactly what you want and don't like to waste time browsing.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order an Italian sub, a tomato soup, potato chips, and a fresh squeezed OJ.",
            background="You have a short lunch break and need to order quickly at Deli Delights.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Tell the agent you would like an Italian sub, a tomato soup, potato chips, and a fresh squeezed OJ.",
                "Confirm the order when summarized.",
                "Provide your name and phone number when asked.",
            ],
            guidelines=[
                "Do not order any items other than one Italian sub, one tomato soup, one potato chips, and one fresh squeezed OJ.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Deli Delights! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": DELI_DELIGHTS_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class DeliGrilledCheeseSoup(RestaurantBaseScenario):
    """Order a grilled cheese with chicken noodle soup and bottled water at Deli Delights."""

    name = "restaurant__deli_grilled_cheese_soup"
    description = "Order a grilled cheese, chicken noodle soup, and water at Deli Delights"
    reference_answer = {
        "items": [
            {"name": "Grilled Cheese", "unit_price": "5.99", "quantity": "1"},
            {"name": "Chicken Noodle Soup", "unit_price": "4.49", "quantity": "1"},
            {"name": "Bottled Water", "unit_price": "1.49", "quantity": "1"},
        ],
        "customer_name": "Mia",
        "customer_phone": "691-470-3826",
        "total_price": "11.97",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Mia",
            background="You are a graduate student studying biology. Your phone number is 691-470-3826.",
            personality="You are cheerful and curious. You like to ask questions about the food before ordering.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a grilled cheese, a chicken noodle soup, and a bottled water.",
            background="You are feeling under the weather and want some comfort food at Deli Delights.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what soups they have.",
                "Ask what sandwiches pair well with soup.",
                "Order one grilled cheese and one chicken noodle soup.",
                "Add a bottled water to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than one grilled cheese, one chicken noodle soup, and one bottled water.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Deli Delights! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": DELI_DELIGHTS_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


@register_eval_scenario
class BurgerVeggieMilkshake(RestaurantBaseScenario):
    """Order a veggie burger and a strawberry milkshake at Burger Barn."""

    name = "restaurant__burger_veggie_milkshake"
    description = "Order a veggie burger and a strawberry milkshake at Burger Barn"
    reference_answer = {
        "items": [
            {"name": "Veggie Burger", "unit_price": "7.99", "quantity": "1"},
            {"name": "Milkshake", "unit_price": "4.99", "quantity": "1"},
        ],
        "customer_name": "Priya",
        "customer_phone": "350-816-2947",
        "total_price": "12.98",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Priya",
            background="You are a software engineer who follows a vegetarian diet. Your phone number is 350-816-2947.",
            personality="You are friendly and inquisitive. You like to confirm details and make sure you understand the options before ordering.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a veggie burger and a strawberry milkshake.",
            background="You are meeting a friend for a casual meal at Burger Barn.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask if the restaurant has any vegetarian burger options.",
                "Order one veggie burger.",
                "Ask about milkshake flavors and order a strawberry milkshake.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=[
                "Do not order any items other than one veggie burger and one milkshake.",
                "When asked about milkshake flavor, say strawberry.",
                "Provide your name and phone number when asked.",
            ],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user by saying 'Welcome to Burger Barn! I'm Lisa, what can I get for you?'.",
                "Ask the user what they would like to order and help them make the order.",
                "If the user orders a milkshake, ask which flavor they would like.",
                "Summarize the order and confirm with the user if the order is correct.",
                "Ask the user for their name and phone number, and associate them with the order.",
                "Place the order using the `PlaceOrderTool` tool, and confirm with the user if the order is placed successfully.",
                "Thank the user for their order and say goodbye, and use the `EndConversationTool` tool to end the conversation.",
            ],
            guidelines=[
                "Do not make up any items not on the menu.",
                "Always use the `PlaceOrderTool` tool to place the final confirmed order.",
                "Always confirm with the user if the order is correct before placing the order.",
                "Use the `EndConversationTool` tool to end the conversation when the user says goodbye or has no other questions.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": BURGER_BARN_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=[
                "You can use the `GetMenuTool` tool to retrieve the restaurant menu.",
                f"When placing an order, the total price should be calculated based on the unit price and quantity of the items. "
                f"The expected order format is: {EXPECTED_ORDER_FORMAT}.",
            ],
        )


# ---------------------------------------------------------------------------
# Waitlist scenarios (demonstrate shared_state between tools)
# ---------------------------------------------------------------------------

# Pre-populated waitlist: 10 people already waiting
_INITIAL_WAITLIST_DATA = [
    {"name": "Alice", "phone": "413-927-0586", "party_size": 2},
    {"name": "Bob", "phone": "728-041-5369", "party_size": 4},
    {"name": "Carol", "phone": "305-862-1947", "party_size": 1},
    {"name": "Dan", "phone": "641-379-8205", "party_size": 3},
    {"name": "Eve", "phone": "937-514-0268", "party_size": 2},
    {"name": "Frank", "phone": "182-046-9357", "party_size": 5},
    {"name": "Grace", "phone": "574-830-6192", "party_size": 2},
    {"name": "Hank", "phone": "890-213-7546", "party_size": 6},
    {"name": "Ivy", "phone": "261-508-4937", "party_size": 1},
    {"name": "Jack", "phone": "709-346-1825", "party_size": 3},
]
INITIAL_WAITLIST = json.dumps(_INITIAL_WAITLIST_DATA)


@register_eval_scenario
class WaitlistJoinThenDrop(RestaurantBaseScenario):
    """User joins the waitlist, asks how many people are ahead, then drops when they hear there are 10."""

    name = "restaurant__waitlist_join_then_drop"
    description = (
        "User joins a busy restaurant waitlist then decides to leave after learning there are 10 people ahead"
    )
    max_duration = 120

    # Both JoinWaitListTool and DropWaitListTool inherit SendScenarioSummaryTool,
    # so the bridge records both actions in order as a list.
    # Each summary includes the full waitlist state at that point.
    _SAM = {"name": "Sam", "phone": "483-926-1057", "party_size": 2}
    reference_answer = [
        {
            "waitlist": _INITIAL_WAITLIST_DATA + [_SAM],
            "action": "join",
            "customer": _SAM,
        },
        {
            "waitlist": _INITIAL_WAITLIST_DATA,
            "action": "drop",
            "customer": {"name": "Sam"},
            "removed": True,
        },
    ]

    ignore_capitalization = True
    ignore_punctuation = True
    clean_text = True

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Sam",
            background="You are a software engineer. Your phone number is 483-926-1057. You are here with a friend, so your party size is 2. You are hungry but impatient.",
            personality="You are friendly but practical. You don't like waiting too long for a table.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Join the waitlist at the restaurant, then decide to leave when you find out how many people are ahead of you.",
            background="You arrived at a popular restaurant called Pizza Palace and it's very busy.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Tell the agent you'd like to join the waitlist.",
                "Provide your name and phone number when asked.",
                "After joining, ask how many people are ahead of you on the waitlist.",
                "When you hear there are many people ahead of you, say that's too long and ask to be removed from the waitlist.",
                "After being removed, thank the agent and say goodbye.",
            ],
            guidelines=[
                "Provide your name as 'Sam', phone number as '483-926-1057', and party size as 2.",
                "If there are more than five people ahead of you, decide the wait is too long and ask to be removed.",
            ],
        )

    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="helpful AI agent",
            name="Lisa",
            background="You are a host at Pizza Palace, managing the restaurant waitlist.",
            personality="You are friendly, apologetic about wait times, and helpful. Always concise and to the point.",
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal="Manage the restaurant waitlist. Help customers join, check their position, or leave the waitlist."
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the customer by saying 'Welcome to Pizza Palace! I'm Lisa, the host. How can I help you?'.",
                "If the customer wants to join the waitlist, ask for their name, phone number, and party size.",
                "Use the `JoinWaitListTool` to add them to the waitlist.",
                "If the customer asks about the waitlist, use the `GetWaitlistTool` to check and tell them their position.",
                "If the customer wants to leave the waitlist, use the `DropWaitListTool` to remove them.",
                "After the customer's request is handled and they say goodbye, use the `EndConversationTool` to end the conversation.",
            ],
            guidelines=[
                "Always use the `JoinWaitListTool` to add customers to the waitlist.",
                "Always use the `GetWaitlistTool` to check the current waitlist when asked.",
                "Always use the `DropWaitListTool` to remove customers from the waitlist.",
                "After the customer says goodbye, use the `EndConversationTool` to end the conversation.",
                "Be apologetic if the wait is long.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "JoinWaitListTool": {"initial_waitlist": INITIAL_WAITLIST},
                "DropWaitListTool": {},
                "GetWaitlistTool": {},
                "EndConversationTool": {},
            },
            information=[
                "The restaurant is Pizza Palace and it is very popular tonight.",
                "The waitlist currently has ten people on it.",
            ],
        )
