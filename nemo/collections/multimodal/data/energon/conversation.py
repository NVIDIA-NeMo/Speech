# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConversationTemplateConfig:
    """Conversation template config related parameters"""

    system: Optional[str] = "".format()  # fmt: off
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    stop_string: Optional[str] = None
    chat_template = None


class LLaVATemplateConfig(BaseConversationTemplateConfig):
    """LLava specific template configuration which extends the base config"""

    system: Optional[str] = (
        "A chat between a curious user and artificial assistant agent. The assistant gives helpful, detailed and polite answers to user's questions.".format()
    )  # fmt: off
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    stop_string: str = "</s>"
    chat_template = """
    {%- for message in messages %}
        {%- if message['role'] == 'system' %}
            {{- message['content'].strip() + ' ' -}}
        {%- elif message['role'] == 'user' %}
            {{- 'USER: ' -}} {{- message['content'].strip() + ' ' -}}
        {%- elif message['role'] == 'assistant' %}
            {{- 'ASSISTANT: ' -}} {{- message['content'].strip() -}}
            {{- '</s>' -}}
        {%- endif %}
    {%- endfor -%}
    """


class MLlamaTemplateConfig(BaseConversationTemplateConfig):
    """LLava specific template configuration which extends the base config"""

    system: Optional[str] = None
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    stop_string: str = None
    chat_template = """
    '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- Find out if there are any images #}\n{% set image_ns = namespace(has_images=false) %}      \n{%- for message in messages %}\n    {%- for content in message[\'content\'] %}\n        {%- if content[\'type\'] == \'image\' %}\n            {%- set image_ns.has_images = true %}\n        {%- endif %}\n    {%- endfor %}\n{%- endfor %}\n\n{#- Error out if there are images and system message #}\n{%- if image_ns.has_images and not system_message == "" %}\n    {{- raise_exception("Prompting with images is incompatible with system messages.") }}\n{%- endif %}\n\n{#- System message if there are no images #}\n{%- if not image_ns.has_images %}\n    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n    {%- if tools is not none %}\n        {{- "Environment: ipython\\n" }}\n    {%- endif %}\n    {{- "Cutting Knowledge Date: December 2023\\n" }}\n    {{- "Today Date: " + date_string + "\\n\\n" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n        {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n        {{- "Do not use variables.\\n\\n" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- "\\n\\n" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- "<|eot_id|>" }}\n{%- endif %}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n    {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\' }}\n        {%- if message[\'content\'] is string %}\n            {{- message[\'content\'] }}\n        {%- else %}\n            {%- for content in message[\'content\'] %}\n                {%- if content[\'type\'] == \'image\' %}\n                    {{- \'<|image|>\' }}\n                {%- elif content[\'type\'] == \'text\' %}\n                    {{- content[\'text\'] }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    """
