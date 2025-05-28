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

import dataclasses
from enum import Enum, auto
from typing import Any, List, Optional, Union

from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""

    CHATML = auto()
    QWEN2VL = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: Optional[str]
    roles: tuple[str, str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.QWEN2VL
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_name_or_path: Any = None
    stop_str: Union[str, List[str]] = "<|im_end|>"
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def process_chat_template(self, tokenizer_name_or_path, messages):
        # pylint: disable=C0115,C0116
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.system is None or len(self.system) == 0:
            chat = []
        else:
            chat = [{"role": "system", "content": self.system}]
        for role, message in messages:
            chat.append({"role": role.lower(), "content": message})
        ret = tokenizer.apply_chat_template(chat, tokenize=False, add_vision_id=True, add_generation_prompt=False)
        return ret

    def get_prompt(self):
        # pylint: disable=C0115,C0116
        messages = self.messages

        if self.sep_style == SeparatorStyle.QWEN2VL:
            """
            refer to: https://github.com/QwenLM/Qwen2-VL#data-preparation
            [
              {
                "system": "You are a helpful assistant.",
                "messages": [
                  {
                    "content": "<image>Who are they?",
                    "role": "user"
                  },
                  {
                    "content": "They're Kane and Gretzka from Bayern Munich.",
                    "role": "assistant"
                  },
                  {
                    "content": "What are they doing?<image>",
                    "role": "user"
                  },
                  {
                    "content": "They are celebrating on the soccer field.",
                    "role": "assistant"
                  }
                ],
                "images": [
                  "mllm_demo_data/1.jpg",
                  "mllm_demo_data/1.jpg"
                ]
              },
            ]
            """
            tokenizer_name_or_path = self.tokenizer_name_or_path or "Qwen/Qwen2-VL-2B-Instruct"
            ret = self.process_chat_template(tokenizer_name_or_path, messages)

        elif self.sep_style == SeparatorStyle.CHATML:
            # FIXME:  To be support video.
            # pylint: disable=C0301
            """
            Input is already in CHATML format.
            <|im_start|>system
            Assistant is an intelligent chatbot designed to help users answer their tax related questions.
            <|im_end|>
            <|im_start|>user
            When do I need to file my taxes by?
            <|im_end|>
            <|im_start|>assistant
            In 2023, you will need to file your taxes by April 18th. The date falls after the usual April 15th deadline because April 15th falls on a Saturday in 2023. For more details, see https://www.irs.gov/filing/individuals/when-to-file
            <|im_end|>
            """
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        # pylint: disable=C0115,C0116
        self.messages.append([role, message])


conv_qwen2vl = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    version="qwen2vl",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.QWEN2VL,
    sep="",
)
conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

default_conversation = conv_qwen2vl
conv_templates = {
    "default": conv_qwen2vl,
    "qwen2vl": conv_qwen2vl,
    "chatml_direct": conv_chatml_direct,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
