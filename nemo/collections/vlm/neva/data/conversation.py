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

import base64
import dataclasses
import re
from collections import defaultdict
from enum import Enum, auto
from io import BytesIO
from typing import Any, List, Optional, Union

from PIL import Image
from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    MISTRAL = auto()
    NVGPT = auto()
    QWEN = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: Optional[str]
    roles: tuple[str, str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_name_or_path: Any = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def process_prompt_with_images(self, messages):
        # Process messages to handle potential image tokens.
        return messages

    def process_chat_template(self, tokenizer_name_or_path, messages):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.system is None:
            chat = []
        else:
            chat = [{"role": "system", "content": self.system}]
        for role, message in messages:
            chat.append({"role": role.lower(), "content": message})
        ret = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return ret

    def get_prompt(self):
        messages = self.messages
        messages = self.process_prompt_with_images(messages)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            """
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ user_message_1 }} ASSISTANT: {{ model_answer_1 }}</s>USER: {{ user_message_2 }}
            """
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.MISTRAL and self.version == "vila":
            """
            <s>[INST] {{ user_message_1 }} [/INST]{{ model_answer_1 }}</s>[INST] {{ user_message_2 }} [/INST]
            """
            wrap_sys = lambda msg: f"{msg}" + ("\n" if msg else "")
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = "<s>"

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += message + self.sep2
                else:
                    ret += ""

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            """
            <s>[INST] <<SYS>>
            You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
            <</SYS>>

            {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_message_2 }} [/INST]
            """
            tokenizer_name_or_path = self.tokenizer_name_or_path or "meta-llama/Llama-2-7b-chat-hf"
            ret = self.process_chat_template(tokenizer_name_or_path, messages)

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {{ user_message_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            {{ model_answer_1 }}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {{ user_message_2 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            tokenizer_name_or_path = self.tokenizer_name_or_path or "meta-llama/Meta-Llama-3-8B-Instruct"
            ret = self.process_chat_template(tokenizer_name_or_path, messages)

        elif self.sep_style == SeparatorStyle.NVGPT:
            ret = self.sep2 + self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + '\n' + message + '\n' + self.sep
                else:
                    ret += role + '\n'

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""

        elif self.sep_style == SeparatorStyle.MISTRAL:
            """
            NOT tested in NeMo!
            """
            tokenizer_name_or_path = self.tokenizer_name_or_path or "mistralai/Mistral-7B-Instruct-v0.2"
            ret = self.process_chat_template(tokenizer_name_or_path, messages)

        elif self.sep_style == SeparatorStyle.CHATML:
            """
            NOT tested in NeMo!
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
            return ret

        elif self.sep_style == SeparatorStyle.MPT:
            """
            NOT tested in NeMo!
            """
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.GEMMA:
            """
            NOT tested in NeMo!
            """
            ret = ""
            for i, (role, message) in enumerate(messages):
                assert role == self.roles[i % 2], "Conversation should alternate user/assistant/user/assistant/..."
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format="PNG"):
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 1008, 672
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False, return_path=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        if not return_path:
                            img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        images.append(img)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = "<image>\n" + msg.replace("<image>", "").strip()
                    else:
                        msg = re.sub(r"(<image>)\n(?=<image>)", r"\1 ", msg)
                    for img in image:
                        img_b64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}"/>'
                        msg = msg.replace("<image>", img_str, 1).strip()
                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


# Conversation Template for NVGPT
conv_nvgpt = Conversation(
    system="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n""",
    roles=("User", "Assistant"),
    version="nvgpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.NVGPT,
    sep="<extra_id_1>",
    sep2=f"<extra_id_0>System\n",
)

conv_nv_dpo = Conversation(
    system="\n",
    roles=("User", "Assistant"),
    version="nv_dpo",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.NVGPT,
    sep="<extra_id_1>",
    sep2=f"<extra_id_0>System\n",
)

conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        [
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ],
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    stop_str="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
    stop_str=" </s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
    stop_str=" </s>",
)

conv_llava_llama_3 = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("user", "assistant"),
    version="llama_v3",
    messages=[],
    offset=0,
    sep="<|eot_id|>",
    sep_style=SeparatorStyle.LLAMA_3,
    tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    stop_str="<|eot_id|>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
    stop_str=" </s>",
)

conv_llava_llama_2_simple = Conversation(
    system="Answer the questions about the visual content that the user provides.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
    stop_str=" </s>",
)

conv_llava_llama_2_mmtag = Conversation(
    system="Answer the questions about the visual content that the user provides."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2_mmtag",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
    stop_str=" </s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_qwen = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_gemma_instruct = Conversation(
    system="",
    roles=("<start_of_turn>user\n", "<start_of_turn>model\n"),
    version="gemma",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.GEMMA,
    sep="<end_of_turn>\n",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="",
    sep2="\n",
    stop_str="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_vila = Conversation(
    system=None,
    roles=("USER", "ASSISTANT"),
    version="vila",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="",
    sep2="</s>",
    stop_str="</s>",
)

conv_mistral_orca = Conversation(
    system="""<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mistral_zephyr = Conversation(
    system="""<|system|>
You are a helpful AI assistant.""",
    roles=("<|user|>\n", "<|assistant|>\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="</s>",
)

conv_mistral_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v1,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "mistral_orca": conv_mistral_orca,
    "mistral_zephyr": conv_mistral_zephyr,
    "mistral_direct": conv_mistral_direct,
    "mistral": conv_mistral_vila,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "chatml_direct": conv_chatml_direct,
    "llava_v0": conv_llava_v0,
    "llava_v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "llava_v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llava_llama_3": conv_llava_llama_3,
    "llava_llama_2_simple": conv_llava_llama_2_simple,
    "llava_llama_2_mmtag": conv_llava_llama_2_mmtag,
    "llava_mistral_instruct": conv_mistral_instruct,
    "mpt": conv_mpt,
    "qwen_1_5": conv_qwen,
    "gemma_instruct": conv_gemma_instruct,
    "nvgpt": conv_nvgpt,
    "nv_steerlm": conv_nvgpt,
    "nv_dpo": conv_nv_dpo,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
