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
import torch
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.lhotse.text_adapters import NeMoSFTExample, SourceTargetTextExample
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import BOS_SLOT, EOS_SLOT, Modality, PromptFormatter


class Llama2PromptFormatter(PromptFormatter):
    """
    This template has been validated to provide identical tokenized results to the official code
    in https://github.com/meta-llama/llama/blob/main/llama/generation.py
    """

    NAME = "llama2"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "system_and_user": {
            "template": f"{BOS_SLOT}[INST] <<SYS>>\n|system|\n<</SYS>>\n\n|message| [/INST]",
            "slots": {
                "system": Modality.Text,
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{BOS_SLOT}[INST] |message| [/INST]",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message| {EOS_SLOT}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, Llama2PromptFormatter)
def llama2(cut: Cut, prompt: Llama2PromptFormatter) -> dict[str, torch.Tensor]:
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut
    if cut.has_custom("context"):
        context = cut.context
    elif cut.has_custom("question"):
        context = cut.question
    else:
        context = cut.default_context

    turns = []
    if cut.has_custom("system_prompt"):
        turns.append({"role": "system_and_user", "slots": {"system": cut.system_prompt, "message": context}})
    else:
        turns.append({"role": "user", "slots": {"message": context}})
    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "slots": {"message": answer}})
    return prompt.encode_dialog(turns)


@registered_prompt_format_fn(SourceTargetTextExample, Llama2PromptFormatter)
def llama2_src_tgt_text_example(example: SourceTargetTextExample, prompt: Llama2PromptFormatter):
    if example.question is not None:
        user_turn = {
            "role": "system_and_user",
            "slots": {"system": example.question.text, "message": example.source.text},
        }
    else:
        user_turn = {
            "role": "user",
            "slots": {"message": example.source.text},
        }
    return prompt.encode_dialog(
        [
            user_turn,
            {"role": prompt.OUTPUT_ROLE, "slots": {"message": example.target.text}},
        ]
    )


@registered_prompt_format_fn(NeMoSFTExample, Llama2PromptFormatter)
def llama2_sft_text_example(example: NeMoSFTExample, prompt: Llama2PromptFormatter):
    first_user_turn = example.data["conversations"][0]["value"]
    if "system" in example.data and example.data["system"]:
        first_turn = {
            "role": "system_and_user",
            "slots": {"system": example.data["system"], "message": first_user_turn},
        }
    else:
        first_turn = {
            "role": "user",
            "slots": {"message": first_user_turn},
        }
    return prompt.encode_dialog(
        [first_turn]
        + [
            {"role": "user" if turn["from"] == "User" else prompt.OUTPUT_ROLE, "slots": {"message": turn["value"]}}
            for turn in example.data["conversations"][1:]
        ]
    )


LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_HEADER_BEGIN = "<|start_header_id|>"
LLAMA3_HEADER_END = "<|end_header_id|>"
LLAMA3_END_OF_TURN = "<|eot_id|>"
LLAMA3_NL = "\n\n"


class Llama3PromptFormatter(PromptFormatter):
    """
    Implemented following the code at:
     https://github.com/meta-llama/llama3/blob/main/llama/test_tokenizer.py#L56
    """

    NAME = "llama3"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "preamble": {
            "template": LLAMA3_BOS,
        },
        "system": {
            "template": f"{LLAMA3_HEADER_BEGIN}system{LLAMA3_HEADER_END}{LLAMA3_NL}|message|{LLAMA3_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{LLAMA3_HEADER_BEGIN}user{LLAMA3_HEADER_END}{LLAMA3_NL}|message|{LLAMA3_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{LLAMA3_HEADER_BEGIN}assistant{LLAMA3_HEADER_END}{LLAMA3_NL}|message|{LLAMA3_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
