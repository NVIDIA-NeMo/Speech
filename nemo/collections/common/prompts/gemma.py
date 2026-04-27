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
"""
Gemma1 prompt format reference:
    https://www.promptingguide.ai/models/gemma#gemma-7b-prompt-format

Gemma4 prompt format reference (multimodal: text + image + audio):
    <|turn>user
    Describe this image: <|image|>
    And translate this audio: <|audio|><turn|>
    <|turn>model
"""
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

GEMMA_BOS = "<start_of_turn>"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_NL = "\n\n"


class GemmaPromptFormatter(PromptFormatter):
    NAME = "gemma"
    OUTPUT_ROLE = "assistant"
    INSERT_BOS = True
    INSERT_EOS = True
    TEMPLATE = {
        "user": {
            "template": f"{GEMMA_BOS}user\n|message|{GEMMA_END_OF_TURN}\n{GEMMA_BOS}model\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            # Note: that trailing NL is bothering me.
            "template": f"|message|{GEMMA_END_OF_TURN}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, GemmaPromptFormatter)
def gemma1(cut: Cut, prompt: GemmaPromptFormatter):
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut
    if cut.has_custom("context"):
        context = cut.context
    elif cut.has_custom("question"):
        context = cut.question
    else:
        context = cut.default_context
    turns = [{"role": "user", "slots": {"message": context}}]
    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "slots": {"message": answer}})
    return prompt.encode_dialog(turns)


GEMMA4_BOT = "<|turn>"           # beginning-of-turn
GEMMA4_EOT = "<turn|>"           # end-of-turn
GEMMA4_IMAGE = "<|image|>"       # image placeholder token
GEMMA4_AUDIO = "<|audio|>"       # audio placeholder token


class Gemma4PromptFormatter(PromptFormatter):
    NAME = "gemma4"
    OUTPUT_ROLE = "assistant"
    INSERT_BOS = True
    INSERT_EOS = True
    TEMPLATE = {
        "user": {
            "template": f"{GEMMA4_BOT}user\n|message|{GEMMA4_EOT}\n{GEMMA4_BOT}model\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|{GEMMA4_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, Gemma4PromptFormatter)
def gemma4(cut: Cut, prompt: Gemma4PromptFormatter):
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut
    if cut.has_custom("context"):
        context = cut.context
    elif cut.has_custom("question"):
        context = cut.question
    else:
        context = cut.default_context
    parts = []
    if context:
        parts.append(context)
    if cut.has_custom("image") and cut.image is not None:
        parts.append(GEMMA4_IMAGE)
    if getattr(cut, "has_recording", False) or cut.has_custom("audio_filepath"):
        parts.append(GEMMA4_AUDIO)
    if cut.has_custom("extra_audios") and cut.extra_audios:
        for _ in cut.extra_audios:
            parts.append(GEMMA4_AUDIO)
    user_message = "\n".join(parts)
    turns = [{"role": "user", "slots": {"message": user_message}}]
    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "slots": {"message": answer}})
    return prompt.encode_dialog(turns)