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
# pylint: disable=missing-function-docstring,missing-class-docstring
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

SYSTEM_BOS = "<SPECIAL_10>"
TURN_BOS = "<SPECIAL_11>"


class NemotronHPromptFormatter(PromptFormatter):
    NAME = "nemotron-h"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"\n{TURN_BOS}Assistant\n"
    TEMPLATE = {
        "system": {
            "template": f"{SYSTEM_BOS}System\n|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"\n{TURN_BOS}User\n|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, NemotronHPromptFormatter)
def nemotron_h(cut: Cut, prompt: NemotronHPromptFormatter):
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut

    turns = []

    system = ""
    if cut.has_custom("system_prompt"):
        system = cut.system_prompt
    turns.append({"role": "system", "content": system})

    if cut.has_custom("context"):
        ctx = cut.context
    else:
        ctx = ""
    turns.append({"role": "user", "content": ctx})

    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "content": answer})

    return prompt.encode_dialog(turns)
