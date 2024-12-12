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
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class PlainPromptFormatter(PromptFormatter):
    """
    Plain prompt formatter: there is nothing being added to the user and assistants turns.
    """

    NAME = "plain"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, PlainPromptFormatter)
def plain(cut: Cut, prompt: PlainPromptFormatter):
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut
    if cut.has_custom("context"):
        ctx = cut.context
    else:
        ctx = ""

    turns = [{"role": "user", "slots": {"message": ctx}}]
    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "slots": {"message": answer}})

    return prompt.encode_dialog(turns)
