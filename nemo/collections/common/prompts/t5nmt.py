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
# pylint: disable=C0116
# pylint: disable=C0301
from collections import defaultdict

import torch
from lhotse import MonoCut
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class T5NMTPromptFormatter(PromptFormatter):
    """
    The default prompt format for Megatron T5 based neural machine translation models.
    Based on: https://github.com/NVIDIA/NeMo/blob/ad5ef750e351edbb5eeb7eb6df2d0c804819600f/nemo/collections/nlp/models/machine_translation/megatron_nmt_model.py#L790
    """

    NAME = "t5nmt"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|target_lang| |message|",
            "slots": {
                "target_lang": Modality.Text,
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

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        # Automatically adds "<" and ">" to target lang token for T5 NMT.
        # Based on: https://github.com/NVIDIA/NeMo/blob/ad5ef750e351edbb5eeb7eb6df2d0c804819600f/nemo/collections/nlp/models/machine_translation/mt_enc_dec_model.py#L307
        if (val := slot_values.get("target_lang")) is not None:
            if not val.startswith("<") or not val.endswith(">"):
                slot_values["target_lang"] = f"<{val}>"
        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


@registered_prompt_format_fn(Cut, T5NMTPromptFormatter)
def t5nmt(cut: Cut, prompt: T5NMTPromptFormatter) -> dict[str, torch.Tensor]:
    ans = defaultdict(list)
    if isinstance(cut, MixedCut):
        cut = cut._first_non_padding_cut
    if not isinstance(cut, MonoCut):
        raise TypeError(
            f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
        )

    if hasattr(cut, "context"):
        context = cut.context
    elif hasattr(cut, "default_context"):
        context = cut.default_context
    else:
        raise RuntimeError("Missing context/default_context custom field in cut: {cut}")

    turns = [
        dict(
            role="user",
            # "message" slot is the audio portion of the cut; currently it is populated inside model's forward.
            slots={"target_lang": context, "message": ""},
        ),
    ]
    if len(cut.supervisions) > 0 and cut.supervisions[0].text is not None:
        turns.append(
            dict(
                role=prompt.OUTPUT_ROLE,
                slots={"message": cut.supervisions[0].text},
            )
        )
    return prompt.encode_dialog(turns)


@registered_prompt_format_fn(SourceTargetTextExample, T5NMTPromptFormatter)
def t5nmt_src_tgt_text_example(example: SourceTargetTextExample, prompt: T5NMTPromptFormatter):
    ctx = f"<{example.target.language}>"
    if example.has_custom("extra_prompt"):
        ctx = f"{ctx} {example.extra_prompt}"
    return prompt.encode_dialog(
        [
            {"role": "user", "slots": {"message": example.source.text, "target_lang": ctx}},
            {"role": prompt.OUTPUT_ROLE, "slots": {"message": example.target.text}},
        ]
    )
