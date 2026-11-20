# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
Implemented following the guide at
https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

MISTRAL_BOS = "<s>"
MISTRAL_PROMPT_BEGIN = "[INST]"
MISTRAL_PROMPT_END = "[/INST]"
MISTRAL_END_OF_TURN = "</s>"
MISTRAL_NL = "\n\n"


class MistralPromptFormatter(PromptFormatter):
    NAME = "mistral"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "preamble": {
            "template": MISTRAL_BOS,
        },
        "user": {
            "template": f"{MISTRAL_PROMPT_BEGIN} |message| {MISTRAL_PROMPT_END} ",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|{MISTRAL_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
