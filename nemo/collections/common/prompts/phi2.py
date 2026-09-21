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
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class Phi2QAPromptFormatter(PromptFormatter):
    NAME = "phi2_qa"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": "Instruct: |message|\nOutput: ",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": "|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


class Phi2ChatPromptFormatter(PromptFormatter):
    NAME = "phi2_chat"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": "Human: |message|\nAI: ",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": "|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


class Phi2CodePromptFormatter(PromptFormatter):
    NAME = "phi2_code"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": "|message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": "|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
