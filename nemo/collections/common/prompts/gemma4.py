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
# pylint: disable=C0115
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

GEMMA4_BOT = "<|turn>"
GEMMA4_EOT = "<turn|>"


class Gemma4PromptFormatter(PromptFormatter):
    """
    Gemma 4 chat format (differs from Gemma 1-3's ``<start_of_turn>``/``<end_of_turn>``)::

        <bos><|turn>system\\n{system}<turn|>\\n<|turn>user\\n{user}<turn|>\\n<|turn>model\\n{answer}<turn|>\\n
    """

    NAME = "gemma4"
    OUTPUT_ROLE = "assistant"
    INSERT_BOS = True
    INSERT_EOS = True
    INFERENCE_PREFIX = f"{GEMMA4_BOT}model\n"
    TEMPLATE = {
        "system": {
            "template": f"{GEMMA4_BOT}system\n|message|{GEMMA4_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{GEMMA4_BOT}user\n|message|{GEMMA4_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|{GEMMA4_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
