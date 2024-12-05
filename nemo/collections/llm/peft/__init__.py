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

from nemo.collections.llm.peft.api import gpt_lora, merge_lora
from nemo.collections.llm.peft.canonical_lora import CanonicalLoRA
from nemo.collections.llm.peft.dora import DoRA
from nemo.collections.llm.peft.lora import LoRA

PEFT_STR2CLS = {
    "LoRA": LoRA,
    "lora": LoRA,
    "DoRA": DoRA,
    "dora": DoRA,
    "CanonicalLoRA": CanonicalLoRA,
    "canonical_lora": CanonicalLoRA,
}

__all__ = ["LoRA", "DoRA", "CanonicalLoRA", "gpt_lora", "PEFT_STR2CLS", "merge_lora"]
