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
# flake8: noqa
from nemo.collections.common.prompts.canary import CanaryPromptFormatter
from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.prompts.gemma import GemmaPromptFormatter
from nemo.collections.common.prompts.llama import Llama2PromptFormatter, Llama3PromptFormatter
from nemo.collections.common.prompts.mistral import MistralPromptFormatter
from nemo.collections.common.prompts.nemotron_h import NemotronHPromptFormatter
from nemo.collections.common.prompts.phi2 import (
    Phi2ChatPromptFormatter,
    Phi2CodePromptFormatter,
    Phi2QAPromptFormatter,
)
from nemo.collections.common.prompts.plain import PlainPromptFormatter
from nemo.collections.common.prompts.qwen import QwenPromptFormatter
from nemo.collections.common.prompts.t5nmt import T5NMTPromptFormatter
