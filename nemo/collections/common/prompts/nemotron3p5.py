# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from lhotse.cut import Cut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.nemotron_nano_v3 import NemotronNanoV3PromptFormatter, nemotron_nano_v3


class Nemotron3p5PromptFormatter(NemotronNanoV3PromptFormatter):
    """Speech prompt formatter for NVIDIA Nemotron 3.5 chat checkpoints.

    Nemotron 3.5 and Nemotron 3 Nano use the same wire format for the
    system/user/plain-assistant turns supported by SpeechLM. They have
    separate upstream chat templates, however, so keep a distinct registered
    name rather than making recipes claim to use the Nano model family.

    The upstream templates differ for structured ``reasoning_content`` and
    tool-call messages. Those fields are outside the SpeechLM formatter's
    current ``message``-slot schema and are intentionally not approximated
    here.
    """

    NAME = "nemotron3p5"


@registered_prompt_format_fn(Cut, Nemotron3p5PromptFormatter)
def nemotron3p5(cut: Cut, prompt: Nemotron3p5PromptFormatter):
    """Format a Lhotse cut with the Nemotron 3.5 SpeechLM wire format."""
    return nemotron_nano_v3(cut, prompt)
