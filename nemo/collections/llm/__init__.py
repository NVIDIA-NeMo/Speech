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

# This is here to import it once, which improves the speed of launch when in debug-mode
from nemo.utils.import_utils import safe_import

safe_import("transformer_engine")

from nemo.collections.llm import peft
from nemo.collections.llm.gpt.data import (
    DollyDataModule,
    FineTuningDataModule,
    MockDataModule,
    PreTrainingDataModule,
    SquadDataModule,
)
from nemo.collections.llm.gpt.data.api import dolly, mock, squad
from nemo.collections.llm.gpt.model import (
    Baichuan2Config,
    Baichuan2Config7B,
    Baichuan2Model,
    BaseMambaConfig1_3B,
    BaseMambaConfig2_7B,
    BaseMambaConfig130M,
    BaseMambaConfig370M,
    BaseMambaConfig780M,
    ChatGLM2Config6B,
    ChatGLM3Config6B,
    ChatGLMConfig,
    ChatGLMModel,
    CodeGemmaConfig2B,
    CodeGemmaConfig7B,
    CodeLlamaConfig7B,
    CodeLlamaConfig13B,
    CodeLlamaConfig34B,
    CodeLlamaConfig70B,
    GemmaConfig,
    GemmaConfig2B,
    GemmaConfig7B,
    GemmaModel,
    GPTConfig,
    GPTConfig5B,
    GPTConfig7B,
    GPTConfig20B,
    GPTConfig40B,
    GPTConfig126M,
    GPTConfig175B,
    GPTModel,
    Llama2Config7B,
    Llama2Config13B,
    Llama2Config70B,
    Llama3Config8B,
    Llama3Config70B,
    Llama31Config8B,
    Llama31Config70B,
    Llama31Config405B,
    LlamaConfig,
    LlamaModel,
    MaskedTokenLossReduction,
    MistralConfig7B,
    MistralModel,
    MixtralConfig8x3B,
    MixtralConfig8x7B,
    MixtralConfig8x22B,
    MixtralModel,
    Nemotron3Config4B,
    Nemotron3Config8B,
    Nemotron4Config15B,
    Nemotron4Config22B,
    Nemotron4Config340B,
    NemotronConfig,
    NemotronModel,
    NVIDIAMambaConfig8B,
    NVIDIAMambaHybridConfig8B,
    Qwen2Config,
    Qwen2Config1P5B,
    Qwen2Config7B,
    Qwen2Config72B,
    Qwen2Config500M,
    Qwen2Model,
    SSMConfig,
    Starcoder2Config,
    Starcoder2Config3B,
    Starcoder2Config7B,
    Starcoder2Config15B,
    Starcoder2Model,
    StarcoderConfig,
    StarcoderConfig15B,
    StarcoderModel,
    gpt_data_step,
    gpt_forward_step,
)
from nemo.collections.llm.t5.model import T5Config, T5Model, t5_data_step, t5_forward_step

__all__ = [
    "MockDataModule",
    "GPTModel",
    "GPTConfig",
    "gpt_data_step",
    "gpt_forward_step",
    "T5Model",
    "T5Config",
    "t5_data_step",
    "t5_forward_step",
    "MaskedTokenLossReduction",
    "MistralConfig7B",
    "MistralModel",
    "MixtralConfig8x3B",
    "MixtralConfig8x7B",
    "MixtralConfig8x22B",
    "MixtralModel",
    "Starcoder2Config15B",
    "Starcoder2Config",
    "Starcoder2Model",
    "NemotronModel",
    "Nemotron3Config4B",
    "Nemotron3Config8B",
    "Nemotron4Config15B",
    "Nemotron4Config22B",
    "Nemotron4Config340B",
    "NemotronConfig",
    "SSMConfig",
    "BaseMambaConfig130M",
    "BaseMambaConfig370M",
    "BaseMambaConfig780M",
    "BaseMambaConfig1_3B",
    "BaseMambaConfig2_7B",
    "NVIDIAMambaConfig8B",
    "NVIDIAMambaHybridConfig8B",
    "LlamaConfig",
    "Llama2Config7B",
    "Llama2Config13B",
    "Llama2Config70B",
    "Llama3Config8B",
    "Llama3Config70B",
    "Llama31Config8B",
    "Llama31Config70B",
    "Llama31Config405B",
    "CodeLlamaConfig7B",
    "CodeLlamaConfig13B",
    "CodeLlamaConfig34B",
    "CodeLlamaConfig70B",
    "LlamaModel",
    "GemmaConfig",
    "GemmaConfig2B",
    "GemmaConfig7B",
    "CodeGemmaConfig2B",
    "CodeGemmaConfig7B",
    "GemmaModel",
    "Baichuan2Config",
    "Baichuan2Config7B",
    "Baichuan2Model",
    "ChatGLMConfig",
    "ChatGLM2Config6B",
    "ChatGLM3Config6B",
    "ChatGLMModel",
    "Qwen2Model",
    "Qwen2Config7B",
    "Qwen2Config",
    "Qwen2Config500M",
    "Qwen2Config1P5B",
    "Qwen2Config72B",
    "PreTrainingDataModule",
    "FineTuningDataModule",
    "SquadDataModule",
    "DollyDataModule",
    "tokenizer",
    "mock",
    "squad",
    "dolly",
    "peft",
]


from nemo.utils import logging

try:
    import nemo_run as run

    from nemo.collections.llm.api import export_ckpt, finetune, generate, import_ckpt, pretrain, train, validate
    from nemo.collections.llm.recipes import *  # noqa

    __all__.extend(
        [
            "train",
            "import_ckpt",
            "export_ckpt",
            "pretrain",
            "validate",
            "finetune",
            "generate",
        ]
    )
except ImportError as error:
    logging.warning(f"Failed to import nemo.collections.llm.[api,recipes]: {error}")

try:
    from nemo.collections.llm.api import deploy

    __all__.append("deploy")
except ImportError as error:
    logging.warning(f"The deploy module could not be imported: {error}")
