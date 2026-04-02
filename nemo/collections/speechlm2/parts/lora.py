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
from omegaconf import DictConfig, ListConfig
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from nemo.collections.speechlm2.parts.pretrained import move_embedding
from nemo.utils import logging


def maybe_install_lora(model):
    """Add LoRA adapters to a model, using HuggingFace PEFT library."""
    if "lora" in model.cfg:
        assert hasattr(model, "cfg") and isinstance(model.cfg, DictConfig)
        assert hasattr(model, "llm") and isinstance(model.llm, PreTrainedModel)
        assert "prevent_freeze_params" in model.cfg and isinstance(model.cfg.prevent_freeze_params, (list, ListConfig))
        model.lora_config = LoraConfig(**model.cfg.lora)
        # PEFT inspects get_input_embeddings() while wrapping the model, so temporarily
        # restore the embedding layer that SALM keeps outside the LLM for FSDP/TP.
        with move_embedding(model):
            model.llm = get_peft_model(model.llm, model.lora_config)
        model.cfg.prevent_freeze_params.append(r"^.+\.lora_.+$")
        logging.info(f"LoRA adapter installed: {model.lora_config}")
