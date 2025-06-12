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

from dataclasses import dataclass, field
from typing import Union

from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.avlm.model.base import AVLMConfig
from nemo.collections.llm import Llama3Config8B
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.collections.vlm.vision.base import HFCLIPVisionConfig, MultimodalProjectorConfig


@dataclass
class AVLMConfig8B(AVLMConfig):
    """
    Configuration class for the 8B parameter variant of the AVLM model.

    """

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama3Config8B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336",
        )
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mlp2x_gelu", input_size=1024, hidden_size=4096, ffn_hidden_size=4096
        )
    )
    audio_transformer_config: TransformerConfig = field(
        default_factory=lambda: ASRModuleConfig(
            _target_="nemo.collections.speechlm.modules.asr_module.ASRModuleConfig",
            use_hf_auto_model=True,
            hf_trust_remote_code=False,
            hf_load_pretrained_weights=True,
            pretrained_model="openai/whisper-large-v3",
            hidden_size=1280,
            target_module="model.encoder",
        )
    )
    audio_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mlp2x_gelu", input_size=1280, hidden_size=4096, ffn_hidden_size=4096
        )
    )


__all__ = [
    "AVLMConfig8B",
]
