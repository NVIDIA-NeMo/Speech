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

from dataclasses import dataclass

from torch import nn

from nemo.collections.llm.peft.lora import LoRA as LLMLoRA


@dataclass
class LoRA(LLMLoRA):
    """
    Built on top of llm.LoRA, vlm.LoRA additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.

    For detailed usage of the LoRA api, see llm.LoRA docstrings.

    Example:
    --------
        >>> from nemo.collections import vlm
        >>> lora = vlm.peft.LoRA(target_modules=["*.language_model.*.linear_qkv"], freeze_vision_model=False, dim=32)
        >>> model = vlm.MLlamaModel(model_transform=lora)
        >>> # (set up trainer and data)
        >>> trainer.fit(model, data)

    References:
    -----------
        Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).
        LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
        https://arxiv.org/abs/2106.09685

    )

    """

    freeze_language_model: bool = True
    freeze_vision_model: bool = True

    def freeze_model(self, model: nn.Module) -> None:
        modules = []
        if self.freeze_language_model and model.module.module.language_model is not None:
            modules.append(model.module.module.language_model)
        if self.freeze_vision_model and model.module.module.vision_model is not None:
            modules.append(model.module.module.vision_model)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
