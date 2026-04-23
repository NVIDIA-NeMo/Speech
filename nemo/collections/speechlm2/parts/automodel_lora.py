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
"""LoRA helpers for models that use NeMo Automodel (not HuggingFace PEFT).

See ``lora.py`` for the HF-PEFT variant used by other speechlm2 models.
"""
from typing import Optional

from omegaconf import DictConfig, ListConfig, open_dict

from nemo.utils import logging

LORA_PARAM_PATTERN = r"^.+\.lora_.+$"


def make_peft_config(lora_cfg: Optional[DictConfig]):
    """Create an Automodel ``PeftConfig`` from the ``model.lora`` config section.

    Automodel's ``ModuleMatcher`` matches target-module patterns against the
    **full** dotted module path (e.g. ``model.layers.0.self_attn.q_proj``).
    Short names like ``q_proj`` won't match because the regex is anchored at
    both ends.  To stay compatible with HuggingFace-style configs that use
    short leaf names, we auto-prepend ``*.`` to any entry that doesn't already
    contain a wildcard (``*``) or a dot (``.``).
    """
    from nemo_automodel.components._peft.lora import PeftConfig

    if lora_cfg is None or not lora_cfg:
        return None
    kwargs = {k: list(v) if isinstance(v, ListConfig) else v for k, v in lora_cfg.items()}
    if "target_modules" in kwargs:
        kwargs["target_modules"] = [f"*.{m}" if "*" not in m and "." not in m else m for m in kwargs["target_modules"]]
    if "exclude_modules" in kwargs:
        kwargs["exclude_modules"] = [
            f"*.{m}" if "*" not in m and "." not in m else m for m in kwargs["exclude_modules"]
        ]
    return PeftConfig(**kwargs)


def maybe_install_lora(model):
    """Apply Automodel-native LoRA adapters to the LLM.

    Returns the ``PeftConfig`` that was applied, or ``None`` if LoRA is not configured.
    """
    if "lora" not in model.cfg:
        return None

    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    peft_config = make_peft_config(model.cfg.lora)
    n_modified = apply_lora_to_linear_modules(model.llm, peft_config)
    logging.info(f"LoRA adapters installed on {n_modified} modules: {peft_config}")

    ensure_lora_trainable(model)
    return peft_config


def ensure_lora_trainable(model):
    """Append the LoRA parameter pattern to ``prevent_freeze_params`` so that
    LoRA weights stay trainable even when the LLM is frozen."""
    with open_dict(model.cfg):
        if "prevent_freeze_params" not in model.cfg:
            model.cfg.prevent_freeze_params = []
        pfp = model.cfg.prevent_freeze_params
        if LORA_PARAM_PATTERN not in pfp:
            pfp.append(LORA_PARAM_PATTERN)
