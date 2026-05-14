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

"""LLM-backbone-specific composition for the NeMo Speech LM (SALM) plugin.

The single registered model class ``NeMoSpeechLMForConditionalGeneration``
delegates everything LLM-specific to a backend object so the model class can
stay backbone-agnostic. Each backend owns three things:

* The architecture name(s) to pass to ``init_vllm_registered_model`` so vLLM
  picks the right inner model class for the language tower.
* Optional pre-rename weight preprocessing (LoRA merge for transformer
  backbones, identity for hybrid).
* The NeMo-checkpoint -> HuggingFace weight name mapping that
  ``AutoWeightsLoader`` consumes.

Hybrid backbones additionally expose vLLM's ``IsHybrid`` mamba state
classmethods. The model class declares ``IsHybrid`` for the NemotronH path;
non-hybrid backbones engage vLLM's ``layer_types`` runtime escape hatch in
``config.py`` so ``ModelConfig.is_hybrid`` returns False at runtime and the
hybrid KV-cache allocator stays out of their way.
"""

import re
from collections.abc import Iterable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from nemo.collections.speechlm2.vllm.salm.audio import _pad_to_vocab_size

logger = init_logger(__name__)


# ── Base backend ────────────────────────────────────────────────────


class _BaseBackend:
    """Common interface every SALM backend implements.

    Backends are stateless beyond ``self.config``; they contain the
    LLM-specific knowledge that varies across backbones (architecture
    name, weight rename rules, optional preprocessing) and nothing else.
    """

    def __init__(self, config: Any):
        self.config = config

    def architectures(self) -> list[str]:
        """Architecture names passed to ``init_vllm_registered_model``."""
        raise NotImplementedError

    def preprocess_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Optional pre-rename pass (e.g. LoRA merge). Default: identity."""
        return weights

    def nemo_to_hf_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Map NeMo checkpoint weight names to the HuggingFace names ``AutoWeightsLoader`` expects."""
        raise NotImplementedError(f"{type(self).__name__} must implement nemo_to_hf_llm_weights")


# ── Transformer backend (Qwen3, etc.) ────────────────────────────────


def _normalize_lora_name(name: str) -> tuple[str, str]:
    """Classify a weight name and return (normalized_name, kind).

    Returns kind: "lora_a", "lora_b", or "base".
    Normalizes both PEFT and NeMo LoRA naming to a common base key.
      PEFT:  q_proj.lora_A.default.weight -> (q_proj.weight, lora_a)
      NeMo:  q_proj.lora_A.weight         -> (q_proj.weight, lora_a)
      PEFT:  q_proj.base_layer.weight     -> (q_proj.weight, base)
      Plain: q_proj.weight                -> (q_proj.weight, base)
    """
    m = re.match(r"(.+)\.lora_(A|B)(?:\.default)?\.weight$", name)
    if m:
        return m.group(1) + ".weight", "lora_" + m.group(2).lower()
    if ".base_layer." in name:
        return name.replace(".base_layer.", "."), "base"
    return name, "base"


def _merge_lora_weights(
    weights: list[tuple[str, torch.Tensor]],
    lora_cfg: dict | None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Merge LoRA A/B into base weights in float32.

    Handles both PEFT and NeMo LoRA naming conventions.
    """
    has_lora = any(".lora_A." in n or ".lora_B." in n for n, _ in weights)
    if not has_lora:
        yield from weights
        return

    if not lora_cfg:
        lora_cfg = {"r": 128, "lora_alpha": 256}
    scaling = lora_cfg.get("lora_alpha", 1) / lora_cfg.get("r", 1)

    base: dict[str, torch.Tensor] = {}
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}

    for name, tensor in weights:
        key, kind = _normalize_lora_name(name)
        if kind == "lora_a":
            lora_a[key] = tensor
        elif kind == "lora_b":
            lora_b[key] = tensor
        else:
            base[key] = tensor

    merged_count = 0
    for key, tensor in base.items():
        if key in lora_a and key in lora_b:
            orig_dtype = tensor.dtype
            a = lora_a.pop(key).float()
            b = lora_b.pop(key).float()
            tensor = (tensor.float() + scaling * (b @ a)).to(orig_dtype)
            merged_count += 1
        yield (key, tensor)

    logger.info(
        "Merged %d LoRA weight pairs (scaling=%.4f)",
        merged_count,
        scaling,
    )
    if lora_a:
        logger.warning("Unmerged LoRA keys: %s", list(lora_a.keys()))


class TransformerBackend(_BaseBackend):
    """Standard decoder-only LLM backbones (e.g. Qwen3).

    Includes inline LoRA merging for PEFT checkpoints; both PEFT-wrapped
    (``llm.base_model.model.model.``) and plain NeMo (``llm.model.``) name
    layouts are renamed to the HuggingFace ``model.`` prefix that
    ``AutoWeightsLoader`` expects.
    """

    def architectures(self) -> list[str]:
        return self.config.llm_architectures

    def preprocess_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        lora_cfg = getattr(self.config, "lora", None)
        return list(_merge_lora_weights(list(weights), lora_cfg))

    def nemo_to_hf_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Map NeMo/PEFT weight names to HuggingFace standard format.

        Handles:
          - PEFT-wrapped: ``llm.base_model.model.model.`` -> ``model.``
          - Plain NeMo:   ``llm.model.``                  -> ``model.``
          - Standalone:   ``embed_tokens.weight``         -> ``model.embed_tokens.weight``
          - LM head:      ``llm.lm_head.weight``          -> ``lm_head.weight``
          - Weight tying:  if no embed_tokens, duplicate lm_head as embed_tokens
        """
        target_vocab = getattr(self.config.text_config, "vocab_size", None)

        has_embed_tokens = False
        lm_head_tensor = None

        for name, tensor in weights:
            if name == "embed_tokens.weight":
                if target_vocab:
                    tensor = _pad_to_vocab_size(tensor, target_vocab)
                yield ("model.embed_tokens.weight", tensor)
                has_embed_tokens = True
                continue

            hf = name
            if hf.startswith("llm.base_model.model."):
                hf = hf[len("llm.base_model.model.") :]
            elif hf.startswith("llm."):
                hf = hf[len("llm.") :]

            if hf.startswith("lm_head") or hf.startswith("model.embed_tokens"):
                if target_vocab:
                    tensor = _pad_to_vocab_size(tensor, target_vocab)
                if hf.startswith("model.embed_tokens"):
                    has_embed_tokens = True

            if hf == "lm_head.weight":
                lm_head_tensor = tensor

            yield (hf, tensor)

        if not has_embed_tokens and lm_head_tensor is not None:
            logger.info("No embed_tokens found; duplicating lm_head (weight tying)")
            if target_vocab:
                lm_head_tensor = _pad_to_vocab_size(lm_head_tensor, target_vocab)
            yield ("model.embed_tokens.weight", lm_head_tensor)


# ── Hybrid backend (NemotronH / Mamba+MoE) ──────────────────────────


class HybridBackend(_BaseBackend):
    """Hybrid Mamba+MoE backbones (e.g. NemotronH).

    Forces vLLM's ``NemotronHForCausalLM`` for the language tower and ships
    the NemotronH-specific MoE expert split / norm rename. The mamba state
    classmethods at the bottom are reached only when vLLM's runtime
    ``ModelConfig.is_hybrid`` returns True; the outer model class delegates
    to them via classmethod passthroughs so vLLM's hybrid KV-cache allocator
    sees a stable interface regardless of which backbone is loaded.
    """

    def architectures(self) -> list[str]:
        # Normalize to vLLM's official NemotronH architecture name regardless of
        # what the backbone config originally declared (NemotronHybridForCausalLM
        # is also accepted upstream but only NemotronHForCausalLM dispatches
        # correctly inside vLLM's hybrid registry).
        return ["NemotronHForCausalLM"]

    def nemo_to_hf_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """NeMo -> HuggingFace NemotronH weight name mapping.

        Handles the MoE expert weight layout (NeMo packs experts into a single
        tensor; HF expects per-expert ``experts.{i}.{up,down}_proj.weight``)
        and the ``backbone.norm`` -> ``backbone.norm_f`` rename.
        """
        target_vocab = getattr(self.config.text_config, "vocab_size", None)
        for name, tensor in weights:
            hf_name = name.replace("llm.model.", "backbone.")
            hf_name = hf_name.replace("llm.lm_head", "lm_head")
            if hf_name == "backbone.norm.weight":
                hf_name = "backbone.norm_f.weight"

            if hf_name.endswith(".experts.down_projs"):
                prefix = hf_name.replace(".experts.down_projs", "")
                for i in range(tensor.shape[0]):
                    yield (
                        f"{prefix}.experts.{i}.down_proj.weight",
                        tensor[i].t(),
                    )
            elif hf_name.endswith(".experts.gate_and_up_projs"):
                prefix = hf_name.replace(".experts.gate_and_up_projs", "")
                for i in range(tensor.shape[0]):
                    yield (
                        f"{prefix}.experts.{i}.up_proj.weight",
                        tensor[i].t(),
                    )
            elif hf_name in (
                "backbone.embed_tokens.weight",
                "lm_head.weight",
            ):
                if target_vocab:
                    tensor = _pad_to_vocab_size(tensor, target_vocab)
                yield (hf_name, tensor)
            else:
                yield (hf_name, tensor)

    # ── vLLM IsHybrid mamba state passthroughs ──
    #
    # vLLM calls these as ``cls.get_mamba_state_*_from_config(vllm_config)`` on
    # the registered model class, and only when ``ModelConfig.is_hybrid``
    # returns True at runtime. The outer model class wires its ``IsHybrid``
    # classmethods directly to these so transformer backbones never reach this
    # path (their ``layer_types`` shim turns ``is_hybrid`` off at runtime).

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> Any:
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> Any:
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> Any:
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_copy_func()


# ── Factory ─────────────────────────────────────────────────────────


def make_backend(config: Any) -> _BaseBackend:
    """Pick the right backend for the given ``NeMoSpeechLMConfig``.

    Hybrid vs transformer is determined once at config-load time
    (``config.is_hybrid``) and then stays fixed for the lifetime of the model.
    """
    if config.is_hybrid:
        return HybridBackend(config)
    return TransformerBackend(config)
