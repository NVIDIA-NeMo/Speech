# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""vLLM plugin registration for NeMo Speech LM (SALM) models.

Registers ``NeMoSpeechLMConfig`` and the single
``NeMoSpeechLMForConditionalGeneration`` model class with vLLM via the
``vllm.general_plugins`` entry point.

A single model class covers every supported backbone family (standard
decoder-only LLMs like Qwen3, hybrid Mamba+MoE like NemotronH).
Backbone-specific behavior is selected at instantiation time.
"""

import logging

_PKG = "nemo.collections.speechlm2.vllm.salm"
_LOGGER = logging.getLogger(__name__)
_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE = None
_AUTOMODEL_DFLASH2_ARCHITECTURES = frozenset(
    {
        "Qwen3DFlash2DraftModel",
        "DFlashQwen3DFlash2DraftModel",
    }
)


def _normalize_dflash2_architecture(hf_config, *, supported_archs=None):
    """Route Automodel DFlash2 exports to vLLM's canonical runtime.

    Automodel keeps its training class in ``config.json`` so the checkpoint can
    be reopened for training. The pinned vLLM DFlash2 implementation dispatches
    both its V2 runner and candidate-selector speculator only when it sees the
    canonical ``DFlash2DraftModel`` architecture. Normalize before vLLM wraps
    the draft in ``EAGLEConfig``; otherwise ``method=dflash`` prefixes the
    Automodel name and silently selects the plain-DFlash speculator.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    if len(architectures) != 1 or architectures[0] not in _AUTOMODEL_DFLASH2_ARCHITECTURES:
        return hf_config

    if supported_archs is None:
        try:
            from vllm.model_executor.models.registry import ModelRegistry
        except (ImportError, RuntimeError):
            return hf_config
        supported_archs = ModelRegistry.get_supported_archs()

    supported_archs = set(supported_archs)
    automodel_arch = architectures[0]
    if "DFlash2DraftModel" in supported_archs and automodel_arch not in supported_archs:
        hf_config.architectures = ["DFlash2DraftModel"]
    return hf_config


def _register_model_aliases(model_registry, native_arch, aliases, supported_archs=None) -> None:
    """Register aliases without assuming vLLM's private registry-entry representation."""
    if supported_archs is None:
        supported_archs = model_registry.get_supported_archs()
    supported_archs = set(supported_archs)
    if native_arch not in supported_archs:
        return

    native_model = model_registry.models[native_arch]
    module_name = getattr(native_model, "module_name", None)
    class_name = getattr(native_model, "class_name", None)
    if isinstance(module_name, str) and isinstance(class_name, str):
        model_ref = f"{module_name}:{class_name}"
    else:
        model_ref = getattr(native_model, "model_cls", None)

    if model_ref is None:
        _LOGGER.warning(
            "Cannot register aliases for %s: unsupported vLLM registry entry %s.",
            native_arch,
            type(native_model).__name__,
        )
        return

    for alias in aliases:
        if alias not in supported_archs:
            model_registry.register_model(alias, model_ref)


def _nemo_speechlm_mtp_hf_config_override(hf_config):
    """Apply SpeechLM speculative-config rewrites, then defer to vLLM.

    This function must remain at module scope: vLLM retains it on the draft
    ``ModelConfig``, which can cross a spawned process boundary. The original
    vLLM callable stays in process-local module state because binding the
    replaced static method inside a closure also makes that method
    unresolvable by standard pickle.
    """
    if hf_config.model_type == "nemo_speechlm":
        mtp_cfg = getattr(hf_config, "mtp", None)
        if not isinstance(mtp_cfg, dict):
            mtp_cfg = {}
        # Match SALMAutomodel's training defaults exactly: retaining a recipe
        # depth does not enable MTP, while an enabled block with no explicit
        # depth constructs one logical head.
        mtp_enabled = bool(mtp_cfg.get("enabled", False))
        n_predict = int(mtp_cfg.get("num_nextn_predict_layers", 1 if mtp_enabled else 0) or 0)
        if mtp_enabled and n_predict > 0:
            use_repeated_layer = bool(mtp_cfg.get("use_repeated_layer", False))
            if n_predict > 1 and not use_repeated_layer:
                raise ValueError(
                    f"NeMo SpeechLM MTP with {n_predict} distinct head layers is not "
                    f"supported: vLLM's NemotronHMultiTokenPredictor builds a single "
                    f"physical MTP layer and reuses it every speculative step. Only "
                    f"checkpoints trained with mtp.use_repeated_layer=true match that "
                    f"execution model."
                )
            hf_config.model_type = "nemo_speechlm_mtp"
            hf_config.update(
                {
                    # vLLM instantiates one physical prediction step and reuses
                    # it for arbitrary speculative K. Repeated-layer training
                    # produces exactly that checkpoint layout.
                    "n_predict": 1,
                    "num_nextn_predict_layers": 1,
                    "architectures": ["NeMoSpeechLMMTPModel"],
                }
            )
            return hf_config

    global _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE
    if _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE is None:
        # A spawn child can import this module while unpickling the function
        # without running the vLLM plugin hook first. In that case the class
        # still exposes its native override, which is safe to capture lazily.
        import vllm.config.speculative as _spec_mod

        current_override = _spec_mod.SpeculativeConfig.hf_config_override
        if current_override is _nemo_speechlm_mtp_hf_config_override:
            raise RuntimeError("NeMo SpeechLM MTP override was installed without preserving vLLM's original hook.")
        _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE = current_override
    hf_config = _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE(hf_config)
    return _normalize_dflash2_architecture(hf_config)


_nemo_speechlm_mtp_hf_config_override._nemo_speechlm_mtp_override = True


def _patch_vllm_for_nemo_speechlm_mtp() -> None:
    """Extend vLLM's speculative-decoding framework for SpeechLM drafts.

    Four patches are applied on supported vLLM releases:

    1. ``MTPModelTypes`` — the Literal type that guards the MTP detection
       branch in ``SpeculativeConfig.__post_init__`` is extended to include
       ``"nemo_speechlm_mtp"``.

    2. ``SpeculativeConfig.hf_config_override`` — the static method that
       rewrites the draft-model HF config is wrapped to detect
       ``nemo_speechlm`` checkpoints that carry enabled MTP heads
       (``mtp.enabled`` and ``mtp.num_nextn_predict_layers > 0``) and redirect
       them to the
       ``NeMoSpeechLMMTPModel`` architecture with the right ``n_predict``.

    3. ``ModelRegistry`` — ``NeMoSpeechLMMTPModel`` is registered so that
       vLLM can resolve and instantiate it as the draft model.

    4. Automodel DFlash2 architecture names are normalized to vLLM's canonical
       ``DFlash2DraftModel`` before ``EAGLEConfig`` wrapping, which activates
       the V2 model runner and candidate-selector speculator.
    """
    from typing import Literal, get_args

    import vllm.config.speculative as _spec_mod

    # Extend vLLM's recognized MTP model types.
    old_args = get_args(_spec_mod.MTPModelTypes)
    if "nemo_speechlm_mtp" not in old_args:
        _spec_mod.MTPModelTypes = Literal[old_args + ("nemo_speechlm_mtp",)]

    # Route SpeechLM MTP checkpoints through SpeculativeConfig.hf_config_override.
    current_override = _spec_mod.SpeculativeConfig.hf_config_override
    if not getattr(current_override, "_nemo_speechlm_mtp_override", False):
        global _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE
        # Preserve the first native hook for the lifetime of this process.
        # Replacing it during later registration could capture a third-party
        # wrapper that already delegates to us and create an override cycle.
        if _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE is None:
            _ORIGINAL_VLLM_HF_CONFIG_OVERRIDE = current_override
        _spec_mod.SpeculativeConfig.hf_config_override = staticmethod(_nemo_speechlm_mtp_hf_config_override)

    # Register the SpeechLM MTP draft architecture with vLLM.
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMMTPModel",
        f"{_PKG}.mtp:NeMoSpeechLMMTP",
    )


def register():
    """Register the NeMo Speech LM model and config with vLLM."""
    from transformers import AutoConfig

    from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig

    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["nemo_speechlm"] = NeMoSpeechLMConfig

    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )

    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from nemo.collections.speechlm2.vllm.salm.config_hook import NeMoSpeechLMForConditionalGenerationConfig

    MODELS_CONFIG_MAP["NeMoSpeechLMForConditionalGeneration"] = NeMoSpeechLMForConditionalGenerationConfig

    _patch_vllm_for_nemo_speechlm_mtp()

    from nemo.collections.speechlm2.vllm.salm.runtime_compat import install_prompt_contract

    install_prompt_contract()

    # DFlash aliases are optional compatibility shims. Install them only after
    # the mandatory SpeechLM model, MTP hook, and prompt contract are active so
    # a future registry representation cannot leave the server half-patched.
    _register_model_aliases(
        ModelRegistry,
        "DFlashDraftModel",
        ("Qwen3DFlashDraftModel", "DFlashQwen3DFlashDraftModel"),
    )
