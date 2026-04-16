# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Factory for creating LLM and TTS inference backends.

NemotronVoiceChat has two components that can each be backed by native
PyTorch or vLLM.  This factory returns the right backend for a given
``engine_type``:

- ``"native_llm"``    -- wraps the PyTorch model directly (LLM component)
- ``"native_eartts"`` -- wraps the PyTorch DuplexEARTTS model (TTS component)
- ``"vllm_llm"``      -- vLLM engine for the LLM (DuplexSTT) component
- ``"vllm_eartts"``   -- vLLM engine for the TTS (EarTTS) component

Usage:
    from nemo.collections.speechlm2.inference.model_wrappers.factory import create_model

    llm = create_model(engine_type="native_llm", model=voicechat_model)
    tts = create_model(engine_type="native_eartts", model=voicechat_model.tts_model)
"""

from typing import Any

from nemo.collections.speechlm2.inference.model_wrappers.backend.interface import ModelInterface


def create_model(
    engine_type: str,
    model=None,
    vllm_config: dict[str, Any] | None = None,
    special_token_ids: set[int] | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    **kwargs
) -> ModelInterface:
    """
    Factory function to create a single inference backend for one component.

    Each call creates a backend for one specific component (LLM or TTS) on
    one specific runtime (native PyTorch or vLLM).  The ``engine_type``
    must be one of the four ``{backend}_{component}`` combinations.

    Note: the user-facing config uses a *combined* ``engine_type`` like
    ``"native"`` or ``"vllm_llm_vllm_eartts"`` which the wrapper
    translates into two ``create_model`` calls (one for LLM, one for TTS).

    Args:
        engine_type: One of "native_llm", "native_eartts", "vllm_llm", "vllm_eartts"
        model: The PyTorch model to wrap.  Required for native backends
            (NemotronVoiceChat for LLM, DuplexEARTTS for TTS).  Not used
            by vLLM backends, which load their own engine from ``vllm_config``.
        vllm_config: Configuration dict for vLLM engines (required for "vllm*")
        special_token_ids: Set of special token IDs (pad, eos, bos) that should bypass sampling.
                          If None (default), will auto-extract from model.tokenizer for tokens:
                          '<s>' (bos), '</s>' (eos), '<SPECIAL_12>' (pad).
        top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
        repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
        temperature: Temperature for sampling. 1.0 = no change, 0.0 = greedy. Default: 1.0
        **kwargs: Additional arguments passed to the backend constructor

    Returns:
        A ModelInterface instance ready for inference.

    Example:
        >>> # Native PyTorch LLM with greedy decoding
        >>> llm = create_model(engine_type="native_llm", model=voicechat_model)
        >>>
        >>> # Native PyTorch EarTTS
        >>> tts = create_model(engine_type="native_eartts", model=voicechat_model.tts_model)
        >>>
        >>> # vLLM LLM engine
        >>> llm = create_model(engine_type="vllm_llm", vllm_config={...})
        >>>
        >>> # vLLM EarTTS engine
        >>> tts = create_model(engine_type="vllm_eartts", vllm_config={...})
    """
    engine_type = engine_type.lower()

    if engine_type == "native_llm":
        from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.model import PyTorchLLM

        if model is None:
            raise ValueError("model must be provided for native engine")
        return PyTorchLLM(
            model=model,
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

    elif engine_type == "native_eartts":
        from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.eartts import PyTorchEarTTS

        if model is None:
            raise ValueError("model (DuplexEARTTS instance) must be provided for native EarTTS engine")
        return PyTorchEarTTS(tts_model=model)

    elif engine_type == "vllm_eartts":
        from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.eartts import VLLMEarTTS

        if vllm_config is None:
            raise ValueError("vllm_config must be provided for vLLM EARTTS engine")
        return VLLMEarTTS(
            **vllm_config,
            model_type="eartts",
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            **kwargs
        )

    elif engine_type == "vllm_llm":
        from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.llm import VLLMLLM

        if vllm_config is None:
            raise ValueError("vllm_config must be provided for vLLM engine")
        return VLLMLLM(
            **vllm_config,
            model_type="llm",
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown engine_type: {engine_type}. "
            f"Supported types: 'native_llm', 'native_eartts', 'vllm_llm', 'vllm_eartts'"
        )
