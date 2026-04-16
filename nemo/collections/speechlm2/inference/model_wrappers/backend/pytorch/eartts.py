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
Native PyTorch backend for the TTS (EarTTS) component of NemotronVoiceChat.

Wraps the DuplexEARTTS model for direct PyTorch inference, providing the
same ``ModelInterface`` API as ``VLLMEarTTS`` so the inference wrapper can
treat both backends uniformly.

Used as ``model_eartts_interface`` in the inference wrapper when
``engine_type="native_eartts"``.
"""

from typing import Any
from dataclasses import dataclass, fields
import torch

from nemo.utils import logging
from nemo.collections.speechlm2.inference.model_wrappers.backend.interface import ModelInterface


@dataclass
class TTSGenerationResult:
    """Result from a single TTS generation step (shared by PyTorch and vLLM backends)."""
    codes: torch.Tensor  # Generated acoustic tokens
    past_key_values: Any  # Updated cache (if applicable)

    def __getitem__(self, item: str | int):
        """Allows for accessing attributes by key or index."""
        if isinstance(item, str):
            return getattr(self, item)
        else:
            return getattr(self, fields(self)[item].name)


class PyTorchEarTTS(ModelInterface):
    """
    Native PyTorch backend for the TTS (EarTTS) component.

    Wraps ``DuplexEARTTS.infer_codes_one_step()`` for per-frame audio codec
    generation, conforming to the ModelInterface contract.
    """

    def __init__(self, tts_model):
        """
        Args:
            tts_model: A ``DuplexEARTTS`` instance (``NemotronVoiceChat.tts_model``).
        """
        super().__init__()
        self.tts_model = tts_model

    def __call__(self, inputs: dict, **kwargs) -> TTSGenerationResult:
        """
        Run one TTS code-generation step via ``infer_codes_one_step``.

        Args:
            inputs: Keyword arguments for ``DuplexEARTTS.infer_codes_one_step``
                (current_subword_id, prev_subword_id, current_subword_mask,
                prev_audio_tokens, past_key_values, guidance_enabled, etc.)

        Returns:
            TTSGenerationResult with generated codes and updated cache.
        """
        codes, cache = self.tts_model.infer_codes_one_step(**inputs)
        return TTSGenerationResult(codes=codes, past_key_values=cache)

    def prefill_prompt(self, init_inputs, prompt_token_ids=None, request_id=None, **kwargs):
        """Prefill TTS with speaker embedding / warmup inputs.

        For native PyTorch, this calls the inner ``tts_model`` directly
        (the actual EarTTS nn.Module, not the DuplexEARTTS wrapper).

        Args:
            init_inputs: Dict of initial TTS inputs from ``get_init_inputs()``.
            prompt_token_ids: Unused for native (vLLM-only parameter).
            request_id: Unused for native (vLLM-only parameter).

        Returns:
            Model outputs (with ``past_key_values`` and ``codes``).
        """
        return self.tts_model.tts_model(**init_inputs)

    def compile(self, **kwargs) -> None:
        """Apply torch.compile to the TTS backbone if available."""
        tts_backbone = getattr(self.tts_model, 'tts_model', None)
        if tts_backbone is not None and hasattr(tts_backbone, 'backbone'):
            mode = kwargs.get('mode', 'default')
            logging.info(f"Compiling TTS backbone with torch.compile(mode='{mode}')...")
            tts_backbone.backbone = torch.compile(tts_backbone.backbone, mode=mode)
            logging.info("  TTS backbone compiled")

    def setup_subword_cache(self, cfg) -> None:
        """Enable TTS subword embedding cache from config flags."""
        from omegaconf import OmegaConf

        tts_inner = getattr(self.tts_model, 'tts_model', None)
        if tts_inner is None or not hasattr(tts_inner, 'config'):
            return
        if bool(cfg.get("use_tts_subword_cache", False)):
            OmegaConf.update(tts_inner.config, "use_tts_subword_cache", True)
            logging.info("TTS speedup enabled: use_tts_subword_cache")
            embed_subword = getattr(tts_inner, 'embed_subword', None)
            if embed_subword is not None and hasattr(embed_subword, 'use_tts_subword_cache'):
                embed_subword.use_tts_subword_cache = True

    def to(self, device_or_dtype: torch.device | torch.dtype) -> 'PyTorchEarTTS':
        """Move underlying TTS model to device or convert dtype."""
        self.tts_model = self.tts_model.to(device_or_dtype)
        return self

    def eval(self) -> 'PyTorchEarTTS':
        """Set underlying TTS model to eval mode."""
        self.tts_model.eval()
        return self

    @property
    def device(self) -> torch.device:
        """Get device of the underlying TTS model."""
        try:
            return next(self.tts_model.parameters()).device
        except StopIteration:
            return torch.device('cpu')
