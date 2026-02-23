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
"""Inference wrapper for StreamingSALM — exposes encode/generate_streaming for pipeline use."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nemo.collections.asr.inference.utils.device_utils import setup_device

if TYPE_CHECKING:
    from nemo.collections.speechlm2.models.streaming_salm import StreamingSALM, StreamingState


class StreamingSALMInferenceWrapper:
    """
    Wraps :class:`StreamingSALM` for streaming inference.

    Unlike :class:`SALMASRInferenceWrapper`, this wrapper does **not** inherit
    from ``ASRInferenceWrapper`` because StreamingSALM has no CTC/RNNT-style
    preprocessor, vocabulary, or subsampling factor.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        device_id: int = 0,
        compute_dtype: str = "bfloat16",
        use_amp: bool = True,
        latency: int = 1,
        context: str | None = None,
    ):
        self.device_str, self.device_id, self.compute_dtype = setup_device(device.strip(), device_id, compute_dtype)
        self.use_amp = use_amp
        self.device = torch.device(self.device_str)
        self.model = self._load_model(model_name, self.device)
        self.model.to(dtype=self.compute_dtype)
        self.latency = latency
        self.context = context

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def blank_token_id(self) -> int:
        return self.model.blank_token_id

    @property
    def word_separator(self) -> str:
        return " "

    @property
    def sample_rate(self) -> int:
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        return MimiEncoder.SAMPLE_RATE

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_name: str, device: torch.device) -> StreamingSALM:
        try:
            from nemo.collections.speechlm2.models import StreamingSALM

            model = StreamingSALM.from_pretrained(model_name).eval()
            model.to(device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load StreamingSALM model {model_name}: {e}") from e

    # ------------------------------------------------------------------
    # Audio encoding
    # ------------------------------------------------------------------

    def encode_audio(self, audio: Tensor, audio_lens: Tensor) -> tuple[Tensor, Tensor]:
        """Encode raw audio to Mimi codes.

        Args:
            audio: ``(B, T_samples)`` at Mimi's native sample rate (24 kHz).
            audio_lens: ``(B,)`` sample counts.

        Returns:
            codes ``(B, num_codebooks, T_frames)`` and code_lens ``(B,)``.
        """
        with (
            torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
        ):
            return self.model.mimi.encode(audio, audio_lens)

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def generate_streaming(
        self,
        audio_codes: Tensor | None,
        model_state: StreamingState | None,
        latency: int = 1,
        context: str | None = None,
    ) -> tuple[list[list[int]], StreamingState]:
        """Thin wrapper around ``model.generate_streaming`` with AMP context."""
        with (
            torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
        ):
            return self.model.generate_streaming(audio_codes, model_state, latency, context)

    # ------------------------------------------------------------------
    # Text decoding
    # ------------------------------------------------------------------

    def ids_to_text(self, token_ids: list[int]) -> str:
        return self.model.tokenizer.ids_to_text(token_ids)
