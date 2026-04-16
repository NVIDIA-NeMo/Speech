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

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions


@dataclass
class S2SStreamingState:
    """Pipeline-level output accumulator for a single S2S stream.

    Collects generated audio chunks and text parts produced by each
    inference step, concatenating lazily to avoid O(n^2) copying.

    This is *not* the model-level decode state (KV caches, token
    workspaces) -- that is :class:`StreamingDecodeState` in
    ``model_wrappers/decode_state.py``.
    """

    # Required init metadata
    device: torch.device
    dtype: torch.dtype
    output_sample_rate: int

    # Per-stream request options (system prompt, sampling overrides, etc.)
    options: S2SRequestOptions = field(default_factory=S2SRequestOptions)

    # Audio chunks accumulated each step; use the ``audio_buffer`` property
    # to get a single concatenated tensor (lazy, O(n) instead of O(n^2)).
    _audio_chunks: list[torch.Tensor] = field(default_factory=list, repr=False)
    _total_audio_samples: int = field(default=0, repr=False)

    # Text parts accumulated each step; use the ``output_text_str`` /
    # ``output_asr_text_str`` properties to get joined strings.
    _text_parts: list[str] = field(default_factory=list, repr=False)
    _asr_text_parts: list[str] = field(default_factory=list, repr=False)

    # Snapshots of full token-ID tensors, saved from StreamingDecodeState
    # before the decode context is destroyed at end-of-stream.
    # Used for post-hoc tokens_to_str conversion and CTM generation.
    final_gen_text: torch.Tensor | None = None
    final_gen_asr_text: torch.Tensor | None = None
    final_gen_function_text: torch.Tensor | None = None
    final_total_frames: int = 0

    @property
    def audio_buffer(self) -> torch.Tensor:
        """Concatenated audio from all steps. Built lazily to avoid O(n^2) copies."""
        if not self._audio_chunks:
            return torch.empty((1, 0), device=self.device, dtype=self.dtype)
        return torch.cat(self._audio_chunks, dim=-1)

    @property
    def output_text_str(self) -> str:
        """Accumulated agent response text. Joined lazily to avoid O(n^2) copies."""
        return "".join(self._text_parts)

    @property
    def output_asr_text_str(self) -> str:
        """Accumulated ASR (user) text. Joined lazily to avoid O(n^2) copies."""
        return "".join(self._asr_text_parts)

    def reset(self) -> None:
        """Reset all accumulated outputs to initial state."""
        self._audio_chunks.clear()
        self._total_audio_samples = 0
        self._text_parts.clear()
        self._asr_text_parts.clear()
        self.final_gen_text = None
        self.final_gen_asr_text = None
        self.final_gen_function_text = None
        self.final_total_frames = 0

    def append_step_output(
        self,
        audio: torch.Tensor,
        text: str | None = None,
        asr_text: str | None = None,
    ) -> None:
        """Append generated audio and optional text from one inference step."""
        if audio is None:
            return
        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor")

        append_tensor = audio
        if append_tensor.dim() > 1:
            append_tensor = append_tensor.reshape(1, -1)
        elif append_tensor.dim() == 1:
            append_tensor = append_tensor.unsqueeze(0)
        self._audio_chunks.append(append_tensor.to(self.device, dtype=self.dtype))
        self._total_audio_samples += int(append_tensor.shape[-1])

        if isinstance(text, str) and text:
            self._text_parts.append(text)

        if isinstance(asr_text, str) and asr_text:
            self._asr_text_parts.append(asr_text)

    def save_token_tensors(
        self,
        gen_text: torch.Tensor,
        gen_asr_text: torch.Tensor,
        total_frames: int,
        gen_function_text: torch.Tensor | None = None,
    ) -> None:
        """Snapshot the full token-ID tensors from the decode context before it is destroyed."""
        self.final_gen_text = gen_text[:, :total_frames].clone().cpu()
        self.final_gen_asr_text = gen_asr_text[:, :total_frames].clone().cpu()
        self.final_total_frames = total_frames
        self.final_gen_function_text = (
            gen_function_text[:, :total_frames].clone().cpu()
            if gen_function_text is not None else None
        )

    def get_token_tensors(self) -> tuple | None:
        """Return (gen_text, gen_asr_text, total_frames, gen_function_text) or None if not saved."""
        if self.final_gen_text is None:
            return None
        return self.final_gen_text, self.final_gen_asr_text, self.final_total_frames, self.final_gen_function_text

    def clear_audio_buffer(self) -> None:
        """Clear the audio buffer (e.g. after sending audio to a client)."""
        self._audio_chunks.clear()
        self._total_audio_samples = 0
