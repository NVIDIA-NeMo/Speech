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
Base interface for S2S model inference backends.

NemotronVoiceChat has two main inference components that can each run on
either native PyTorch or vLLM:

- **LLM** (inside DuplexSTT): takes audio embeddings, produces text and ASR
  tokens.  Wrapped as ``model_llm_interface`` in the inference wrapper.
- **TTS** (EarTTS): takes text tokens, produces audio codec codes.
  Wrapped as ``model_eartts_interface`` in the inference wrapper.

This module defines the abstract ``ModelInterface`` that both backends
implement, with shared sampling utilities (top-p, repetition penalty,
temperature).
"""

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

from nemo.utils import logging


class ModelInterface(ABC):
    """
    Abstract base class for LLM and TTS inference backends.

    Concrete implementations wrap either the LLM component (DuplexSTT backbone
    that predicts text/ASR tokens from audio embeddings) or the TTS component
    (EarTTS that generates audio codec codes from text tokens).  Each component
    can run on native PyTorch or vLLM.

    Provides shared sampling utilities (top-p, repetition penalty, temperature)
    and lifecycle methods (compile, prefill, subword cache) so the inference
    wrapper can treat all backends uniformly.
    """

    def __init__(
        self,
        special_token_ids: set[int] | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize base interface with sampling parameters.

        Args:
            special_token_ids: Set of special token IDs (pad, eos, bos) that should bypass sampling.
                               These tokens will use greedy decoding and won't be penalized.
            top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
            repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
            temperature: Temperature for sampling. 1.0 = no change, <1.0 = sharper, >1.0 = flatter.
                        0.0 = greedy (argmax). Default: 1.0
        """
        if not math.isfinite(temperature):
            raise ValueError(f"temperature must be finite, got {temperature}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {temperature}")

        self.special_token_ids = special_token_ids if special_token_ids is not None else set()
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

        # Pre-built tensor for special-token filtering in repetition penalty.
        # Lazily moved to the right device on first use (see _sample_text_token).
        self._special_ids_tensor: torch.Tensor | None = (
            torch.tensor(sorted(self.special_token_ids), dtype=torch.long) if self.special_token_ids else None
        )

    def _sample_text_token(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        current_step: int,
        sampling_params: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """
        Sample text tokens with optional top-p sampling and repetition penalty.
        Special tokens (pad, eos, bos) bypass sampling - if they have highest probability, return them directly.

        Args:
            logits: Logits tensor of shape (B, V) for vocabulary V.
            generated_tokens: Previously generated tokens of shape (B, T).
            current_step: Current decoding step (used to slice generated_tokens).
            sampling_params: Optional per-request overrides for ``top_p``,
                ``temperature``, and ``repetition_penalty``.  Missing keys
                fall back to ``self.*`` (the pipeline-level defaults).

        Returns:
            Sampled token ids of shape (B,).
        """
        top_p = sampling_params.get("top_p", self.top_p) if sampling_params else self.top_p
        temperature = sampling_params.get("temperature", self.temperature) if sampling_params else self.temperature
        rep_penalty = (
            sampling_params.get("repetition_penalty", self.repetition_penalty)
            if sampling_params
            else self.repetition_penalty
        )

        B, V = logits.shape
        device = logits.device

        # First check greedy tokens (on original logits)
        greedy_tokens = logits.argmax(dim=-1)  # (B,)

        # If no sampling needed (all disabled), return greedy
        if top_p >= 1.0 and rep_penalty == 1.0 and (temperature == 1.0 or temperature == 0.0):
            return greedy_tokens

        # temperature=0 means greedy
        if temperature == 0.0:
            return greedy_tokens

        # For each batch, if greedy is special token, use greedy; otherwise sample
        sampled_tokens = greedy_tokens.clone()

        # Ensure cached special-token tensor is on the right device (once).
        if self._special_ids_tensor is not None and self._special_ids_tensor.device != device:
            self._special_ids_tensor = self._special_ids_tensor.to(device)

        for b in range(B):
            # If greedy token is a special token, keep it (no sampling)
            if greedy_tokens[b].item() in self.special_token_ids:
                continue

            # Not a special token - apply repetition penalty and sampling
            batch_logits = logits[b].clone()  # (V,)

            # Apply repetition penalty (vectorized, no Python loop)
            if rep_penalty != 1.0 and current_step > 0:
                unique_prev = generated_tokens[b, :current_step].unique()
                # Exclude special tokens from penalty
                if self._special_ids_tensor is not None:
                    ids_t = self._special_ids_tensor
                    if ids_t.device != unique_prev.device:
                        ids_t = ids_t.to(unique_prev.device)
                    unique_prev = unique_prev[~torch.isin(unique_prev, ids_t)]

                if unique_prev.numel() > 0:
                    if unique_prev.device != batch_logits.device:
                        unique_prev = unique_prev.to(batch_logits.device)
                    prev_logits = batch_logits[unique_prev]
                    # Positive logits are divided, negative logits are multiplied
                    # (same as the standard repetition_penalty convention)
                    batch_logits[unique_prev] = torch.where(
                        prev_logits > 0,
                        prev_logits / rep_penalty,
                        prev_logits * rep_penalty,
                    )

            # Apply temperature scaling
            if temperature != 1.0:
                batch_logits = batch_logits / temperature

            # Fall back to greedy if logits are non-finite before top-p
            # (top-p intentionally introduces -inf, so check must happen first)
            if not torch.isfinite(batch_logits).all():
                logging.warning(
                    f"_sample_text_token: logits contain NaN or inf at step {current_step}, batch {b}: "
                    f"nan={batch_logits.isnan().sum().item()}, "
                    f"inf={batch_logits.isinf().sum().item()}, "
                    f"min={batch_logits[~batch_logits.isnan()].min().item() if not batch_logits.isnan().all() else 'all_nan'}, "
                    f"max={batch_logits[~batch_logits.isnan()].max().item() if not batch_logits.isnan().all() else 'all_nan'}"
                )
                sampled_tokens[b] = greedy_tokens[b]
                continue

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(batch_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative prob > top_p, keeping at least one
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep the first token that exceeds threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                # Set to -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                batch_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(batch_logits, dim=-1)
            sampled_tokens[b] = torch.multinomial(probs, num_samples=1).item()

        return sampled_tokens

    @abstractmethod
    def __call__(self, input_embeds: torch.Tensor, cache: Any | None = None, **kwargs) -> dict[str, Any]:
        """
        Perform model inference.

        Args:
            input_embeds: Input embeddings tensor of shape [batch, seq_len, hidden_dim]
            cache: Optional cache object (e.g., DynamicCache for transformers)
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing:
                - 'text_logits': Logits for text generation [batch, seq_len, vocab_size]
                - 'cache': Updated cache object (if cache was provided)
                - Additional model-specific outputs
        """
        pass

    @abstractmethod
    def to(self, device_or_dtype: torch.device | torch.dtype) -> 'ModelInterface':
        """Move model to specified device or convert to specified dtype."""
        pass

    @abstractmethod
    def eval(self) -> 'ModelInterface':
        """Set model to evaluation mode."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device of the model."""
        pass

    def compile(self, **kwargs) -> None:
        """Apply torch.compile optimizations. No-op by default; override in subclasses."""
        pass

    def setup_subword_cache(self, cfg) -> None:
        """Enable TTS subword embedding cache. No-op by default; override in subclasses."""
        pass

    def create_cache(self, **kwargs) -> Any:
        """Create inference cache for this backend. Returns None by default.

        Override in LLM backends that manage their own cache (e.g. PyTorchLLM
        creates a DynamicCache for standard transformer models).  vLLM backends
        manage cache internally and inherit the default.
        """
        return None

    def prefill_prompt(self, embeddings, **kwargs):
        """Prefill the model with prompt embeddings before streaming begins.

        Override in subclasses to implement engine-specific prefill logic.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement prefill_prompt")

    def abort_request(self, request_id: str) -> bool:
        """Abort an in-flight streaming request.

        No-op for backends without request lifecycle management (e.g. native
        PyTorch).  vLLM backends override this to cancel engine-side requests.
        """
        return False
