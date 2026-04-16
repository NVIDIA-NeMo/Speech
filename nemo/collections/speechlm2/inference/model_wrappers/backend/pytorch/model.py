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
Native PyTorch backend for the LLM component of NemotronVoiceChat.

Wraps the DuplexSTT model (which contains the Nemotron LLM backbone) for
direct PyTorch inference with top-p sampling and repetition penalty support.

Used as ``model_llm_interface`` in the inference wrapper when
``engine_type="native_llm"``.
"""

from typing import Any
import torch

from nemo.utils import logging
from nemo.collections.speechlm2.inference.model_wrappers.backend.interface import ModelInterface


class PyTorchLLM(ModelInterface):
    """
    Native PyTorch backend for the LLM (DuplexSTT) component.

    Wraps the DuplexSTT model's forward pass (``stt_model()``) to produce
    text/ASR token predictions, conforming to the ModelInterface contract.
    Supports top-p sampling and repetition penalty.
    """

    def __init__(
        self,
        model,
        special_token_ids: set[int] | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize with an existing model.

        Args:
            model: The DuplexS2SExternalSpeechDecoderModel instance
            special_token_ids: Set of special token IDs (pad, eos, bos) that should bypass sampling.
                               These tokens will use greedy decoding and won't be penalized.
                               If None, will try to extract from model.tokenizer for tokens:
                               '<s>' (bos), '</s>' (eos), '<SPECIAL_12>' (pad).
                               You can also manually provide: {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
            top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
            repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
                               Recommended value when enabling: 1.2
            temperature: Temperature for sampling. 1.0 = no change, <1.0 = sharper, >1.0 = flatter.
                        0.0 = greedy (argmax). Default: 1.0
        """
        DEFAULT_SPECIAL_TOKEN_IDS = {1, 2, 12}

        if special_token_ids is None:
            special_token_ids = self._extract_special_token_ids_from_nemo(model)
        if not special_token_ids:
            special_token_ids = DEFAULT_SPECIAL_TOKEN_IDS

        super().__init__(
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        self.model = model

        logging.debug(f"Special token IDs: {self.special_token_ids}")

        sampling_active = top_p < 1.0 or repetition_penalty != 1.0 or (temperature != 1.0 and temperature != 0.0)
        if sampling_active and not self.special_token_ids:
            import warnings
            warnings.warn(
                "Sampling is enabled but special_token_ids is empty. "
                "Could not auto-extract from model.tokenizer. "
                "Please provide special_token_ids manually to ensure special tokens use greedy decoding. "
                "Otherwise, EOS tokens may be randomly sampled and generation may not stop properly!"
            )

    def __call__(
        self,
        input_embeds: torch.Tensor,
        cache: Any | None = None,
        cache_position: torch.Tensor | None = None,
        generated_tokens: torch.Tensor | None = None,
        current_step: int = 0,
        return_logits: bool = False,
        sampling_params: dict[str, float] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Perform inference using the native model.

        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_dim]
            cache: Optional DynamicCache or HybridMambaAttentionDynamicCache
            cache_position: Optional position tensor for Nemotron models
            generated_tokens: Previously generated tokens [batch, num_generated].
                             Required for repetition_penalty. If None, creates empty tensor.
            current_step: Current decoding step. Used for repetition penalty.
            sampling_params: Optional per-request overrides for sampling
                (top_p, temperature, repetition_penalty).
            **kwargs: Additional arguments passed to the model

        Returns:
            Dictionary with 'predicted_token', 'asr_predicted_token', and 'cache'
        """
        result = self.model.stt_model(input_embeds, cache=cache, cache_position=cache_position, **kwargs)

        if not isinstance(result, dict):
            raise TypeError(f"Model returned {type(result)}, expected dict")

        if 'text_logits' not in result:
            raise KeyError("Model output must contain 'text_logits' key")

        text_logits = result["text_logits"][:, -1]  # [batch, vocab_size]
        batch_size = text_logits.shape[0]

        if generated_tokens is None:
            gen_tokens = torch.empty(batch_size, 0, device=text_logits.device, dtype=torch.long)
        else:
            gen_tokens = generated_tokens

        predicted_token = self._sample_text_token(
            logits=text_logits,
            generated_tokens=gen_tokens,
            current_step=current_step,
            sampling_params=sampling_params,
        )

        # ASR tokens use greedy decoding (no sampling)
        asr_predicted_token = result["asr_logits"][:, -1].argmax(dim=-1)

        ans = {
            "predicted_token": predicted_token,
            "asr_predicted_token": asr_predicted_token,
            "cache": result.get("cache", None),
        }
        if return_logits:
            ans["text_logits"] = result["text_logits"]
            ans["asr_logits"] = result.get("asr_logits")
            if "function_logits" in result:
                ans["function_logits"] = result["function_logits"]
        if "function_logits" in result:
            ans["function_predicted_token"] = result["function_logits"][:, -1].argmax(dim=-1)
        return ans

    @staticmethod
    def _extract_special_token_ids_from_nemo(model) -> set[int]:
        """
        Extract special token IDs from NeMo model's tokenizer.

        NeMo tokenizer uses bos_token, eos_token, pad_token (not bos_token_id).
        Then converts token strings to IDs using token_to_id method.

        Args:
            model: The DuplexS2SExternalSpeechDecoderModel instance

        Returns:
            Set of special token IDs, or empty set if extraction fails
        """
        special_ids = set()
        try:
            tokenizer = model.stt_model.tokenizer
        except AttributeError:
            logging.debug("Cannot extract special token IDs: model has no stt_model.tokenizer")
            return special_ids

        for attr in ('bos_token', 'eos_token', 'pad_token'):
            token = getattr(tokenizer, attr, None)
            if token is not None and hasattr(tokenizer, 'token_to_id'):
                tid = tokenizer.token_to_id(token)
                if tid is not None and isinstance(tid, int):
                    special_ids.add(tid)

        return special_ids

    def to(self, device_or_dtype: torch.device | torch.dtype) -> 'PyTorchLLM':
        """Move underlying model to device or convert dtype."""
        self.model = self.model.to(device_or_dtype)
        return self

    def eval(self) -> 'PyTorchLLM':
        """Set underlying model to eval mode."""
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        """Get device of the underlying model."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def prefill_prompt(self, embeddings, cache=None, cache_position=None, **kwargs):
        """Prefill the native LLM with prompt embeddings to warm up the KV cache.

        Args:
            embeddings: Prompt embeddings [batch, seq_len, hidden_dim].
            cache: KV cache object to update in-place.
            cache_position: Position tensor for the prompt tokens.

        Returns:
            Dictionary with updated 'cache'.
        """
        result = self.model.stt_model(embeddings, cache=cache, cache_position=cache_position, **kwargs)
        if not isinstance(result, dict):
            raise TypeError(f"Model returned {type(result)}, expected dict")
        return {"cache": result.get("cache", cache)}

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying model.

        This allows transparent access to model attributes like
        perception, tokenizer, etc.
        """
        if name in ('model', '__dict__', '__class__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self.model, name)
