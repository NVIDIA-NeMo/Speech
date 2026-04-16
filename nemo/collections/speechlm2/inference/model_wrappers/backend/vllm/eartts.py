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
vLLM backend for the TTS (EarTTS) component of NemotronVoiceChat.

EarTTS generates audio codec codes from text tokens: given the current
subword ID, mask, previous audio codes, and speaker latent, it predicts
the next frame of RVQ acoustic tokens.  This module wraps it in a
CustomInputAsyncVLLMEngine for accelerated inference.

Used as ``model_eartts_interface`` in the inference wrapper.
"""

from typing import Any
import os
import torch

from nemo.utils import logging
from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.base import VLLMModelBase
from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.eartts import TTSGenerationResult


class VLLMEarTTS(VLLMModelBase):
    """
    vLLM backend for the TTS (EarTTS) component.

    Accepts dictionary inputs with codes, subword IDs, masks, and speaker
    latents, and returns ``TTSGenerationResult`` with generated acoustic tokens.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._speaker_latent_dim = None
        logging.info("VLLMEarTTS initialized with EARTTS-specific settings.")

    def _convert_ckpt(self, save_path: str):
        """Convert EARTTS checkpoint to vLLM format."""
        from nemo.collections.speechlm2.inference.vllm.scripts.convert_eartts_checkpoint import convert
        ckpt_dir = os.path.normpath(self.model_path)
        config_file = os.path.join(ckpt_dir, "config.json")
        model_ckpt = os.path.join(ckpt_dir, "model.safetensors")
        convert(save_path, config_file, model_ckpt)

    def __call__(
        self,
        inputs: dict[str, torch.Tensor] | None = None,
        request_id: str | None = None,
        prompt_token_ids: list | None = None,
        **kwargs
    ) -> TTSGenerationResult:
        """
        Perform TTS inference using vLLM streaming engine.

        Supports two calling conventions:
        1. model(inputs_dict, request_id="id")  - pass dict as first positional arg
        2. model(**inputs_dict)  - unpack dict as keyword arguments

        Args:
            inputs: Optional dict of model inputs (if None, uses **kwargs)
            request_id: Optional request identifier
            prompt_token_ids: Optional list of prompt token IDs for prefill
            **kwargs: Model inputs as keyword arguments (used if inputs is None)

        Returns:
            TTSGenerationResult containing generated acoustic tokens and cache
        """
        if inputs is not None:
            input_dict = inputs
        else:
            if request_id is None:
                request_id = kwargs.pop('request_id', None)
            input_dict = kwargs

        if request_id is None:
            request_id = 'tts_request_id_1'

        result = self._loop.run_until_complete(
            self._async_inference(input_dict, request_id, prompt_token_ids=prompt_token_ids)
        )

        return result

    async def _process_inputs_to_outputs(
        self,
        inputs: dict[str, torch.Tensor],
        request_id: str,
        prompt_token_ids: list | None = None,
    ) -> dict[str, Any]:
        """
        Process tensor inputs to generate acoustic tokens via vLLM engine.

        Args:
            inputs: Dictionary with code, context_hidden_state, subword_ids,
                    subword_mask, and optional non_prompt_mask / audio_prompt_lantent.
            request_id: Request identifier.
            prompt_token_ids: Optional prompt token IDs for prefill.

        Returns:
            TTSGenerationResult with generated acoustic tokens.
        """

        assert inputs["context_hidden_state"] is None, "EARTTS vllm model does not support context_hidden_state input"

        codes = inputs["code"].squeeze(0)  # T x 31
        if codes.shape[0] > 1:
            # In prefill stage, shift acoustic tokens for vLLM to replicate
            # the NeMo logic for teacher-forced input construction.
            codes = torch.nn.functional.pad(codes[:-1], [0, 0, 1, 0])
        input_tensors = [
            codes,
            inputs["subword_ids"].squeeze(0),
            inputs["subword_mask"].squeeze(0),
        ]

        if "non_prompt_mask" in inputs:
            # Apply edge detection to match native model's BOS placement logic:
            # BOS should only be applied at the FIRST position where non_prompt_mask is True
            non_prompt_mask = inputs["non_prompt_mask"].squeeze(0)  # T
            padded_prev = torch.nn.functional.pad(non_prompt_mask[:-1], [1, 0], value=False)
            bos_mask = (non_prompt_mask & (~padded_prev)).to(dtype=getattr(torch, self._dtype))
            input_tensors.append(bos_mask)

        else:
            current_subword_id = input_tensors[1]
            # Use a tiny epsilon instead of exact 0 so the vLLM model's
            # (bos_mask == 0) check is False during decoding.  This prevents
            # use_audio_prompt_frozen_projection from incorrectly applying the
            # speaker-prompt projection to every decoding step.  The epsilon is
            # small enough that bos_mask * bos_emb remains negligible.
            bos_mask = torch.full_like(current_subword_id, 1e-20, dtype=getattr(torch, self._dtype))
            input_tensors.append(bos_mask)

        # Pass speaker_latent: the pre-extracted speaker embedding.
        # During prefill with speaker_name: audio_prompt_lantent is [1, T, hidden_size]
        # During decode or speaker_reference: pass zeros so the model falls back
        # to computing the latent from acoustic tokens.
        if "audio_prompt_lantent" in inputs and inputs["audio_prompt_lantent"] is not None:
            speaker_latent = inputs["audio_prompt_lantent"].squeeze(0)  # T x hidden_size
            self._speaker_latent_dim = speaker_latent.shape[-1]
            input_tensors.append(speaker_latent.to(dtype=getattr(torch, self._dtype)))
        else:
            if self._speaker_latent_dim is None:
                import json as _json
                dir_name = os.path.basename(os.path.normpath(self.model_path))
                converted_config_path = os.path.join("/tmp", dir_name + "_vllm_converted_eartts", "config.json")
                if os.path.exists(converted_config_path):
                    with open(converted_config_path) as _f:
                        self._speaker_latent_dim = _json.load(_f)["hidden_size"]
                else:
                    raise RuntimeError(
                        f"Cannot determine speaker_latent_dim: converted config not found at {converted_config_path}. "
                        "Run a prefill with audio_prompt_lantent first, or ensure the converted checkpoint exists."
                    )
            num_tokens = codes.shape[0]
            speaker_latent = torch.zeros(num_tokens, self._speaker_latent_dim, dtype=getattr(torch, self._dtype))
            input_tensors.append(speaker_latent)

        result = await self.engine.generate_next_token(input_tensors, prompt_token_ids=prompt_token_ids, request_id=request_id)
        acoustic_tokens = result.custom_outputs["acoustic_tokens"]  # T x 31
        step_acoustic_tokens = acoustic_tokens[-1:]  # 1 x 31
        return TTSGenerationResult(
            codes=step_acoustic_tokens.unsqueeze(0).cuda(),  # Add batch dim back: 1 x 1 x 31
            past_key_values=None  # vLLM manages cache internally
        )

    def prefill_prompt(self, init_inputs, prompt_token_ids=None, request_id=None, **kwargs):
        """Prefill vLLM EarTTS engine with speaker embedding context.

        Args:
            init_inputs: Dict of initial TTS inputs (codes, subword_ids, etc.).
            prompt_token_ids: List of prompt token IDs for the speaker embedding.
            request_id: Unique request identifier.

        Returns:
            TTSGenerationResult with codes from the prefill step, or None.
        """
        import copy
        inputs_copy = copy.deepcopy(init_inputs)
        return self(inputs_copy, request_id=request_id, prompt_token_ids=prompt_token_ids)
