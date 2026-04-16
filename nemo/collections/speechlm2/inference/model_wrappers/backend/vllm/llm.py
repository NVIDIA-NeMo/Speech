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
vLLM backend for the LLM component of NemotronVoiceChat.

The LLM lives inside DuplexSTT: it takes audio frame embeddings (from the
perception encoder) and produces text tokens, ASR tokens, and optional
function-call tokens at each step.  This module wraps it in a
CustomInputAsyncVLLMEngine for accelerated inference.

Used as ``model_llm_interface`` in the inference wrapper.
"""

from typing import Any
import torch

from nemo.utils import logging
from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.base import VLLMModelBase


class VLLMLLM(VLLMModelBase):
    """
    vLLM backend for the LLM (DuplexSTT) component.

    Accepts audio frame embeddings, runs them through the vLLM streaming
    engine one step at a time, and returns text/ASR token predictions with
    optional post-hoc sampling (top-p, repetition penalty).
    """

    def _convert_ckpt(self, save_path: str):
        """Convert existing DuplexSTT checkpoint to vLLM-compatible HF format."""
        from nemo.collections.speechlm2.inference.vllm.scripts.convert_nemotronllm_checkpoint import convert_nemo_to_hf_format

        convert_nemo_to_hf_format(
            checkpoint_path=self.model_path,
            output_dir=save_path,
            pretrained_llm=self.pretrained_llm,
            dtype=self._dtype
        )
        logging.info(f"Converted model saved to {save_path}")

    def __call__(
        self,
        input_embeds: torch.Tensor,
        request_id: str | None = "request_id_1",
        **kwargs
    ) -> dict[str, Any]:
        """
        Perform inference using vLLM streaming engine.

        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_dim]
            request_id: Unique request identifier for this generation
            **kwargs: Additional arguments (decode_steps, generated_tokens, etc.)

        Returns:
            Dictionary containing predicted_token, asr_predicted_token, cache,
            is_finished, and request_id.
        """
        result = self._loop.run_until_complete(
            self._async_inference(input_embeds, request_id, **kwargs)
        )
        return result

    async def _process_inputs_to_outputs(
        self,
        input_embeds: torch.Tensor,
        request_id: str,
        decode_steps: int = 1,
        prompt_token_ids: list | None = None,
        generated_tokens: torch.Tensor | None = None,
        current_step: int = 0,
        sampling_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Process embeddings sequentially to generate text and ASR tokens.

        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_dim]
            request_id: Request identifier
            decode_steps: Number of decoding steps to perform; 0 means prefill only
            prompt_token_ids: Optional list of prompt token IDs for prefill
            generated_tokens: Previously generated tokens [batch, num_generated].
                             Required for repetition_penalty. If None, creates empty tensor.
            current_step: Current decoding step. Used for repetition penalty.
            sampling_params: Optional per-request overrides for sampling.
        """

        if decode_steps == 0:
            input_embeds = input_embeds.flatten(0, 1)  # [seq_len, hidden_dim]
            result = await self.engine.generate_next_token([input_embeds],
                                                            prompt_token_ids,
                                                            request_id=request_id)
            return True if result is not None else False

        text_token_ids = []
        asr_token_ids = []
        result = None
        for i in range(decode_steps):
            single_embed = input_embeds[:, i:i+1, :].squeeze(1)  # [batch, hidden_dim]

            result = await self.engine.generate_next_token([single_embed], request_id=request_id)
            if result is None:
                break

            text_token_ids.append(result.token_id)
            asr_token_ids.append(result.custom_outputs["asr_tokens"])

            if result.is_finished:
                break

        assert len(text_token_ids) <= decode_steps, "Generated more tokens than input embeddings"
        is_finished = False
        if text_token_ids:
            is_finished = len(text_token_ids) < decode_steps or (result and result.is_finished)

        text_logits = result.custom_outputs["text_logits"] if result else None

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

        ans = {
            "predicted_token": predicted_token,
            "asr_predicted_token": asr_token_ids[-1],
            "cache": None,  # vLLM manages cache internally
            "is_finished": is_finished,
            "request_id": request_id
        }
        if result and result.custom_outputs and "function_tokens" in result.custom_outputs:
            ans["function_predicted_token"] = result.custom_outputs["function_tokens"]
        return ans

    def prefill_prompt(self, embeddings: torch.Tensor, request_id: str = None, **kwargs) -> bool:
        """Prefill vLLM LLM engine with prompt embeddings in a single bulk step.

        Args:
            embeddings: Prompt embeddings [batch, seq_len, hidden_dim].
            request_id: Unique request identifier.

        Returns:
            True if prefill succeeded.
        """
        return self._loop.run_until_complete(
            self._async_inference(embeddings, request_id, decode_steps=0)
        )
