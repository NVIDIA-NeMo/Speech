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


import torch
from torch import Tensor

from nemo.collections.asr.inference.model_wrappers.asr_inference_wrapper import ASRInferenceWrapper
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel


class RNNTInferenceWrapper(ASRInferenceWrapper):
    """
    Provides a unified interface to work with RNNT/TDT/Hybrid models.
    """

    def __post_init__(self) -> None:
        """
        Additional post initialization step
        Checks if the model is a rnnt model and sets the decoding strategy to rnnt.
        """
        if not isinstance(self.asr_model, (EncDecRNNTModel, EncDecHybridRNNTCTCModel)):
            raise ValueError(
                "Provided model is not a RNNT type. You are trying to use a RNNT transcriber with a non-RNNT model."
            )

        decoder_type = 'rnnt'
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.cur_decoder = decoder_type

        # reset the decoding strategy
        self.reset_decoding_strategy(decoder_type)
        self.set_decoding_strategy(decoder_type)

        self.cast_dtype = torch.float32 if self.use_amp else self.compute_dtype
        self.asr_model.to(self.cast_dtype)

    def get_blank_id(self) -> int:
        """
        Returns id of the blank token.
        Returns:
            (int) blank id for the model.
        """
        blank_id = len(self.asr_model.joint.vocabulary)
        return blank_id

    def get_vocabulary(self) -> list[str]:
        """
        Returns the list of vocabulary tokens.
        Returns:
            (list[str]) list of vocabulary tokens.
        """
        return self.asr_model.joint.vocabulary

    def get_subsampling_factor(self) -> int:
        """
        Returns the subsampling factor for the ASR encoder.
        Returns:
            (int) subsampling factor for the ASR encoder model.
        """
        return self.asr_model.encoder.subsampling_factor

    def encode(
        self, processed_signal: Tensor, processed_signal_length: Tensor, prompt_vectors: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Get encoder output from the model. It is used for streaming inference.
        Args:
            processed_signal: (Tensor) processed signal. Shape is torch.Size([B, C, T]).
            processed_signal_length: (Tensor) processed signal length. Shape is torch.Size([B]).
            prompt_vectors: (Tensor | None) Optional prompt vectors for multilingual models.
                Shape can be torch.Size([B, num_prompts]) or torch.Size([B, T_enc, num_prompts]) if already expanded.
        Returns:
            (tuple[Tensor, Tensor]) encoder output and encoder output length of shape torch.Size([B, T, D]), torch.Size([B]).
        """
        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)

        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)

        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):

            # Prepare model arguments
            model_args = {
                'processed_signal': processed_signal.to(self.cast_dtype),
                'processed_signal_length': processed_signal_length,
            }
            if prompt_vectors is not None:
                model_args['prompt'] = prompt_vectors

            forward_outs = self.asr_model(**model_args)

        encoded, encoded_len = forward_outs
        return encoded, encoded_len

    def decode(self, encoded: Tensor, encoded_len: Tensor, partial_hypotheses: list) -> list:
        """
        Decode the encoder output using the RNNT decoder.
        Args:
            encoded: (Tensor) encoder output.
            encoded_len: (Tensor) encoder output length.
            partial_hypotheses: (list) list of partial hypotheses for stateful decoding.
        Returns:
            (list) list of best hypotheses.
        """
        best_hyp = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded.to(self.cast_dtype), encoded_len, return_hypotheses=True, partial_hypotheses=partial_hypotheses
        )
        return best_hyp

    def encode_with_prompts(
        self, processed_signal: Tensor, processed_signal_length: Tensor, prompt_vectors: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Convenience wrapper for prompt-enabled encoding.
        Expands prompt vectors across the time dimension before calling encode.
        Args:
            processed_signal: (Tensor) processed signal. Shape is torch.Size([B, C, T]).
            processed_signal_length: (Tensor) processed signal length. Shape is torch.Size([B]).
            prompt_vectors: (Tensor) prompt vectors. Shape is torch.Size([B, num_prompts]).
        Returns:
            (tuple[Tensor, Tensor]) encoder output and encoder output length.
        """
        encoder_time_steps = processed_signal.shape[2] // self.get_subsampling_factor()
        # Expand prompts: [B, num_prompts] -> [B, T_enc, num_prompts]
        prompt_vectors = prompt_vectors.unsqueeze(1).expand(-1, encoder_time_steps, -1)
        return self.encode(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            prompt_vectors=prompt_vectors,
        )
