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
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel


class CTCInferenceWrapper(ASRInferenceWrapper):
    """
    Provides a unified interface to work with CTC/Hybrid-CTC models.
    """

    def __post_init__(self) -> None:
        """
        Additional post initialization step
        Checks if the model is a ctc model and sets the decoding strategy to ctc.
        """
        if not isinstance(self.asr_model, (EncDecCTCModel, EncDecHybridRNNTCTCModel)):
            raise ValueError(
                "Provided model is not a CTC type. You are trying to use a CTC transcriber with a non-CTC model."
            )

        decoder_type = 'ctc'
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
        if isinstance(self.asr_model, EncDecCTCModel):
            blank_id = len(self.asr_model.decoder.vocabulary)
        else:
            blank_id = len(self.asr_model.ctc_decoder.vocabulary)
        return blank_id

    def get_vocabulary(self) -> list[str]:
        """
        Returns the list of vocabulary tokens.
        Returns:
            (list[str]) list of vocabulary tokens.
        """
        if isinstance(self.asr_model, EncDecCTCModel):
            return self.asr_model.decoder.vocabulary
        else:
            return self.asr_model.ctc_decoder.vocabulary

    def get_subsampling_factor(self) -> int:
        """
        Returns the subsampling factor for the ASR encoder.
        Returns:
            (int) subsampling factor for the ASR encoder model.
        """
        return self.asr_model.encoder.subsampling_factor

    def get_logprobs(self, processed_signal: Tensor, processed_signal_length: Tensor) -> Tensor:
        """
        Get log probabilities from the model. It is used for streaming inference.
        Args:
            processed_signal: (Tensor) processed signal. Shape is torch.Size([B, C, T]).
            processed_signal_length: (Tensor) processed signal length. Shape is torch.Size([B]).
        Returns:
            (Tensor) log probabilities. Shape is torch.Size([B, T, V+1]).
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

            forward_outs = self.asr_model(
                processed_signal=processed_signal.to(self.cast_dtype), processed_signal_length=processed_signal_length
            )

        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            encoded, encoded_len = forward_outs
            log_probs = self.asr_model.ctc_decoder(encoder_output=encoded.clone())
        else:
            log_probs, encoded_len, predictions = forward_outs
        return log_probs
