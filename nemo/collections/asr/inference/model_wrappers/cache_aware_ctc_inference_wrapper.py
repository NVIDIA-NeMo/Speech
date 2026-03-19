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

from nemo.collections.asr.inference.model_wrappers.cache_aware_asr_inference_wrapper import (
    CacheAwareASRInferenceWrapper,
)
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder


class CacheAwareCTCInferenceWrapper(CacheAwareASRInferenceWrapper):
    """
    Provides a unified interface to work with Cache-Aware CTC models.
    """

    def __post_init__(self) -> None:
        """
        Additional post initialization step
        Checks if the model is a ctc model and sets the decoding strategy to ctc.
        """

        if not isinstance(self.asr_model, (EncDecCTCModel, EncDecHybridRNNTCTCModel)):
            raise ValueError(
                "Provided model is not a CTC type. You are trying to use a CTC Inference with a non-CTC model."
            )

        if not isinstance(self.asr_model.encoder, StreamingEncoder):
            raise NotImplementedError("Encoder of this model does not support streaming!")

        decoder_type = 'ctc'
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.cur_decoder = decoder_type

        # reset the decoding strategy
        self.reset_decoding_strategy(decoder_type)
        self.set_decoding_strategy(decoder_type)

        # setup streaming parameters
        if self.asr_model.encoder.streaming_cfg is None:
            self.asr_model.encoder.setup_streaming_params()

        self.drop_extra_pre_encoded = self.get_drop_extra_pre_encoded()

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

    def execute_step(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        context: CacheAwareContext,
        drop_extra_pre_encoded: int | None,
        keep_all_outputs: bool,
        drop_left_context: int | None = None,
        valid_out_len: int | None = None,
        return_tail_result: bool = False,
    ) -> tuple[Tensor, Tensor | None, CacheAwareContext]:
        """
        Executes a single streaming step.
        Args:
            processed_signal: (Tensor) input signal tensor.
            processed_signal_length: (Tensor) input signal length tensor.
            context: (CacheAwareContext) context object.
            drop_extra_pre_encoded: (int | None) number of extra pre-encoded frames to drop.
            keep_all_outputs: (bool) whether to keep all outputs or not.
            drop_left_context: (int | None) number of left context frames to drop.
            valid_out_len: (int | None) number of valid output frames.
            return_tail_result: (bool) whether to return tail result or not.
        Returns:
            (tuple[Tensor, Tensor | None, CacheAwareContext]) log probabilities, tail log probabilities and new context.
        """

        (
            encoded,
            encoded_len,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        ) = self.asr_model.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=context.cache_last_channel,
            cache_last_time=context.cache_last_time,
            cache_last_channel_len=context.cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )

        if drop_left_context:
            # drop left context
            encoded = encoded[:, :, drop_left_context:]

        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            all_log_probs = self.asr_model.ctc_decoder(encoder_output=encoded)
        else:
            all_log_probs = self.asr_model.decoder(encoder_output=encoded)

        tail_log_probs = None
        if valid_out_len and not keep_all_outputs:
            # drop right context if any
            log_probs = all_log_probs[:, :valid_out_len, :]
            if return_tail_result:
                tail_log_probs = all_log_probs[:, valid_out_len:, :]
        else:
            log_probs = all_log_probs

        # create a new context
        new_context = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        return log_probs, tail_log_probs, new_context

    def stream_step(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        context: CacheAwareContext = None,
        drop_extra_pre_encoded: int | None = None,
        keep_all_outputs: bool = False,
        drop_left_context: int | None = None,
        valid_out_len: int | None = None,
        return_tail_result: bool = False,
    ) -> tuple[Tensor, Tensor | None, CacheAwareContext]:
        """
        Executes a single streaming step.
        Args:
            processed_signal: (Tensor) input signal tensor.
            processed_signal_length: (Tensor) input signal length tensor.
            context: (CacheAwareContext) context object.
            drop_extra_pre_encoded: (int | None) number of extra pre-encoded frames to drop.
            keep_all_outputs: (bool) whether to keep all outputs or not.
            drop_left_context: (int | None) number of left context frames to drop.
            valid_out_len: (int | None) number of valid output frames.
            return_tail_result: (bool) whether to return tail result or not.
        Returns:
            (tuple[Tensor, Tensor | None, CacheAwareContext]) log probabilities, tail log probabilities and new context.
        """

        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)

        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)

        if context is None:
            # create a dummy context
            context = CacheAwareContext()

        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):

            log_probs, tail_log_probs, new_context = self.execute_step(
                processed_signal,
                processed_signal_length,
                context,
                drop_extra_pre_encoded,
                keep_all_outputs,
                drop_left_context,
                valid_out_len,
                return_tail_result,
            )
        return log_probs, tail_log_probs, new_context
