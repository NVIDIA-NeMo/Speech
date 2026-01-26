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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.model_wrappers.cache_aware_ctc_inference_wrapper import (
    CacheAwareCTCInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_ctc_decoder import CTCGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_ctc_endpointing import CTCGreedyEndpointing
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.cache_aware_ctc_state import CacheAwareCTCStreamingState
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    drop_trailing_features,
    get_confidence_utils,
    normalize_log_probs,
)

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator


class CacheAwareCTCPipeline(BasePipeline):
    """Cache Aware CTC pipeline."""

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: CacheAwareCTCInferenceWrapper,
        itn_model: AlignmentPreservingInverseNormalizer | None = None,
        nmt_model: LLMTranslator | None = None,
    ):
        """
        Initialize the CacheAwareCTCPipeline.
        Args:
            cfg: (DictConfig) Configuration parameters.
            asr_model: (CacheAwareCTCInferenceWrapper) ASR model.
            itn_model: (AlignmentPreservingInverseNormalizer | None) Inverse Text Normalization model.
        """
        self.copy_asr_model_attributes(asr_model)
        self.init_parameters(cfg)
        self.init_context_manager()
        self.init_bufferer_for_cache_aware_streaming()
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)
        self.init_bpe_decoder()
        self.init_greedy_ctc_decoder()
        self.init_endpointer()
        self.init_text_processor(cfg, itn_model)
        self.init_nmt_model(nmt_model)
        super().__init__()

    def init_parameters(self, cfg: DictConfig) -> None:
        """
        Initialize the configuration parameters.
        Args:
            cfg: (DictConfig) Configuration parameters.
        """
        if cfg.streaming.att_context_size is not None:
            self.asr_model.set_default_att_context_size(att_context_size=cfg.streaming.att_context_size)
        self.sample_rate = cfg.streaming.sample_rate
        self.asr_output_granularity = cfg.asr_output_granularity

        self.use_cache = cfg.streaming.use_cache
        self.use_feat_cache = cfg.streaming.use_feat_cache
        self.batch_size = cfg.streaming.batch_size
        self.num_slots = cfg.streaming.num_slots
        if self.num_slots < self.batch_size:
            raise ValueError(
                f"Number of slots in the context manager must be >= batch_size: {self.num_slots} < {self.batch_size}"
            )
        self.request_type = RequestType.from_str(cfg.streaming.request_type)
        self.word_boundary_tolerance = cfg.streaming.word_boundary_tolerance
        self.stop_history_eou_in_milliseconds = cfg.endpointing.stop_history_eou
        self.residue_tokens_at_end = cfg.endpointing.residue_tokens_at_end
        self.return_tail_result = cfg.return_tail_result

        self.pre_encode_cache_size = self.asr_model.get_pre_encode_cache_size()
        self.model_chunk_size = self.asr_model.get_chunk_size()
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]

        if cfg.streaming.get("chunk_size_in_secs", None) is not None:
            self.chunk_size_in_secs = cfg.streaming.chunk_size_in_secs
            self.tokens_per_frame = math.ceil(
                np.trunc(self.chunk_size_in_secs / self.window_stride) / self.subsampling_factor
            )
            # overwrite the encoder streaming params with proper shift size for cache aware streaming
            self.asr_model.setup_streaming_params(
                chunk_size=self.model_chunk_size // self.subsampling_factor, shift_size=self.tokens_per_frame
            )
        else:
            self.chunk_size_in_secs = self.model_chunk_size * self.window_stride
            self.tokens_per_frame = math.ceil(self.model_chunk_size / self.subsampling_factor)

        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * self.window_stride

        model_chunk_size_in_secs = self.model_chunk_size * self.window_stride

        if self.use_cache:
            # if using cache, we need to pad some samples for pre_encode
            self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs
            self.drop_left_context = None
            self.valid_out_len = None
        else:
            # if not using cache, we need to keep left context in buffer, but no extra padding in pre_encode
            left_context_size = self.asr_model.get_att_context_size()[0]
            if left_context_size < 0:
                raise ValueError(f"Left context size should not be a negative value: {left_context_size}")
            self.buffer_size_in_secs = (
                model_chunk_size_in_secs + left_context_size * self.subsampling_factor * self.window_stride
            )
            self.drop_left_context = left_context_size
            self.valid_out_len = self.tokens_per_frame

        # Expected feature buffer length for trimming (safeguard for feature buffer inputs)
        self.expected_feature_buffer_len = int(self.buffer_size_in_secs / self.window_stride)

    def init_greedy_ctc_decoder(self) -> None:
        """Initialize the CTC decoder."""
        check_existance_of_required_attributes(self, ['vocabulary', 'conf_func'])
        self.greedy_ctc_decoder = CTCGreedyDecoder(vocabulary=self.vocabulary, conf_func=self.conf_func)

    def init_endpointer(self) -> None:
        """Initialize the endpointer."""
        check_existance_of_required_attributes(
            self,
            [
                'vocabulary',
                'model_stride_in_milliseconds',
                'stop_history_eou_in_milliseconds',
                'residue_tokens_at_end',
            ],
        )

        self.endpointer = CTCGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milliseconds,
            stop_history_eou=self.stop_history_eou_in_milliseconds,
            residue_tokens_at_end=self.residue_tokens_at_end,
        )

    def create_state(self, options: ASRRequestOptions) -> CacheAwareCTCStreamingState:
        """
        Create new empty state.
        Args:
            options: (ASRRequestOptions) Request options for particular stream.
        Returns:
            (CacheAwareCTCStreamingState) New empty state.
        """
        state = CacheAwareCTCStreamingState()
        state.set_global_offset(0)
        new_options = options.augment_with_defaults(
            default_enable_itn=self.text_processor.is_itn_enabled(),
            default_enable_pnc=self.text_processor.is_pnc_enabled(),
            default_enable_nmt=self.nmt_enabled,
            default_source_language=self.nmt_model.source_language if self.nmt_enabled else None,
            default_target_language=self.nmt_model.target_language if self.nmt_enabled else None,
            default_stop_history_eou=self.stop_history_eou_in_milliseconds,
            default_asr_output_granularity=self.asr_output_granularity,
        )

        eou_label_buffer_size = 0
        if new_options.stop_history_eou > 0:
            eou_label_buffer_size = millisecond_to_frames(
                new_options.stop_history_eou, math.ceil(self.model_stride_in_milliseconds)
            )
            eou_label_buffer_size += self.residue_tokens_at_end
        state.setup_label_buffer(eou_label_buffer_size, self.blank_id)
        state.set_options(new_options)
        return state

    def get_sep(self) -> str:
        """Return the separator for the text processor."""
        return self.sep

    def preprocess(self, buffers: list[Tensor], right_paddings: list[int] | None = None) -> tuple[Tensor, Tensor]:
        """
        Preprocess the feature buffers by stacking them and computing the lengths
        Args:
            buffers: (list[Tensor]) List of feature buffers.
            right_paddings: (list[int] | None) List of right paddings.
        Returns:
            (tuple[Tensor, Tensor]) Processed feature buffers and their lengths.
        """
        feature_buffers = [f_buffer.unsqueeze_(0) for f_buffer in buffers]
        # Trim to expected feature buffer length (safeguard for external feature buffer inputs)
        feature_buffers = [
            drop_trailing_features(f_buffer, self.expected_feature_buffer_len) for f_buffer in feature_buffers
        ]
        feature_buffer_lens = torch.tensor([f_buffer.shape[2] for f_buffer in feature_buffers], device=self.device)
        if right_paddings is not None:
            right_paddings = torch.tensor(right_paddings, device=feature_buffer_lens.device)
            feature_buffer_lens = feature_buffer_lens - right_paddings
        feature_buffers = torch.cat(feature_buffers).to(self.device)
        return feature_buffers, feature_buffer_lens

    def run_greedy_decoder(self, state: CacheAwareCTCStreamingState, request: Request, log_probs: Tensor):
        """
        Run the greedy CTC decoder on the log_probs and update the state
        Args:
            state: (CacheAwareCTCStreamingState) The state of the stream
            request: (Request) The current request (frame or feature buffer)
            log_probs: (Tensor) The log probabilities of the current request
        Returns:
            (bool) Whether EOU is detected.
        """
        eou_detected = request.is_last
        last_token = state.label_buffer[-1] if len(state.label_buffer) > 0 else self.blank_id
        cur_output = self.greedy_ctc_decoder(log_probs, compute_confidence=True, previous=last_token)
        state.update_label_buffer(cur_output["labels"])

        if not eou_detected:
            emissions = state.get_label_buffer()
            pivot_point = len(emissions) - 1
            eou_detected, _ = self.endpointer.detect_eou_near_pivot(
                emissions, pivot_point, stop_history_eou=state.options.stop_history_eou
            )

        state.update_state(cur_output, eou_detected=eou_detected)
        state.increment_global_offset(self.tokens_per_frame)
        return eou_detected

    def decode_log_probs(
        self,
        requests: list[Request],
        log_probs: Tensor,
        tail_log_probs: Tensor | None,
        ready_state_ids: set,
    ) -> None:
        """
        Decode the log probabilities and update the state
        Args:
            requests: (list[Request]) List of requests (frames or feature buffers) to transcribe.
            log_probs: (Tensor) Log probabilities.
            tail_log_probs: (Tensor | None) Tail log probabilities.
            ready_state_ids: (set) Set of ready state IDs.
        """

        for idx, request in enumerate(requests):
            state = self.get_state(request.stream_id)
            eou_detected = self.run_greedy_decoder(state, request, log_probs[idx])

            if eou_detected:
                self.bpe_decoder.decode_bpe_tokens(state)
                state.cleanup_after_eou()
                ready_state_ids.add(request.stream_id)

            if tail_log_probs is not None:
                last_token = state.label_buffer[-1] if len(state.label_buffer) > 0 else self.blank_id
                tail_output = self.greedy_ctc_decoder(
                    tail_log_probs[idx], compute_confidence=False, previous=last_token
                )
                state.set_incomplete_segment_tokens(tail_output["tokens"])

    def cache_aware_transcribe_step(
        self,
        requests: list[Request],
        buffered_features: list[Tensor],
        right_paddings: list[int] | None,
        ready_state_ids: set,
        keep_all_outputs: bool = False,
    ) -> None:
        """
        Cache Aware Transcribe Step
        It receives a list of requests (Frame or FeatureBuffer) and features and do the following:

        1. Preprocess the features by stacking them and computing the lengths
        2. Get the context and mapping from the context manager for cache aware streaming
        3. Perform a streaming step with the ASR model
        4. Update the cache and reset the cache slots for the streams that has ended
        5. Decode the log probabilities and update the state

        Args:
            requests: (list[Request]) List of requests (frames or feature buffers) to transcribe.
            buffered_features: (list[Tensor]) List of buffered features.
            right_paddings: (list[int] | None) List of right paddings.
            ready_state_ids: (set) Set of ready state IDs.
            keep_all_outputs: (bool) Whether to keep all outputs or not.
        """
        feature_buffers, feature_buffer_lens = self.preprocess(buffered_features, right_paddings)

        stream_ids = [request.stream_id for request in requests]
        eos_flags = [request.is_last for request in requests]
        context, mapping = self.context_manager.get_context(stream_ids)

        drop_extra_pre_encoded = 0 if not self.use_cache else self.asr_model.drop_extra_pre_encoded
        log_probs, tail_log_probs, new_context = self.asr_model.stream_step(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            context=context,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            keep_all_outputs=keep_all_outputs,
            drop_left_context=self.drop_left_context,
            valid_out_len=self.valid_out_len,
            return_tail_result=self.return_tail_result,
        )

        if log_probs is not None:
            log_probs = normalize_log_probs(log_probs)
        self.context_manager.update_cache(stream_ids, new_context, mapping)
        self.context_manager.reset_slots(stream_ids, eos_flags)
        self.decode_log_probs(requests, log_probs, tail_log_probs, ready_state_ids)

    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """
        Transcribes the frames in a streaming manner.
        After detecting EOU, it updates the state and run text processor.
        If there are multiple streams, it waits until all states are ready to run text processor.
        Args:
            frames: (list[Frame]) List of frames to transcribe.
        """
        all_fbuffers, right_paddings = self.bufferer.update(frames)

        ready_state_ids = set()
        if len(all_fbuffers) > 0:
            nonfinal_frames, nonfinal_fbuffers = [], []
            final_frames, final_fbuffers = [], []
            final_right_paddings = []
            for jdx, bfeature in enumerate(all_fbuffers):
                frame = frames[jdx]
                if frame.is_last:
                    final_frames.append(frame)
                    final_fbuffers.append(bfeature)
                    final_right_paddings.append(right_paddings[jdx])
                else:
                    nonfinal_frames.append(frame)
                    nonfinal_fbuffers.append(bfeature)

            if len(nonfinal_frames) > 0:
                self.cache_aware_transcribe_step(
                    nonfinal_frames, nonfinal_fbuffers, None, ready_state_ids, keep_all_outputs=False
                )
            if len(final_frames) > 0:
                self.cache_aware_transcribe_step(
                    final_frames, final_fbuffers, final_right_paddings, ready_state_ids, keep_all_outputs=True
                )

        # Postprocess the ready states
        if len(ready_state_ids) > 0:
            self.text_processor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
            ready_state_ids.clear()

        self.update_partial_transcript(frames, self.tokenizer, self.leading_regex_pattern)

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """
        Transcribes the feature buffers in a streaming manner.
        After detecting EOU, it updates the state and run text processor.
        If there are multiple streams, it waits until all states are ready to run text processor.
        Args:
            fbuffers: (list[FeatureBuffer]) List of feature buffers to transcribe.
        """
        ready_state_ids = set()

        final_fbuffers, final_features = [], []
        nonfinal_fbuffers, nonfinal_features = [], []
        final_right_paddings = []

        for fbuffer in fbuffers:
            feature = fbuffer.features
            right_padding = max(0, self.expected_feature_buffer_len - fbuffer.valid_size)

            if fbuffer.is_last:
                final_fbuffers.append(fbuffer)
                final_features.append(feature)
                final_right_paddings.append(right_padding)
            else:
                nonfinal_fbuffers.append(fbuffer)
                nonfinal_features.append(feature)

        if len(nonfinal_fbuffers) > 0:
            self.cache_aware_transcribe_step(
                nonfinal_fbuffers, nonfinal_features, None, ready_state_ids, keep_all_outputs=False
            )

        if len(final_fbuffers) > 0:
            self.cache_aware_transcribe_step(
                final_fbuffers, final_features, final_right_paddings, ready_state_ids, keep_all_outputs=True
            )

        if len(ready_state_ids) > 0:
            self.text_processor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
            ready_state_ids.clear()

        self.update_partial_transcript(fbuffers, self.tokenizer, self.leading_regex_pattern)

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """
        Initialize the request generator.
        Returns:
            (ContinuousBatchedRequestStreamer) Request generator.
        """
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size_in_secs,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            request_type=self.request_type,
            preprocessor=self.preprocessor,
            buffer_size_in_secs=self.buffer_size_in_secs,
            device=self.device,
            pad_last_frame=True,
        )
        return request_generator
