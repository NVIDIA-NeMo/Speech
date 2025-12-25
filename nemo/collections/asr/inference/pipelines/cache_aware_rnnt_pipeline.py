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

from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
    CacheAwareRNNTInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import RNNTGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_rnnt_endpointing import RNNTGreedyEndpointing
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTStreamingState
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    get_confidence_utils,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator


class CacheAwareRNNTPipeline(BasePipeline):
    """Cache Aware RNNT pipeline."""

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: CacheAwareRNNTInferenceWrapper,
        itn_model: AlignmentPreservingInverseNormalizer | None = None,
        nmt_model: LLMTranslator | None = None,
    ):
        """
        Initialize the CacheAwareRNNTPipeline.
        Args:
            cfg: (DictConfig) Configuration parameters.
            asr_model: (CacheAwareRNNTInferenceWrapper) ASR model.
            itn_model: (AlignmentPreservingInverseNormalizer | None) Inverse Text Normalization model.
            nmt_model: (LLMTranslator | None) LLM based translation model.
        """
        self.copy_asr_model_attributes(asr_model)
        self.init_prompt_support()
        self.init_parameters(cfg)
        self.init_context_manager()
        self.init_bufferer_for_cache_aware_streaming()
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)
        self.init_bpe_decoder()
        self.init_greedy_rnnt_decoder()
        self.init_endpointer()
        self.init_text_processor(cfg, itn_model)
        self.init_nmt_model(nmt_model)
        super().__init__()

    def init_parameters(self, cfg: DictConfig) -> None:
        """
        Initialize the parameters.
        Args:
            cfg: (DictConfig) Configuration parameters.
        """
        if cfg.streaming.att_context_size is not None:
            self.asr_model.set_default_att_context_size(att_context_size=cfg.streaming.att_context_size)

        self.sample_rate = cfg.streaming.sample_rate
        self.asr_output_granularity = cfg.asr_output_granularity
        self.pre_encode_cache_size = self.asr_model.get_pre_encode_cache_size()
        self.model_chunk_size = self.asr_model.get_chunk_size()
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]

        self.use_cache = cfg.streaming.use_cache
        self.use_feat_cache = cfg.streaming.use_feat_cache

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

        # Context Manager
        self.batch_size = cfg.streaming.batch_size
        self.num_slots = cfg.streaming.num_slots
        if self.num_slots < self.batch_size:
            raise ValueError(
                f"Number of slots in the context manager must be >= batch_size: {self.num_slots} < {self.batch_size}"
            )
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

        self.stop_history_eou_in_milliseconds = cfg.endpointing.stop_history_eou
        self.residue_tokens_at_end = cfg.endpointing.residue_tokens_at_end
        self.word_boundary_tolerance = cfg.streaming.word_boundary_tolerance
        self.return_tail_result = cfg.return_tail_result

        self.request_type = RequestType.from_str(cfg.streaming.request_type)
        if self.request_type is not RequestType.FRAME:
            raise ValueError(f"Request type {self.request_type} is not supported for cache-aware streaming.")

    def init_greedy_rnnt_decoder(self) -> None:
        """Initialize the RNNT decoder."""
        check_existance_of_required_attributes(self, ['vocabulary', 'conf_func'])
        self.greedy_rnnt_decoder = RNNTGreedyDecoder(vocabulary=self.vocabulary, conf_func=self.conf_func)

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

        self.endpointer = RNNTGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milliseconds,
            stop_history_eou=self.stop_history_eou_in_milliseconds,
            residue_tokens_at_end=self.residue_tokens_at_end,
        )

    def create_state(self, options: ASRRequestOptions) -> CacheAwareRNNTStreamingState:
        """
        Create new empty state.
        Args:
            options: (ASRRequestOptions) Request options for particular stream.
        Returns:
            (CacheAwareRNNTStreamingState) New empty state.
        """
        state = CacheAwareRNNTStreamingState()
        state.set_global_offset(0)
        new_options = options.augment_with_defaults(
            default_enable_itn=self.text_processor.is_itn_enabled(),
            default_enable_pnc=self.text_processor.is_pnc_enabled(),
            default_enable_nmt=self.nmt_enabled,
            default_source_language=self.nmt_model.source_language if self.nmt_enabled else None,
            default_target_language=self.nmt_model.target_language if self.nmt_enabled else None,
            default_stop_history_eou=self.stop_history_eou_in_milliseconds,
            default_asr_output_granularity=self.asr_output_granularity,
            default_language_code="en-US" if self.prompt_enabled else None,
        )

        eou_label_buffer_size = 0
        if new_options.stop_history_eou > 0:
            eou_label_buffer_size = millisecond_to_frames(
                new_options.stop_history_eou, math.ceil(self.model_stride_in_milliseconds)
            )
            eou_label_buffer_size += self.residue_tokens_at_end
        state.setup_label_buffer(eou_label_buffer_size, self.blank_id)
        state.set_previous_hypothesis(None)
        state.set_options(new_options)

        # Create per-stream prompt index for prompt-enabled models
        if self.prompt_enabled:
            lang_code = getattr(new_options, "language_code", None)
            if not isinstance(lang_code, str) or len(lang_code) == 0:
                raise ValueError("Prompt-enabled model requires a valid language_code in request options.")
            prompt_idx = self._resolve_prompt_index(lang_code)
            state.set_prompt_index(prompt_idx)

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
        feature_buffer_lens = torch.tensor([f_buffer.shape[2] for f_buffer in feature_buffers], device=self.device)
        if right_paddings is not None:
            right_paddings = torch.tensor(right_paddings, device=feature_buffer_lens.device)
            feature_buffer_lens = feature_buffer_lens - right_paddings
        feature_buffers = torch.cat(feature_buffers).to(self.device)
        return feature_buffers, feature_buffer_lens

    def run_greedy_decoder(self, state: CacheAwareRNNTStreamingState, frame: Frame, hyp: Hypothesis) -> bool:
        """
        Run the greedy RNNT decoder on the hypothesis and update the state
        Args:
            state: (CacheAwareRNNTStreamingState) The state of the stream
            frame: (Frame) The current frame
            hyp: (Hypothesis) The hypothesis of the current frame
        Returns:
            (bool) Whether EOU is detected.
        """
        eou_detected = frame.is_last
        cur_output, cur_labels, new_offset = self.greedy_rnnt_decoder(
            global_timestamps=hyp.timestamp,
            tokens=hyp.y_sequence,
            length=self.tokens_per_frame,
            offset=state.offset,
        )
        state.set_offset(new_offset)

        # cur labels contains blank tokens as well, it is needed for EOU detection
        state.update_label_buffer(cur_labels)

        if not eou_detected:
            emissions = state.get_label_buffer()
            pivot_point = len(emissions) - 1
            eou_detected, _ = self.endpointer.detect_eou_near_pivot(
                emissions, pivot_point, stop_history_eou=state.options.stop_history_eou
            )

        state.update_state(cur_output, eou_detected=eou_detected)
        return eou_detected

    def cache_aware_transcribe_step(
        self,
        frames: list[Frame],
        features: list[Tensor],
        right_paddings: list[int],
        ready_state_ids: set,
        keep_all_outputs: bool = False,
    ) -> None:
        """
        Cache Aware Transcribe Step
        It receives a list of frames and features and do the following:

        1. Preprocess the features by stacking them and computing the lengths
        2. Collecting previous hypotheses for stateful decoding
        3. Get the context and mapping from the context manager for cache aware streaming
        4. Perform a streaming step with the ASR model
        5. Update the cache and reset the cache slots for the streams that has ended
        6. Update the previous hypothesis and reset the previous hypothesis for the streams that has ended
        7. Perform greedy RNNT decoding to get the best hypothesis and update the states
        8. Update the ready states to indicate that the state is ready for text post-processing
        Args:
            frames: (list[Frame]) List of frames to transcribe.
            features: (list[Tensor]) List of feature buffers.
            right_paddings: (list[int] | None) List of right paddings.
            ready_state_ids: (set) Set of ready state IDs.
            keep_all_outputs: (bool) Whether to keep all outputs or not.
        """

        feature_buffers, feature_buffer_lens = self.preprocess(features, right_paddings)
        states, stream_ids, eos_flags = [], [], []
        for frame in frames:
            states.append(self.get_state(frame.stream_id))
            stream_ids.append(frame.stream_id)
            eos_flags.append(frame.is_last)

        previous_hypotheses = [state.get_previous_hypothesis() for state in states]
        context, mapping = self.context_manager.get_context(stream_ids)

        prompt_vectors = None
        if self.prompt_enabled:
            prompt_vectors = self._build_prompt_vectors(states)

        drop_extra_pre_encoded = 0 if not self.use_cache else self.asr_model.drop_extra_pre_encoded
        best_hyp, new_context = self.asr_model.stream_step(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            context=context,
            previous_hypotheses=previous_hypotheses,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            keep_all_outputs=keep_all_outputs,
            drop_left_context=self.drop_left_context,
            valid_out_len=self.valid_out_len,
            prompt_vectors=prompt_vectors,
        )

        # update the cache and reset the cache slots for the streams that has ended
        self.context_manager.update_cache(stream_ids, new_context, mapping)
        self.context_manager.reset_slots(stream_ids, eos_flags)

        # update the previous hypothesis and reset the previous hypothesis for the streams that has ended
        for state, hyp, eos in zip(states, best_hyp, eos_flags):
            if eos:
                state.reset_previous_hypothesis()
            else:
                state.set_previous_hypothesis(hyp)

        # run greedy decoder for each frame-state-hypothesis tuple
        for frame, state, hyp in zip(frames, states, best_hyp):
            eou_detected = self.run_greedy_decoder(state, frame, hyp)
            if eou_detected:
                self.bpe_decoder.decode_bpe_tokens(state)
                state.cleanup_after_eou()
                ready_state_ids.add(frame.stream_id)

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        raise NotImplementedError("Feature buffer type is not supported for cache aware streaming.")

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

        # streams that contains multiple frames
        if len(all_fbuffers) > 0:
            final_frames, final_fbuffers = [], []
            nonfinal_frames, nonfinal_fbuffers = [], []
            final_right_paddings = []
            for jdx, bfeature in enumerate(all_fbuffers):
                bframe = frames[jdx]

                if bframe.is_last:
                    final_frames.append(bframe)
                    final_fbuffers.append(bfeature)
                    final_right_paddings.append(right_paddings[jdx])
                else:
                    nonfinal_frames.append(bframe)
                    nonfinal_fbuffers.append(bfeature)

            if len(nonfinal_frames) > 0:
                self.cache_aware_transcribe_step(
                    nonfinal_frames, nonfinal_fbuffers, None, ready_state_ids, keep_all_outputs=False
                )

            if len(final_frames) > 0:
                self.cache_aware_transcribe_step(
                    final_frames, final_fbuffers, final_right_paddings, ready_state_ids, keep_all_outputs=True
                )

        # post-process the ready states
        if len(ready_state_ids) > 0:
            self.text_processor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
            ready_state_ids.clear()

        self.update_partial_transcript(frames, self.tokenizer, self.leading_regex_pattern)

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """
        Initialize the request generator.
        Returns:
            (ContinuousBatchedRequestStreamer) Request generator.
        """
        # for cache aware streaming we need to process one frame at a time -> n_frames_per_stream=1
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size_in_secs,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            request_type=self.request_type,
            preprocessor=None,
            buffer_size_in_secs=None,
            device=None,
            pad_last_frame=True,
        )
        return request_generator
