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
import os
import re
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import ClippedRNNTGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_rnnt_endpointing import RNNTGreedyEndpointing
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.rnnt_state import RNNTStreamingState
from nemo.collections.asr.inference.utils.enums import FeatureBufferPaddingMode, RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    adjust_vad_segments,
    check_existance_of_required_attributes,
    drop_trailing_features,
    get_confidence_utils,
    normalize_features,
    update_punctuation_and_language_tokens_timestamps,
)
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    BatchedBeamHyps,
    batched_beam_hyps_to_hypotheses,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis as NemoHypothesis
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator


class BufferedRNNTPipeline(BasePipeline):
    """Buffered RNN-T/TDT pipeline."""

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: RNNTInferenceWrapper,
        itn_model: AlignmentPreservingInverseNormalizer | None = None,
        nmt_model: LLMTranslator | None = None,
    ):
        """
        Initialize the BufferedRNNTPipeline.
        Args:
            cfg: (DictConfig) Configuration parameters.
            asr_model: (RNNTInferenceWrapper) ASR model.
            itn_model: (AlignmentPreservingInverseNormalizer | None) Inverse Text Normalization model.
            nmt_model: (LLMTranslator | None) LLM based translation model.
        """

        self.copy_asr_model_attributes(asr_model)
        self.init_prompt_support()
        self.init_parameters(cfg)
        self.init_bufferer_for_buffered_streaming()
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)
        self.init_endpointer()
        self.init_greedy_rnnt_decoder()
        self.init_bpe_decoder()
        self.init_decoding_computer()
        self.init_text_processor(cfg, itn_model)
        self.init_nmt_model(nmt_model)
        super().__init__()

    def init_parameters(self, cfg: DictConfig) -> None:
        """
        Initialize the configuration parameters.
        Args:
            cfg: (DictConfig) Configuration parameters.
        """
        self.asr_output_granularity = cfg.asr_output_granularity
        self.sample_rate = cfg.streaming.sample_rate
        self.stateful = cfg.streaming.stateful
        self.stateless = not self.stateful
        self.batch_size = cfg.streaming.batch_size

        self.chunk_size = cfg.streaming.chunk_size
        self.left_padding_size = cfg.streaming.left_padding_size
        self.right_padding_size = cfg.streaming.right_padding_size
        self.buffer_size_in_secs = self.chunk_size + self.left_padding_size + self.right_padding_size
        self.expected_feature_buffer_len = int(self.buffer_size_in_secs / self.window_stride)

        self.mid_delay = math.ceil((self.chunk_size + self.right_padding_size) / self.model_stride_in_secs)
        self.tokens_per_frame_float = self.chunk_size / self.model_stride_in_secs
        self.tokens_per_left_padding_float = self.left_padding_size / self.model_stride_in_secs
        self.tokens_per_right_padding_float = self.right_padding_size / self.model_stride_in_secs
        self.tokens_per_frame = math.ceil(self.tokens_per_frame_float)
        self.tokens_per_left_padding = math.ceil(self.tokens_per_left_padding_float)
        self.tokens_per_right_padding = math.ceil(self.tokens_per_right_padding_float)

        if self.stateful:
            self.initial_delay = self.right_padding_size / self.model_stride_in_secs
        else:
            self.initial_delay = (self.left_padding_size + self.right_padding_size) / self.model_stride_in_secs

        if self.stateful and (
            abs(self.tokens_per_frame_float - self.tokens_per_frame) > 1e-5
            or abs(self.tokens_per_left_padding_float - self.tokens_per_left_padding) > 1e-5
            or abs(self.tokens_per_right_padding_float - self.tokens_per_right_padding) > 1e-5
        ):
            self.tokens_per_frame_float = self.tokens_per_frame
            self.tokens_per_left_padding_float = self.tokens_per_left_padding
            self.left_padding_size = self.tokens_per_left_padding * self.model_stride_in_secs
            self.chunk_size = self.tokens_per_frame * self.model_stride_in_secs
            self.right_padding_size = self.tokens_per_right_padding * self.model_stride_in_secs
            self.buffer_size_in_secs = self.chunk_size + self.left_padding_size + self.right_padding_size

        self.request_type = RequestType.from_str(cfg.streaming.request_type)
        self.padding_mode = FeatureBufferPaddingMode.from_str(cfg.streaming.padding_mode)
        self.right_padding = self.padding_mode is FeatureBufferPaddingMode.RIGHT
        self.stop_history_eou_in_milliseconds = cfg.endpointing.stop_history_eou
        self.residue_tokens_at_end = cfg.endpointing.residue_tokens_at_end
        self.word_boundary_tolerance = cfg.streaming.word_boundary_tolerance
        self.return_tail_result = cfg.return_tail_result
        self.tokens_to_move = self.punctuation_ids.union(self.language_token_ids)

        # Beam-search collapse strategy. ``False`` (default): commit the chunk's top
        # beam at every chunk boundary. ``True``: keep ``beam_size`` divergent beams in
        # the carried state across all chunks of an utterance and only collapse them at
        # end-of-utterance. Published transcripts use the per-chunk argmax beam in both
        # modes; no effect for greedy.
        self.collapse_on_eou_only = bool(cfg.streaming.get("collapse_on_eou_only", False))

        self.zero_encoded = self.init_zero_enc() if self.right_padding else None

    def init_endpointer(self) -> None:
        """Initialize the endpointer."""
        check_existance_of_required_attributes(
            self,
            [
                'stateful',
                'chunk_size',
                'right_padding_size',
                'buffer_size_in_secs',
                'vocabulary',
                'model_stride_in_milliseconds',
                'stop_history_eou_in_milliseconds',
                'residue_tokens_at_end',
            ],
        )

        if self.stateful:
            effective_buffer_size_in_secs = self.chunk_size + self.right_padding_size
        else:
            effective_buffer_size_in_secs = self.buffer_size_in_secs

        self.endpointer = RNNTGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milliseconds,
            effective_buffer_size_in_secs=effective_buffer_size_in_secs,
            stop_history_eou=self.stop_history_eou_in_milliseconds,
            residue_tokens_at_end=self.residue_tokens_at_end,
        )

    def init_greedy_rnnt_decoder(self) -> None:
        """Initialize the greedy RNNT decoder."""
        check_existance_of_required_attributes(self, ['vocabulary', 'conf_func', 'endpointer', 'tokens_per_frame'])
        self.greedy_rnnt_decoder = ClippedRNNTGreedyDecoder(
            vocabulary=self.vocabulary,
            conf_func=self.conf_func,
            endpointer=self.endpointer,
            tokens_per_frame=self.tokens_per_frame,
        )

    def init_decoding_computer(self) -> None:
        """Initialize the decoding computer."""
        check_existance_of_required_attributes(self, ['stateful', 'asr_model'])
        self.decoding_computer = None
        if self.stateful:
            self.decoding_computer = self.asr_model.asr_model.decoding.decoding.decoding_computer

    def init_zero_enc(self) -> Tensor:
        """
        Initialize the encoder output for the zero buffer.
        Returns:
            (Tensor) Encoder output for the zero buffer.
        """
        check_existance_of_required_attributes(
            self, ['buffer_size_in_secs', 'sample_rate', 'device', 'expected_feature_buffer_len']
        )
        buffer_size_in_samples = int(self.buffer_size_in_secs * self.sample_rate)
        zero_buffer = torch.zeros(1, buffer_size_in_samples, device=self.device)
        zero_features, zero_features_len = self.preprocess(
            buffers=zero_buffer,
            buffer_lens=torch.tensor([zero_buffer.shape[1]], device=self.device),
            expected_feature_buffer_len=self.expected_feature_buffer_len,
        )

        if self.prompt_enabled:
            # Use "en-US" as the default prompt for zero encoding
            # This region is sliced out before decoding, so language choice doesn't matter
            default_prompt_idx = self._resolve_prompt_index("en-US")
            prompt_indices = torch.tensor([default_prompt_idx], device=self.device, dtype=torch.long)
            prompt_vector = self._create_one_hot_prompts(prompt_indices)  # [1, num_prompts]

            zero_encoded, _ = self.asr_model.encode_with_prompts(
                processed_signal=zero_features,
                processed_signal_length=zero_features_len,
                prompt_vectors=prompt_vector,
            )
        else:
            zero_encoded, _ = self.asr_model.encode(
                processed_signal=zero_features, processed_signal_length=zero_features_len
            )

        return zero_encoded[0]

    def create_state(self, options: ASRRequestOptions) -> RNNTStreamingState:
        """
        Create new empty state.
        Args:
            options: (ASRRequestOptions) Request options for particular stream.
        Returns:
            (RNNTStreamingState) New empty state.
        """
        state = RNNTStreamingState()
        state.set_global_offset(-self.initial_delay)
        new_options = options.fill_defaults(
            default_enable_itn=self.text_processor.itn_enabled,
            default_enable_nmt=self.nmt_enabled,
            default_source_language=self.nmt_model.source_language if self.nmt_enabled else None,
            default_target_language=self.nmt_model.target_language if self.nmt_enabled else None,
            default_stop_history_eou=self.stop_history_eou_in_milliseconds,
            default_asr_output_granularity=self.asr_output_granularity,
            default_language_code="en-US" if self.prompt_enabled else None,
        )
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

    def preprocess(
        self, buffers: Tensor, buffer_lens: Tensor, expected_feature_buffer_len: int
    ) -> tuple[Tensor, Tensor]:
        """
        Preprocess the buffered frames and extract features.
        Args:
            buffers: (Tensor) Audio buffers.
            buffer_lens: (Tensor) Lengths of the audio buffers.
            expected_feature_buffer_len: (int) Expected length of the feature buffers.
        Returns:
            (tuple[Tensor, Tensor]) Processed feature buffers and their lengths.
        """
        feature_buffers, feature_buffer_lens = self.preprocessor(input_signal=buffers, length=buffer_lens)
        feature_buffers = drop_trailing_features(feature_buffers, expected_feature_buffer_len)
        feature_buffers = normalize_features(feature_buffers, feature_buffer_lens)
        feature_buffer_lens = feature_buffer_lens.clamp(max=feature_buffers.shape[2])
        return feature_buffers, feature_buffer_lens

    def get_cut_off_range(self, T: int, is_last: bool) -> tuple[int, int]:
        """
        Compute the start and end indices to clip.
        Args:
            T: (int) Time dimension of the alignment.
            is_last: (bool) Whether the last frame is reached.
        Returns:
            (tuple[int, int]) Start and end indices to clip.
        """
        start = max(T - 1 - self.mid_delay, 0)
        end = T if is_last else min(start + self.tokens_per_frame, T)
        return start, end

    def encode_raw_signals(
        self, frames: list[Frame], raw_signals: list[Tensor], left_paddings: list[int]
    ) -> tuple[Tensor, Tensor]:
        """
        Run Encoder part on the audio buffers.
        Args:
            frames: (list[Frame]) Frames to transcribe.
            raw_signals: (list[Tensor]) Audio buffers.
            left_paddings: (list[int]) Left paddings for audio buffers.
        Returns:
            (tuple[Tensor, Tensor]) Encoded signals and their lengths.
        """

        if self.right_padding:
            left_paddings = torch.tensor(left_paddings, dtype=torch.int64, device=self.device)

        buffers = []
        for i in range(len(raw_signals)):
            buffer = raw_signals[i]
            if self.right_padding:
                # Roll the buffered frames to the left by the left padding
                # This is done to avoid the padding at the beginning of the buffered frames
                # which can cause the performance degradation
                lpad = left_paddings[i].item()
                if lpad > 0:
                    buffer = buffer.roll(shifts=-lpad)
            buffers.append(buffer.unsqueeze_(0))

        # Only final frames have right padding
        # Calculate right paddings
        right_paddings = torch.tensor([frame.size - frame.valid_size for frame in frames], device=self.device).clamp(
            min=0
        )

        # Create and adjust the buffer lens
        buffer_lens = torch.tensor([buffers[0].size(1)] * len(buffers), device=self.device)
        buffer_lens = buffer_lens - right_paddings
        if self.right_padding:
            buffer_lens = buffer_lens - left_paddings

        feature_buffers, feature_buffer_lens = self.preprocess(
            buffers=torch.cat(buffers).to(self.device),
            buffer_lens=buffer_lens,
            expected_feature_buffer_len=self.expected_feature_buffer_len,
        )

        # Build prompt vectors if prompts are enabled
        if self.prompt_enabled:
            requests_states = [self.get_state(f.stream_id) for f in frames]
            prompt_vectors = self._build_prompt_vectors(requests_states)

            # Use encode_with_prompts which handles dimension expansion
            encoded, encoded_len = self.asr_model.encode_with_prompts(
                processed_signal=feature_buffers,
                processed_signal_length=feature_buffer_lens,
                prompt_vectors=prompt_vectors,
            )
        else:
            encoded, encoded_len = self.asr_model.encode(
                processed_signal=feature_buffers, processed_signal_length=feature_buffer_lens
            )
        encoded = encoded.clone()
        encoded_len = encoded_len.clone()

        # Roll back the encoded signals to the right
        if self.right_padding:
            for i in range(encoded.shape[0]):
                lpad = left_paddings[i]
                if lpad > 0:
                    lpad = int(lpad / self.sample_rate / self.model_stride_in_secs)
                    encoded[i] = encoded[i].roll(lpad, dims=1)
                    encoded[i][:, :lpad] = self.zero_encoded[:, :lpad]
                    encoded_len[i] = encoded_len[i] + lpad

        return encoded, encoded_len

    def encode_processed_signals(
        self, fbuffers: list[FeatureBuffer], processed_signals: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        Run Encoder part on the feature buffers.
        Args:
            fbuffers: (list[FeatureBuffer]) Feature buffers.
            processed_signals: (list[Tensor]) Processed buffers.
        Returns:
            (tuple[Tensor, Tensor]) Encoder output and their lengths.
        """

        processed_signals = torch.cat([sig.unsqueeze_(0) for sig in processed_signals]).to(self.device)
        processed_signals = drop_trailing_features(processed_signals, self.expected_feature_buffer_len)
        processed_signal_lengths = torch.tensor([f.valid_size for f in fbuffers], device=self.device)
        processed_signals = normalize_features(processed_signals, processed_signal_lengths)
        processed_signal_lengths = processed_signal_lengths.clamp(max=processed_signals.shape[2])

        # Build prompt vectors if prompts are enabled
        if self.prompt_enabled:
            requests_states = [self.get_state(f.stream_id) for f in fbuffers]
            prompt_vectors = self._build_prompt_vectors(requests_states)

            # Use encode_with_prompts which handles dimension expansion
            encoded, encoded_len = self.asr_model.encode_with_prompts(
                processed_signal=processed_signals,
                processed_signal_length=processed_signal_lengths,
                prompt_vectors=prompt_vectors,
            )
        else:
            encoded, encoded_len = self.asr_model.encode(
                processed_signal=processed_signals, processed_signal_length=processed_signal_lengths
            )
        encoded = encoded.clone()
        encoded_len = encoded_len.clone()

        if self.right_padding:
            for i in range(encoded.shape[0]):
                lpad = int(fbuffers[i].roll_size / self.subsampling_factor)
                if lpad > 0:
                    encoded[i] = encoded[i].roll(lpad, dims=1)
                    encoded[i][:, :lpad] = self.zero_encoded[:, :lpad]
                    encoded_len[i] = encoded_len[i] + lpad
        return encoded, encoded_len

    def encode_frames(self, frames: list[Frame]) -> tuple[Tensor, Tensor]:
        """
        Encode the frames using the Encoder part of the ASR model.
        Args:
            frames: (list[Frame]) Frames to transcribe.
        Returns:
            (tuple[Tensor, Tensor]) Encoder output and their lengths.
        """
        raw_signals, left_paddings = self.bufferer.update(frames)
        encs, enc_lens = None, None
        if len(raw_signals) > 0:
            encs, enc_lens = self.encode_raw_signals(frames, raw_signals, left_paddings)
        return encs, enc_lens

    def encode_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> tuple[Tensor, Tensor]:
        """
        Encode the feature buffers using the Encoder part of the ASR model.
        Args:
            fbuffers: (list[FeatureBuffer]) Feature buffers to transcribe.
        Returns:
            (tuple[Tensor, Tensor]) Encoder output and their lengths.
        """
        processed_signals = self.bufferer.update(fbuffers)
        encs, enc_lens = None, None
        if len(processed_signals) > 0:
            encs, enc_lens = self.encode_processed_signals(fbuffers, processed_signals)
        return encs, enc_lens

    def run_greedy_decoder(
        self,
        state: RNNTStreamingState,
        request: Request,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        start: int,
        end: int,
        alignment_length: int,
        timestamp_offset: int = 0,
        vad_segments: torch.Tensor = None,
    ) -> bool:
        """
        Greedy RNN-T decoder.
        Args:
            state: (RNNTStreamingState) Current state for the particular stream.
            request: (Request) Current request for the particular stream.
            timesteps: (Tensor) Timesteps.
            tokens: (Tensor) Tokens.
            start: (int) Start index.
            end: (int) End index.
            alignment_length: (int) Length of the alignment.
            timestamp_offset: (int) Timestamp offset.
            vad_segments: (Tensor) VAD segments.
        Returns:
            (bool) Whether EOU is detected.
        """
        if self.stateful and vad_segments is not None:
            vad_segments = adjust_vad_segments(vad_segments, self.left_padding_size)

        clipped_output, tail_output, eou_detected, start_idx, end_idx = self.greedy_rnnt_decoder(
            global_timesteps=timesteps,
            tokens=tokens,
            alignment_length=alignment_length,
            clip_start=start,
            clip_end=end,
            is_last=request.is_last,
            is_start=request.is_first,
            return_tail_result=self.return_tail_result,
            state_start_idx=state.decoder_start_idx,
            state_end_idx=state.decoder_end_idx,
            timestamp_offset=timestamp_offset,
            vad_segments=vad_segments,
            stop_history_eou=state.options.stop_history_eou,
        )
        state.update_state(clipped_output, eou_detected)
        state.update_from_decoder_results(start_idx, end_idx)
        if self.stateless:
            # For stateless mode, we need to set the last token, it will be used for filtering duplicate token
            state.set_last_token(clipped_output["last_token"], clipped_output["last_token_idx"])
            # For stateless mode, we need to increment the global offset
            state.increment_global_offset(self.tokens_per_frame_float)
        state.set_incomplete_segment_tokens(tail_output["tokens"])
        return eou_detected

    def stateless_transcribe_step(
        self, requests: list[Request], encs: Tensor, enc_lens: Tensor, ready_state_ids: set
    ) -> None:
        """
        Stateless transcribe step.
        Stateless assumes that we don't keep track of partial hypotheses (partial_hypotheses=None).
        Args:
            requests: (list[Request]) List of requests to transcribe.
            encs: (Tensor) Encoder output.
            enc_lens: (Tensor) Encoder output lengths.
            ready_state_ids: (set) Set of ready state IDs.
        """
        states = [self.get_state(request.stream_id) for request in requests]
        best_hyp = self.asr_model.decode(encs, enc_lens, partial_hypotheses=None)
        # For stateless mode, use zero timestamp offsets since we don't track timestamps
        ready_states = self.decode_step(best_hyp, requests, states)
        ready_state_ids.update(ready_states)

    def stateful_transcribe_step(
        self, requests: list[Request], encs: Tensor, enc_lens_chunk: Tensor, enc_lens: Tensor, ready_state_ids: set
    ) -> None:
        """
        Stateful transcribe step.
        Stateful assumes that we keep track of partial hypotheses.
        Args:
            requests: (list[Request]) List of requests to transcribe.
            encs: (Tensor) Encoder output.
            enc_lens_chunk: (Tensor) Encoder output lengths for the chunk.
            enc_lens: (Tensor) Encoder output lengths.
            ready_state_ids: (set) Set of ready state IDs.
        """
        states = [self.get_state(request.stream_id) for request in requests]
        partial_hypotheses, rnnt_states = [], []
        all_rnnt_states_are_none = True
        all_multi_biasing_models_empty = True
        multi_biasing_ids = np.full([len(states)], fill_value=-1)
        for i, state in enumerate(states):
            hyp_state = state.hyp_decoding_state
            rnnt_states.append(hyp_state)
            if hyp_state is not None:
                all_rnnt_states_are_none = False
            if state.has_biasing_request():
                if state.options.biasing_cfg.multi_model_id is not None:
                    all_multi_biasing_models_empty = False
                    multi_biasing_ids[i] = state.options.biasing_cfg.multi_model_id
                elif state.options.biasing_cfg.auto_manage_multi_model:
                    state.options.biasing_cfg.add_to_multi_model(
                        tokenizer=self.asr_model.tokenizer,
                        biasing_multi_model=self.decoding_computer.biasing_multi_model,
                    )
                    multi_biasing_ids[i] = state.options.biasing_cfg.multi_model_id
                    all_multi_biasing_models_empty = False
                else:
                    logging.warning("Biasing request is not empty, not auto managed and not compiled. Skipping")
            if hyp_state is not None or state.has_biasing_request():
                partial_hypotheses.append(
                    NemoHypothesis(
                        score=0.0,
                        y_sequence=torch.zeros([0], dtype=torch.long),
                        dec_state=hyp_state,
                        biasing_cfg=state.options.biasing_cfg,
                    )
                )
            else:
                partial_hypotheses.append(None)

        batched_rnnt_states = None
        if not all_rnnt_states_are_none:
            batched_rnnt_states = self.decoding_computer.merge_to_batched_state(rnnt_states)

        self._dbg_log_carry_in(batched_rnnt_states, requests)
        # Snapshot the carry-in (scores, lengths) BEFORE the decoder mutates it in place.
        # Used by ``_dbg_log_chunk_delta`` to show per-beam token/score increments per chunk.
        carry_snapshot_lens, carry_snapshot_scores = (None, None)
        if self._dbg_enabled() and batched_rnnt_states is not None and batched_rnnt_states.batched_hyps is not None:
            carry_snapshot_lens = batched_rnnt_states.batched_hyps.current_lengths_nb.cpu().tolist()
            carry_snapshot_scores = batched_rnnt_states.batched_hyps.scores.cpu().tolist()

        if all_multi_biasing_models_empty:
            multi_biasing_ids = None
        else:
            multi_biasing_ids = torch.from_numpy(multi_biasing_ids).to(device=enc_lens_chunk.device)

        encs_dim_last = encs.transpose(1, 2)

        def _to_hypotheses(out):
            if isinstance(out, BatchedBeamHyps):
                return batched_beam_hyps_to_hypotheses(out, batch_size=enc_lens.shape[0])
            return batched_hyps_to_hypotheses(out, batch_size=enc_lens.shape[0])

        # decode chunk
        with torch.inference_mode(), torch.no_grad():
            best_batched_hyps_chunk, _, batched_state = self.decoding_computer(
                encs_dim_last,
                enc_lens_chunk,
                batched_rnnt_states,
                multi_biasing_ids=multi_biasing_ids,
            )

            is_beam = isinstance(best_batched_hyps_chunk, BatchedBeamHyps)

            self._dbg_log_beams("post-decode (pre-collapse)", best_batched_hyps_chunk, requests)
            # Delta vs the carry-in: how many tokens were added this chunk per beam
            # and how much cumulative score moved. Reveals whether all K survivors
            # gained tokens (real diversity) or only the empty-prefix survivors did
            # (silence-K-carry contamination).
            self._dbg_log_chunk_delta(carry_snapshot_lens, carry_snapshot_scores, best_batched_hyps_chunk, requests)

            # In on_eou mode, snapshot the K beams' chunk-local tokens, chunk-local
            # delta scores, and chunk-local delta lengths BEFORE the publish collapse.
            # Used at EOU detection (inside ``decode_step``) to rerank the K beams by
            # chunk-local length-normalised score and replace just the chunk-N portion
            # of ``state.tokens`` with the norm-winner's chunk-N tokens. For all
            # non-EOU chunks the standard raw-argmax winner publishes unchanged.
            if is_beam and self.collapse_on_eou_only:
                self._snapshot_eou_rerank_candidates(best_batched_hyps_chunk, batched_rnnt_states, states)

            # Collapse ``batched_state`` to the per-chunk raw-argmax beam. Drives the
            # published transcript, seeds the RC pass, and is what gets carried to the
            # next chunk. Same code path used by per_chunk and on_eou modes - the only
            # difference is that on_eou mode may later override the chunk-N portion of
            # ``state.tokens`` inside ``decode_step`` when EOU is flagged. No-op for
            # greedy.
            if is_beam:
                beam_indices = best_batched_hyps_chunk.scores.argmax(dim=-1)
                self.decoding_computer.collapse_batched_state_to_beams_(
                    batched_state, best_batched_hyps_chunk, beam_indices
                )

        best_hyps = _to_hypotheses(best_batched_hyps_chunk)

        # Carry forward the collapsed (single-winner) state, both for per_chunk and
        # on_eou modes. on_eou mode no longer keeps a K-diverse cross-chunk carry:
        # the K beams only live within the chunk that produced them, the snapshot
        # above is consumed inside ``decode_step`` if EOU fires, and discarded
        # otherwise (the next chunk starts from a single committed prefix).
        carry_items = self.decoding_computer.split_batched_state(batched_state)
        for state, rnnt_state in zip(states, carry_items):
            state.hyp_decoding_state = rnnt_state

        if self.tokens_per_right_padding > 0:
            # decode right context
            _, max_time, feat_dim = encs_dim_last.shape
            device = encs.device
            # we are indexing `encs_dim_last` with `shift_indices` to get a tensor where right context is at the start
            # everything after right context is padded with `0` index (first encoder vector)
            # padding will be ignored by decoder_computer since we pass the lengths
            shift_indices = torch.arange(max_time, device=device, dtype=torch.long)[None, :] + enc_lens_chunk[:, None]
            # pad with zeros everything beyond needed context
            shift_indices = torch.where(shift_indices < max_time, shift_indices, torch.zeros_like(shift_indices))
            with torch.inference_mode(), torch.no_grad():
                best_batched_hyps_rc, _, _ = self.decoding_computer(
                    torch.gather(encs_dim_last, dim=1, index=shift_indices[:, :, None].expand(-1, -1, feat_dim)),
                    enc_lens - enc_lens_chunk,
                    batched_state,
                    multi_biasing_ids=multi_biasing_ids,
                )
                # Collapse the right-context beam too, so the merged published transcript
                # and timestamps come from a single hypothesis per stream. Right-context
                # state is not carried forward (it's a lookahead-only decode), so we only
                # need to collapse the prefix tree itself.
                if isinstance(best_batched_hyps_rc, BatchedBeamHyps):
                    rc_beam_indices = best_batched_hyps_rc.scores.argmax(dim=-1)
                    best_batched_hyps_rc.keep_beam_(rc_beam_indices)
                best_hyps_rc = _to_hypotheses(best_batched_hyps_rc)
            # merge right context to chunk hypothesis
            for hyp, hyp_rc in zip(best_hyps, best_hyps_rc):
                hyp.merge_(hyp_rc)

        ready_states = self.decode_step(best_hyps, requests, states)
        # ``_collapse_eou_state`` is no longer needed: ``stateful_transcribe_step``
        # always carries the collapsed (single-winner) state, so the next
        # utterance already starts conditioned on a single committed prefix.
        for curr_state in states:
            curr_state.timestamp_offset += self.tokens_per_frame_float
        ready_state_ids.update(ready_states)

        for request, state in zip(requests, states):
            # only the first request contains biasing options; biasing options for the stream are stored in state
            if request.is_last and state.has_biasing_request():
                if state.options.biasing_cfg.auto_manage_multi_model:
                    state.options.biasing_cfg.remove_from_multi_model(
                        biasing_multi_model=self.decoding_computer.biasing_multi_model
                    )

    def decode_step(self, best_hyp: list, requests: list[Request], states: list[RNNTStreamingState]) -> set:
        """
        Perform greedy RNNT decoding to get the best hypothesis and update the state.
        If EOU is detected, push the words to the state and cleanup the state.
        Args:
            best_hyp: (list) Best hypothesis.
            requests: (list[Request]) List of requests to transcribe.
            states: (list[RNNTStreamingState]) List of states.
        Returns:
            (set) Set of ready state IDs.
        """
        ready_state_ids = set()
        for idx, hyp in enumerate(best_hyp):
            state = states[idx]
            request = requests[idx]
            # Perform timestamp based decoding for the hypothesis
            if self.stateful:
                alignment_length = self.tokens_per_right_padding + self.tokens_per_frame
            else:
                if self.request_type is RequestType.FEATURE_BUFFER:
                    alignment_length = math.ceil(request.size / self.subsampling_factor)
                else:  # RequestType.FRAME
                    alignment_length = math.ceil(self.expected_feature_buffer_len / self.subsampling_factor)

            if self.stateful:
                start, end = 0, self.tokens_per_frame
            else:
                # For stateless mode
                if request.is_first and request.is_last:
                    start, end = 0, alignment_length
                else:
                    start, end = self.get_cut_off_range(alignment_length, request.is_last)

            timestamp = hyp.timestamp
            tokens = hyp.y_sequence
            timestamp = torch.tensor(timestamp) if isinstance(timestamp, list) else timestamp
            tokens = torch.tensor(tokens) if isinstance(tokens, list) else tokens
            timestamp = update_punctuation_and_language_tokens_timestamps(
                tokens, timestamp, self.tokens_to_move, self.underscore_id
            )
            vad_segments = request.vad_segments
            # DEBUG: log what came in (hyp) before the time-clip and overlap-dedup.
            if self._dbg_enabled():
                try:
                    hyp_text = self.asr_model.tokenizer.ids_to_text([int(t) for t in tokens.tolist() if int(t) >= 0])
                except Exception:
                    hyp_text = "<decode failed>"
                self._dbg(
                    f"decode_step IN  sid={request.stream_id} is_first={request.is_first} "
                    f"is_last={request.is_last} hyp_len={len(tokens)} clip=[{start},{end}] "
                    f"state.dec_start={state.decoder_start_idx} state.dec_end={state.decoder_end_idx} "
                    f"timestamp_offset={state.timestamp_offset:.1f} hyp_text={hyp_text!r}"
                )
                # Full state snapshot BEFORE the greedy decoder mutates it, so we can see
                # what tokens the current-utterance buffer has accumulated from prior chunks.
                self._dbg_log_state("state IN ", state, request)

            # Snapshot the cumulative-buffer length before the greedy decoder appends
            # this chunk's raw-argmax tokens. Used by ``_maybe_rerank_eou_transcript``
            # to (a) detect "this is the first chunk of the current utterance"
            # via ``prev_tokens_len == 0`` (``state.tokens`` is cleared by
            # ``cleanup_after_eou`` after every EOU commit, and starts empty for
            # the stream's very first utterance) and (b) overwrite *just* the
            # just-appended chunk-N segment of ``state.tokens`` when the
            # length-norm rerank picks a non-raw winner among the K beams of
            # the EOU chunk.
            prev_tokens_len = len(state.tokens)

            eou_detected = self.run_greedy_decoder(
                state=state,
                request=request,
                timesteps=timestamp,
                tokens=tokens,
                start=start,
                end=end,
                alignment_length=alignment_length,
                timestamp_offset=state.timestamp_offset,
                vad_segments=vad_segments,
            )

            self._dbg_log_state(
                "decode_step OUT",
                state,
                request,
                extras={"eou": eou_detected, "dec_end": state.decoder_end_idx},
            )

            if eou_detected:
                self._dbg(f"EOU sid={request.stream_id} committing state.tokens to transcript")
                # In ``collapse_on_eou_only`` mode, replace the chunk-N portion of
                # ``state.tokens`` (just appended by ``run_greedy_decoder``) with the
                # chunk-local length-normalised winner among the K beams of the EOU
                # chunk. No-op when no EOU snapshot is available (greedy / per_chunk
                # mode, or stream without a beam search step this chunk).
                self._maybe_rerank_eou_transcript(state, request, prev_tokens_len)
                # State right before the BPE decoder consumes ``state.tokens`` and we then
                # clear the per-utterance buffer in ``cleanup_after_eou``. Lets us verify
                # that nothing meaningful is being silently discarded by the reset.
                self._dbg_log_state("state EOU pre-cleanup ", state, request)
                self.bpe_decoder.decode_bpe_tokens(state)
                state.cleanup_after_eou()
                # Confirms ``tokens`` / ``timesteps`` were cleared and that the carried
                # bookkeeping (``decoder_*_idx``, ``global_offset``, ``timestamp_offset``,
                # ``last_token*``) is preserved as intended for the next utterance.
                self._dbg_log_state("state EOU post-cleanup", state, request)
                ready_state_ids.add(request.stream_id)
        return ready_state_ids

    # ---------------------------------------------------------------------
    # Debug helpers.
    # ---------------------------------------------------------------------
    # Env vars:
    #   NEMO_STREAMING_DBG=1                       enable debug logging
    #   NEMO_STREAMING_DBG_FILE=/path/to/log       write to this file (line-buffered)
    #                                              instead of stderr; greatly reduces IDE
    #                                              terminal hangs on large runs
    #   NEMO_STREAMING_DBG_SIDS=0,6,12             only emit lines containing ``sid=K``
    #                                              where K is in this comma-separated set
    #                                              (lines without a ``sid=`` token are always
    #                                              emitted, e.g. carry-in headers)
    # Implementation notes:
    #   The file handle is cached on the class; we reopen only if the path changes.
    #   We deliberately do NOT pass flush=True per call - that was the main reason the
    #   IDE terminal lagged. Line-buffered file IO already flushes per newline; stderr
    #   in an IDE is line-buffered too. The OS may buffer writes if the process aborts
    #   abnormally, but a normal exit will flush.
    _dbg_file = None
    _dbg_file_path = None
    _dbg_sids_cache: tuple[str, frozenset[int]] | None = None

    @staticmethod
    def _dbg_enabled() -> bool:
        return os.environ.get("NEMO_STREAMING_DBG", "0") == "1"

    @classmethod
    def _dbg_stream(cls):
        path = os.environ.get("NEMO_STREAMING_DBG_FILE")
        if not path:
            return sys.stderr
        if cls._dbg_file_path != path:
            if cls._dbg_file is not None:
                try:
                    cls._dbg_file.close()
                except Exception:
                    pass
            cls._dbg_file = open(path, "a", buffering=1)
            cls._dbg_file_path = path
        return cls._dbg_file

    @classmethod
    def _dbg_sid_filter(cls) -> frozenset[int] | None:
        raw = os.environ.get("NEMO_STREAMING_DBG_SIDS", "")
        if not raw:
            return None
        if cls._dbg_sids_cache is None or cls._dbg_sids_cache[0] != raw:
            try:
                sids = frozenset(int(x) for x in raw.split(",") if x.strip())
            except ValueError:
                sids = frozenset()
            cls._dbg_sids_cache = (raw, sids)
        return cls._dbg_sids_cache[1]

    @staticmethod
    def _dbg(msg: str) -> None:
        cls = BufferedRNNTPipeline
        if not cls._dbg_enabled():
            return
        sids = cls._dbg_sid_filter()
        if sids is not None:
            m = re.search(r"sid=(\d+)", msg)
            if m is not None and int(m.group(1)) not in sids:
                return
        print(f"[DBG] {msg}", file=cls._dbg_stream())

    def _dbg_log_beams(self, label: str, batched_hyps, requests: list[Request]) -> None:
        """Dump K beams' text + scores per stream. No-op if not in beam mode or debug off.

        ``transcript_wb`` is a prefix tree, not a flat per-beam token array; the real
        tokens for beam ``k`` must be reconstructed by walking ``transcript_wb_prev_ptr``
        from the leaf. We do that the cheap way by cloning the ``BatchedBeamHyps`` and
        calling :meth:`to_nbest_hyps_list`, which performs the walk via
        :meth:`flatten_sort_`. Cloning is essential: that method is in-place.
        """
        if not self._dbg_enabled():
            return
        from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps

        if not isinstance(batched_hyps, BatchedBeamHyps):
            return

        cloned = batched_hyps.clone()
        nbest_list = cloned.to_nbest_hyps_list(score_norm=False)
        K = batched_hyps.beam_size
        for b in range(batched_hyps.batch_size):
            sid = requests[b].stream_id if b < len(requests) else b
            raw_scores = batched_hyps.scores[b].tolist()
            lens_nb = batched_hyps.current_lengths_nb[b].tolist()
            argmax = int(batched_hyps.scores[b].argmax().item())
            self._dbg(
                f"{label} sid={sid} argmax_beam={argmax} "
                f"raw_scores={[f'{s:.3f}' for s in raw_scores]} len_nb={lens_nb}"
            )
            # ``nbest_list[b].n_best_hypotheses`` is sorted by ``to_nbest_hyps_list`` (raw
            # since ``score_norm=False``), so its order matches ``raw_scores`` after sort.
            sorted_idx = sorted(range(K), key=lambda k: -raw_scores[k])
            for rank, hyp in enumerate(nbest_list[b].n_best_hypotheses):
                k = sorted_idx[rank] if rank < len(sorted_idx) else rank
                try:
                    ids = [int(t) for t in hyp.y_sequence.tolist() if int(t) >= 0]
                    txt = self.asr_model.tokenizer.ids_to_text(ids) if ids else ""
                except Exception as e:
                    txt = f"<decode-err {type(e).__name__}>"
                win = " <-WIN(raw)" if k == argmax else ""
                self._dbg(
                    f"  {label}   beam[{k}]: score={raw_scores[k]:.3f} len_nb={lens_nb[k]} "
                    f"text={txt!r}{win} sid={sid}"
                )

    def _dbg_log_carry_in(self, batched_state, requests: list[Request]) -> None:
        """At chunk start, dump the K beams of the carry-in state for each stream.

        Shows, per stream, what the K diverse hypothesis seeds look like before this
        chunk's MALSD search runs. Lets us see whether the carry actually preserves
        meaningful alternatives or is mostly silence-induced near-duplicates.
        """
        if not self._dbg_enabled():
            return
        if batched_state is None or batched_state.batched_hyps is None:
            self._dbg("carry-in: <no carry> (fresh streams - K identical SOS seeds)")
            return
        from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps

        bh = batched_state.batched_hyps
        if not isinstance(bh, BatchedBeamHyps):
            return
        cloned = bh.clone()
        nbest_list = cloned.to_nbest_hyps_list(score_norm=False)
        K = bh.beam_size
        for b in range(bh.batch_size):
            sid = requests[b].stream_id if b < len(requests) else b
            raw_scores = bh.scores[b].tolist()
            lens_nb = bh.current_lengths_nb[b].tolist()
            self._dbg(f"carry-in sid={sid} raw_scores={[f'{s:.3f}' for s in raw_scores]} len_nb={lens_nb}")
            sorted_idx = sorted(range(K), key=lambda k: -raw_scores[k])
            for rank, hyp in enumerate(nbest_list[b].n_best_hypotheses):
                k = sorted_idx[rank] if rank < len(sorted_idx) else rank
                try:
                    ids = [int(t) for t in hyp.y_sequence.tolist() if int(t) >= 0]
                    txt = self.asr_model.tokenizer.ids_to_text(ids) if ids else ""
                except Exception as e:
                    txt = f"<decode-err {type(e).__name__}>"
                self._dbg(f"  carry-in   beam[{k}]: score={raw_scores[k]:.3f} text={txt!r} sid={sid}")

    def _dbg_log_chunk_delta(self, before_lens, before_scores, batched_hyps_after, requests: list[Request]) -> None:
        """Show per-beam (delta_len_nb, delta_score) added in this chunk.

        ``before_lens`` / ``before_scores`` are snapshots taken from the carry-in
        state just before the decoder ran (MALSD mutates the carry in place, so a
        live reference would already be the after-state). ``batched_hyps_after`` is
        the K post-decode beams. Beam indices are NOT directly comparable (MALSD
        permutes slots inside the chunk) - the deltas are *per beam slot*, so a
        slot that "gained nothing" usually means no chunk-N parent's lineage
        survived in that slot. The aggregate of ``delta_len_nb`` across the K slots
        tells us how spread out this chunk's new tokens were across beam slots.
        """
        if not self._dbg_enabled():
            return
        from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps

        if not isinstance(batched_hyps_after, BatchedBeamHyps):
            return
        after_lens = batched_hyps_after.current_lengths_nb.tolist()
        after_scores = batched_hyps_after.scores.tolist()
        if before_lens is None:
            # Fresh streams: pretend pre-state was all zeros / 0.0.
            B, K = batched_hyps_after.batch_size, batched_hyps_after.beam_size
            before_lens = [[0] * K for _ in range(B)]
            before_scores = [[0.0] * K for _ in range(B)]
        for b in range(batched_hyps_after.batch_size):
            sid = requests[b].stream_id if b < len(requests) else b
            d_lens = [a - bf for a, bf in zip(after_lens[b], before_lens[b])]
            d_scores = [a - bf if bf != float("-inf") else None for a, bf in zip(after_scores[b], before_scores[b])]
            unique_lens = len(set(d_lens))
            self._dbg(
                f"chunk-delta sid={sid} d_len_nb={d_lens} (unique={unique_lens}) "
                f"d_score={[f'{s:.3f}' if s is not None else 'n/a' for s in d_scores]}"
            )

    def _dbg_log_state(
        self, label: str, state: RNNTStreamingState, request: Request, extras: dict | None = None
    ) -> None:
        """Dump streaming state's accumulated tokens / text and bookkeeping indices.

        ``state.tokens`` is the *current-utterance* buffer (cleared on EOU by
        :meth:`StreamingState.cleanup_after_eou`). This helper renders both the
        buffer (as text and as a tail slice of timesteps) and the per-state
        bookkeeping (decoder index window, global timestamp offset, last-token
        de-dup info, EOU latch) so we can see exactly what survives the EOU reset
        and what does not.
        """
        if not self._dbg_enabled():
            return
        toks = list(state.tokens) if state.tokens is not None else []
        text = ""
        try:
            ids = [int(t) for t in toks if int(t) >= 0]
            if ids:
                text = self.asr_model.tokenizer.ids_to_text(ids)
        except Exception:
            text = "<decode failed>"
        ts = list(state.timesteps) if state.timesteps is not None else []
        extra_str = ""
        if extras:
            extra_str = " " + " ".join(f"{k}={v}" for k, v in extras.items())
        self._dbg(
            f"{label} sid={request.stream_id} chunk_is_last={request.is_last} "
            f"n_tokens={len(toks)} text={text!r} tokens_tail={toks[-8:]} timesteps_tail={ts[-6:]} "
            f"dec=[{state.decoder_start_idx},{state.decoder_end_idx}] "
            f"global_offset={state.global_offset:.1f} ts_offset={state.timestamp_offset:.1f} "
            f"last_token={state.last_token} last_tok_idx={state.last_token_idx} "
            f"eou_before={state.eou_detected_before}"
            f"{extra_str}"
        )

    def _snapshot_eou_rerank_candidates(
        self,
        batched_hyps,
        batched_rnnt_states,
        states: list[RNNTStreamingState],
    ) -> None:
        """Snapshot the K beams' chunk-local hypotheses for use by
        ``_maybe_rerank_eou_transcript`` if EOU later fires for this chunk.

        Each beam ends up as a small dict on ``state._eou_rerank``:

        - ``tokens``     : chunk-local non-blank token ids
        - ``timesteps``  : chunk-local non-blank timestamps
        - ``delta_score``: chunk-local score change vs the carry-in
        - ``delta_len_nb``: chunk-local non-blank count (= ``len(tokens)``)

        Implementation notes:
        - We delegate the prefix-tree walk + blank-filter to the existing
          ``to_nbest_hyps_list`` helper. The ``BatchedBeamHyps.transcript_wb``
          buffer is reset at the start of each chunk by ``clear_chunk_local_``,
          so ``y_sequence`` from that helper is *already* chunk-local; the only
          subtraction we need is for the score.
        - With per_chunk-style carry (now used unconditionally) all K beams of
          chunk-N descend from slot 0 of the carry-in, so a single per-stream
          carry-in score is enough.
        - We clone first so ``flatten_sort_`` (inside ``to_nbest_hyps_list``)
          does not mutate the live tree that the publish-collapse + RC pass +
          carry-out still need.
        - When the underlying MALSD computer is running with
          ``allow_cuda_graphs=true``, the kernels launched here (clone's
          ``copy_from_``, ``flatten_sort_``'s many small per-token ``copy_``
          ops, and the ``.tolist()`` CPU readbacks) interleave badly with the
          captured graph's mempool: the next replay (RC pass for this chunk,
          and the next chunk's main pass) sees corrupted tail-row state, which
          leaks back into the active rows via the captured top-k kernel and
          produces a large WER regression specific to ``on_eou`` mode. A full
          device sync at the end of the snapshot fixes this. It is cheap
          relative to the snapshot itself (we already do many CPU syncs via
          ``.tolist()``) and is a no-op when CUDA is not in use.
        """
        from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps

        live_b = min(int(batched_hyps.batch_size), len(states)) if isinstance(batched_hyps, BatchedBeamHyps) else 0
        if live_b <= 0:
            for state in states:
                state._eou_rerank = None
            return

        # Clone first so ``flatten_sort_`` (inside ``to_nbest_hyps_list``) does not
        # mutate the live tree that the publish-collapse + RC pass + carry-out still
        # need. Trim to ``live_b`` so the returned list aligns 1:1 with ``states``.
        nbest = batched_hyps.clone(batch_size=live_b).to_nbest_hyps_list(score_norm=False)

        if batched_rnnt_states is not None and batched_rnnt_states.batched_hyps is not None:
            carry_scores = batched_rnnt_states.batched_hyps.scores[:live_b, 0].tolist()
        else:
            carry_scores = [0.0] * live_b

        for b in range(live_b):
            candidates: list[dict] = []
            for hyp in nbest[b].n_best_hypotheses:
                tokens = [int(t) for t in hyp.y_sequence.tolist()]
                timesteps = [float(t) for t in hyp.timestamp.tolist()] if hyp.timestamp is not None else []
                candidates.append(
                    {
                        "tokens": tokens,
                        "timesteps": timesteps,
                        "delta_score": float(hyp.score) - carry_scores[b],
                        "delta_len_nb": len(tokens),
                    }
                )
            states[b]._eou_rerank = candidates

        for state in states[live_b:]:
            state._eou_rerank = None

        # See docstring: required to keep the captured MALSD graph's mempool consistent
        # across replays. Removing this re-introduces the on_eou+cuda_graphs regression.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _maybe_rerank_eou_transcript(self, state: RNNTStreamingState, request: Request, prev_tokens_len: int) -> None:
        """At EOU, replace the chunk-N portion of ``state.tokens`` with the
        chunk-local length-normalised winner among the K beams of the EOU chunk.

        Non-EOU chunks already published the raw-argmax winner unchanged via
        ``run_greedy_decoder`` (it sits in ``state.tokens[prev_tokens_len:]``);
        this only fires when EOU has been flagged for the current chunk and a
        snapshot is available. Candidates are sorted raw-descending (the
        ``to_nbest_hyps_list`` upstream uses ``score_norm=False``), so the
        raw winner is index 0 by construction.

        Args:
            state: stream whose EOU is being committed.
            request: the EOU request (used for debug logging only).
            prev_tokens_len: length of ``state.tokens`` immediately BEFORE
                ``run_greedy_decoder`` appended chunk-N's raw-argmax tokens.
        """
        candidates = state._eou_rerank
        state._eou_rerank = None
        if not candidates:
            return

        # First-chunk-of-utterance guard: when EOU fires on the first chunk of
        # a fresh utterance (``state.tokens`` was empty before
        # ``run_greedy_decoder`` ran, i.e. ``prev_tokens_len == 0``) the K
        # chunk-local beams differ mostly by a hallucinated leading filler
        # ("okay yeah ...") emitted on leading silence/noise, and the linear
        # ``delta_score / (delta_len_nb + 1)`` denominator reliably picks the
        # longer decoy. Skip the rerank there and publish the raw-argmax
        # top-1 instead (same as the per_chunk path). The rerank still runs
        # for EOUs that fire on chunk N>0 of an utterance, where the K beams
        # share a real-speech prefix already committed earlier in the
        # utterance and the length bias is much weaker.
        if prev_tokens_len == 0:
            if self._dbg_enabled():
                self._dbg(
                    f"EOU-rerank sid={request.stream_id} SKIP (first chunk of utterance, "
                    f"prev_tokens_len=0): keeping raw top-1"
                )
            return

        # Chunk-local length-norm: chunk's score delta / (chunk's non-blank
        # delta + 1). Applied to every EOU after the stream's first one.
        # Non-EOU chunks always publish the raw-argmax top-1 via the
        # per-chunk collapse and never see this rerank.
        norms = [c["delta_score"] / (c["delta_len_nb"] + 1) for c in candidates]
        norm_winner = max(range(len(candidates)), key=norms.__getitem__)

        if self._dbg_enabled():
            disagree = " (NORM-DISAGREES)" if norm_winner != 0 else ""
            self._dbg(
                f"EOU-rerank sid={request.stream_id} prev_tokens_len={prev_tokens_len} "
                f"raw_winner=0 norm_winner={norm_winner}{disagree}"
            )
            for k, c in enumerate(candidates):
                try:
                    txt = self.asr_model.tokenizer.ids_to_text([int(t) for t in c["tokens"] if int(t) >= 0])
                except Exception:
                    txt = "<decode-err>"
                mark = " <-NORM-WIN" if k == norm_winner else (" <-RAW-WIN" if k == 0 else "")
                self._dbg(
                    f"  EOU-rerank sid={request.stream_id} beam[{k}]: dscore={c['delta_score']:.3f} "
                    f"norm={norms[k]:.3f} dlen_nb={c['delta_len_nb']} text={txt!r}{mark}"
                )

        if norm_winner == 0:
            return

        winner = candidates[norm_winner]
        state.tokens = state.tokens[:prev_tokens_len] + winner["tokens"]
        state.timesteps = state.timesteps[:prev_tokens_len] + winner["timesteps"]
        if state.confidences is not None:
            state.confidences = state.confidences[:prev_tokens_len] + [0.0] * len(winner["tokens"])
        if winner["tokens"]:
            state.last_token = winner["tokens"][-1]
            state.last_token_idx = winner["timesteps"][-1] if winner["timesteps"] else state.last_token_idx

    def shared_transcribe_step_stateful(self, requests: list[Request], encs: Tensor, enc_lens: Tensor) -> None:
        """
        Stateful transcribe step.
        After detecting EOU, it updates the state and run text processor.
        If there are multiple streams, it waits until all states are ready to run text processor.
        Args:
            requests: (list[Request]) List of requests to transcribe.
            encs: (Tensor) Encoder output.
            enc_lens: (Tensor) Encoder output lengths.
        """
        tokens_per_left_padding_tensor = torch.tensor(self.tokens_per_left_padding, device=self.device)
        tokens_per_frame_tensor = torch.tensor(self.tokens_per_frame, device=self.device)
        postponed_requests = [(ridx, request.stream_id) for ridx, request in enumerate(requests)]
        next_postponed_requests = []
        ready_state_ids = set()
        while len(postponed_requests) > 0:
            request_ids_to_process = []
            for ridx, stream_id in postponed_requests:
                if stream_id in ready_state_ids:
                    next_postponed_requests.append((ridx, stream_id))
                    continue
                request_ids_to_process.append(ridx)
            if len(request_ids_to_process) > 0:
                requests_to_process = [requests[jdx] for jdx in request_ids_to_process]
                request_is_last = torch.tensor(
                    [request.is_last for request in requests_to_process], dtype=torch.bool, device=self.device
                )
                enc_lens_dec = enc_lens - tokens_per_left_padding_tensor
                enc_lens_dec_trimmed = torch.where(
                    request_is_last,
                    enc_lens_dec,
                    torch.minimum(enc_lens_dec, tokens_per_frame_tensor.expand_as(enc_lens_dec)),
                )
                self.stateful_transcribe_step(
                    requests_to_process,
                    encs[request_ids_to_process][:, :, self.tokens_per_left_padding :],
                    enc_lens_dec_trimmed,
                    enc_lens_dec,
                    ready_state_ids,
                )
            if len(ready_state_ids) > 0:
                self.text_processor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
                ready_state_ids.clear()
            postponed_requests = next_postponed_requests.copy()
            next_postponed_requests.clear()

        self.update_partial_transcript(requests, self.tokenizer, self.leading_regex_pattern)

    def shared_transcribe_step(self, requests: list[Request], encs: Tensor, enc_lens: Tensor) -> None:
        """
        Stateless transcribe step.
        After detecting EOU, it updates the state and run text processor.
        If there are multiple streams, it waits until all stated are ready to run text processor.
        Args:
            requests: (list[Request]) List of requests to transcribe.
            encs: (Tensor) Encoder output.
            enc_lens: (Tensor) Encoder output lengths.
        """
        postponed_requests = [(ridx, request.stream_id) for ridx, request in enumerate(requests)]
        next_postponed_requests = []
        ready_state_ids = set()

        while len(postponed_requests) > 0:

            request_ids_to_process = []
            for ridx, stream_id in postponed_requests:

                if stream_id in ready_state_ids:
                    # Skip if the state is already ready
                    next_postponed_requests.append((ridx, stream_id))
                    continue

                request_ids_to_process.append(ridx)

            if len(request_ids_to_process) > 0:
                requests_to_process = [requests[jdx] for jdx in request_ids_to_process]
                self.stateless_transcribe_step(
                    requests_to_process,
                    encs=encs[request_ids_to_process],
                    enc_lens=enc_lens[request_ids_to_process],
                    ready_state_ids=ready_state_ids,
                )

            if len(ready_state_ids) > 0:
                self.text_processor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
                ready_state_ids.clear()

            postponed_requests = next_postponed_requests.copy()
            next_postponed_requests.clear()

        self.update_partial_transcript(requests, self.tokenizer, self.leading_regex_pattern)

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """
        Transcribe a step for feature buffers.
        Args:
            fbuffers: (list[FeatureBuffer]) List of feature buffers to transcribe.
        """
        encs, enc_lens = self.encode_feature_buffers(fbuffers)
        if encs is not None:
            if self.stateful:
                self.shared_transcribe_step_stateful(requests=fbuffers, encs=encs, enc_lens=enc_lens)
            else:
                self.shared_transcribe_step(requests=fbuffers, encs=encs, enc_lens=enc_lens)

    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """
        Transcribe a step for frames.
        Args:
            frames: (list[Frame]) List of frames to transcribe.
        """
        encs, enc_lens = self.encode_frames(frames)
        if encs is not None:
            if self.stateful:
                self.shared_transcribe_step_stateful(requests=frames, encs=encs, enc_lens=enc_lens)
            else:
                self.shared_transcribe_step(requests=frames, encs=encs, enc_lens=enc_lens)

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """
        Initialize the request generator.
        Returns:
            (ContinuousBatchedRequestStreamer) Request generator.
        """
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            request_type=self.request_type,
            preprocessor=self.preprocessor,
            buffer_size_in_secs=self.buffer_size_in_secs,
            device=self.device,
            pad_last_frame=True,
            right_pad_features=self.right_padding,
        )
        return request_generator
