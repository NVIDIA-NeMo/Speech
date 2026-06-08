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
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import (
    CacheAwareRNNTMALSDStreamingState,
    CacheAwareRNNTStreamingState,
)
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    drop_trailing_features,
    get_confidence_utils,
)
from nemo.collections.asr.parts.submodules.rnnt_malsd_batched_computer import ModifiedALSDBatchedRNNTComputer
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    BatchedBeamHyps,
    export_batched_beam_hyps_to_cpu_lists,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging

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
        self.init_decoding_computer()
        super().__init__()

    def init_decoding_computer(self) -> None:
        """
        Probe the model's decoding stack once and stash the resulting computer
        on ``self`` so per-chunk code can branch on it without re-doing the
        attribute-chain dive.

        Exactly one of ``self.decoding_computer`` (MALSD beam-search) and
        ``self.greedy_decoding_computer`` (greedy, used for per-stream biasing
        detection) is non-``None`` for any supported decoding stack; both are
        ``None`` if the stack exposes no ``decoding_computer`` at all.
        """
        try:
            decoding_computer = self.asr_model.asr_model.decoding.decoding.decoding_computer
        except AttributeError:
            decoding_computer = None
        if isinstance(decoding_computer, ModifiedALSDBatchedRNNTComputer):
            self.decoding_computer: ModifiedALSDBatchedRNNTComputer | None = decoding_computer
            self.greedy_decoding_computer = None
        else:
            self.decoding_computer = None
            self.greedy_decoding_computer = decoding_computer

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

        # Expected feature buffer length for trimming (safeguard for feature buffer inputs)
        self.expected_feature_buffer_len = int(self.buffer_size_in_secs / self.window_stride)

        self.stop_history_eou_in_milliseconds = cfg.endpointing.stop_history_eou
        self.residue_tokens_at_end = cfg.endpointing.residue_tokens_at_end
        self.word_boundary_tolerance = cfg.streaming.word_boundary_tolerance
        self.return_tail_result = cfg.return_tail_result

        self.request_type = RequestType.from_str(cfg.streaming.request_type)

        # MALSD beam-search streaming knobs. ``chunks_per_beam_reset == 1`` collapses
        # the K-beam state down to a single top-1 hypothesis after every chunk, which
        # matches the original cache-aware behaviour. Higher values preserve beam
        # diversity across multiple chunks before collapsing - currently only the
        # ``== 1`` path is fully wired up; larger values fall back to it.
        self.chunks_per_beam_reset = int(cfg.streaming.get("chunks_per_beam_reset", 1))

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
            (CacheAwareRNNTStreamingState) New empty state. Returns the MALSD subclass
            when the pipeline is configured for beam-search decoding.
        """
        state = (
            CacheAwareRNNTMALSDStreamingState() if self.decoding_computer is not None else CacheAwareRNNTStreamingState()
        )
        state.set_global_offset(0)
        new_options = options.fill_defaults(
            default_enable_itn=self.text_processor.itn_enabled,
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

    def _streaming_step(
        self,
        states: list[CacheAwareRNNTStreamingState],
        feature_buffers: Tensor,
        feature_buffer_lens: Tensor,
        context,
        previous_hypotheses: list[Hypothesis | None],
        drop_extra_pre_encoded: int,
        keep_all_outputs: bool,
        prompt_vectors: Tensor | None,
        biasing_enabled: bool,
    ) -> tuple[list[Hypothesis], object]:
        """
        Dispatcher between the greedy single-shot path and the MALSD beam path.

        For greedy (``self.decoding_computer is None``) this just calls the existing
        ``asr_model.stream_step``. For MALSD it runs the encoder once and drives
        :class:`ModifiedALSDBatchedRNNTComputer` with the per-stream beam carry.
        """
        if self.decoding_computer is None:
            return self.asr_model.stream_step(
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
        return self._malsd_stream_step(
            states=states,
            feature_buffers=feature_buffers,
            feature_buffer_lens=feature_buffer_lens,
            context=context,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            keep_all_outputs=keep_all_outputs,
            biasing_enabled=biasing_enabled,
        )

    def _malsd_stream_step(
        self,
        states: list[CacheAwareRNNTMALSDStreamingState],
        feature_buffers: Tensor,
        feature_buffer_lens: Tensor,
        context,
        drop_extra_pre_encoded: int,
        keep_all_outputs: bool,
        biasing_enabled: bool,
    ) -> tuple[list[Hypothesis], object]:
        """
        One streaming step for the MALSD beam-search path:

        1. Encoder-only pass - the decoder is driven by this pipeline, not by
           the model's built-in decoding wrapper.
        2. Merge per-stream ``MALSDStateItem``s into a batched MALSD state.
        3. Run :class:`ModifiedALSDBatchedRNNTComputer` for this chunk.
        4. Update per-stream windowed-beam tracking from this chunk's emissions.
        5. Optionally collapse to the chunk's top-1 (current default behaviour).
        6. Split the batched MALSD state back into per-stream carries.
        7. Build a cumulative ``Hypothesis`` per stream from
           ``window_committed + window_beam_tokens[top1]``.

        Returns a list of cumulative ``Hypothesis`` per stream and the new
        encoder cache context, matching the shape of ``stream_step``.
        """
        # Per-stream multi-biasing ids: not yet supported on the MALSD streaming
        # path. Greedy-side per-stream biasing knobs stay independent.
        multi_biasing_ids = None
        if biasing_enabled:
            logging.warning(
                "Per-stream biasing is not yet wired up on the MALSD cache-aware "
                "streaming path; ignoring biasing requests for this chunk."
            )

        # Merge per-stream carries into a batched MALSD state. ``None`` entries
        # (fresh streams) are filled with the after-SOS state inside ``merge_to_batched_state``.
        carries = [state.hyp_decoding_state for state in states]
        if all(c is None for c in carries):
            batched_state = None
        else:
            batched_state = self.decoding_computer.merge_to_batched_state(carries)

        # All MALSD GPU work (encoder, decoder, windowed walk, collapse, split)
        # shares one ``inference_mode`` region: ``collapse_batched_state_to_beams_``
        # and ``split_batched_state`` mutate the inference tensors returned by
        # ``decoding_computer(...)`` in place, which is illegal once we've left
        # the captured ``inference_mode`` region.
        with (
            torch.amp.autocast(
                device_type=self.asr_model.device_str,
                dtype=self.asr_model.compute_dtype,
                enabled=self.asr_model.use_amp,
            ),
            torch.inference_mode(),
        ):
            encoded, encoded_len, new_context = self.asr_model.encode_step(
                processed_signal=feature_buffers,
                processed_signal_length=feature_buffer_lens,
                context=context,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
                keep_all_outputs=keep_all_outputs,
                drop_left_context=self.drop_left_context,
                valid_out_len=self.valid_out_len,
            )
            # ``encoded`` from the encoder wrapper is shaped [B, D, T]; the MALSD
            # computer expects [B, T, D] (matches the rest of the decoding stack).
            encs_dim_last = encoded.transpose(1, 2).contiguous()

            best_batched_hyps, batched_state = self.decoding_computer(
                encs_dim_last, encoded_len, batched_state
            )

            self._update_windowed_beam_state(states=states, best_batched_hyps=best_batched_hyps)

            # Capture pre-collapse argmax + scores. After ``collapse_batched_state_to_beams_``
            # runs, ``scores[:, 1:]`` is forced to ``INACTIVE_SCORE`` and ``scores[:, 0]``
            # carries the winner - so any post-collapse argmax returns 0 unconditionally.
            # We need the PRE-collapse slot index to index ``window_beam_tokens`` (which
            # was just computed against the diverged pre-collapse slots).
            beam_indices_cpu = best_batched_hyps.scores.argmax(dim=-1).detach().cpu().tolist()
            scores_pre_collapse = best_batched_hyps.scores.detach().cpu()

            # Collapse the K-beam state at the configured cadence. For now we always
            # collapse every chunk (``chunks_per_beam_reset == 1``); the multi-chunk
            # window is a follow-up that needs full prefix-tree carry across chunks.
            for state in states:
                state._malsd_chunk_count += 1
            do_collapse = self.chunks_per_beam_reset <= 1 or any(
                state._malsd_chunk_count >= self.chunks_per_beam_reset for state in states
            )
            if do_collapse:
                beam_indices = best_batched_hyps.scores.argmax(dim=-1).to(torch.long)
                self.decoding_computer.collapse_batched_state_to_beams_(
                    batched_state, best_batched_hyps, beam_indices
                )

            carry_items = self.decoding_computer.split_batched_state(batched_state)
            for state, carry in zip(states, carry_items):
                state.hyp_decoding_state = carry

        # Build per-stream cumulative ``Hypothesis`` from the windowed state,
        # then (on collapse chunks) promote the chosen beam's window tokens into
        # the committed prefix and clear the window. The published hypothesis
        # is identical pre/post-collapse promotion - just with everything moved
        # into ``committed`` afterwards.
        hyps: list[Hypothesis] = []
        for b, state in enumerate(states):
            top1_slot = beam_indices_cpu[b]
            window_tokens = (
                state.window_beam_tokens[top1_slot] if state.window_beam_tokens else []
            )
            window_ts = (
                state.window_beam_timestamps[top1_slot] if state.window_beam_timestamps else []
            )
            cum_tokens = state.window_committed_tokens + list(window_tokens)
            cum_ts = state.window_committed_timestamps + list(window_ts)

            hyps.append(
                Hypothesis(
                    score=float(scores_pre_collapse[b, top1_slot].item()),
                    y_sequence=cum_tokens,
                    timestamp=cum_ts,
                    length=len(cum_tokens),
                )
            )

            if do_collapse:
                state._malsd_chunk_count = 0
                state.window_committed_tokens = list(cum_tokens)
                state.window_committed_timestamps = list(cum_ts)
                state.window_beam_tokens = None
                state.window_beam_timestamps = None

        return hyps, new_context

    def _update_windowed_beam_state(
        self,
        states: list[CacheAwareRNNTMALSDStreamingState],
        best_batched_hyps: BatchedBeamHyps,
    ) -> None:
        """
        Extend each state's per-slot ``window_beam_tokens[k]`` with the chunk-local
        emissions of the slot that originated from carry slot ``k`` at chunk start.

        The helper exposes per-(batch, beam) chunk-local tokens/timestamps and the
        chunk-start -> chunk-end descent map; the permute-then-append windowed-beam
        policy lives here.
        """
        chunk_tokens, chunk_timestamps, root_ptrs = export_batched_beam_hyps_to_cpu_lists(best_batched_hyps)
        beam_size = best_batched_hyps.beam_size
        for state, ct, cts, rp in zip(states, chunk_tokens, chunk_timestamps, root_ptrs):
            prev_t = state.window_beam_tokens or [[] for _ in range(beam_size)]
            prev_ts = state.window_beam_timestamps or [[] for _ in range(beam_size)]
            state.window_beam_tokens = [prev_t[int(rp[k])] + ct[k] for k in range(beam_size)]
            state.window_beam_timestamps = [prev_ts[int(rp[k])] + cts[k] for k in range(beam_size)]

    def run_malsd_decoder(
        self, state: CacheAwareRNNTMALSDStreamingState, request: Request, hyp: Hypothesis
    ) -> bool:
        """
        MALSD counterpart to :meth:`run_greedy_decoder`.

        Reuses the greedy decoder for EOU detection, label-buffer rolling and
        offset bookkeeping. Then RESYNCS ``state.tokens`` / ``state.timesteps`` /
        ``state.confidences`` with the current top-1's cumulative slice
        (``hyp.y_sequence[_malsd_utterance_start:]``).

        The resync is the load-bearing step that distinguishes MALSD from
        greedy: between chunks, MALSD's raw-argmax top-1 can switch beams with
        incompatible token histories (beam A: ``["I"]`` at chunk t, beam B:
        ``["I", "I"]`` at chunk t+1). ``run_greedy_decoder`` appends
        ``hyp.y_sequence[offset:]`` onto whatever was already in ``state.tokens``,
        which would splice A's prefix with B's new tokens into a Frankenstein
        transcript. Overwriting with the actual current top-1 belief keeps the
        published transcript consistent with whichever beam currently wins.

        On EOU we bump ``_malsd_utterance_start`` to the current cumulative
        length so the next utterance's resync slice starts past the cleared
        previous utterance.
        """
        eou_detected = self.run_greedy_decoder(state, request, hyp)

        # Resync state.tokens / state.timesteps / state.confidences with the
        # current top-1's cumulative slice for this utterance.
        all_tokens = list(hyp.y_sequence) if hyp.y_sequence is not None else []
        all_timestamps = list(hyp.timestamp) if hyp.timestamp is not None else []
        start = max(0, int(state._malsd_utterance_start))
        start = min(start, len(all_tokens))
        tokens_list = all_tokens[start:]
        timestamps_list = all_timestamps[start:]

        state.tokens = list(tokens_list)
        state.timesteps = list(timestamps_list)
        state.confidences = [0.0] * len(tokens_list)
        if tokens_list:
            state.last_token = tokens_list[-1]
            state.last_token_idx = timestamps_list[-1] if timestamps_list else None

        if eou_detected:
            # mark the boundary so the next utterance's slice starts past the
            # tokens we just finalised
            state._malsd_utterance_start = len(all_tokens)
        return eou_detected

    def run_greedy_decoder(self, state: CacheAwareRNNTStreamingState, request: Request, hyp: Hypothesis) -> bool:
        """
        Run the greedy RNNT decoder on the hypothesis and update the state
        Args:
            state: (CacheAwareRNNTStreamingState) The state of the stream
            request: (Request) The current request (frame or feature buffer)
            hyp: (Hypothesis) The hypothesis of the current request
        Returns:
            (bool) Whether EOU is detected.
        """
        eou_detected = request.is_last
        # Per-token non-blank confidence precomputed during RNN-T decoding (aligned with `hyp.y_sequence`).
        # Populated only when `asr.decoding.greedy.preserve_frame_confidence=true`; otherwise None.
        cur_output, cur_labels, new_offset = self.greedy_rnnt_decoder(
            global_timestamps=hyp.timestamp,
            tokens=hyp.y_sequence,
            length=self.tokens_per_frame,
            offset=state.offset,
            confidences=hyp.non_blank_step_confidence_precomputed,
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
        requests: list[Request],
        features: list[Tensor],
        right_paddings: list[int],
        ready_state_ids: set,
        keep_all_outputs: bool = False,
    ) -> None:
        """
        Cache Aware Transcribe Step
        It receives a list of requests (Frame or FeatureBuffer) and features and do the following:

        1. Preprocess the features by stacking them and computing the lengths
        2. Collecting previous hypotheses for stateful decoding
        3. Get the context and mapping from the context manager for cache aware streaming
        4. Perform a streaming step with the ASR model
        5. Update the cache and reset the cache slots for the streams that has ended
        6. Update the previous hypothesis and reset the previous hypothesis for the streams that has ended
        7. Perform greedy RNNT decoding to get the best hypothesis and update the states
        8. Update the ready states to indicate that the state is ready for text post-processing
        Args:
            requests: (list[Request]) List of requests (frames or feature buffers) to transcribe.
            features: (list[Tensor]) List of feature buffers.
            right_paddings: (list[int] | None) List of right paddings.
            ready_state_ids: (set) Set of ready state IDs.
            keep_all_outputs: (bool) Whether to keep all outputs or not.
        """

        feature_buffers, feature_buffer_lens = self.preprocess(features, right_paddings)
        states, stream_ids, eos_flags = [], [], []
        for request in requests:
            states.append(self.get_state(request.stream_id))
            stream_ids.append(request.stream_id)
            eos_flags.append(request.is_last)

        previous_hypotheses = [state.get_previous_hypothesis() for state in states]

        # Per-stream biasing is only wired up on the greedy decoder. When MALSD
        # is active ``self.greedy_decoding_computer`` is ``None`` (see
        # :meth:`init_decoding_computer`) so ``biasing_enabled`` falls back to
        # ``False`` and the warning in ``_malsd_stream_step`` covers the rest.
        biasing_enabled = (
            self.greedy_decoding_computer is not None and self.greedy_decoding_computer.per_stream_biasing_enabled
        )

        if not biasing_enabled and any(state.has_biasing_request() for state in states):
            logging.warning("Biasing request is not empty, but decoder does not support per-stream biasing. Skipping")

        # Handle per-stream biasing: add biasing models to multi_model if needed
        if biasing_enabled:
            for i, (request, state, previous_hyp) in enumerate(zip(requests, states, previous_hypotheses)):
                if state.has_biasing_request():
                    if state.options.biasing_cfg.multi_model_id is None:
                        if state.options.biasing_cfg.auto_manage_multi_model:
                            state.options.biasing_cfg.add_to_multi_model(
                                tokenizer=self.asr_model.tokenizer,
                                biasing_multi_model=self.greedy_decoding_computer.biasing_multi_model,
                            )
                        else:
                            logging.warning(
                                "Biasing request is not empty, not auto managed and not compiled. Skipping"
                            )
                    if previous_hyp is None:
                        previous_hypotheses[i] = Hypothesis.empty_with_biasing_cfg(state.options.biasing_cfg)
                    else:
                        previous_hyp.biasing_cfg = state.options.biasing_cfg

        context, mapping = self.context_manager.get_context(stream_ids)

        prompt_vectors = None
        if self.prompt_enabled:
            prompt_vectors = self._build_prompt_vectors(states)

        drop_extra_pre_encoded = 0 if not self.use_cache else self.asr_model.drop_extra_pre_encoded
        best_hyp, new_context = self._streaming_step(
            states=states,
            feature_buffers=feature_buffers,
            feature_buffer_lens=feature_buffer_lens,
            context=context,
            previous_hypotheses=previous_hypotheses,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            keep_all_outputs=keep_all_outputs,
            prompt_vectors=prompt_vectors,
            biasing_enabled=biasing_enabled,
        )

        # update the cache and reset the cache slots for the streams that has ended
        self.context_manager.update_cache(stream_ids, new_context, mapping)
        self.context_manager.reset_slots(stream_ids, eos_flags)

        # update the previous hypothesis for non-eos streams. For greedy this is the
        # ``Hypothesis`` returned by ``rnnt_decoder_predictions_tensor``; for MALSD
        # it is the cumulative ``Hypothesis`` built in ``_malsd_stream_step``. The
        # eos reset is deferred to *after* the per-request decoder loop below so
        # that ``run_malsd_decoder`` can still see the current utterance start.
        for state, hyp, eos in zip(states, best_hyp, eos_flags):
            if not eos:
                state.set_previous_hypothesis(hyp)

        # run per-request decoder for each request-state-hypothesis tuple
        for request, state, hyp in zip(requests, states, best_hyp):
            if self.decoding_computer is not None:
                eou_detected = self.run_malsd_decoder(state, request, hyp)
            else:
                eou_detected = self.run_greedy_decoder(state, request, hyp)
            if eou_detected:
                self.bpe_decoder.decode_bpe_tokens(state)
                state.cleanup_after_eou()
                ready_state_ids.add(request.stream_id)

        # Deferred eos reset - now safe to clear MALSD per-stream carry too.
        for state, eos in zip(states, eos_flags):
            if eos:
                state.reset_previous_hypothesis()

        # Cleanup per-stream biasing models when stream ends (greedy path only;
        # ``biasing_enabled`` is True only when ``self.greedy_decoding_computer`` is set).
        if biasing_enabled:
            for request, state in zip(requests, states):
                # only the first request contains biasing options; biasing options for the stream are stored in state
                if request.is_last and state.has_biasing_request():
                    if state.options.biasing_cfg.auto_manage_multi_model:
                        state.options.biasing_cfg.remove_from_multi_model(
                            biasing_multi_model=self.greedy_decoding_computer.biasing_multi_model
                        )

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
            self._debug_print_finals(ready_state_ids)
            ready_state_ids.clear()

        self.update_partial_transcript(fbuffers, self.tokenizer, self.leading_regex_pattern)
        self._debug_print_partials(fbuffers)

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
            self._debug_print_finals(ready_state_ids)
            ready_state_ids.clear()

        self.update_partial_transcript(frames, self.tokenizer, self.leading_regex_pattern)
        self._debug_print_partials(frames)

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
            preprocessor=self.preprocessor,
            buffer_size_in_secs=self.buffer_size_in_secs,
            device=self.device,
            pad_last_frame=True,
        )
        return request_generator
