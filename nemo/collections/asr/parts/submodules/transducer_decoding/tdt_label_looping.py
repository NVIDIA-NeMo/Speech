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


from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    BatchedLabelLoopingState,
    GreedyBatchedLabelLoopingComputerBase,
    LabelLoopingStateItem,
    SeparateGraphsLabelLooping,
)
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.core.utils.cuda_python_utils import NeMoCUDAPythonException, cu_call, run_nvrtc, with_conditional_node
from nemo.utils import logging

try:
    from cuda.bindings import runtime as cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


class LabelLoopingState:
    """
    State for Loop Labels algorithm. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors

    model_durations: torch.Tensor

    encoder_output_projected: torch.Tensor  # projected output from the encoder for decoding algorithm
    encoder_output_length: torch.Tensor  # length of the (projected) output from the encoder

    labels: torch.Tensor  # storage for current labels
    scores: torch.Tensor  # storage for current scores
    durations: torch.Tensor  # storage for current predicted durations

    batch_indices: torch.Tensor  # indices of elements in batch (constant, range [0, batch_size-1])

    time_indices: torch.Tensor  # current time indices for each element in batch
    safe_time_indices: torch.Tensor  # current time indices, but guaranteed to be < encoder_output_length
    time_indices_current_labels: torch.Tensor  # time indices for found labels (corresponding to `labels` field)
    last_timesteps: torch.Tensor  # indices of the last timesteps for each element (encoder_output_length - 1)

    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    advance_mask: torch.Tensor  # mask for "advancing" hypotheses (blank is found for the element on the current step)
    blank_mask: torch.Tensor  # if the element is blank
    active_mask_prev: torch.Tensor  # if the element was active on the previous step
    found_labels_mask: torch.Tensor  # mask for found labels (non-blank)

    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')
    advance_mask_any: torch.Tensor  # 0-dim bool tensor, condition for inner loop ('should advance any index')

    decoder_state: Any  # current decoder state
    decoder_output: torch.Tensor  # output from the decoder (projected)

    decoder_state_after_sos: Any  # decoder state after _SOS symbol (for initialization)
    decoder_output_after_sos: (
        torch.Tensor
    )  # output from the decoder (projected) after _SOS symbol (for initialization)

    batched_hyps: rnnt_utils.BatchedHyps  # batched hypotheses - decoding result
    alignments: Optional[rnnt_utils.BatchedAlignments] = None  # batched alignments

    fusion_states_list: Optional[List[torch.Tensor]] = None
    fusion_states_candidates_list: Optional[List[torch.Tensor]] = None
    fusion_scores_list: Optional[List[torch.Tensor]] = None

    def __init__(
        self,
        batch_size: int,
        max_time: int,
        encoder_dim: int,
        max_symbols: int,
        device: torch.device,
        float_dtype: torch.dtype,
        logits_dim: int,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        include_duration_confidence: bool = False,
        include_duration: bool = False,
    ):
        """

        Args:
            batch_size: batch size for encoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            max_symbols: max symbols per step (to avoid infinite looping and pre-allocate storage)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
            logits_dim: output dimension for Joint
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
            include_duration: if predicted token durations are needed to be added to the Hypothesis object
        """
        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.max_time = max_time
        self.include_duration = include_duration

        self.encoder_output_projected = torch.zeros(
            (self.batch_size, self.max_time, encoder_dim),
            dtype=float_dtype,
            device=self.device,
        )
        self.encoder_output_length = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)

        self.labels = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)
        self.scores = torch.zeros([self.batch_size], dtype=float_dtype, device=self.device)

        # indices of elements in batch (constant)
        self.batch_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

        self.time_indices = torch.zeros_like(self.batch_indices)
        self.safe_time_indices = torch.zeros_like(self.batch_indices)
        self.time_indices_current_labels = torch.zeros_like(self.time_indices)
        self.last_timesteps = torch.zeros_like(self.time_indices)
        self.durations = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)

        self.active_mask = torch.zeros([self.batch_size], dtype=torch.bool, device=self.device)
        self.advance_mask = torch.zeros_like(self.active_mask)
        self.blank_mask = torch.zeros_like(self.active_mask)
        self.active_mask_prev = torch.zeros_like(self.active_mask)
        self.found_labels_mask = torch.zeros_like(self.active_mask)

        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)
        self.advance_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=self.batch_size,
            init_length=self.max_time * max_symbols,
            device=self.device,
            float_dtype=float_dtype,
            is_with_durations=include_duration,
        )
        if preserve_alignments or preserve_frame_confidence:
            self.alignments = rnnt_utils.BatchedAlignments(
                batch_size=batch_size,
                logits_dim=logits_dim,
                init_length=max_time * (max_symbols + 1),
                device=self.device,
                float_dtype=self.float_dtype,
                store_alignments=preserve_alignments,
                store_frame_confidence=preserve_frame_confidence,
                with_duration_confidence=include_duration_confidence,
            )
        else:
            self.alignments = None

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


class GreedyBatchedTDTLabelLoopingComputer(GreedyBatchedLabelLoopingComputerBase, ConfidenceMethodMixin):
    """
    Label-Looping algorithm implementation https://arxiv.org/abs/2406.06220 for optimized batched greedy decoding.
    Iterates over labels, on each step finding the next non-blank label
    (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
    to prediction network (with maximum possible batch size),
    which makes it especially useful for scaling the prediction network.
    During decoding all active hypotheses ("texts") have the same lengths.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_label_looping_conditional_tdt.cu"

    separate_graphs: Optional[SeparateGraphsLabelLooping]
    full_graph: Optional[torch.cuda.CUDAGraph]
    state: Optional[LabelLoopingState]
    fusion_models: Optional[List[NGramGPULanguageModel]]

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        durations: list[int] | ListConfig[int],
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        include_duration: bool = False,
        include_duration_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
        allow_cuda_graphs: bool = True,
        fusion_models: Optional[List[NGramGPULanguageModel]] = None,
        fusion_models_alpha: Optional[List[float]] = None,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            durations: list of TDT durations, e.g., [0, 1, 2, 4, 8]
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration: if predicted token durations are needed to be added to the Hypothesis object
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
            confidence_method_cfg: config for the confidence
            fusion_models: list of fusion models
            fusion_models_alpha: list of weights for the fusion models scores
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        # keep durations on CPU to avoid side effects in multi-gpu environment
        self.durations = torch.tensor(list(durations), device="cpu").to(torch.long)
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.preserve_alignments = preserve_alignments or preserve_frame_confidence
        self.allow_cuda_graphs = allow_cuda_graphs
        self.include_duration = include_duration
        self.include_duration_confidence = include_duration_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

        self.state = None
        self.full_graph = None
        self.separate_graphs = None

        self.cuda_graphs_mode = None
        self.cuda_graphs_allow_fallback = True
        self.maybe_enable_cuda_graphs()

        self.fusion_models = fusion_models
        self.fusion_models_alpha = fusion_models_alpha

    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        self.state = None
        self.full_graph = None
        self.separate_graphs = None

    def _get_frame_confidence(self, logits: torch.Tensor, num_durations: int) -> Optional[torch.Tensor]:
        float_dtype = logits.dtype
        return (
            torch.stack(
                (
                    self._get_confidence_tensor(F.log_softmax(logits[:, :-num_durations], dim=-1)).to(
                        dtype=float_dtype
                    ),
                    self._get_confidence_tensor(F.log_softmax(logits[:, -num_durations:], dim=-1)).to(
                        dtype=float_dtype
                    ),
                ),
                dim=-1,
            )
            if self.include_duration_confidence
            else (
                self._get_confidence_tensor(F.log_softmax(logits[:, :-num_durations], dim=-1)).to(dtype=float_dtype)
                if self.preserve_frame_confidence
                else None
            )
        )

    def torch_impl(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
    ) -> tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], BatchedLabelLoopingState]:
        """
        Pure PyTorch implementation

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
            prev_batched_state: previous batched decoding state
        """
        batch_size, max_time, _unused = encoder_output.shape
        device = encoder_output.device
        if self.fusion_models is not None:
            for fusion_model in self.fusion_models:
                fusion_model.to(device)  # fusion_models is nn.Module, but self is not; need to move manually

        # do not recalculate joint projection, project only once
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
            is_with_durations=self.include_duration,
        )
        # init alignments if necessary
        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        # always use alignments variable - for torch.jit adaptation, but keep it as minimal as possible
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2 if use_alignments else 1,  # blank for each timestep + text tokens
            device=device,
            float_dtype=float_dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
            with_duration_confidence=self.include_duration_confidence,
        )

        # durations
        model_durations = self.durations.to(device, non_blocking=True)
        num_durations = model_durations.shape[0]

        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

        # time indices
        last_timesteps = torch.maximum(encoder_output_length - 1, torch.zeros_like(encoder_output_length))
        time_indices = (
            torch.zeros_like(batch_indices) if prev_batched_state is None else prev_batched_state.time_jumps.clone()
        )
        safe_time_indices = torch.minimum(time_indices, last_timesteps)  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices)

        # masks for utterances in batch
        active_mask: torch.Tensor = time_indices < encoder_output_length
        active_mask_prev = active_mask.clone()
        advance_mask = torch.empty_like(active_mask)

        if prev_batched_state is None:
            # initial state, needed for torch.jit to compile (cannot handle None)
            state = self.decoder.initialize_state(encoder_output_projected)
            # last found labels - initially <SOS> (<blank>) symbol
            labels = torch.full_like(batch_indices, fill_value=self._SOS)
            decoder_output, state, *_ = self.decoder.predict(
                labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # fusion models
            if self.fusion_models is not None:
                fusion_states_list = []
                for fusion_model in self.fusion_models:
                    fusion_states_list.append(fusion_model.get_init_states(batch_size=batch_size, bos=True))
            else:
                fusion_states_list = None
        else:
            decoder_output = prev_batched_state.predictor_outputs
            state = prev_batched_state.predictor_states
            fusion_states_list = prev_batched_state.fusion_states_list

        # loop while there are active utterances
        while active_mask.any():
            # stage 1: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            active_mask_prev.copy_(active_mask)

            # stage 1.1: get first joint output
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices, safe_time_indices].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            scores, labels = logits[:, :-num_durations].max(dim=-1)

            if self.fusion_models is not None:
                fusion_scores_list, fusion_states_candidates_list = [], []
                logits_with_fusion = logits.clone()
                for fusion_idx, fusion_model in enumerate(self.fusion_models):
                    fusion_scores, fusion_states_candidates = fusion_model.advance(
                        states=fusion_states_list[fusion_idx],
                    )
                    fusion_scores = fusion_scores.to(dtype=float_dtype)
                    # combine logits with fusion model without blank
                    logits_with_fusion[:, : -num_durations - 1] += self.fusion_models_alpha[fusion_idx] * fusion_scores
                    # save fusion scores and states candidates
                    fusion_scores_list.append(fusion_scores)
                    fusion_states_candidates_list.append(fusion_states_candidates)
                # get max scores and labels without blank
                fusion_scores_max, fusion_labels_max = logits_with_fusion[:, : -num_durations - 1].max(dim=-1)
                # preserve "blank" / "non-blank" category
                torch.where(labels == self._blank_index, labels, fusion_labels_max, out=labels)
                torch.where(labels == self._blank_index, scores, fusion_scores_max, out=scores)

            jump_durations_indices = logits[:, -num_durations:].argmax(dim=-1)
            durations = model_durations[jump_durations_indices]

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            # for blank labels force duration >= 1
            durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
            time_indices_current_labels.copy_(time_indices)
            if use_alignments:
                alignments.add_results_masked_(
                    active_mask=active_mask,
                    time_indices=time_indices_current_labels,
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=self._get_frame_confidence(logits=logits, num_durations=num_durations),
                )

            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            time_indices += durations * active_mask
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, encoder_output_length, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # stage 1.2: inner loop - find next non-blank labels (if exist)
            while advance_mask.any():
                # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
                # store current time indices to use further for storing the results
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                logits = (
                    self.joint.joint_after_projection(
                        encoder_output_projected[batch_indices, safe_time_indices].unsqueeze(1),
                        decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits[:, :-num_durations].max(dim=-1)

                if self.fusion_models is not None:
                    logits_with_fusion = logits.clone()
                    for fusion_idx, fusion_scores in enumerate(fusion_scores_list):
                        # combined scores with fusion model - without blank
                        logits_with_fusion[:, : -num_durations - 1] += (
                            self.fusion_models_alpha[fusion_idx] * fusion_scores
                        )
                    # get max scores and labels without blank
                    more_scores_w_fusion, more_labels_w_fusion = logits_with_fusion[:, : -num_durations - 1].max(
                        dim=-1
                    )
                    # preserve "blank" / "non-blank" category
                    torch.where(more_labels == self._blank_index, more_labels, more_labels_w_fusion, out=more_labels)

                # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
                torch.where(advance_mask, more_labels, labels, out=labels)
                # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
                torch.where(advance_mask, more_scores, scores, out=scores)
                jump_durations_indices = logits[:, -num_durations:].argmax(dim=-1)
                durations = model_durations[jump_durations_indices]

                if use_alignments:
                    alignments.add_results_masked_(
                        active_mask=advance_mask,
                        time_indices=time_indices_current_labels,
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=self._get_frame_confidence(logits=logits, num_durations=num_durations),
                    )

                blank_mask = labels == self._blank_index
                # for blank labels force duration >= 1
                durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
                # same as time_indices[advance_mask] += durations[advance_mask], but non-blocking
                torch.where(advance_mask, time_indices + durations, time_indices, out=time_indices)
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, encoder_output_length, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # NB: difference between RNN-T and TDT here, at the end of utterance:
            # For RNN-T, if we found a non-blank label, the utterance is active (need to find blank to stop decoding)
            # For TDT, we could find a non-blank label, add duration, and the utterance may become inactive
            found_labels_mask = torch.logical_and(active_mask_prev, labels != self._blank_index)
            # store hypotheses
            if self.max_symbols is not None:
                # pre-allocated memory, no need for checks
                batched_hyps.add_results_masked_no_checks_(
                    active_mask=found_labels_mask,
                    labels=labels,
                    time_indices=time_indices_current_labels,
                    scores=scores,
                    token_durations=durations if self.include_duration else None,
                )
            else:
                # auto-adjusted storage
                batched_hyps.add_results_masked_(
                    active_mask=found_labels_mask,
                    labels=labels,
                    time_indices=time_indices_current_labels,
                    scores=scores,
                    token_durations=durations if self.include_duration else None,
                )

            # stage 3: get decoder (prediction network) output with found labels
            # NB: if active_mask is False, this step is redundant;
            # but such check will require device-to-host synchronization, so we avoid it
            # preserve state/decoder_output for inactive elements
            prev_state = state
            prev_decoder_output = decoder_output
            decoder_output, state, *_ = self.decoder.predict(
                labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # preserve correct states/outputs for inactive elements
            self.decoder.batch_replace_states_mask(
                src_states=prev_state,
                dst_states=state,
                mask=~found_labels_mask,
            )
            torch.where(
                found_labels_mask.unsqueeze(-1).unsqueeze(-1), decoder_output, prev_decoder_output, out=decoder_output
            )

            if self.fusion_models is not None:
                for fusion_idx, fusion_states_candidates in enumerate(fusion_states_candidates_list):
                    torch.where(
                        active_mask,
                        fusion_states_candidates[batch_indices, labels * active_mask],
                        fusion_states_list[fusion_idx],
                        out=fusion_states_list[fusion_idx],
                    )

            # stage 4: to avoid infinite looping, go to the next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index,
                            batched_hyps.last_timestamp_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestamp == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                # update safe_time_indices, non-blocking
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                # same as: active_mask = time_indices < encoder_output_length
                torch.less(time_indices, encoder_output_length, out=active_mask)

        # fix timestamps for iterative decoding
        if prev_batched_state is not None:
            batched_hyps.timestamps += prev_batched_state.decoded_lengths.unsqueeze(1)
            if use_alignments:
                alignments.timestamps += prev_batched_state.decoded_lengths.unsqueeze(1)
        # NB: last labels can not exist (nothing decoded on this step).
        # return the last labels from the previous state in this case
        last_labels = batched_hyps.get_last_labels(pad_id=self._SOS)
        decoding_state = BatchedLabelLoopingState(
            predictor_states=state,
            predictor_outputs=decoder_output,
            labels=(
                torch.where(last_labels == self._SOS, prev_batched_state.labels, last_labels)
                if prev_batched_state is not None
                else last_labels
            ),
            decoded_lengths=(
                encoder_output_length.clone()
                if prev_batched_state is None
                else encoder_output_length + prev_batched_state.decoded_lengths
            ),
            fusion_states_list=fusion_states_list,
            time_jumps=time_indices - encoder_output_length,
        )
        if use_alignments:
            return batched_hyps, alignments, decoding_state
        return batched_hyps, None, decoding_state

    def _get_decoding_state_item_after_sos(self, device: torch.device | str) -> LabelLoopingStateItem:
        """Get decoding state item after <SOS> symbol, used for initialization from empty hypotheses."""
        batched_state = self._get_batched_decoding_state_after_sos(device=device, batch_size=1)
        return self.split_batched_state(batched_state)[0]

    def _get_batched_decoding_state_after_sos(
        self, device: torch.device | str, batch_size: int
    ) -> BatchedLabelLoopingState:
        """Get batched decoding state after <SOS> symbol, used for initialization from empty hypotheses."""
        labels = torch.full([batch_size], fill_value=self._SOS, dtype=torch.long, device=device)
        decoder_output, state, *_ = self.decoder.predict(
            labels.unsqueeze(1), None, add_sos=False, batch_size=batch_size
        )
        decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

        if self.fusion_models is not None:
            fusion_states_list = []
            for fusion_model in self.fusion_models:
                fusion_states_list.append(fusion_model.get_init_states(batch_size=batch_size, bos=True))

        state = BatchedLabelLoopingState(
            predictor_states=state,
            predictor_outputs=decoder_output,
            labels=labels,
            decoded_lengths=torch.zeros([batch_size], dtype=torch.long, device=device),
            fusion_states_list=(
                [
                    fusion_model.get_init_states(batch_size=batch_size, bos=True).to(device)
                    for fusion_model in self.fusion_models
                ]
                if self.fusion_models is not None
                else None
            ),
            time_jumps=torch.zeros([batch_size], dtype=torch.long, device=device),
        )
        return state

    def reset_state_by_mask(self, state: BatchedLabelLoopingState, mask: torch.Tensor) -> BatchedLabelLoopingState:
        """
        Reset state for masked elements in the batched state.
        This is used to reset state for elements that are not active anymore to start a new decoding session.

        Args:
            state: batched decoding state
            mask: mask for elements to reset
        """
        state_after_sos = self._get_batched_decoding_state_after_sos(
            device=state.predictor_outputs.device, batch_size=state.labels.shape[0]
        )
        self.decoder.batch_replace_states_mask(
            src_states=state_after_sos.predictor_states, dst_states=state.predictor_states, mask=mask
        )
        torch.where(
            mask[:, None, None],
            state_after_sos.predictor_outputs,
            state.predictor_outputs,
            out=state.predictor_outputs,
        )
        torch.where(mask, state_after_sos.labels, state.labels, out=state.labels)
        torch.where(mask, state_after_sos.decoded_lengths, state.decoded_lengths, out=state.decoded_lengths)
        if self.fusion_models is not None:
            for fusion_idx, fusion_states in enumerate(state.fusion_states_list):
                torch.where(
                    mask,
                    state_after_sos.fusion_states_list[fusion_idx],
                    fusion_states,
                    out=state.fusion_states_list[fusion_idx],
                )
        torch.where(mask, state_after_sos.time_jumps, state.time_jumps, out=state.time_jumps)
        return state

    def split_batched_state(self, state: BatchedLabelLoopingState) -> list[LabelLoopingStateItem]:
        """
        Split batched state into list of items, each item contains state for one hypothesis.
        This is used to pass state between invocations of the algorithm.

        Args:
            state: batched decoding state
        """
        state_items: list[LabelLoopingStateItem] = []
        for i, predictor_state in enumerate(self.decoder.batch_split_states(state.predictor_states)):
            state_items.append(
                LabelLoopingStateItem(
                    predictor_state=predictor_state,
                    predictor_output=state.predictor_outputs[i],
                    label=state.labels[i],
                    decoded_length=state.decoded_lengths[i],
                    fusion_state_list=(
                        [fusion_state[i] for fusion_state in state.fusion_states_list]
                        if state.fusion_states_list is not None
                        else None
                    ),
                    time_jump=state.time_jumps[i],
                )
            )
        return state_items

    def merge_to_batched_state(self, state_items: list[LabelLoopingStateItem | None]) -> BatchedLabelLoopingState:
        """
        Merge list of items into batched state, each item contains state for one hypothesis.
        This is used to pass state between invocations of the algorithm.

        Args:
            state_items: list of items to merge
        """
        if any(item is None for item in state_items):
            not_none_item = next(item for item in state_items if item is not None)
            assert not_none_item is not None
            device = not_none_item.predictor_output.device
            start_item = self._get_decoding_state_item_after_sos(device=device)
            for i, item in enumerate(state_items):
                if item is None:
                    state_items[i] = start_item

        if self.fusion_models is not None:
            fusion_states_list = []
            for fusion_idx in range(len(self.fusion_models)):
                fusion_states_list.append(
                    torch.stack([item.fusion_state_list[fusion_idx] for item in state_items])
                    if any(item.fusion_state_list[fusion_idx] is not None for item in state_items)
                    else None
                )
        else:
            fusion_states_list = None

        batched_state = BatchedLabelLoopingState(
            predictor_states=self.decoder.batch_unsplit_states([item.predictor_state for item in state_items]),
            predictor_outputs=torch.stack([item.predictor_output for item in state_items]),
            labels=torch.stack([item.label for item in state_items]),
            decoded_lengths=torch.stack([item.decoded_length for item in state_items]),
            fusion_states_list=fusion_states_list,
            time_jumps=torch.stack([item.time_jump for item in state_items]),
        )
        return batched_state

    def cuda_graphs_impl(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
    ) -> tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], BatchedLabelLoopingState]:
        """
        Implementation with CUDA graphs.

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
            prev_batched_state: previous batched decoding state
        """
        assert self.cuda_graphs_mode is not None

        # do not recalculate joint projection, project only once
        encoder_output = self.joint.project_encoder(encoder_output)
        current_batch_size = encoder_output.shape[0]
        current_max_time = encoder_output.shape[1]

        if torch.is_autocast_enabled():
            encoder_output = encoder_output.to(torch.get_autocast_gpu_dtype())
        else:
            # since autocast could be enabled outside and disallowed here,
            # we need to cast encoder output to dtype of params
            float_dtype = next(self.joint.parameters()).dtype
            encoder_output = encoder_output.to(float_dtype)

        # init or reinit graph
        if self.state is None or self.state.need_reinit(encoder_output):
            self._graph_reinitialize(encoder_output, encoder_output_length)

        # copy (projected) encoder output and lengths
        self.state.encoder_output_projected[:current_batch_size, :current_max_time, ...].copy_(encoder_output)
        self.state.encoder_output_length[: encoder_output_length.shape[0]].copy_(encoder_output_length)
        # set length to zero for elements outside the current batch
        self.state.encoder_output_length[current_batch_size:].fill_(0)

        self._init_decoding_state(current_batch_size=current_batch_size, prev_batched_state=prev_batched_state)

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs.before_outer_loop.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs.before_inner_loop.replay()
                while self.state.advance_mask_any.item():
                    self.separate_graphs.inner_loop_code.replay()
                self.separate_graphs.after_inner_loop.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            self._before_outer_loop()
            while self.state.active_mask_any.item():
                self._before_inner_loop_get_joint_output()
                while self.state.advance_mask_any.item():
                    self._inner_loop_step_find_next_non_blank()
                self._after_inner_loop_step()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        if prev_batched_state is not None:
            self._fix_timestamps_for_iterative_decoding(
                current_batch_size=current_batch_size, prev_batched_state=prev_batched_state
            )
        # NB: last labels can not exist (nothing decoded on this step).
        # return the last labels from the previous state in this case
        last_labels = self.state.batched_hyps.get_last_labels(pad_id=self._SOS)
        pad_batch_size = (
            self.state.batch_size - prev_batched_state.labels.shape[-1] if prev_batched_state is not None else 0
        )

        if self.fusion_models is not None:
            fusion_states_list = []
            for fusion_states in self.state.fusion_states_list:
                fusion_states_list.append(fusion_states.clone())

        decoding_state = BatchedLabelLoopingState(
            predictor_states=self.decoder.clone_state(self.state.decoder_state),
            predictor_outputs=self.state.decoder_output.clone(),
            labels=(
                torch.where(
                    last_labels == self._SOS,
                    F.pad(prev_batched_state.labels, (0, pad_batch_size), value=self._SOS),
                    last_labels,
                )
                if prev_batched_state is not None
                else last_labels
            ),
            decoded_lengths=(
                self.state.encoder_output_length.clone()
                if prev_batched_state is None
                else self.state.encoder_output_length
                + F.pad(prev_batched_state.decoded_lengths, (0, pad_batch_size), value=0)
            ),
            fusion_states_list=fusion_states_list if self.fusion_models is not None else None,
            time_jumps=self.state.time_indices - self.state.encoder_output_length,
        )

        # NB: return an independent copy of hyps/alignments/state
        # to avoid any manipulations with allocated memory outside the decoder
        return (
            self.state.batched_hyps.clone(),
            self.state.alignments.clone() if self.preserve_alignments else None,
            decoding_state,
        )

    @classmethod
    def _create_outer_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

        extern "C" __global__
        void outer_label_looping_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
        {
         cudaGraphSetConditional(handle, *active_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"outer_label_looping_conditional", cls.CUDA_PROGRAM_NAME)

    @classmethod
    def _create_inner_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the inner loop body (not all non-blank labels found).
        Condition: while(advance_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

        extern "C" __global__
        void inner_find_non_blank_conditional(cudaGraphConditionalHandle handle, const bool *advance_mask_any)
        {
         cudaGraphSetConditional(handle, *advance_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"inner_find_non_blank_conditional", cls.CUDA_PROGRAM_NAME)

    def _graph_reinitialize(
        self,
        encoder_output_projected: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ):
        batch_size, max_time, encoder_dim = encoder_output_projected.shape

        self.state = LabelLoopingState(
            batch_size=batch_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            encoder_dim=encoder_dim,
            max_symbols=self.max_symbols,
            device=encoder_output_projected.device,
            float_dtype=encoder_output_projected.dtype,
            logits_dim=self.joint.num_classes_with_blank,
            preserve_alignments=self.preserve_alignments,
            preserve_frame_confidence=self.preserve_frame_confidence,
            include_duration_confidence=self.include_duration_confidence,
            include_duration=self.include_duration,
        )
        self.state.model_durations = self.durations.to(self.state.device, non_blocking=True)

        # init decoder state
        self.state.labels.fill_(self._SOS)
        decoder_output, new_state, *_ = self.decoder.predict(
            self.state.labels.unsqueeze(1),
            self.decoder.initialize_state(self.state.encoder_output_projected),
            add_sos=False,
            batch_size=self.state.batch_size,
        )
        self.state.decoder_state_after_sos = new_state
        self.state.decoder_state = self.decoder.initialize_state(encoder_output_projected)
        self.decoder.batch_replace_states_all(
            src_states=self.state.decoder_state_after_sos, dst_states=self.state.decoder_state
        )
        # to avoid recalculation of joint projection, store decoder output in state
        self.state.decoder_output_after_sos = self.joint.project_prednet(decoder_output)
        self.state.decoder_output = self.state.decoder_output_after_sos.clone()

        if self.fusion_models is not None:
            # init fusion models states and scores
            self.state.fusion_states_list = []
            self.state.fusion_states_candidates_list = []
            self.state.fusion_scores_list = []
            device = encoder_output_projected.device
            float_dtype = encoder_output_projected.dtype

            for fusion_model in self.fusion_models:
                vocab_size = fusion_model.vocab_size
                fusion_model.to(device)  # fusion_models is nn.Module, but self is not; need to move manually
                self.state.fusion_states_list.append(
                    fusion_model.get_init_states(batch_size=self.state.batch_size, bos=True)
                )
                self.state.fusion_states_candidates_list.append(
                    torch.zeros([batch_size, vocab_size], dtype=torch.long, device=device)
                )

                self.state.fusion_scores_list.append(
                    torch.zeros([batch_size, vocab_size], dtype=float_dtype, device=device)
                )

        # warmup before graph compilation
        if self.cuda_graphs_mode is not self.CudaGraphsMode.NO_GRAPHS:
            self._warmup_for_cuda_graphs()

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            try:
                self._full_graph_compile()
            except NeMoCUDAPythonException as e:
                if not self.cuda_graphs_allow_fallback:
                    raise RuntimeError("Full CUDA graph decoding failed. Mode is forced, raising exception") from e
                logging.warning(
                    f"Full CUDA graph compilation failed: {e}. "
                    "Falling back to native PyTorch CUDA graphs. Decoding will be slower."
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_WHILE_LOOPS
                self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # no graphs needed
            pass
        else:
            raise NotImplementedError

    def _warmup_for_cuda_graphs(self):
        """Warmup before compiling CUDA graphs"""
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        # 11 warmup steps required in DDP mode
        # see https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel
        num_runs = 11 if is_ddp else 3
        self.state.encoder_output_projected.fill_(0.0)
        self.state.encoder_output_length.fill_(1)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(num_runs):
                self._before_outer_loop()
                self._before_inner_loop_get_joint_output()
                self._inner_loop_step_find_next_non_blank()
                self._after_inner_loop_step()
        torch.cuda.current_stream().wait_stream(s)
        self.state.encoder_output_length.fill_(0)

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsLabelLooping()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_outer_loop, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_outer_loop()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_inner_loop, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_inner_loop_get_joint_output()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.inner_loop_code, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._inner_loop_step_find_next_non_blank()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.after_inner_loop, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._after_inner_loop_step()

    def _full_graph_compile(self):
        """Compile full graph for decoding"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.full_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            self._before_outer_loop()

            capture_status, _, graph, _, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # capture: while self.active_mask_any:
            (outer_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            outer_loop_kernel = self._create_outer_while_loop_kernel()
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            outer_loop_args = np.array(
                [outer_loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )

            # loop while there are active utterances
            # while self.active_mask_any:
            with with_conditional_node(
                outer_loop_kernel, outer_loop_args, outer_loop_conditional_handle, device=self.state.device
            ):
                self._before_inner_loop_get_joint_output()
                # capture: while self.advance_mask_any.item():
                inner_while_loop_kernel = self._create_inner_while_loop_kernel()
                (inner_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                advance_mask_any_ptr = np.array([self.state.advance_mask_any.data_ptr()], dtype=np.uint64)
                inner_loop_args = np.array(
                    [
                        inner_loop_conditional_handle.getPtr(),
                        advance_mask_any_ptr.ctypes.data,
                    ],
                    dtype=np.uint64,
                )
                # while self.advance_mask_any.item():
                with with_conditional_node(
                    inner_while_loop_kernel, inner_loop_args, inner_loop_conditional_handle, device=self.state.device
                ):
                    self._inner_loop_step_find_next_non_blank()
                self._after_inner_loop_step()

    def _init_decoding_state(
        self, current_batch_size: int, prev_batched_state: Optional[BatchedLabelLoopingState] = None
    ):
        # NB: we can speedup the case when prev_batched_state is None by using CUDA graphs
        if prev_batched_state is None:
            # last found labels - initially <SOS> (<blank>) symbol
            self.state.labels.fill_(self._SOS)
            self.decoder.batch_replace_states_all(
                src_states=self.state.decoder_state_after_sos, dst_states=self.state.decoder_state
            )
            self.state.decoder_output.copy_(self.state.decoder_output_after_sos)

            # init fusion models states
            if self.fusion_models is not None:
                for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                    self.state.fusion_states_list[fusion_model_idx].copy_(
                        fusion_model.get_init_states(batch_size=self.state.batch_size, bos=True)
                    )

            self.state.time_indices.fill_(0)
        else:
            # labels
            self.state.labels[:current_batch_size].copy_(prev_batched_state.labels[:current_batch_size])
            # initial state
            self.decoder.batch_replace_states_all(
                src_states=prev_batched_state.predictor_states,
                dst_states=self.state.decoder_state,
                batch_size=current_batch_size,
            )
            self.state.decoder_output[:current_batch_size].copy_(
                prev_batched_state.predictor_outputs[:current_batch_size]
            )

            # init fusion models states
            if self.fusion_models is not None:
                for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                    self.state.fusion_states_list[fusion_model_idx][:current_batch_size].copy_(
                        prev_batched_state.fusion_states_list[fusion_model_idx][:current_batch_size]
                    )

            self.state.time_indices[:current_batch_size].copy_(prev_batched_state.time_jumps[:current_batch_size])

    def _before_outer_loop(self):
        """Clear state and compute initial active mask"""
        self.state.batched_hyps.clear_()
        if self.state.alignments is not None:
            self.state.alignments.clear_()

        self.state.scores.fill_(0.0)

        # time indices
        self.state.time_indices_current_labels.copy_(self.state.time_indices)
        # safe time indices: guaranteed to be < encoder_output_length
        torch.sub(self.state.encoder_output_length, 1, out=self.state.last_timesteps)
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)

        # masks for utterances in batch
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)

        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)
        self.state.durations.fill_(0)

    def _before_inner_loop_get_joint_output(self):
        """Get Joint output after decoder output, prepare inner loop to search for all next non-blank labels"""
        # stage 1: get joint output, iteratively seeking for non-blank labels
        # blank label in `labels` tensor means "end of hypothesis" (for this index)
        self.state.active_mask_prev.copy_(self.state.active_mask)
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[self.state.batch_indices, self.state.safe_time_indices].unsqueeze(
                    1
                ),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # same as: scores, labels = logits[:, : -self.state.model_durations.shape[0]].max(-1)
        torch.max(
            logits[:, : -self.state.model_durations.shape[0]], dim=-1, out=(self.state.scores, self.state.labels)
        )

        if self.fusion_models is not None:
            for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                # get fusion scores/states
                fusion_scores, fusion_states_candidates = fusion_model.advance(
                    states=self.state.fusion_states_list[fusion_model_idx]
                )
                self.state.fusion_states_candidates_list[fusion_model_idx].copy_(fusion_states_candidates)
                self.state.fusion_scores_list[fusion_model_idx].copy_(fusion_scores.to(dtype=self.state.float_dtype))
                # update logits with fusion scores
                logits[:, : -self.state.model_durations.shape[0] - 1] += (
                    self.fusion_models_alpha[fusion_model_idx] * fusion_scores
                )
            # get labels (greedy) and scores from current logits, replace labels/scores with new
            scores_w_fusion, labels_w_fusion = logits[:, : -self.state.model_durations.shape[0] - 1].max(dim=-1)
            # preserve "blank" / "non-blank" category
            torch.where(
                self.state.labels == self._blank_index, self.state.labels, labels_w_fusion, out=self.state.labels
            )
            torch.where(
                self.state.labels == self._blank_index, self.state.scores, scores_w_fusion, out=self.state.scores
            )

        jump_durations_indices = logits[:, -self.state.model_durations.shape[0] :].argmax(dim=-1)
        self.state.durations.copy_(self.state.model_durations[jump_durations_indices])

        # search for non-blank labels using joint, advancing time indices for blank labels
        # checking max_symbols is not needed, since we already forced advancing time indices for such cases
        torch.eq(self.state.labels, self._blank_index, out=self.state.blank_mask)
        # blank_mask = self.labels == self._blank_index
        self.state.time_indices_current_labels.copy_(self.state.time_indices)
        # for blank labels force duration >= 1
        self.state.durations.masked_fill_(torch.logical_and(self.state.durations == 0, self.state.blank_mask), 1)

        if self.state.alignments is not None:
            self.state.alignments.add_results_masked_no_checks_(
                active_mask=self.state.active_mask,
                time_indices=self.state.time_indices_current_labels,
                logits=logits if self.preserve_alignments else None,
                labels=self.state.labels if self.preserve_alignments else None,
                confidence=self._get_frame_confidence(
                    logits=logits, num_durations=self.state.model_durations.shape[0]
                ),
            )

        # advance_mask is a mask for current batch for searching non-blank labels;
        # each element is True if non-blank symbol is not yet found AND we can increase the time index
        self.state.time_indices.add_(self.state.durations * self.state.active_mask)
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)
        torch.logical_and(self.state.active_mask, self.state.blank_mask, out=self.state.advance_mask)

        # inner loop: find next non-blank labels (if exist)
        # same as: self.advance_mask_any = advance_mask.any()
        torch.any(self.state.advance_mask, out=self.state.advance_mask_any)

    def _inner_loop_step_find_next_non_blank(self):
        """Find next non-blank labels - one iteration"""
        # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
        # store current time indices to use further for storing the results
        torch.where(
            self.state.advance_mask,
            self.state.time_indices,
            self.state.time_indices_current_labels,
            out=self.state.time_indices_current_labels,
        )
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[self.state.batch_indices, self.state.safe_time_indices].unsqueeze(
                    1
                ),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # get labels (greedy) and scores from current logits, replace labels/scores with new
        # labels[advance_mask] are blank, and we are looking for non-blank labels
        more_scores, more_labels = logits[:, : -self.state.model_durations.shape[0]].max(-1)

        if self.fusion_models is not None:
            for fusion_model_idx, fusion_scores in enumerate(self.state.fusion_scores_list):
                # update logits with fusion scores
                logits[:, : -self.state.model_durations.shape[0] - 1] += (
                    self.fusion_models_alpha[fusion_model_idx] * fusion_scores
                )
            # # get labels (greedy) and scores from current logits, replace labels/scores with new
            more_scores_w_fusion, more_labels_w_fusion = logits[:, : -self.state.model_durations.shape[0] - 1].max(
                dim=-1
            )
            # preserve "blank" / "non-blank" category
            torch.where(more_labels == self._blank_index, more_labels, more_labels_w_fusion, out=more_labels)
            torch.where(more_labels == self._blank_index, more_scores, more_scores_w_fusion, out=more_scores)

        jump_durations_indices = logits[:, -self.state.model_durations.shape[0] :].argmax(dim=-1)
        more_durations = self.state.model_durations[jump_durations_indices]
        # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
        torch.where(self.state.advance_mask, more_labels, self.state.labels, out=self.state.labels)
        # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
        torch.where(self.state.advance_mask, more_scores, self.state.scores, out=self.state.scores)

        if self.state.alignments is not None:
            self.state.alignments.add_results_masked_no_checks_(
                active_mask=self.state.advance_mask,
                time_indices=self.state.time_indices_current_labels,
                logits=logits if self.preserve_alignments else None,
                labels=more_labels if self.preserve_alignments else None,
                confidence=self._get_frame_confidence(
                    logits=logits, num_durations=self.state.model_durations.shape[0]
                ),
            )

        # blank_mask = self.labels == self._blank_index
        torch.eq(self.state.labels, self._blank_index, out=self.state.blank_mask)
        # for blank labels force duration >= 1
        more_durations.masked_fill_(torch.logical_and(more_durations == 0, self.state.blank_mask), 1)
        # self.time_indices += self.blank_mask
        torch.where(
            self.state.advance_mask,
            self.state.time_indices + more_durations,
            self.state.time_indices,
            out=self.state.time_indices,
        )

        torch.where(self.state.advance_mask, more_durations, self.state.durations, out=self.state.durations)

        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)
        torch.logical_and(self.state.active_mask, self.state.blank_mask, out=self.state.advance_mask)
        torch.any(self.state.advance_mask, out=self.state.advance_mask_any)

    def _after_inner_loop_step(self):
        """After inner loop: store labels, query decoder/fusion models, force max symbols"""
        self._after_inner_loop_store_labels()
        self._after_inner_loop_select_fusion_states()
        self._after_inner_loop_get_decoder_output()
        self._after_inner_loop_force_max_symbols()

    def _after_inner_loop_store_labels(self):
        """Stage 3.1: Store hypotheses, update decoder state"""
        self.state.found_labels_mask.copy_(
            torch.logical_and(self.state.active_mask_prev, self.state.labels != self._blank_index)
        )
        self.state.batched_hyps.add_results_masked_no_checks_(
            active_mask=self.state.found_labels_mask,
            labels=self.state.labels,
            time_indices=self.state.time_indices_current_labels,
            scores=self.state.scores,
            token_durations=self.state.durations if self.include_duration else None,
        )

    def _after_inner_loop_select_fusion_states(self):
        """Stage 3.2: Select fusion states with new labels"""
        if self.fusion_models is not None:
            for fusion_model_idx, fusion_states_candidates in enumerate(self.state.fusion_states_candidates_list):
                # select necessary fusion states based on chosen labels
                torch.where(
                    self.state.active_mask,
                    fusion_states_candidates[self.state.batch_indices, self.state.labels * self.state.active_mask],
                    self.state.fusion_states_list[fusion_model_idx],
                    out=self.state.fusion_states_list[fusion_model_idx],
                )

    def _after_inner_loop_get_decoder_output(self):
        """Stage 3.3: Get decoder (prediction network) output using new labels"""
        decoder_output, new_state, *_ = self.decoder.predict(
            self.state.labels.unsqueeze(1), self.state.decoder_state, add_sos=False, batch_size=self.state.batch_size
        )
        self.decoder.batch_replace_states_mask(
            src_states=new_state, dst_states=self.state.decoder_state, mask=self.state.found_labels_mask
        )
        decoder_output_projected = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        torch.where(
            self.state.found_labels_mask.unsqueeze(-1).unsqueeze(-1),
            decoder_output_projected,
            self.state.decoder_output,
            out=self.state.decoder_output,
        )

    def _after_inner_loop_force_max_symbols(self):
        """Stage 4: to avoid looping, go to next frame after max_symbols emission"""
        # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
        # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
        force_blank_mask = torch.logical_and(
            self.state.active_mask,
            torch.logical_and(
                torch.logical_and(
                    self.state.labels != self._blank_index,
                    self.state.batched_hyps.last_timestamp_lasts >= self.max_symbols,
                ),
                self.state.batched_hyps.last_timestamp == self.state.time_indices,
            ),
        )
        self.state.time_indices.add_(force_blank_mask)  # emit blank => advance time indices
        # update safe_time_indices, non-blocking
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        # same as: active_mask = time_indices < encoder_output_length
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _fix_timestamps_for_iterative_decoding(
        self, current_batch_size: int, prev_batched_state: BatchedLabelLoopingState
    ):
        """
        Fix timestamps: if we are in iterative decoding mode,
        we need to add the length of the previous batch to current timestamps
        """
        self.state.batched_hyps.timestamps[:current_batch_size] += prev_batched_state.decoded_lengths[
            :current_batch_size
        ].unsqueeze(1)
        if self.state.alignments is not None:
            self.state.alignments.timestamps[:current_batch_size] -= prev_batched_state.decoded_lengths[
                :current_batch_size
            ].unsqueeze(1)
