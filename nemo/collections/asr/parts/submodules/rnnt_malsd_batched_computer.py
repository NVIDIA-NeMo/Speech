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
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.context_biasing.biasing_multi_model import (
    GPUBiasingMultiModel,
    GPUBiasingMultiModelBase,
)
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import BatchedLabelLoopingState
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    INACTIVE_SCORE,
    NON_EXISTENT_LABEL_VALUE,
    BatchedBeamHyps,
    BlankLMScoreMode,
    PruningMode,
)
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.utils.cuda_python_utils import (
    NeMoCUDAPythonException,
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)
from nemo.core.utils.optional_libs import CUDA_PYTHON_AVAILABLE, cuda_python_required
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum

if CUDA_PYTHON_AVAILABLE:
    from cuda.bindings import runtime as cudart


class MALSDState:
    """
    State for batched ALSD algorithm for RNN-T models. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors
    beam_size: int  # (maximum) length of internal storage for beam dimension
    blank_index: int  # the index of the blank token

    NON_EXISTENT_LABEL: torch.Tensor  # tensor for non existent label constant
    BLANK_TENSOR: torch.Tensor  # tensor for non blank constant
    INACTIVE_SCORE: torch.Tensor  # tensor for inactive score constant

    encoder_output_projected: torch.Tensor  # projected output from the encoder for decoding algorithm
    encoder_output_length: torch.Tensor  # length of the (projected) output from the encoder

    next_labels: torch.Tensor  # storage for next labels
    next_scores: torch.Tensor  # storage for next scores
    next_idx: torch.Tensor  # storage for next scores

    batch_indices: torch.Tensor  # indices of elements in batch (constant, range [0, batch_size-1])
    beam_indices: torch.Tensor  # indices of elements in batch (constant, range [0, beam_size-1])

    time_indices: torch.Tensor  # current time indices for each element in batch
    safe_time_indices: torch.Tensor  # current time indices, but guaranteed to be < encoder_output_length
    last_timesteps: torch.Tensor  # indices of the last timesteps for each element (encoder_output_length - 1)
    last_labels_wb: torch.Tensor  # last labels with blank
    hyp_scores: torch.Tensor  # scores for hypotheses

    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    blank_mask: torch.Tensor  # if the element is blank
    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')

    last_decoder_state: Any  # last state from the decoder, needed for the output
    decoder_state: Any  # current decoder state
    decoder_output: torch.Tensor  # output from the decoder (projected)
    prev_decoder_state: Any  # current decoder state
    prev_decoder_output: torch.Tensor  # output from the decoder (projected)
    init_decoder_state: Any  # current decoder state
    init_decoder_output: torch.Tensor  # output from the decoder (projected)

    batched_hyps: BatchedBeamHyps  # batched hypotheses - decoding result

    # fusion models related fields
    fusion_models: Optional[List[NGramGPULanguageModel]] = None  # list of fusion models
    fusion_models_alpha: Optional[List[float]] = None  # list of weights for the fusion models scores
    fusion_states_list: Optional[List[torch.Tensor]] = None  # list of fusion states
    fusion_states_candidates_list: Optional[List[torch.Tensor]] = None  # list of fusion states candidates
    fusion_scores_list: Optional[List[torch.Tensor]] = None  # list of fusion scores
    fusion_states_prev_list: Optional[List[torch.Tensor]] = None  # list of previous fusion states
    init_fusion_states_list: Optional[List[torch.Tensor]] = None  # list of initial fusion states
    init_fusion_states_candidates_list: Optional[List[torch.Tensor]] = None  # list of initial fusion states candidates
    init_fusion_scores_list: Optional[List[torch.Tensor]] = None  # list of initial fusion scores

    # per-stream biasing: model IDs (kept separate from fusion state lists)
    multi_biasing_ids: Optional[torch.Tensor] = None  # model IDs for per-stream biasing [batch_size]
    multi_biasing_ids_expanded: Optional[torch.Tensor] = None  # expanded from [B] to [B * beam_size]

    # Streaming state fields
    is_continuation: torch.Tensor  # flag indicating if this is a continuation from previous chunk
    is_first_chunk: torch.Tensor  # complement of ``is_continuation``; both feed the captured graph's IF nodes

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        max_time: int,
        encoder_dim: int,
        max_symbols: int,
        device: torch.device,
        float_dtype: torch.dtype,
        blank_index: int,
    ):
        """
        Args:
            batch_size: batch size for encoder output storage
            beam_size: beam size for decoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            max_symbols: max symbols per step (to avoid infinite looping and pre-allocate storage)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
            blank_index: index of the blank symbol
        """
        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_time = max_time
        self.blank_index = blank_index

        self.NON_EXISTENT_LABEL = torch.tensor(NON_EXISTENT_LABEL_VALUE, device=self.device, dtype=torch.long)
        self.BLANK_TENSOR = torch.tensor(self.blank_index, device=self.device, dtype=torch.long)
        self.INACTIVE_SCORE = torch.tensor(INACTIVE_SCORE, device=self.device, dtype=float_dtype)

        self.encoder_output_projected = torch.zeros(
            (self.batch_size, self.max_time, encoder_dim),
            dtype=float_dtype,
            device=self.device,
        )
        self.encoder_output_length = torch.zeros(
            [self.batch_size, self.beam_size], dtype=torch.long, device=self.device
        )

        self.next_idx = torch.zeros([self.batch_size, self.beam_size], dtype=torch.long, device=self.device)
        self.next_labels = torch.zeros([self.batch_size, self.beam_size], dtype=torch.long, device=self.device)
        self.next_scores = torch.zeros([self.batch_size, self.beam_size], dtype=float_dtype, device=self.device)

        self.last_labels_wb = torch.full(
            [self.batch_size, self.beam_size], device=self.device, dtype=torch.long, fill_value=self.blank_index
        )
        self.hyp_scores = torch.full(
            [self.batch_size, self.beam_size], fill_value=self.INACTIVE_SCORE, device=self.device, dtype=float_dtype
        )

        # indices of elements in batch and beam (constant)
        self.batch_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
            .expand(batch_size, self.beam_size)
            .clone()
        )  # size: batch_size x beam_size
        self.beam_indices = (
            torch.arange(self.beam_size, dtype=torch.long, device=self.device)[None, :, None]
            .expand(self.batch_size, -1, self.beam_size)
            .clone()
        )  # size: batch_size x beam_size x beam_size

        self.time_indices = torch.zeros_like(self.batch_indices)
        self.safe_time_indices = torch.zeros_like(self.batch_indices)
        self.last_timesteps = torch.zeros_like(self.batch_indices)

        self.active_mask = torch.zeros_like(self.batch_indices, dtype=torch.bool)
        self.blank_mask = torch.zeros_like(self.active_mask, dtype=torch.bool)
        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self.blank_index,
            init_length=max_time * (max_symbols + 1) if max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )

        # Streaming state fields. The captured FULL_GRAPH reads ``is_first_chunk`` and
        # ``is_continuation`` to route to the first-chunk vs. continuation prologue at replay
        # time. The two flags are kept in lockstep by ``modified_alsd_cuda_graphs`` before
        # each replay (they are mutually exclusive).
        self.is_continuation = torch.tensor(False, device=self.device, dtype=torch.bool)
        self.is_first_chunk = torch.tensor(True, device=self.device, dtype=torch.bool)

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


@dataclass
class SeparateGraphsMALSD:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    before_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    before_loop_continuation: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    loop_body: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    loop_update_decoder: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


@dataclass
class MALSDStateItem:
    """
    Per-stream decoding state for ``ModifiedALSDBatchedRNNTComputer``.

    Used by streaming pipelines that maintain per-stream state. Mirrors
    ``LabelLoopingStateItem`` (greedy), with two beam-search specifics:

    - tensors are shaped ``[beam_size, ...]`` instead of scalar/``[D]``;
    - ``batched_hyps_item`` carries the per-stream prefix tree as a
      ``batch_size == 1`` ``BatchedBeamHyps``.
    """

    predictor_state: Any  # opaque per-stream predictor state of size beam_size
    predictor_output: torch.Tensor  # [beam_size, 1, D]
    label: torch.Tensor  # [beam_size]
    decoded_length: torch.Tensor  # scalar
    fusion_state_list: list[torch.Tensor] = field(default_factory=list)  # each [beam_size, ...]
    batched_hyps_item: Any = None  # BatchedBeamHyps with batch_size == 1


class ModifiedALSDBatchedRNNTComputer(WithOptionalCudaGraphs, ConfidenceMethodMixin):
    """
    Batched Alignment-Length Synchronous Decoding implementation. Callable.
    Based on https://ieeexplore.ieee.org/document/9053040 with the following modficiations:
        - does not support prediction network caching
        - does not employ transcript length estimation, instead, limits the number of expansions for every frame.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_malsd_batch_conditional_rnnt.cu"

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        NO_WHILE_LOOPS = "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    separate_graphs: Optional[SeparateGraphsMALSD]
    full_graph: Optional[torch.cuda.CUDAGraph]
    cuda_graphs_mode: Optional[CudaGraphsMode]
    state: Optional[MALSDState]
    fusion_models: list[NGramGPULanguageModel]

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        beam_size: int,
        max_symbols_per_step: Optional[int] = 10,
        preserve_alignments=False,
        fusion_models: Optional[List[NGramGPULanguageModel]] = None,
        fusion_models_alpha: Optional[List[float]] = None,
        blank_lm_score_mode: Optional[str | BlankLMScoreMode] = None,
        pruning_mode: Optional[str | PruningMode] = None,
        allow_cuda_graphs: bool = True,
        enable_per_stream_biasing: bool = False,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            beam_size: beam size
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            fusion_models: list of fusion models (ngram_lm_model and boosting_tree_model)
            fusion_models_alpha: list of weights for the fusion models scores
            blank_lm_score_mode: mode for scoring blank symbol with fusion models
            pruning_mode: mode for pruning hypotheses with fusion models
            allow_cuda_graphs: whether to allow CUDA graphs
            enable_per_stream_biasing: whether to enable per-stream biasing via multi-boosting tree
        """

        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index

        self.beam_size = beam_size
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self._SOS = self._blank_index
        self.allow_cuda_graphs = allow_cuda_graphs

        if self.preserve_alignments:
            raise NotImplementedError("Preserve alignments is not supported")

        self.state = None
        self.full_graph = None
        self.separate_graphs = None

        self.cuda_graphs_mode = None
        self.cuda_graphs_allow_fallback = True
        self.maybe_enable_cuda_graphs()

        self.biasing_multi_model: GPUBiasingMultiModel | None = (
            GPUBiasingMultiModel(vocab_size=self._blank_index, reallocation_callback_fn=self.reset_cuda_graphs_state)
            if enable_per_stream_biasing
            else None
        )

        self.fusion_models: list[NGramGPULanguageModel] = fusion_models if fusion_models is not None else []
        self.fusion_models_alpha: list[float] = fusion_models_alpha if fusion_models_alpha is not None else []

        if self.fusion_models or self.per_stream_biasing_enabled:
            expected_blank_index = self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            if self._blank_index != expected_blank_index:
                raise ValueError(f"Invalid blank index: expected {expected_blank_index}, got {self._blank_index}")

            self.pruning_mode = PruningMode.EARLY if pruning_mode is None else PruningMode(pruning_mode)
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.blank_lm_score_mode = None

    @property
    def per_stream_biasing_enabled(self) -> bool:
        return self.biasing_multi_model is not None

    @property
    def has_fusion_models(self) -> bool:
        return bool(self.fusion_models) or self.per_stream_biasing_enabled

    def _all_fusion_models(
        self, with_multi_model: bool = True
    ) -> list[NGramGPULanguageModel | GPUBiasingMultiModelBase]:
        if with_multi_model and self.per_stream_biasing_enabled:
            return self.fusion_models + [self.biasing_multi_model]
        return self.fusion_models

    def _advance_all_fusion_models(
        self,
        fusion_states_list: list[torch.Tensor],
        float_dtype: torch.dtype,
        multi_biasing_ids_expanded: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Advance all fusion models (including biasing multi-model) and return lists of (scores, states_candidates).

        Scores are returned with shape [batch_size, beam_size, vocab], with alpha already applied for ngram models
        (biasing model applies alpha internally).
        """
        batch_size = fusion_states_list[0].shape[0] // self.beam_size
        all_scores = []
        all_states_candidates = []
        biasing_index = len(self.fusion_models)  # biasing is always after regular fusion models
        for idx, fusion_model in enumerate(self._all_fusion_models()):
            states = fusion_states_list[idx]
            if idx == biasing_index:
                # biasing multi-model: pass model_ids, alpha is applied internally
                scores, states_candidates = fusion_model.advance(states=states, model_ids=multi_biasing_ids_expanded)
                scores = scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1)
            else:
                scores, states_candidates = fusion_model.advance(states=states)
                scores = (
                    scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.fusion_models_alpha[idx]
                )
            all_scores.append(scores)
            all_states_candidates.append(states_candidates.view(batch_size, self.beam_size, -1))
        return all_scores, all_states_candidates

    def force_cuda_graphs_mode(self, mode: Optional[Union[str, CudaGraphsMode]]):
        """
        Method to set graphs mode. Use only for testing purposes.
        For debugging the algorithm use "no_graphs" mode, since it is impossible to debug CUDA graphs directly.
        """
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
        self.cuda_graphs_allow_fallback = False
        self.state = None

    def maybe_enable_cuda_graphs(self) -> bool:
        """Enable CUDA graphs if conditions met"""
        if self.cuda_graphs_mode is not None:
            # CUDA graphs are already enabled
            return False

        if not self.allow_cuda_graphs:
            self.cuda_graphs_mode = None
        else:
            # cuda graphs are allowed
            # check basic requirements for cuda graphs
            if self.max_symbols is None:
                logging.warning("Max symbols per step is None, which is not allowed with Cuda graphs. Setting to `10`")
                self.max_symbols = 10
            # basic requirements met, need to check while loops
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
                self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
            except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
                logging.warning(
                    "No conditional node support for Cuda.\n"
                    "Cuda graphs with while loops are disabled, decoding speed will be slower\n"
                    f"Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_WHILE_LOOPS
        self.reset_cuda_graphs_state()
        return self.cuda_graphs_mode is not None

    def disable_cuda_graphs(self) -> bool:
        """Disable CUDA graphs, can be used to disable graphs temporary, e.g., in training process"""
        if self.cuda_graphs_mode is None:
            # nothing to disable
            return False
        self.cuda_graphs_mode = None
        self.reset_cuda_graphs_state()
        return True

    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        self.state = None
        self.full_graph = None
        self.separate_graphs = None

    def modified_alsd_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
        multi_biasing_ids: Optional[torch.Tensor] = None,
    ) -> tuple[BatchedBeamHyps, Optional[rnnt_utils.BatchedAlignments], BatchedLabelLoopingState]:
        """
        Pytorch implementation of the batched ALSD algorithm for RNN-T.
        Args:
            encoder_output (torch.Tensor): The output from the encoder network with shape
                [batch_size, max_time, encoder_dim].
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch
                with shape [batch_size].
            multi_biasing_ids (torch.Tensor, optional): Model IDs for per-stream biasing [batch_size].
        Returns:
            BatchedBeamHyps: Batched beam hypotheses.
        """
        batch_size, max_time, _ = encoder_output.shape
        device = encoder_output.device

        if torch.is_autocast_enabled():
            encoder_output = encoder_output.to(torch.get_autocast_gpu_dtype())

        # do not recalculate joint projection, project only once
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        batch_beam_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
            .expand(batch_size, self.beam_size)
            .clone()
        )  # size: batch_size x beam_size
        batch_beam_beam_indices = (
            torch.arange(self.beam_size, dtype=torch.long, device=device)[None, :, None]
            .expand(batch_size, -1, self.beam_size)
            .clone()
        )  # size: batch_size x beam_size x beam_size

        # Always create a fresh ``BatchedBeamHyps`` for this chunk so the returned
        # transcripts / timestamps are chunk-local (and can then be shifted to global
        # before returning). Cross-chunk per-beam state (scores, last_label,
        # transcript_hash, current_lengths_nb, last_timestamp_lasts) is inherited from
        # ``prev_batched_state.batched_hyps`` via ``copy_from_`` + ``clear_chunk_local_``,
        # mirroring greedy ``loop_labels_torch`` which also returns a fresh BatchedHyps per call.
        batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time * (self.max_symbols + 1) if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )
        if prev_batched_state is not None and prev_batched_state.batched_hyps is not None:
            batched_hyps.copy_from_(prev_batched_state.batched_hyps)
            batched_hyps.clear_chunk_local_()

        time_indices = torch.zeros_like(batch_beam_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        # Streaming safety: in mixed batches some rows may have `encoder_output_length == 0`
        # (e.g. the right-context pass for a stream whose chunk has 0 RC frames). Without clamping
        # `last_timesteps` becomes -1 and any indexing into `encoder_output_projected` triggers a
        # device-side assert. Mirrors greedy `loop_labels_torch`.
        last_timesteps = torch.clamp_min(encoder_output_length - 1, 0)[:, None].expand_as(batch_beam_indices)
        active_mask = (encoder_output_length > 0)[:, None].expand_as(batch_beam_indices) & (
            time_indices <= last_timesteps
        )

        # setup fusion models and/or biasing multi-model
        if self.per_stream_biasing_enabled:
            if multi_biasing_ids is None:
                multi_biasing_ids = torch.full([batch_size], fill_value=-1, dtype=torch.long, device=device)
            multi_biasing_ids_expanded = multi_biasing_ids.repeat_interleave(self.beam_size)
        else:
            multi_biasing_ids_expanded = None

        if self.has_fusion_models:
            if prev_batched_state is None or not prev_batched_state.fusion_states_list:
                # fresh start: initial states for all fusion models (incl. biasing if enabled)
                fusion_states_list = []
                for fusion_model in self._all_fusion_models():
                    fusion_model.to(device)
                    fusion_states_list.append(
                        fusion_model.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
                    )
            else:
                # Continuation: reuse fusion states (incl. biasing) from previous chunk.
                # ``prev_batched_state.fusion_states_list[i]`` is stored as
                # ``[B, K]`` (see the ``s.view(batch_size, self.beam_size)`` step
                # below). ``_advance_all_fusion_models`` expects the flat
                # ``[B * K]`` layout, so reshape on the way in. The local
                # ``fusion_states_list`` is replaced again below with the
                # reshaped-to-``[B, K]`` view used by the rest of the loop.
                fusion_states_list = [s.reshape(-1) for s in prev_batched_state.fusion_states_list]
                for fusion_model in self._all_fusion_models():
                    fusion_model.to(device)
            fusion_scores_list, fusion_states_candidates_list = self._advance_all_fusion_models(
                fusion_states_list, float_dtype, multi_biasing_ids_expanded
            )
            fusion_states_list = [s.view(batch_size, self.beam_size) for s in fusion_states_list]
        else:
            fusion_states_list = None
            fusion_states_candidates_list = None
            fusion_scores_list = None

        if prev_batched_state is None:    
            last_labels_wb = torch.full(
                [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
            )
            decoder_state = self.decoder.initialize_state(
                torch.empty(
                    [
                        batch_size * self.beam_size,
                    ],
                    dtype=float_dtype,
                    device=device,
                )
            )

            decoder_output, state, *_ = self.decoder.predict(
                last_labels_wb.view(-1, 1), None, add_sos=False, batch_size=batch_size * self.beam_size
            )
            # do not recalculate joint projection
            decoder_output = self.joint.project_prednet(decoder_output)  # size: [(batch_size x beam_size), 1, Dim]
            self.decoder.batch_replace_states_all(state, dst_states=decoder_state)
        else: 
            # Continuing from previous chunk - batched_hyps already contains all state
            decoder_output = prev_batched_state.predictor_outputs
            decoder_state = prev_batched_state.predictor_states
        step1=0
        while active_mask.any():
            # step 1: get joint output + fuse with fusion models (if present)
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[batch_beam_indices.view(-1), safe_time_indices.view(-1)].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            log_probs = F.log_softmax(logits, dim=-1, dtype=float_dtype).view(
                batch_size, self.beam_size, -1
            )  # [(B x Beam), V]

            if self.has_fusion_models:
                log_probs_top_k, labels_top_k = self.topk_fusion_model(fusion_scores_list, log_probs)
            else:
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )

            # step 2: Make hyps candidates. Add new scores to hyps, force blank if necessary, recombine hyps, prune
            # step 2.1: hyps candidates
            log_probs_blank = log_probs[
                ..., self._blank_index
            ]  # blank scores              size: batch_size x beam_size
            hyps_scores = batched_hyps.scores  # previous hyp scores       size: batch_size x beam_size
            hyps_candidates_prob = (
                hyps_scores.unsqueeze(-1) + log_probs_top_k
            )  # hyps with top-k labels    size: batch_size x beam_size x beam_size
            hyps_candidates_prob_forced_blank = (
                hyps_scores + log_probs_blank
            )  # hyps with forced blank    size: batch_size x beam_size

            # step 2.2 force add final (fully decoded) hyps with to the beam (without updating the score)
            # mask inactive (final) hyps with -inf
            hyps_candidates_prob = torch.where(
                active_mask.unsqueeze(-1),
                hyps_candidates_prob,
                INACTIVE_SCORE,
            )
            # keep inactive (final hypotheses) at the first position in beam
            hyps_candidates_prob[..., 0] = torch.where(
                active_mask,
                hyps_candidates_prob[..., 0],
                hyps_scores,
            )
            # mark the labels corresponding to final hypotheses with negative label (e.g., -1)
            labels_top_k = torch.where(active_mask.unsqueeze(-1), labels_top_k, NON_EXISTENT_LABEL_VALUE)

            # step 2.3: force blank extension with respect to self.max_symbols
            if self.max_symbols is not None:
                force_blank = (batched_hyps.last_timestamp_lasts >= self.max_symbols) & active_mask
            else:
                force_blank = torch.full_like(active_mask, fill_value=False)
            # mask beams if forced blank
            hyps_candidates_prob = torch.where(force_blank.unsqueeze(-1), INACTIVE_SCORE, hyps_candidates_prob)
            # keep hypotheses with forced blank at the first position in beam
            hyps_candidates_prob[..., 0] = torch.where(
                force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
            )
            # change labels to blank if forced blank
            labels_top_k = torch.where(force_blank.unsqueeze(-1), self._blank_index, labels_top_k)

            # step 2.4: final pruning - get top-beam from (beam_size x beam_size) hyps
            next_hyps_prob, hyps_candidates_indices = torch.topk(
                hyps_candidates_prob.view(batch_size, -1), k=self.beam_size, largest=True, sorted=True
            )
            hyps_indices = torch.gather(
                batch_beam_beam_indices.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices
            )  # indices in beam extended with new label
            next_labels = torch.gather(
                labels_top_k.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices
            )  # labels for extended hypotheses

            # step 3: store results
            if self.max_symbols is None:
                batched_hyps.add_results_(hyps_indices, next_labels, next_hyps_prob)
            else:
                batched_hyps.add_results_no_checks_(hyps_indices, next_labels, next_hyps_prob)

            # step 4: recombine hypotheses: sum probabilities of identical hypotheses.
            batched_hyps.recombine_hyps_()

            # step 5: update decoder state + decoder output (+ fusion models state/scores)
            # step 5.1: mask invalid value labels with blank to avoid errors (refer to step 2.2)
            last_labels_wb = torch.where(next_labels >= 0, next_labels, self._blank_index)
            preserve_state = last_labels_wb == self._blank_index

            # size: decoder_output [(B x Beam), 1, Dim]
            # size: state tuple, each is of [Layers, (BxBeam), Dim]
            # step 5.2: update decoder + fusion models state
            # step 5.2.1: storing current decoder output and states of extended hypotheses
            prev_decoder_output = torch.gather(
                decoder_output.view(batch_size, self.beam_size, 1, -1),
                dim=1,
                index=hyps_indices[:, :, None, None].expand(batch_size, self.beam_size, 1, decoder_output.shape[-1]),
            ).view(batch_size * self.beam_size, 1, -1)
            prev_decoder_state = self.decoder.batch_aggregate_states_beam(
                decoder_state, batch_size, self.beam_size, hyps_indices
            )

            # step 5.2.2: get next decoder output and states for extended hypotheses
            decoder_output, decoder_state, *_ = self.decoder.predict(
                last_labels_wb.view(-1).unsqueeze(1),
                prev_decoder_state,
                add_sos=False,
                batch_size=batch_size * self.beam_size,
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # step 5.2.3: update decoder state and output only for non-blank and active hypotheses
            decoder_output = torch.where(preserve_state.view(-1)[:, None, None], prev_decoder_output, decoder_output)
            self.decoder.batch_replace_states_mask(
                src_states=prev_decoder_state, dst_states=decoder_state, mask=preserve_state.view(-1)
            )

            if self.has_fusion_models:
                # fusion_states_list[i]: [batch_size, beam_size]
                # fusion_states_candidates_list[i]: [batch_size, beam_size, vocab_no_blank]
                last_labels_wb_blank_replaced = torch.where(preserve_state, 0, last_labels_wb)
                for fusion_model_idx in range(len(fusion_states_list)):
                    fusion_states_candidates_list[fusion_model_idx] = torch.gather(
                        fusion_states_candidates_list[fusion_model_idx],
                        dim=1,
                        index=hyps_indices[:, :, None].expand(
                            batch_size, self.beam_size, fusion_states_candidates_list[fusion_model_idx].shape[-1]
                        ),
                    )
                    fusion_states_prev = torch.gather(fusion_states_list[fusion_model_idx], dim=1, index=hyps_indices)
                    fusion_states_list[fusion_model_idx] = torch.where(
                        preserve_state,
                        fusion_states_prev,
                        torch.gather(
                            fusion_states_candidates_list[fusion_model_idx],
                            dim=-1,
                            index=last_labels_wb_blank_replaced.unsqueeze(-1),
                        ).squeeze(-1),
                    )
                # advance all fusion models at once (alpha applied inside helper)
                fusion_scores_list, fusion_states_candidates_list = self._advance_all_fusion_models(
                    [s.reshape(-1) for s in fusion_states_list], float_dtype, multi_biasing_ids_expanded
                )

            # step 6: update time indices + active mask
            time_indices = torch.gather(time_indices, dim=-1, index=hyps_indices) + (next_labels == self._blank_index)
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            active_mask = time_indices <= last_timesteps
            
            step1 += 1
        
        # NB: last labels can not exist (nothing decoded on this step).
        # return the last labels from the previous state in this case
        last_labels = batched_hyps.get_last_labels(pad_id=self._SOS)
        batched_hyps.next_timestamp.fill_(0)

        # Make ``batched_hyps.timestamps`` global by adding the cumulative encoder frame
        # offset (number of frames already consumed by previous chunks) to this chunk's
        # writes. Mirrors greedy ``loop_labels_torch`` (see
        # ``rnnt_label_looping.py::loop_labels_torch`` line 523).
        if prev_batched_state is not None:
            batched_hyps.timestamps += prev_batched_state.decoded_lengths.unsqueeze(1).unsqueeze(2)

        decoding_state = BatchedLabelLoopingState(
            predictor_states=decoder_state,
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
            fusion_states_list=fusion_states_list if self.has_fusion_models else None,
            time_jumps=None,
            batched_hyps=batched_hyps,  # Save batched_hyps object for next chunk
        )

        return batched_hyps, None, decoding_state

    def topk_fusion_model(self, fusion_scores_list, log_probs, eps=1e-2):
        """
        Computes the top-k log probabilities and corresponding labels for hypotheses,
        incorporating fusion models scores based on the pruning and blank scoring modes.

        Args:
            fusion_scores_list (List[torch.Tensor]): List of fusion model scores for hypotheses, shape [batch_size, beam_size, vocab_size].
            log_probs (torch.Tensor): Log probabilities from the joint network, shape [batch_size, beam_size, vocab_size].
            eps (float): Epsilon value for numerical stability. Default is 1e-2 for bf16 precision.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_probs_top_k: Top-k log probabilities, shape [batch_size, beam_size, beam_size].
                - labels_top_k: Corresponding top-k labels, shape [batch_size, beam_size, beam_size].
        """

        fusion_scores_sum = sum(fusion_scores_list)
        fusion_scores_alpha_sum = sum(self.fusion_models_alpha)

        match self.pruning_mode, self.blank_lm_score_mode:
            case PruningMode.LATE, BlankLMScoreMode.NO_SCORE:
                log_probs[..., :-1] += fusion_scores_sum
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )

            case PruningMode.LATE, BlankLMScoreMode.LM_WEIGHTED_FULL:
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(
                    -torch.clamp(torch.exp(blank_logprob), max=1.0 - eps)
                )  # 1e-2 is used here instead of 1e-6 to address numerical instability with bf16 precision.
                log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) * fusion_scores_alpha_sum + fusion_scores_sum
                log_probs[..., -1] *= 1 + fusion_scores_alpha_sum
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )

            case PruningMode.EARLY, BlankLMScoreMode.NO_SCORE:
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )
                masked_labels = torch.where(labels_top_k == self._blank_index, 0, labels_top_k)
                log_probs_top_k = torch.where(
                    labels_top_k == self._blank_index,
                    log_probs_top_k,
                    log_probs_top_k + torch.gather(fusion_scores_sum, dim=-1, index=masked_labels),
                )

            case PruningMode.EARLY, BlankLMScoreMode.LM_WEIGHTED_FULL:
                log_probs_top_k, labels_top_k = log_probs.topk(self.beam_size, dim=-1, largest=True, sorted=True)

                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - eps))

                masked_labels = torch.where(labels_top_k == self._blank_index, 0, labels_top_k)
                log_probs_top_k = torch.where(
                    labels_top_k == self._blank_index,
                    log_probs_top_k * (1 + fusion_scores_alpha_sum),
                    log_probs_top_k
                    + non_blank_logprob.unsqueeze(-1) * fusion_scores_alpha_sum
                    + torch.gather(fusion_scores_sum, dim=-1, index=masked_labels),
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported pruning mode {self.pruning_mode} or blank LM score mode {self.blank_lm_score_mode}"
                )

        return log_probs_top_k, labels_top_k

    def modified_alsd_cuda_graphs(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
        multi_biasing_ids: Optional[torch.Tensor] = None,
    ) -> tuple[BatchedBeamHyps, Optional[rnnt_utils.BatchedAlignments], BatchedLabelLoopingState]:
        """
        Cuda-Graphs implementation of the batched ALSD algorithm.
        Args:
            encoder_output (torch.Tensor): The output from the encoder network with shape
                [batch_size, max_time, encoder_dim].
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch
                with shape [batch_size].
            prev_batched_state (Optional[BatchedLabelLoopingState]): The previous batched state.
            multi_biasing_ids (torch.Tensor, optional): Model IDs for per-stream biasing [batch_size].
        Returns:
            tuple: (BatchedBeamHyps, None, BatchedLabelLoopingState)
        """

        assert self.cuda_graphs_mode is not None

        # do not recalculate joint projection, project only once
        encoder_output = self.joint.project_encoder(encoder_output)
        current_batch_size = encoder_output.shape[0]
        current_max_time = encoder_output.shape[1]

        if torch.is_autocast_enabled():
            encoder_output = encoder_output.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self.state is None or self.state.need_reinit(encoder_output):
            self._graph_reinitialize(encoder_output, encoder_output_length)

        # Set continuation flag and restore state from previous chunk if provided
        is_continuation = prev_batched_state is not None
        self.state.is_continuation.fill_(is_continuation)
        # Mirror into the inverse flag so the captured graph's IF nodes can route to
        # the right prologue. Both tensors are read by ``loop_conditional``-style
        # condition kernels baked into the graph at capture time.
        self.state.is_first_chunk.fill_(not is_continuation)

        if is_continuation:
            # Restore state from previous chunk
            self._restore_state_from_prev(prev_batched_state, current_batch_size)

        # set length to zero for elements outside the current batch
        self.state.encoder_output_length.fill_(0)
        # copy (projected) encoder output and lengths
        self.state.encoder_output_projected[:current_batch_size, :current_max_time, ...].copy_(
            encoder_output[:current_batch_size, :current_max_time, ...]
        )
        self.state.encoder_output_length[:current_batch_size].copy_(encoder_output_length.unsqueeze(-1))

        # copy biasing model IDs (kept stable across the replayed graph via state tensors)
        if self.per_stream_biasing_enabled:
            if multi_biasing_ids is None:
                multi_biasing_ids = torch.full(
                    [current_batch_size], fill_value=-1, dtype=torch.long, device=encoder_output.device
                )
            self.state.multi_biasing_ids[:current_batch_size].copy_(multi_biasing_ids)
            self.state.multi_biasing_ids[current_batch_size:].fill_(-1)
            self.state.multi_biasing_ids_expanded.copy_(self.state.multi_biasing_ids.repeat_interleave(self.beam_size))

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            # Single graph dispatches between first-chunk and continuation prologues internally
            # via captured IF nodes that read ``is_first_chunk`` / ``is_continuation``.
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            # Use continuation before_loop graph if continuing from previous chunk
            if is_continuation:
                self.separate_graphs.before_loop_continuation.replay()
            else:
                self.separate_graphs.before_loop.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs.loop_body.replay()
                self.separate_graphs.loop_update_decoder.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # manual loop instead of using graphs
            if is_continuation:
                self._before_loop_continuation()
            else:
                self._before_loop()
            while self.state.active_mask_any.item():
                self._loop_body()
                self._loop_update_decoder()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        if prev_batched_state is not None:
            self.state.batched_hyps.timestamps[:current_batch_size] += (
                prev_batched_state.decoded_lengths[:current_batch_size].unsqueeze(-1).unsqueeze(-1)
            )

        # Create decoding state for next chunk (already a clone of ``self.state.batched_hyps``).
        decoding_state = self._create_decoding_state(encoder_output_length, prev_batched_state)

        # IMPORTANT: return a clone, not the live captured-graph buffer. The caller
        # (streaming pipeline) may keep a reference and convert to hypotheses *after*
        # the next chunk's loop has already mutated ``self.state.batched_hyps``.
        # Trim to the live batch: ``self.state.batched_hyps`` is sized at the
        # capture-time max which can exceed ``current_batch_size`` when streams finish.
        return self.state.batched_hyps.clone(batch_size=current_batch_size), None, decoding_state

    @classmethod
    def _create_loop_body_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

        extern "C" __global__
        void loop_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
        {
         cudaGraphSetConditional(handle, *active_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"loop_conditional", cls.CUDA_PROGRAM_NAME)

    def _graph_reinitialize(
        self,
        encoder_output_projected: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ):
        """
        Reinitializes the graph state for the MALSD computation.
        This method sets up the internal state required for the decoding process, including initializing
        decoder outputs, decoder states, and optional n-gram language model states. It also handles CUDA
        graph compilation based on the specified mode.
        Args:
            encoder_output_projected (torch.Tensor): The projected encoder output tensor of shape
                (batch_size, max_time, encoder_dim).
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch.
        Raises:
            NotImplementedError: If an unsupported CUDA graph mode is specified.
        """

        batch_size, max_time, encoder_dim = encoder_output_projected.shape

        self.state = MALSDState(
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            encoder_dim=encoder_dim,
            max_symbols=self.max_symbols,
            device=encoder_output_projected.device,
            float_dtype=encoder_output_projected.dtype,
            blank_index=self._blank_index,
        )

        self.state.decoder_state = self.decoder.initialize_state(
            torch.empty(
                [
                    batch_size * self.beam_size,
                ],
                dtype=encoder_output_projected.dtype,
                device=encoder_output_projected.device,
            )
        )
        self.state.prev_decoder_state = self.decoder.initialize_state(
            torch.empty(
                [
                    batch_size * self.beam_size,
                ],
                dtype=encoder_output_projected.dtype,
                device=encoder_output_projected.device,
            )
        )

        init_decoder_output, self.state.init_decoder_state, *_ = self.decoder.predict(
            self.state.last_labels_wb.view(-1, 1), None, add_sos=False, batch_size=batch_size * self.beam_size
        )
        self.state.init_decoder_output = self.joint.project_prednet(init_decoder_output).to(
            dtype=self.state.float_dtype
        )  # do not recalculate joint projection

        self.decoder.batch_replace_states_all(self.state.init_decoder_state, dst_states=self.state.decoder_state)
        self.state.decoder_output = self.state.init_decoder_output.clone()

        self.decoder.batch_replace_states_all(self.state.init_decoder_state, dst_states=self.state.prev_decoder_state)
        self.state.prev_decoder_output = self.state.init_decoder_output.clone()

        if self.per_stream_biasing_enabled:
            device = encoder_output_projected.device
            self.state.multi_biasing_ids = torch.full(
                [self.state.batch_size], fill_value=-1, dtype=torch.long, device=device
            )
            self.state.multi_biasing_ids_expanded = torch.full(
                [self.state.batch_size * self.beam_size], fill_value=-1, dtype=torch.long, device=device
            )

        if self.has_fusion_models:
            device = encoder_output_projected.device
            # initialize all fusion models (including biasing multi-model as last element)
            self.state.init_fusion_states_list = []
            for fusion_model in self._all_fusion_models():
                fusion_model.to(device)
                self.state.init_fusion_states_list.append(
                    fusion_model.get_init_states(batch_size=self.state.batch_size * self.beam_size, bos=True).view(
                        self.state.batch_size, self.beam_size
                    )
                )
            self.state.init_fusion_scores_list, self.state.init_fusion_states_candidates_list = (
                self._advance_all_fusion_models(
                    [s.view(-1) for s in self.state.init_fusion_states_list],
                    self.state.float_dtype,
                    self.state.multi_biasing_ids_expanded,
                )
            )

            self.state.fusion_states_list = [s.clone() for s in self.state.init_fusion_states_list]
            self.state.fusion_states_candidates_list = [
                s.clone() for s in self.state.init_fusion_states_candidates_list
            ]
            self.state.fusion_scores_list = [s.clone() for s in self.state.init_fusion_scores_list]
            self.state.fusion_states_prev_list = [s.clone() for s in self.state.init_fusion_states_list]

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
        """Warmup before compiling CUDA graphs.

        Runs a few eager iterations of both the first-chunk and continuation paths so that
        cuBLAS / cuDNN handles and workspaces are allocated and stable before any graph
        capture begins. Mirrors the warmup pattern used by the greedy label-looping decoder.
        """
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        # 11 warmup steps required in DDP mode
        # see https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel
        num_runs = 11 if is_ddp else 3
        self.state.encoder_output_projected.fill_(0.0)
        self.state.encoder_output_length.fill_(1)
        s = torch.cuda.Stream(self.state.device)
        s.wait_stream(torch.cuda.current_stream(device=self.state.device))
        with torch.cuda.stream(s), torch.inference_mode():
            # Warm up the first-chunk path.
            for _ in range(num_runs):
                self._before_loop()
                self._loop_body()
                self._loop_update_decoder()
            # Warm up the continuation path so its prologue and any kernels it touches
            # are primed too. Both captures share a mempool, so any allocator activity
            # they trigger needs to settle before either is captured.
            for _ in range(num_runs):
                self._before_loop_continuation()
                self._loop_body()
                self._loop_update_decoder()
        torch.cuda.current_stream(device=self.state.device).wait_stream(s)
        self.state.encoder_output_length.fill_(0)

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsMALSD()
        
        # Compile before_loop graph for first chunk
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_loop, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_loop()

        # Compile before_loop_continuation graph for streaming
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_loop_continuation, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_loop_continuation()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.loop_body, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._loop_body()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.loop_update_decoder, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._loop_update_decoder()

    @cuda_python_required
    def _full_graph_compile(self):
        """Compile a single CUDA graph that handles both first-chunk and continuation paths.

        The graph contains three conditional sub-graphs in order:
            1. IF (``is_first_chunk``) → ``_before_loop()``
            2. IF (``is_continuation``) → ``_before_loop_continuation()``
            3. WHILE (``active_mask_any``) → ``_loop_body()`` + ``_loop_update_decoder()``

        At replay time the caller toggles ``is_first_chunk`` / ``is_continuation`` so
        exactly one prologue executes. This avoids needing two coexisting CUDAGraph
        objects (which observed to cause cudaErrorIllegalAddress on replay due to
        mempool interaction between the two captures).
        """
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        # Drain any work pending on the default stream (e.g. the warmup that ran just above in
        # ``_graph_reinitialize``) before we start capturing.
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.full_graph = torch.cuda.CUDAGraph()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            # The condition-setter kernel (created lazily by ``_create_loop_body_kernel``) is
            # signature-compatible with any 0-d bool*; we reuse it for all three conditional nodes.
            cond_kernel = self._create_loop_body_kernel()

            # NB: depending on cuda-python version, cudaStreamGetCaptureInfo can return either 5 or 6 elements
            capture_status, _, graph, *_ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # --- IF (is_first_chunk): run first-chunk prologue ---
            (first_chunk_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            is_first_chunk_ptr = np.array([self.state.is_first_chunk.data_ptr()], dtype=np.uint64)
            first_chunk_args = np.array(
                [first_chunk_handle.getPtr(), is_first_chunk_ptr.ctypes.data],
                dtype=np.uint64,
            )
            with with_conditional_node(
                cond_kernel, first_chunk_args, first_chunk_handle, device=self.state.device, cond_type="if"
            ):
                self._before_loop()

            # --- IF (is_continuation): run continuation prologue ---
            (continuation_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            is_continuation_ptr = np.array([self.state.is_continuation.data_ptr()], dtype=np.uint64)
            continuation_args = np.array(
                [continuation_handle.getPtr(), is_continuation_ptr.ctypes.data],
                dtype=np.uint64,
            )
            with with_conditional_node(
                cond_kernel, continuation_args, continuation_handle, device=self.state.device, cond_type="if"
            ):
                self._before_loop_continuation()

            # --- WHILE (active_mask_any): main decoding loop ---
            (loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            loop_args = np.array(
                [loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )
            with with_conditional_node(
                cond_kernel, loop_args, loop_conditional_handle, device=self.state.device, cond_type="while"
            ):
                self._loop_body()
                self._loop_update_decoder()

    def _before_loop(self):
        """
        Clears state and compute initial active mask
        """

        self.state.batched_hyps.clear_()

        # initial state for fusion models (including biasing as last element if enabled)
        if self.has_fusion_models:
            biasing_index = len(self.fusion_models)
            for fusion_idx in range(len(self.state.fusion_states_list)):
                if fusion_idx == biasing_index and self.per_stream_biasing_enabled:
                    # re-initialize biasing states using current model_ids
                    init_states = self.biasing_multi_model.get_init_states(
                        batch_size=self.state.batch_size * self.beam_size, bos=True
                    ).view(self.state.batch_size, self.beam_size)
                    self.state.fusion_states_list[fusion_idx].copy_(init_states)
                else:
                    self.state.fusion_states_list[fusion_idx].copy_(self.state.init_fusion_states_list[fusion_idx])
                self.state.fusion_states_prev_list[fusion_idx].copy_(self.state.fusion_states_list[fusion_idx])
            # advance all to get initial scores and candidates
            scores_list, candidates_list = self._advance_all_fusion_models(
                [s.view(-1) for s in self.state.fusion_states_list],
                self.state.float_dtype,
                self.state.multi_biasing_ids_expanded,
            )
            for fusion_idx in range(len(self.state.fusion_states_list)):
                self.state.fusion_scores_list[fusion_idx].copy_(scores_list[fusion_idx])
                self.state.fusion_states_candidates_list[fusion_idx].copy_(candidates_list[fusion_idx])

        # set decoder state and output to initial values
        self.state.decoder_output.copy_(self.state.init_decoder_output)
        self.state.decoder_state[0].copy_(self.state.init_decoder_state[0])
        self.state.decoder_state[1].copy_(self.state.init_decoder_state[1])

        # last found labels - initially <SOS> (<blank>) symbol
        self.state.last_labels_wb.fill_(self._SOS)

        self._before_loop_common()

    def _before_loop_continuation(self):
        """
        Prologue for a continuation chunk: preserves cross-chunk per-beam state on
        ``batched_hyps`` (scores, last_label, transcript_hash, current_lengths_nb,
        last_timestamp_lasts) and resets only the chunk-local prefix-tree buffers
        (transcript_wb / transcript_wb_prev_ptr / timestamps / current_lengths_wb)
        that the captured loop body would otherwise overflow.

        Decoder and fusion states are already restored by ``_restore_state_from_prev``.
        The caller is responsible for snapshotting / merging the chunk-local transcripts
        before the next chunk (see :meth:`BatchedBeamHyps.merge_` with
        ``is_chunk_continuation=True``).
        """
        self.state.batched_hyps.clear_chunk_local_()
        self._before_loop_common()

    def _before_loop_common(self):
        """
        Common initialization for both first chunk and continuation.
        Resets temporary variables and computes active mask.
        """
        self.state.next_scores.fill_(0.0)
        self.state.next_labels.fill_(0.0)
        self.state.next_idx.fill_(0.0)

        # time indices - reset for current chunk
        self.state.time_indices.fill_(0)
        self.state.safe_time_indices.fill_(0)  # safe time indices: guaranteed to be < encoder_output_length

        torch.sub(self.state.encoder_output_length, 1, out=self.state.last_timesteps)
        torch.clamp_min_(self.state.last_timesteps, 0)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.encoder_output_length, 0, out=self.state.active_mask)

        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

        # set previous decoder state and output to initial values
        self.state.prev_decoder_output.fill_(0)
        self.state.prev_decoder_state[0].fill_(0)
        self.state.prev_decoder_state[1].fill_(0)

    def _loop_body(self):
        """Perform a single iteration of the batched RNN-T decoding loop."""
        # step 1: get joint output + fuse with fusion models (if present)
        logits = self.joint.joint_after_projection(
            self.state.encoder_output_projected[
                self.state.batch_indices.view(-1), self.state.safe_time_indices.view(-1)
            ].unsqueeze(1),
            self.state.decoder_output,
        ).squeeze()
        log_probs = F.log_softmax(logits, dim=-1, dtype=self.state.float_dtype).view(
            self.state.batch_size, self.beam_size, -1
        )  # [(B x Beam), V]

        if self.has_fusion_models:
            log_probs_top_k, labels_top_k = self.topk_fusion_model(self.state.fusion_scores_list, log_probs)
        else:
            log_probs_top_k, labels_top_k = torch.topk(log_probs, self.beam_size, dim=-1, largest=True, sorted=True)

        # step 2: Make hyps candidates. Add new scores to hyps, force blank if necessary, recombine hyps, prune
        # step 2.1: hyps candidates
        log_probs_blank = log_probs[..., self._blank_index]  # blank scores              size: batch_size x beam_size
        hyps_scores = self.state.batched_hyps.scores  # previous hyp scores       size: batch_size x beam_size
        hyps_candidates_prob = (
            hyps_scores.unsqueeze(-1) + log_probs_top_k
        )  # hyps with top-k labels    size: batch_size x beam_size x beam_size
        hyps_candidates_prob_forced_blank = (
            hyps_scores + log_probs_blank
        )  # hyps with forced blank    size: batch_size x beam_size

        # step 2.2 force add final (fully decoded) hyps with to the beam (without updating the score)
        # mask inactive (final) hyps with -inf
        torch.where(
            self.state.active_mask.unsqueeze(-1),
            hyps_candidates_prob,
            self.state.INACTIVE_SCORE,
            out=hyps_candidates_prob,
        )
        # keep inactive (final hypotheses) at the first position in beam
        torch.where(
            self.state.active_mask, hyps_candidates_prob[..., 0], hyps_scores, out=hyps_candidates_prob[..., 0]
        )
        # mark the labels corresponding to final hypotheses with negative label (e.g., -1)
        torch.where(
            self.state.active_mask.unsqueeze(-1), labels_top_k, self.state.NON_EXISTENT_LABEL, out=labels_top_k
        )

        # step 2.3: force blank extension with respect to self.max_symbols
        if self.max_symbols is not None:
            force_blank = (self.state.batched_hyps.last_timestamp_lasts >= self.max_symbols) & self.state.active_mask
        else:
            force_blank = torch.full_like(self.state.active_mask, fill_value=False)
        # mask beams if forced blank
        torch.where(
            force_blank.unsqueeze(-1), self.state.INACTIVE_SCORE, hyps_candidates_prob, out=hyps_candidates_prob
        )
        # keep hypotheses with forced blank at the first position in beam
        torch.where(
            force_blank,
            hyps_candidates_prob_forced_blank,
            hyps_candidates_prob[..., 0],
            out=hyps_candidates_prob[..., 0],
        )
        # change labels to blank if forced blank
        torch.where(force_blank.unsqueeze(-1), self.state.BLANK_TENSOR, labels_top_k, out=labels_top_k)

        # step 2.4: final pruning - get top-beam from (beam x beam) hyps
        next_hyps_prob, hyps_candidates_indices = torch.topk(
            hyps_candidates_prob.view(self.state.batch_size, -1), k=self.beam_size, largest=True, sorted=True
        )
        torch.gather(
            self.state.beam_indices.reshape(self.state.batch_size, -1),
            dim=-1,
            index=hyps_candidates_indices,
            out=self.state.next_idx,
        )  # indices in beam extended with new label
        torch.gather(
            labels_top_k.reshape(self.state.batch_size, -1),
            dim=-1,
            index=hyps_candidates_indices,
            out=self.state.next_labels,
        )  # labels for extended hypotheses
        self.state.next_scores.copy_(next_hyps_prob)

        # step 3: store results
        if self.max_symbols is None:
            self.state.batched_hyps.add_results_(self.state.next_idx, self.state.next_labels, self.state.next_scores)
        else:
            self.state.batched_hyps.add_results_no_checks_(
                self.state.next_idx, self.state.next_labels, self.state.next_scores
            )

        # step 4: recombine hypotheses: sum probabilities of identical hypotheses.
        self.state.batched_hyps.recombine_hyps_()

    def _loop_update_decoder(self):
        """
        Updates the decoder state, decoder output, and optionally the fusion models state
        for the next iteration of the decoding loop in a batched RNNT (Recurrent Neural Network Transducer) setup.
        """
        # step 5: update decoder state + decoder output (+ fusion models state/scores)
        # step 5.1: mask invalid value labels with blank to avoid errors (refer to step 2.2)
        torch.where(
            self.state.next_labels >= 0, self.state.next_labels, self.state.BLANK_TENSOR, out=self.state.last_labels_wb
        )
        preserve_state = self.state.last_labels_wb == self._blank_index

        # size: decoder_output [(B x Beam), 1, Dim]
        # size: state tuple, each is of [Layers, (BxBeam), Dim]
        # step 5.2: update decoder + fusion models state
        # step 5.2.1: storing current decoder output and states of extended hypotheses
        torch.gather(
            self.state.decoder_output.view(self.state.batch_size, self.beam_size, 1, -1),
            dim=1,
            index=self.state.next_idx[:, :, None, None].expand(
                self.state.batch_size, self.beam_size, 1, self.state.decoder_output.shape[-1]
            ),
            out=self.state.prev_decoder_output.view(self.state.batch_size, self.beam_size, 1, -1),
        )
        self.decoder.batch_aggregate_states_beam(
            self.state.decoder_state,
            self.state.batch_size,
            self.beam_size,
            self.state.next_idx,
            self.state.prev_decoder_state,
        )

        # step 5.2.2: get next decoder output and states for extended hypotheses
        decoder_output, decoder_state, *_ = self.decoder.predict(
            self.state.last_labels_wb.view(-1, 1),
            self.state.prev_decoder_state,
            add_sos=False,
            batch_size=self.state.batch_size * self.beam_size,
        )

        # step 5.2.3: update decoder state and output only for non-blank and active hypotheses
        torch.where(
            preserve_state.view(-1)[:, None, None],
            self.state.prev_decoder_output,
            self.joint.project_prednet(decoder_output),
            out=self.state.decoder_output,
        )
        self.decoder.batch_replace_states_mask(
            src_states=self.state.prev_decoder_state,
            dst_states=self.state.decoder_state,
            mask=preserve_state.view(-1),
            other_src_states=decoder_state,
        )

        if self.has_fusion_models:
            # fusion_states_list[i]: [B, beam], fusion_states_candidates_list[i]: [B, beam, V_no_blank]
            last_labels_wb_blank_replaced = torch.where(preserve_state, 0, self.state.last_labels_wb)
            for fusion_idx in range(len(self.state.fusion_states_list)):
                self.state.fusion_states_candidates_list[fusion_idx].copy_(
                    torch.gather(
                        self.state.fusion_states_candidates_list[fusion_idx],
                        dim=1,
                        index=self.state.next_idx[:, :, None].expand(
                            self.state.batch_size,
                            self.beam_size,
                            self.state.fusion_states_candidates_list[fusion_idx].shape[-1],
                        ),
                    )
                )
                torch.gather(
                    self.state.fusion_states_list[fusion_idx],
                    dim=1,
                    index=self.state.next_idx,
                    out=self.state.fusion_states_prev_list[fusion_idx],
                )
                torch.gather(
                    self.state.fusion_states_candidates_list[fusion_idx],
                    dim=-1,
                    index=last_labels_wb_blank_replaced.unsqueeze(-1),
                    out=self.state.fusion_states_list[fusion_idx].unsqueeze(-1),
                )
                torch.where(
                    preserve_state,
                    self.state.fusion_states_prev_list[fusion_idx],
                    self.state.fusion_states_list[fusion_idx],
                    out=self.state.fusion_states_list[fusion_idx],
                )
            # advance all fusion models at once (alpha applied inside helper)
            scores_list, candidates_list = self._advance_all_fusion_models(
                [self.state.fusion_states_list[i].view(-1) for i in range(len(self.state.fusion_states_list))],
                self.state.float_dtype,
                self.state.multi_biasing_ids_expanded,
            )
            for fusion_idx in range(len(self.state.fusion_states_list)):
                self.state.fusion_states_candidates_list[fusion_idx].copy_(candidates_list[fusion_idx])
                self.state.fusion_scores_list[fusion_idx].copy_(scores_list[fusion_idx])

        # step 6: update time indices + active mask
        self.state.time_indices.copy_(self.state.batched_hyps.next_timestamp)
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less_equal(self.state.time_indices, self.state.last_timesteps, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _restore_state_from_prev(
        self, prev_batched_state: BatchedLabelLoopingState, current_batch_size: int
    ):
        """
        Restore decoder state, fusion states, and batched_hyps from previous chunk's state.
        Used for streaming/chunked decoding.
        
        Args:
            prev_batched_state: State from previous chunk
            current_batch_size: Current batch size
        """
        # Restore decoder output and state
        if prev_batched_state.predictor_outputs is not None:
            self.state.decoder_output[:current_batch_size * self.beam_size].copy_(
                prev_batched_state.predictor_outputs.view(-1, 1, prev_batched_state.predictor_outputs.shape[-1])
            )
        
        if prev_batched_state.predictor_states is not None:
            # Copy decoder states (assuming tuple of tensors)
            for i, state_tensor in enumerate(prev_batched_state.predictor_states):
                if state_tensor is not None:
                    self.state.decoder_state[i][:, :current_batch_size * self.beam_size].copy_(
                        state_tensor[:, :current_batch_size * self.beam_size]
                    )
        
        # Restore fusion states (including biasing as last element) if present
        if prev_batched_state.fusion_states_list is not None and self.has_fusion_models:
            for fusion_idx, fusion_state in enumerate(prev_batched_state.fusion_states_list):
                if fusion_state is not None:
                    self.state.fusion_states_list[fusion_idx][:current_batch_size].copy_(
                        fusion_state[:current_batch_size]
                    )
            # Recompute fusion scores and candidates from the restored states for the
            # whole preallocated batch slot. Out-of-batch slots are advanced too, but the
            # captured graph only reads the [:current_batch_size] prefix.
            scores_list, candidates_list = self._advance_all_fusion_models(
                [s.view(-1) for s in self.state.fusion_states_list],
                self.state.float_dtype,
                self.state.multi_biasing_ids_expanded,
            )
            for fusion_idx in range(len(self.state.fusion_states_list)):
                self.state.fusion_scores_list[fusion_idx].copy_(scores_list[fusion_idx])
                self.state.fusion_states_candidates_list[fusion_idx].copy_(candidates_list[fusion_idx])
        
        # Restore batched_hyps from previous state
        if prev_batched_state.batched_hyps is not None:
            self.state.batched_hyps.copy_from_(prev_batched_state.batched_hyps)

    def _create_decoding_state(
        self,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState],
    ) -> BatchedLabelLoopingState:
        """
        Create BatchedLabelLoopingState for the next chunk.
        
        Args:
            encoder_output_length: Length of current encoder output
            prev_batched_state: State from previous chunk (if any)
            
        Returns:
            BatchedLabelLoopingState containing current decoding state
        """
        current_batch_size = encoder_output_length.shape[0]

        # Get last labels from batched_hyps. ``self.state.batched_hyps`` has the capture-time
        # batch dim (>= ``current_batch_size``); slice to the real batch so the returned
        # state matches the rest of the pipeline (and ``predictor_states`` / ``decoded_lengths``
        # which we already slice below).
        last_labels = self.state.batched_hyps.get_last_labels(pad_id=self._SOS)[:current_batch_size]

        # Reset next_timestamp for next chunk
        self.state.batched_hyps.next_timestamp.fill_(0)

        # Calculate accumulated decoded lengths
        if prev_batched_state is None:
            decoded_lengths = encoder_output_length.clone()
        else:
            decoded_lengths = encoder_output_length + prev_batched_state.decoded_lengths[:current_batch_size]
        
        # Handle labels - if nothing decoded this chunk, use previous labels
        if prev_batched_state is not None:
            last_labels = torch.where(
                last_labels == self._SOS,
                prev_batched_state.labels[:current_batch_size],
                last_labels,
            )
        
        # Get fusion states (including biasing) if present
        fusion_states_list = None
        if self.has_fusion_models and self.state.fusion_states_list is not None:
            fusion_states_list = [
                state[:current_batch_size].clone() for state in self.state.fusion_states_list
            ]
        
        return BatchedLabelLoopingState(
            predictor_states=(
                self.state.decoder_state[0][:, :current_batch_size * self.beam_size].clone(),
                self.state.decoder_state[1][:, :current_batch_size * self.beam_size].clone(),
            ),
            predictor_outputs=self.state.decoder_output[:current_batch_size * self.beam_size].clone(),
            labels=last_labels,
            decoded_lengths=decoded_lengths,
            fusion_states_list=fusion_states_list,
            time_jumps=None,
            # Trim to current batch (graph buffers are sized to the capture-time max,
            # which can exceed the live batch when streams finish). Keeps this state's
            # ``batched_hyps.batch_size`` consistent with ``labels.shape[0]``.
            batched_hyps=self.state.batched_hyps.clone(batch_size=current_batch_size),
        )

    def split_batched_state(self, state: BatchedLabelLoopingState) -> list[MALSDStateItem]:
        """
        Split a batched MALSD state into per-stream ``MALSDStateItem``s.

        Mirrors ``GreedyBatchedLabelLoopingComputerBase.split_batched_state`` but on
        beam-search shapes:

        - the predictor state was created with batch dimension ``B * beam_size``;
          we slice it into ``B`` groups of ``beam_size`` consecutive rows and
          re-batch each group with ``decoder.batch_unsplit_states``.
        - ``labels`` / ``decoded_lengths`` are split along the batch axis.
        - ``fusion_states_list`` has each element as ``[B, beam_size, ...]``.
        - ``batched_hyps`` is sliced row-by-row via :meth:`BatchedBeamHyps.slice_row`.
        """
        if state is None:
            return []
        batch_size = state.labels.shape[0]
        beam_size = self.beam_size

        # `predictor_states` was created for batch_size * beam_size. Split into per-row items
        # then re-batch each `beam_size` contiguous chunk.
        per_row_states = self.decoder.batch_split_states(state.predictor_states)
        assert len(per_row_states) == batch_size * beam_size, (
            f"Expected predictor states with batch dim {batch_size * beam_size}, "
            f"got {len(per_row_states)} per-row items"
        )

        items: list[MALSDStateItem] = []
        for i in range(batch_size):
            stream_predictor_state = self.decoder.batch_unsplit_states(
                per_row_states[i * beam_size : (i + 1) * beam_size]
            )
            items.append(
                MALSDStateItem(
                    # Clone tensors so the per-stream snapshot is independent of any further
                    # in-place mutations to the source batched tensors (e.g. by the
                    # right-context decoding pass that reuses the same ``BatchedLabelLoopingState``).
                    predictor_state=stream_predictor_state,
                    predictor_output=state.predictor_outputs[i * beam_size : (i + 1) * beam_size].clone(),
                    label=state.labels[i].clone(),
                    decoded_length=state.decoded_lengths[i].clone(),
                    fusion_state_list=(
                        # ``state.fusion_states_list[k]`` is stored as ``[B, K]``
                        # (see ``modified_alsd_torch``'s
                        # ``s.view(batch_size, self.beam_size)`` step), NOT as the
                        # flat ``[B*K, ...]`` layout used by ``predictor_*``. Slice
                        # along dim 0 with the per-stream index to get ``[K]``.
                        [fs[i].clone() for fs in state.fusion_states_list]
                        if state.fusion_states_list
                        else []
                    ),
                    batched_hyps_item=(
                        state.batched_hyps.slice_row(i) if state.batched_hyps is not None else None
                    ),
                )
            )
        return items

    def merge_to_batched_state(
        self, state_items: list[Optional[MALSDStateItem]]
    ) -> BatchedLabelLoopingState:
        """
        Merge a list of per-stream ``MALSDStateItem``s into a single batched MALSD state.

        ``None`` entries (e.g. fresh streams that joined a batch mid-flight) are
        replaced with a freshly-initialised after-SOS state.

        Mirrors ``GreedyBatchedLabelLoopingComputerBase.merge_to_batched_state``.
        """
        if any(item is None for item in state_items):
            not_none_item = next(item for item in state_items if item is not None)
            assert not_none_item is not None, "merge_to_batched_state needs at least one non-None item"
            device = not_none_item.predictor_output.device
            start_item = self._get_state_item_after_sos(device=device)
            state_items = [item if item is not None else start_item for item in state_items]

        # Re-batch predictor state. Each `item.predictor_state` was built for `beam_size` rows;
        # we split it back to per-row items and then re-batch all `batch_size * beam_size`.
        per_row_states: list[Any] = []
        for item in state_items:
            per_row_states.extend(self.decoder.batch_split_states(item.predictor_state))
        batched_predictor_state = self.decoder.batch_unsplit_states(per_row_states)

        predictor_outputs = torch.cat([item.predictor_output for item in state_items], dim=0)
        labels = torch.stack([item.label for item in state_items], dim=0)
        decoded_lengths = torch.stack([item.decoded_length for item in state_items], dim=0)

        num_fusion = len(state_items[0].fusion_state_list)
        fusion_states_list = []
        for fusion_idx in range(num_fusion):
            # Per-stream ``fusion_state_list[fusion_idx]`` is ``[K]`` (from
            # ``split_batched_state``'s ``fs[i]`` slice or from
            # ``_get_state_item_after_sos``'s ``get_init_states(batch_size=K)``).
            # We need the batched ``[B, K]`` layout that ``modified_alsd_torch``
            # uses for the inner loop (``s.view(batch_size, self.beam_size)``),
            # so stack along a new dim 0 rather than ``cat`` which would
            # produce the flat ``[B*K]`` shape that triggers downstream
            # shape mismatches in ``collapse_batched_state_to_beams_`` and the
            # continuation reshape.
            fusion_states_list.append(
                torch.stack([item.fusion_state_list[fusion_idx] for item in state_items], dim=0)
            )

        batched_hyps_items = [item.batched_hyps_item for item in state_items]
        if all(bh is not None for bh in batched_hyps_items):
            batched_hyps = BatchedBeamHyps.stack_rows(batched_hyps_items)
        else:
            batched_hyps = None

        return BatchedLabelLoopingState(
            predictor_states=batched_predictor_state,
            predictor_outputs=predictor_outputs,
            labels=labels,
            decoded_lengths=decoded_lengths,
            fusion_states_list=fusion_states_list,
            time_jumps=None,
            batched_hyps=batched_hyps,
        )

    def _get_state_item_after_sos(self, device: torch.device | str) -> MALSDStateItem:
        """
        Build a fresh per-stream state corresponding to an after-SOS hypothesis.
        Used by :meth:`merge_to_batched_state` to fill in ``None`` items.
        """
        beam_size = self.beam_size
        # Predictor for a single stream needs `beam_size` rows (mirrors the init path
        # in `modified_alsd_torch` with batch_size=1).
        sos_labels = torch.full([beam_size], fill_value=self._SOS, dtype=torch.long, device=device)
        decoder_output, predictor_state, *_ = self.decoder.predict(
            sos_labels.unsqueeze(1), None, add_sos=False, batch_size=beam_size
        )
        decoder_output = self.joint.project_prednet(decoder_output)  # [beam_size, 1, D]

        fusion_state_list: list[torch.Tensor] = []
        for fusion_model in self._all_fusion_models():
            fusion_state_list.append(
                fusion_model.get_init_states(batch_size=beam_size, bos=True).to(device)
            )

        return MALSDStateItem(
            predictor_state=predictor_state,
            predictor_output=decoder_output,
            label=torch.full([beam_size], fill_value=self._SOS, dtype=torch.long, device=device),
            decoded_length=torch.zeros([], dtype=torch.long, device=device),
            fusion_state_list=fusion_state_list,
            batched_hyps_item=None,
        )

    def collapse_batched_state_to_beams_(
        self,
        state: BatchedLabelLoopingState,
        batched_hyps: BatchedBeamHyps,
        beam_indices: torch.Tensor,
    ) -> None:
        """
        In-place: collapse each row of a batched MALSD state and its associated
        :class:`BatchedBeamHyps` to a single surviving beam, replicated across all
        ``beam_size`` slots.

        After the call, every per-beam tensor on ``state``, on ``state.batched_hyps``
        and on ``batched_hyps`` carries the chosen beam's value at slot 0 and identical
        clones at slots 1..beam_size-1; ``scores[:, 1:]`` is set to ``INACTIVE_SCORE``
        on the prefix-tree buffers so the next chunk's top-k repopulates the slots
        through normal expansion of the surviving beam, mirroring the SOS-time init in
        :meth:`modified_alsd_torch`.

        Used by streaming pipelines after each chunk to commit the per-chunk raw-score
        best beam as the definitive history before the next chunk - this keeps the
        emitted transcript, the EOU decision, and the carried predictor/decoder state
        all derived from the same hypothesis (no inconsistency between "what we
        published" and "what we conditioned the next chunk on").

        Args:
            state: batched MALSD state to collapse in place. ``state.predictor_states``
                is replaced; ``predictor_outputs`` / ``labels`` / ``fusion_states_list``
                are replaced with permuted views/copies. ``state.batched_hyps`` is
                mutated in place via :meth:`BatchedBeamHyps.keep_beam_`.
            batched_hyps: the prefix-tree object returned alongside ``state`` from the
                computer call. Mutated in place via :meth:`BatchedBeamHyps.keep_beam_`.
                Note: in the torch path this aliases ``state.batched_hyps`` (same
                object), in which case ``keep_beam_`` is invoked only once.
            beam_indices: ``[batch_size]`` long tensor giving the beam to keep per row,
                computed from raw scores **before** this method is called (any
                in-place mutation here would invalidate the indices).
        """
        batch_size = state.labels.shape[0]
        beam_size = self.beam_size
        if beam_indices.shape != (batch_size,):
            raise ValueError(
                f"beam_indices must have shape [batch_size={batch_size}], got {tuple(beam_indices.shape)}"
            )

        device = state.labels.device
        beam_indices = beam_indices.to(dtype=torch.long, device=device)

        # Flat row indices into the [B*K]-batched buffers: for batch row b we want all
        # K target slots to point at row (b*K + beam_indices[b]).
        row_offsets = torch.arange(batch_size, device=device, dtype=torch.long) * beam_size
        chosen_flat_idx = row_offsets + beam_indices  # [B]
        flat_perm = chosen_flat_idx.unsqueeze(-1).expand(batch_size, beam_size).reshape(-1)  # [B*K]

        per_row = self.decoder.batch_split_states(state.predictor_states)
        if len(per_row) != batch_size * beam_size:
            raise AssertionError(
                f"Expected predictor states with batch dim {batch_size * beam_size}, "
                f"got {len(per_row)} per-row items"
            )
        replicated_per_row = [per_row[int(idx)] for idx in flat_perm.tolist()]
        state.predictor_states = self.decoder.batch_unsplit_states(replicated_per_row)

        state.predictor_outputs = state.predictor_outputs.index_select(0, flat_perm).contiguous()

        beam_perm = beam_indices.unsqueeze(-1).expand(batch_size, beam_size)
        state.labels = torch.gather(state.labels, dim=1, index=beam_perm).contiguous()

        if state.fusion_states_list:
            # Fusion states (n-gram LM, GPU boosting tree, biasing multi-model)
            # are reshaped to ``[B, K]`` inside ``modified_alsd_torch`` (see the
            # ``s.view(batch_size, self.beam_size)`` step right after the initial
            # ``_advance_all_fusion_models`` call). They are NOT in the
            # ``[B*K, ...]`` flat layout used by ``predictor_states`` /
            # ``predictor_outputs``, so ``index_select(0, flat_perm)`` (whose
            # indices range over ``[0, B*K)``) goes out of bounds and trips a
            # vectorized-gather CUDA assert. Use the per-stream ``beam_perm``
            # gather along the beam axis instead - same pattern as ``state.labels``
            # above. All current fusion models store a single state-id per
            # ``(b, k)`` cell so ``fs.ndim == 2``; assert that here to fail loudly
            # if a future fusion model breaks that assumption.
            for fs in state.fusion_states_list:
                if fs.ndim != 2:
                    raise NotImplementedError(
                        f"collapse_batched_state_to_beams_ only supports rank-2 [B, K] "
                        f"fusion states; got shape {tuple(fs.shape)}"
                    )
            state.fusion_states_list = [
                torch.gather(fs, dim=1, index=beam_perm).contiguous() for fs in state.fusion_states_list
            ]

        # `keep_beam_` on the prefix tree. In the torch path `state.batched_hyps is
        # batched_hyps`, so we only collapse once to avoid double-permuting.
        batched_hyps.keep_beam_(beam_indices)
        if state.batched_hyps is not None and state.batched_hyps is not batched_hyps:
            state.batched_hyps.keep_beam_(beam_indices)

    def collapse_state_item_to_beam_(self, item: MALSDStateItem) -> None:
        """
        In-place per-stream version of :meth:`collapse_batched_state_to_beams_`.

        Collapses the ``beam_size`` parallel hypotheses on a single ``MALSDStateItem`` to
        the raw-score best beam, replicating it across all ``beam_size`` slots and setting
        the other slots' scores to ``INACTIVE_SCORE`` so the next chunk's top-k repopulates
        them through normal expansion of the surviving beam.

        Intended for the "collapse-on-EOU-only" mode of the streaming pipeline: per-chunk
        beam diversity is preserved across the utterance, and the beams are only collapsed
        at the natural utterance boundary so the carried state stays consistent with the
        finalised transcript.

        Args:
            item: per-stream MALSD state. Mutated in place. If ``item.batched_hyps_item``
                is ``None`` (e.g. a stream that never went through a beam decode), this is
                a no-op.
        """
        if item.batched_hyps_item is None:
            return
        beam_size = self.beam_size
        if beam_size <= 1:
            return

        # Pick the surviving beam from the per-stream prefix tree (batch dim == 1).
        chosen = int(item.batched_hyps_item.scores[0].argmax().item())
        device = item.label.device

        # Predictor state: replicate the chosen row across all ``beam_size`` rows.
        per_row = self.decoder.batch_split_states(item.predictor_state)
        if len(per_row) != beam_size:
            raise AssertionError(
                f"Expected predictor state with batch dim {beam_size}, got {len(per_row)} per-row items"
            )
        item.predictor_state = self.decoder.batch_unsplit_states([per_row[chosen]] * beam_size)

        # Predictor output / fusion states: gather the chosen row, then broadcast back.
        chosen_idx = torch.full([beam_size], chosen, dtype=torch.long, device=device)
        item.predictor_output = item.predictor_output.index_select(0, chosen_idx).contiguous()
        item.label = item.label[chosen].expand(beam_size).contiguous()
        if item.fusion_state_list:
            item.fusion_state_list = [fs.index_select(0, chosen_idx).contiguous() for fs in item.fusion_state_list]

        # Prefix tree: ``keep_beam_`` expects per-row indices, shape [batch_size=1].
        chosen_row = torch.tensor([chosen], dtype=torch.long, device=item.batched_hyps_item.device)
        item.batched_hyps_item.keep_beam_(chosen_row)

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
        multi_biasing_ids: Optional[torch.Tensor] = None,
    ) -> tuple[BatchedBeamHyps, Optional[rnnt_utils.BatchedAlignments], BatchedLabelLoopingState]:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self.modified_alsd_cuda_graphs(
                    encoder_output=x,
                    encoder_output_length=out_len,
                    prev_batched_state=prev_batched_state,
                    multi_biasing_ids=multi_biasing_ids,
                )

        return self.modified_alsd_torch(
            encoder_output=x,
            encoder_output_length=out_len,
            prev_batched_state=prev_batched_state,
            multi_biasing_ids=multi_biasing_ids,
        )
