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

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
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
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum

try:
    from cuda.bindings import runtime as cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


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
    loop_body: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    loop_update_decoder: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


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
    fusion_models: Optional[List[NGramGPULanguageModel]]

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
        self.maybe_enable_cuda_graphs()

        if fusion_models is not None:
            expected_blank_index = self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            if self._blank_index != expected_blank_index:
                raise ValueError(f"Invalid blank index: expected {expected_blank_index}, got {self._blank_index}")

            self.fusion_models = fusion_models
            self.fusion_models_alpha = fusion_models_alpha

            self.pruning_mode = PruningMode.EARLY if pruning_mode is None else PruningMode(pruning_mode)
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.fusion_models = None
            self.blank_lm_score_mode = None

    def force_cuda_graphs_mode(self, mode: Optional[Union[str, CudaGraphsMode]]):
        """
        Method to set graphs mode. Use only for testing purposes.
        For debugging the algorithm use "no_graphs" mode, since it is impossible to debug CUDA graphs directly.
        """
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
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
    ) -> BatchedBeamHyps:
        """
        Pytorch implementation of the batched ALSD algorithm for RNN-T.
        Args:
            encoder_output (torch.Tensor): The output from the encoder network with shape
                [batch_size, max_time, encoder_dim].
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch
                with shape [batch_size].
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

        # init empty batched beam hypotheses
        batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time * (self.max_symbols + 1) if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )

        last_labels_wb = torch.full(
            [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
        )

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

        time_indices = torch.zeros_like(batch_beam_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        last_timesteps = (encoder_output_length - 1)[:, None].expand_as(batch_beam_indices)
        active_mask = time_indices <= last_timesteps

        # setup fusion models if available
        if self.fusion_models is not None:
            fusion_states_list = []
            fusion_states_candidates_list = []
            fusion_scores_list = []
            for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                fusion_model.to(device)
                fusion_states = fusion_model.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
                fusion_scores, fusion_states_candidates = fusion_model.advance(
                    states=fusion_states
                )  # vocab_size_no_blank

                fusion_scores = (
                    fusion_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1)
                    * self.fusion_models_alpha[fusion_model_idx]
                )
                fusion_states_list.append(fusion_states)
                fusion_states_candidates_list.append(fusion_states_candidates)
                fusion_scores_list.append(fusion_scores)

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

            if self.fusion_models is not None:
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

            if self.fusion_models is not None:
                # fusion_states: size: [(batch_size x beam_size)]
                # fusion_states_candidates: [(batch_size x beam_size) x V (without blank)]
                for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                    fusion_states_candidates = torch.gather(
                        fusion_states_candidates_list[fusion_model_idx].view(batch_size, self.beam_size, -1),
                        dim=1,
                        index=hyps_indices[:, :, None].expand(
                            batch_size, self.beam_size, fusion_states_candidates_list[fusion_model_idx].shape[-1]
                        ),
                    )
                    fusion_states_prev = torch.gather(
                        fusion_states_list[fusion_model_idx].view(batch_size, self.beam_size),
                        dim=1,
                        index=hyps_indices,
                    )
                    last_labels_wb_blank_replaced = torch.where(preserve_state, 0, last_labels_wb)

                    fusion_states = torch.gather(
                        fusion_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
                    ).squeeze(-1)
                    fusion_states = torch.where(preserve_state, fusion_states_prev, fusion_states).view(-1)

                    fusion_scores, fusion_states_candidates = fusion_model.advance(states=fusion_states)
                    fusion_scores = (
                        fusion_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1)
                        * self.fusion_models_alpha[fusion_model_idx]
                    )
                    fusion_states_list[fusion_model_idx] = fusion_states
                    fusion_states_candidates_list[fusion_model_idx] = fusion_states_candidates
                    fusion_scores_list[fusion_model_idx] = fusion_scores

            # step 6: update time indices + active mask
            time_indices = batched_hyps.next_timestamp
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            active_mask = time_indices <= last_timesteps

        return batched_hyps

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
    ) -> BatchedBeamHyps:
        """
        Cuda-Graphs implementation of the batched ALSD algorithm.
        Args:
            encoder_output (torch.Tensor): The output from the encoder network with shape
                [batch_size, max_time, encoder_dim].
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch
                with shape [batch_size].
        Returns:
            BathcedBeamHyps: Batched beam hypotheses.
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

        # set length to zero for elements outside the current batch
        self.state.encoder_output_length.fill_(0)
        # copy (projected) encoder output and lenghts
        self.state.encoder_output_projected[:current_batch_size, :current_max_time, ...].copy_(encoder_output)
        self.state.encoder_output_length[:current_batch_size].copy_(encoder_output_length.unsqueeze(-1))
        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs.before_loop.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs.loop_body.replay()
                self.separate_graphs.loop_update_decoder.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            self._before_loop()
            while self.state.active_mask_any.item():
                self._loop_body()
                self._loop_update_decoder()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        return self.state.batched_hyps

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

        if self.fusion_models is not None:

            device = encoder_output_projected.device

            self.state.init_fusion_states_list = []
            self.state.init_fusion_states_candidates_list = []
            self.state.init_fusion_scores_list = []

            self.state.fusion_states_list = []
            self.state.fusion_states_candidates_list = []
            self.state.fusion_scores_list = []
            self.state.fusion_states_prev_list = []

            for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                fusion_model.to(device)

                init_fusion_states = fusion_model.get_init_states(
                    batch_size=self.state.batch_size * self.beam_size, bos=True
                ).view(self.state.batch_size, self.beam_size)
                init_fusion_scores, init_fusion_states_candidates = fusion_model.advance(
                    states=init_fusion_states.view(-1)
                )
                self.state.init_fusion_scores_list.append(
                    init_fusion_scores.to(dtype=self.state.float_dtype).view(self.state.batch_size, self.beam_size, -1)
                    * self.fusion_models_alpha[fusion_model_idx]
                )
                self.state.init_fusion_states_candidates_list.append(
                    init_fusion_states_candidates.view(self.state.batch_size, self.beam_size, -1)
                )
                self.state.init_fusion_states_list.append(init_fusion_states)

                self.state.fusion_states_list.append(init_fusion_states.clone())
                self.state.fusion_states_candidates_list.append(
                    self.state.init_fusion_states_candidates_list[fusion_model_idx].clone()
                )
                self.state.fusion_scores_list.append(self.state.init_fusion_scores_list[fusion_model_idx].clone())
                self.state.fusion_states_prev_list.append(init_fusion_states.clone())

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self._full_graph_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # no graphs needed
            pass
        else:
            raise NotImplementedError

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsMALSD()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_loop, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_loop()

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

    def _full_graph_compile(self):
        """Compile full graph for decoding"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        self.full_graph = torch.cuda.CUDAGraph()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            self._before_loop()
            capture_status, _, graph, _, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )

            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # capture: while self.active_mask_any:
            (loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            loop_kernel = self._create_loop_body_kernel()
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            loop_args = np.array(
                [loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )
            # loop while there are active utterances
            with with_conditional_node(loop_kernel, loop_args, loop_conditional_handle, device=self.state.device):
                self._loop_body()
                self._loop_update_decoder()

    def _before_loop(self):
        """
        Clears state and compute initial active mask
        """

        self.state.batched_hyps.clear_()

        # initial state for fusion models
        if self.fusion_models is not None:
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                self.state.fusion_states_list[fusion_idx].copy_(self.state.init_fusion_states_list[fusion_idx])
                self.state.fusion_states_candidates_list[fusion_idx].copy_(
                    self.state.init_fusion_states_candidates_list[fusion_idx]
                )
                self.state.fusion_scores_list[fusion_idx].copy_(self.state.init_fusion_scores_list[fusion_idx])
                self.state.fusion_states_prev_list[fusion_idx].copy_(self.state.init_fusion_states_list[fusion_idx])

        # last found labels - initially <SOS> (<blank>) symbol
        self.state.last_labels_wb.fill_(self._SOS)
        self.state.next_scores.fill_(0.0)
        self.state.next_labels.fill_(0.0)
        self.state.next_idx.fill_(0.0)

        # time indices
        self.state.time_indices.fill_(0)
        self.state.safe_time_indices.fill_(0)  # safe time indices: guaranteed to be < encoder_output_length

        torch.sub(self.state.encoder_output_length, 1, out=self.state.last_timesteps)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.encoder_output_length, 0, out=self.state.active_mask)

        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

        # set decoder state and output to initial values
        self.state.decoder_output.copy_(self.state.init_decoder_output)
        self.state.decoder_state[0].copy_(self.state.init_decoder_state[0])
        self.state.decoder_state[1].copy_(self.state.init_decoder_state[1])

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

        if self.fusion_models is not None:
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

        if self.fusion_models is not None:
            # fusion_states: size: [(batch_size x beam_size)]
            # fusion_states_candidates: [(batch_size x beam_size) x V (without blank)]
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
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
                last_labels_wb_blank_replaced = torch.where(preserve_state, 0, self.state.last_labels_wb)

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
                fusion_scores, fusion_states_candidates = fusion_model.advance(
                    states=self.state.fusion_states_list[fusion_idx].view(-1)
                )
                fusion_scores = (
                    fusion_scores.to(dtype=self.state.float_dtype).view(self.state.batch_size, self.beam_size, -1)
                    * self.fusion_models_alpha[fusion_idx]
                )
                self.state.fusion_states_candidates_list[fusion_idx].copy_(
                    fusion_states_candidates.view(self.state.batch_size, self.state.beam_size, -1)
                )
                self.state.fusion_scores_list[fusion_idx].copy_(fusion_scores)

        # step 6: update time indices + active mask
        self.state.time_indices.copy_(self.state.batched_hyps.next_timestamp)
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less_equal(self.state.time_indices, self.state.last_timesteps, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> BatchedBeamHyps:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self.modified_alsd_cuda_graphs(encoder_output=x, encoder_output_length=out_len)

        return self.modified_alsd_torch(encoder_output=x, encoder_output_length=out_len)
