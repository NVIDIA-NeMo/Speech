# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.parts.context_biasing import GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.utils.cuda_python_utils import (
    NeMoCUDAPythonException,
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


NON_EXISTENT_LABEL_VALUE = -1
INACTIVE_SCORE = float("-inf")


class BacthedBeamCTCState:
    """
    State for Batched Beam Search for CTC models. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors
    beam_size: int  # (maximum) length of internal storage for beam dimension
    blank_index: int  # the index of the blank token

    decoder_outputs: torch.Tensor  # logprobs from decoder
    decoder_output_lengths: torch.Tensor  # lengths of the decoder outputs (i.e. max time for each utterance)
    last_timesteps: torch.Tensor  # last time step for each utterance (used to check if the decoding is finished)

    vocab: torch.Tensor  # vocabulary of the model. Constant
    vocab_blank_mask: torch.Tensor  # mask for blank token in the vocabulary. Constant

    curr_frame_idx: torch.Tensor  # current frame index for each utterance (used to check if the decoding is finished)
    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')

    batched_hyps: BatchedBeamHyps  # batched hypotheses - decoding result

    # fusion models related fields
    fusion_models: Optional[List[NGramGPULanguageModel]] = None
    fusion_states_list: Optional[List[torch.Tensor]] = None
    fusion_states_candidates_list: Optional[List[torch.Tensor]] = None

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        max_time: int,
        vocab_size: int,
        device: torch.device,
        float_dtype: torch.dtype,
        blank_index: int,
    ):
        """
        Args:
            batch_size: batch size for encoder output storage
            beam_size: beam size for decoder output storage
            max_time: maximum time for encoder output storage
            vocab_size: vocabulary size of the model including blank
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
        self.vocab_size = vocab_size

        self.NON_EXISTENT_LABEL = torch.tensor(NON_EXISTENT_LABEL_VALUE, device=self.device, dtype=torch.long)
        self.BLANK_TENSOR = torch.tensor(self.blank_index, device=self.device, dtype=torch.long)
        self.INACTIVE_SCORE = torch.tensor(INACTIVE_SCORE, device=self.device, dtype=float_dtype)

        self.decoder_outputs = torch.zeros(
            (self.batch_size, self.max_time, self.vocab_size),
            dtype=float_dtype,
            device=self.device,
        )
        self.decoder_output_lengths = torch.zeros(
            (self.batch_size, self.beam_size), dtype=torch.long, device=self.device
        )
        self.last_timesteps = torch.zeros((self.batch_size, self.beam_size), dtype=torch.long, device=self.device)

        self.vocab = torch.arange(self.vocab_size, device=self.device, dtype=torch.long)
        self.vocab_blank_mask = torch.eq(self.vocab, self.blank_index)

        self.curr_frame_idx = torch.zeros([self.beam_size], device=self.device, dtype=torch.long)
        self.active_mask = torch.zeros((batch_size, self.beam_size), device=self.device, dtype=torch.bool)
        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self.blank_index,
            init_length=max_time + 1,
            device=device,
            float_dtype=float_dtype,
            model_type='ctc',
        )

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


@dataclass
class SeparateGraphsBatchedBeamCTC:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    _before_process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    _process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    _after_process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


class BatchedBeamCTCComputer(WithOptionalCudaGraphs, ConfidenceMethodMixin):
    """
    Batched beam search implementation for CTC models.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_beam_batch_conditional_ctc.cu"

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        NO_WHILE_LOOPS = "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    separate_graphs: Optional[SeparateGraphsBatchedBeamCTC]
    full_graph: Optional[torch.cuda.CUDAGraph]
    cuda_graphs_mode: Optional[CudaGraphsMode]
    state: Optional[BacthedBeamCTCState]
    fusion_models: Optional[List[NGramGPULanguageModel]]
    fusion_models_alpha: Optional[List[float]]

    def __init__(
        self,
        blank_index: int,
        beam_size: int,
        return_best_hypothesis: bool = True,
        preserve_alignments=False,
        compute_timestamps: bool = False,
        fusion_models: List[NGramGPULanguageModel] = None,
        fusion_models_alpha: List[float] = None,
        beam_beta: float = 0.0,
        beam_threshold: float = 20.0,
        allow_cuda_graphs: bool = True,
    ):
        """
        Init method.
        Args:
            blank_index: index of blank symbol.
            beam_size: beam size.
            return_best_hypothesis: whether to return the best hypothesis or N-best hypotheses.
            preserve_alignments: if alignments are needed. Defaults to False.
            compute_timestamps: if timestamps are needed. Defaults to False.
            fusion_models: list of fusion models.
            fusion_models_alpha: list of weights for the fusion models.
            beam_beta: word insertion weight.
            beam_threshold: threshold for pruning candidates.
            allow_cuda_graphs: whether to allow CUDA graphs. Defaults to True.
        """

        super().__init__()
        self._blank_index = blank_index

        self.beam_size = beam_size
        self.preserve_alignments = preserve_alignments
        self.compute_timestamps = compute_timestamps
        self.allow_cuda_graphs = allow_cuda_graphs
        self.return_best_hypothesis = return_best_hypothesis

        self.beam_beta = beam_beta
        self.beam_threshold = beam_threshold

        assert not self.preserve_alignments, "Preserve aligments is not supported"

        self.state = None
        self.full_graph = None
        self.separate_graphs = None

        self.cuda_graphs_mode = None
        self.cuda_graphs_allow_fallback = True
        self.maybe_enable_cuda_graphs()

        self.fusion_models = fusion_models
        self.fusion_models_alpha = fusion_models_alpha

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
            # check while loops
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
                self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
            except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
                logging.warning(
                    "No conditional node support for Cuda.\n"
                    "Cuda graphs with while loops are disabled, decoding speed will be slower\n"
                    f"Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_GRAPHS
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

    @torch.no_grad()
    def batched_beam_search_torch(
        self, decoder_outputs: torch.Tensor, decoder_output_lengths: torch.Tensor
    ) -> BatchedBeamHyps:
        """
        Pure PyTorch implementation of the batched beam search algorithm.

        Args:
            decoder_outputs (torch.Tensor): Tensor of shape [B, T, V+1], where B is the batch size,
                T is the maximum sequence length, and V is the vocabulary size. The tensor contains log-probabilities.
            decoder_output_lengths (torch.Tensor): Tensor of shape [B], contains lengths of each sequence in the batch.
        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """

        curr_batch_size, curr_max_time, vocab_size = decoder_outputs.shape

        vocab = torch.arange(vocab_size, device=decoder_outputs.device, dtype=torch.long)
        vocab_blank_mask = vocab == self._blank_index

        batched_beam_hyps = BatchedBeamHyps(
            batch_size=curr_batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=curr_max_time + 1,
            device=decoder_outputs.device,
            float_dtype=decoder_outputs.dtype,
            model_type='ctc',
        )

        # init fusion models
        if self.fusion_models is not None:
            fusion_states_list = []
            fusion_states_candidates_list = []
            for fusion_model in self.fusion_models:
                fusion_model.to(decoder_outputs.device)
                fusion_states_list.append(
                    fusion_model.get_init_states(batch_size=curr_batch_size * self.beam_size, bos=True)
                )
                fusion_states_candidates_list.append(None)

        for frame_idx in range(curr_max_time):
            active_mask = frame_idx < decoder_output_lengths.unsqueeze(1)
            repeated_mask = batched_beam_hyps.last_label[:, :, None] == vocab[None, None, :]
            repeated_or_blank_mask = repeated_mask | vocab_blank_mask[None, None, :]

            # step 2.1: getting the log probs and updating with fusion scores
            log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1).repeat(1, self.beam_size, 1)
            log_probs += batched_beam_hyps.scores.unsqueeze(-1)

            # step 2.2: updating non-blank and non-repeating token scores with `beam_beta`
            log_probs = torch.where(repeated_or_blank_mask, log_probs, log_probs + self.beam_beta)

            if self.fusion_models is not None:
                for fusion_idx, fusion_model in enumerate(self.fusion_models):
                    fusion_scores, fusion_states_candidates = fusion_model.advance(
                        states=fusion_states_list[fusion_idx].view(-1)
                    )
                    fusion_scores = torch.where(
                        repeated_mask[..., :-1], 0, fusion_scores.view(curr_batch_size, self.beam_size, -1)
                    )
                    log_probs[..., :-1] += self.fusion_models_alpha[fusion_idx] * fusion_scores.view(
                        curr_batch_size, self.beam_size, -1
                    )
                    fusion_states_candidates_list[fusion_idx] = fusion_states_candidates

            # step 2.3: getting `beam_size` best candidates
            next_scores, next_candidates_indices = torch.topk(
                log_probs.view(curr_batch_size, -1), k=self.beam_size, largest=True, sorted=True
            )
            next_indices = next_candidates_indices // vocab_size
            next_labels = next_candidates_indices % vocab_size

            # step 2.3: pruning candidates with threshold `beam_threshold`
            batch_next_scores = next_scores.view(curr_batch_size, -1)
            max_next_score = batch_next_scores.max(dim=-1, keepdim=True).values
            batch_next_scores.masked_fill_(batch_next_scores <= max_next_score - self.beam_threshold, INACTIVE_SCORE)
            next_scores.view(curr_batch_size, self.beam_size, -1)

            # step 2.4: preserving updated fusion states
            if self.fusion_models is not None:
                last_labels = torch.gather(batched_beam_hyps.last_label, dim=-1, index=next_indices)
                blank_mask = next_labels == self._blank_index
                repeating_mask = next_labels == last_labels
                preserve_state_mask = repeating_mask | blank_mask | ~active_mask

                # step 2.4.1: masking blanks and inactive labels to pass to fusion models, as fusion models do not support blanks
                next_labels_masked = torch.where(blank_mask, 0, next_labels)

                # step 2.4.2: gathering fusion states of extended hypotheses
                # batch_fusion_states: [(BxBeam)]
                # batch_fusion_states_candidates: [(BxBeam) x V (without blank)]

                for fusion_idx, fusion_model in enumerate(self.fusion_models):
                    next_indices_extended = next_indices[:, :, None].expand(
                        curr_batch_size, self.beam_size, fusion_states_candidates_list[fusion_idx].shape[-1]
                    )
                    fusion_states_candidates = fusion_states_candidates_list[fusion_idx].view(
                        curr_batch_size, self.beam_size, -1
                    )
                    fusion_states_candidates = torch.gather(
                        fusion_states_candidates, dim=1, index=next_indices_extended
                    )
                    fusion_states_prev = torch.gather(
                        fusion_states_list[fusion_idx].view(curr_batch_size, self.beam_size), dim=1, index=next_indices
                    )
                    fusion_states = torch.gather(
                        fusion_states_candidates, dim=-1, index=next_labels_masked.unsqueeze(-1)
                    ).squeeze(-1)

                    fusion_states_list[fusion_idx] = torch.where(
                        preserve_state_mask, fusion_states_prev, fusion_states
                    ).view(-1)

            # step 2.5: masking inactive hypotheses, updating + recombining batched beam hypoteses
            next_labels = torch.where(active_mask, next_labels, NON_EXISTENT_LABEL_VALUE)
            batched_beam_hyps.add_results_(next_indices, next_labels, next_scores)
            batched_beam_hyps.recombine_hyps_()

        # step 3: updating fusoin scores with eos scores
        if self.fusion_models is not None:
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                # only GPUBoostingTreeModel does not support eos scores for CTC models by default
                if not isinstance(fusion_model, GPUBoostingTreeModel):
                    eos_score = fusion_model.get_final(fusion_states_list[fusion_idx]).view(
                        batched_beam_hyps.scores.shape
                    )
                    batched_beam_hyps.scores += eos_score * self.fusion_models_alpha[fusion_idx]

        return batched_beam_hyps

    def batched_beam_search_cuda_graphs(
        self,
        decoder_outputs: torch.Tensor,
        decoder_output_lengths: torch.Tensor,
    ) -> BatchedBeamHyps:
        """
        Cuda-Graphs implementation of the batched beam search algorithm.

        Args:
            decoder_outputs (torch.Tensor): Tensor of shape [B, T, V+1], where B is the batch size,
                T is the maximum sequence length, and V is the vocabulary size. The tensor contains log-probabilities.
            decoder_output_lengths (torch.Tensor): Tensor of shape [B], contains lengths of each sequence in the batch.
        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """

        assert self.cuda_graphs_mode is not None

        curr_batch_size, curr_max_time, _ = decoder_outputs.shape

        if torch.is_autocast_enabled():
            decoder_outputs = decoder_outputs.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self.state is None or self.state.need_reinit(decoder_outputs):
            self._graph_reinitialize(decoder_outputs, decoder_output_lengths)

        # set length to zero for elements outside the current batch
        self.state.decoder_output_lengths.fill_(0)
        # copy (projected) encoder output and lenghts
        self.state.decoder_outputs[:curr_batch_size, :curr_max_time, ...].copy_(decoder_outputs)
        self.state.decoder_output_lengths[:curr_batch_size].copy_(decoder_output_lengths.unsqueeze(-1))
        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs._before_process_batch.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs._process_batch.replay()
            self.separate_graphs._after_process_batch.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            self._before_process_batch()
            while self.state.active_mask_any.item():
                self._process_batch()
            self._after_process_batch()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        return self.state.batched_hyps

    @classmethod
    def _create_process_batch_kernel(cls):
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
        decoder_outputs: torch.Tensor,
        decoder_output_lengths: torch.Tensor,
    ):
        """
        Reinitializes the graph state for the Beam Search computation.
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

        batch_size, max_time, vocab_size = decoder_outputs.shape

        self.state = BacthedBeamCTCState(
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            vocab_size=vocab_size,
            device=decoder_outputs.device,
            float_dtype=decoder_outputs.dtype,
            blank_index=self._blank_index,
        )

        # init fusion models
        if self.fusion_models is not None:
            self.state.fusion_states_list = []
            self.state.fusion_states_candidates_list = []
            for fusion_model in self.fusion_models:
                fusion_model.to(decoder_outputs.device)
                self.state.fusion_states_list.append(
                    fusion_model.get_init_states(batch_size=batch_size * self.beam_size, bos=True).view(
                        batch_size, self.beam_size
                    )
                )
                self.state.fusion_states_candidates_list.append(
                    torch.zeros([batch_size, fusion_model.vocab_size], dtype=torch.long, device=self.state.device)
                )

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

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsBatchedBeamCTC()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._before_process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_process_batch()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._process_batch()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._after_process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._after_process_batch()

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
            self._before_process_batch()
            # NB: depending on cuda-python version, cudaStreamGetCaptureInfo can return either 5 or 6 elements
            capture_status, _, graph, *_ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )

            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # capture: while self.active_mask_any:
            (loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            loop_kernel = self._create_process_batch_kernel()
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            loop_args = np.array(
                [loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )
            # loop while there are active utterances
            with with_conditional_node(loop_kernel, loop_args, loop_conditional_handle, device=self.state.device):
                self._process_batch()

            self._after_process_batch()

    def _before_process_batch(self):
        """
        Clears state and setups fusion models.
        """
        # step 1.1: reset state
        self.state.batched_hyps.clear_()
        self.state.curr_frame_idx.fill_(0)

        # maximum time step for each utterance
        torch.sub(self.state.decoder_output_lengths, 1, out=self.state.last_timesteps)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.decoder_output_lengths, 0, out=self.state.active_mask)

        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

        # step 1.2: setup fusion models
        if self.fusion_models is not None:
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                fusion_model.to(self.state.device)
                fusion_states = fusion_model.get_init_states(
                    batch_size=self.state.batch_size * self.beam_size, bos=True
                )
                # self.state.fusion_states_list[fusion_idx].copy_(fusion_states.view(self.state.batch_size, self.beam_size))
                self.state.fusion_states_list[fusion_idx].copy_(
                    fusion_states.view(self.state.batch_size, self.beam_size)
                )
                self.state.fusion_states_candidates_list[fusion_idx] = torch.empty(
                    (self.state.batch_size, self.state.beam_size, fusion_model.vocab_size),
                    device=self.state.device,
                    dtype=torch.long,
                )

    def _process_batch(self):
        """
        Performs a decoding step.
        """
        repeated_mask = self.state.batched_hyps.last_label[:, :, None] == self.state.vocab[None, None, :]
        repeated_or_blank_mask = repeated_mask | self.state.vocab_blank_mask[None, None, :]

        # step 2.1: getting the log probs and updating with fusion scores
        log_probs = self.state.decoder_outputs.index_select(dim=1, index=self.state.curr_frame_idx)
        log_probs += self.state.batched_hyps.scores[:, :, None]

        # step 2.2: updating non-blank and non-repeating token scores with `beam_beta`
        log_probs = torch.where(repeated_or_blank_mask, log_probs, log_probs + self.beam_beta)

        if self.fusion_models is not None:
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                fusion_scores, fusion_states_candidates = fusion_model.advance(
                    states=self.state.fusion_states_list[fusion_idx].view(-1)
                )
                fusion_scores = torch.where(
                    repeated_mask[..., :-1], 0, fusion_scores.view(log_probs.shape[0], self.beam_size, -1)
                )
                log_probs[..., :-1] += self.fusion_models_alpha[fusion_idx] * fusion_scores.view(
                    log_probs.shape[0], self.beam_size, -1
                )
                self.state.fusion_states_candidates_list[fusion_idx].copy_(
                    fusion_states_candidates.view(self.state.batch_size, self.beam_size, -1)
                )

        # step 2.3: getting `beam_size` best candidates
        next_scores, next_candidates_indices = torch.topk(
            log_probs.view(self.state.batch_size, -1), k=self.beam_size, largest=True, sorted=True
        )
        next_indices = next_candidates_indices // self.state.vocab_size
        next_labels = next_candidates_indices % self.state.vocab_size

        # step 2.3: pruning candidates with threshold `beam_threshold`
        batch_next_scores = next_scores.view(self.state.batch_size, -1)
        max_next_score = batch_next_scores.max(dim=-1, keepdim=True).values
        batch_next_scores.masked_fill_(batch_next_scores <= max_next_score - self.beam_threshold, INACTIVE_SCORE)
        next_scores.view(self.state.batch_size, self.beam_size, -1)

        # step 2.4: preserving updated fusion states
        if self.fusion_models is not None:
            last_labels = torch.gather(self.state.batched_hyps.last_label, dim=-1, index=next_indices)
            blank_mask = next_labels == self._blank_index
            repeating_mask = next_labels == last_labels
            preserve_state_mask = repeating_mask | blank_mask | ~self.state.active_mask

            # step 2.4.1: masking blanks and inactive labels to pass to fusion model, as fusion model does not support blanks
            next_labels_masked = torch.where(blank_mask, 0, next_labels)

            # step 2.4.2: gathering fusion states of extended hypotheses
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                # fusion_states: [(BxBeam)]
                # fusion_states_candidates: [(BxBeam) x V (without blank)]
                next_indices_extended = next_indices[:, :, None].expand(
                    self.state.fusion_states_candidates_list[fusion_idx].shape
                )
                fusion_states_candidates = torch.gather(
                    self.state.fusion_states_candidates_list[fusion_idx], dim=1, index=next_indices_extended
                )
                fusion_states_prev = torch.gather(self.state.fusion_states_list[fusion_idx], dim=1, index=next_indices)
                fusion_states = torch.gather(
                    fusion_states_candidates, dim=-1, index=next_labels_masked.unsqueeze(-1)
                ).squeeze()

                # step 2.4.3: update fusion states in State
                self.state.fusion_states_candidates_list[fusion_idx].copy_(fusion_states_candidates)
                torch.where(
                    preserve_state_mask,
                    fusion_states_prev,
                    fusion_states,
                    out=self.state.fusion_states_list[fusion_idx],
                )

        # step 2.5: masking inactive hypotheses, updating + recombining batched beam hypoteses
        torch.where(self.state.active_mask, next_labels, self.state.NON_EXISTENT_LABEL, out=next_labels)
        self.state.batched_hyps.add_results_no_checks_(next_indices, next_labels, next_scores)
        self.state.batched_hyps.recombine_hyps_()

        # step 2.6: updating frame idx and active masks
        self.state.curr_frame_idx.add_(1)
        torch.greater_equal(self.state.last_timesteps, self.state.curr_frame_idx, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _after_process_batch(self):
        """
        Finalizes the decoding process by updating the fusion scores with the end-of-sequence (eos) scores.
        """

        # step 3: updating fusion scores with eos scores
        if self.fusion_models is not None:
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                if not isinstance(fusion_model, GPUBoostingTreeModel):
                    eos_score = fusion_model.get_final(self.state.fusion_states_list[fusion_idx]).view(
                        self.state.batched_hyps.scores.shape
                    )
                    self.state.batched_hyps.scores += eos_score * self.fusion_models_alpha[fusion_idx]

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> BatchedBeamHyps:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            return self.batched_beam_search_cuda_graphs(decoder_outputs=x, decoder_output_lengths=out_len)

        return self.batched_beam_search_torch(decoder_outputs=x, decoder_output_lengths=out_len)
