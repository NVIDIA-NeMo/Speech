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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.utils.cuda_python_utils import check_cuda_python_cuda_graphs_conditional_nodes_supported
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum


@dataclass
class SeparateGraphsLabelLooping:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    before_outer_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    before_inner_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    inner_loop_code: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    after_inner_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


@dataclass
class BatchedLabelLoopingState:
    """Decoding state to pass between invocations"""

    predictor_states: Any
    predictor_outputs: torch.Tensor
    labels: torch.Tensor
    decoded_lengths: torch.Tensor
    fusion_states_list: list[torch.Tensor] | None = None
    time_jumps: torch.Tensor | None = None


@dataclass
class LabelLoopingStateItem:
    """Decoding state to pass between invocations"""

    predictor_state: Any
    predictor_output: torch.Tensor
    label: torch.Tensor
    decoded_length: torch.Tensor
    fusion_state_list: list[torch.Tensor] | None = None
    time_jump: torch.Tensor | None = None


class GreedyBatchedLabelLoopingComputerBase(WithOptionalCudaGraphs, ABC):
    """
    Base class for Label-Looping algorithm implementation https://arxiv.org/abs/2406.06220
    for optimized batched greedy decoding.
    Iterates over labels, on each step finding the next non-blank label
    (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
    to prediction network (with maximum possible batch size),
    which makes it especially useful for scaling the prediction network.
    During decoding all active hypotheses ("texts") have the same lengths.
    """

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        NO_WHILE_LOOPS = "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    cuda_graphs_mode: Optional[CudaGraphsMode] = None
    max_symbols: Optional[int]
    allow_cuda_graphs: bool

    def force_cuda_graphs_mode(self, mode: Optional[str | CudaGraphsMode]):
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

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        raise NotImplementedError

    @abstractmethod
    def split_batched_state(self, state: BatchedLabelLoopingState) -> list[LabelLoopingStateItem]:
        """
        Split batched state into list of items, each item contains state for one hypothesis.
        This is used to pass state between invocations of the algorithm.

        Args:
            state: batched decoding state
        """
        raise NotImplementedError

    @abstractmethod
    def merge_to_batched_state(self, state_items: list[LabelLoopingStateItem | None]) -> BatchedLabelLoopingState:
        """
        Merge list of items into batched state, each item contains state for one hypothesis.
        This is used to pass state between invocations of the algorithm.

        Args:
            state_items: list of items to merge
        """
        raise NotImplementedError

    def reset_state_by_mask(self, state: BatchedLabelLoopingState, mask: torch.Tensor) -> BatchedLabelLoopingState:
        """
        Reset state for masked elements in the batched state.
        This is used to reset state for elements that are not active anymore to start a new decoding session.

        Args:
            state: batched decoding state
            mask: mask for elements to reset
        """
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
    ) -> tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], BatchedLabelLoopingState]:
        """
        Entry point for the decoding algorithm

        Args:
            x: encoder output
            out_len: encoder output length
            prev_batched_state: previous batched decoding state
        """
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            # disable CUDA graphs if Mixed Precision is used due to incorrect behavior
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # TODO(vbataev): fix issue with mixed precision, remove this restriction
                return self.cuda_graphs_impl(
                    encoder_output=x, encoder_output_length=out_len, prev_batched_state=prev_batched_state
                )

        return self.torch_impl(encoder_output=x, encoder_output_length=out_len, prev_batched_state=prev_batched_state)
