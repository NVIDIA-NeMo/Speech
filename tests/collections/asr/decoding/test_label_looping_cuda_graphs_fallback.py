# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Regression test: maybe_enable_cuda_graphs() must fall back gracefully, not
crash, when cuda-python is importable but there is no real NVIDIA driver to
talk to (e.g. ROCm/AMD builds of torch, or CPU-only boxes with cuda-python
installed transitively).
"""
from unittest.mock import patch

import pytest

from nemo.collections.asr.parts.submodules.transducer_decoding import label_looping_base
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    GreedyBatchedLabelLoopingComputerBase,
)


class _MinimalLabelLoopingComputer(GreedyBatchedLabelLoopingComputerBase):
    """Concrete stub implementing only what's needed to exercise
    maybe_enable_cuda_graphs() in isolation, without any real ASR model."""

    def __init__(self, allow_cuda_graphs: bool, max_symbols: int = 10):
        self.allow_cuda_graphs = allow_cuda_graphs
        self.max_symbols = max_symbols
        self.cuda_graphs_mode = None
        self.cuda_graphs_allow_fallback = False

    def reset_cuda_graphs_state(self):
        pass

    def torch_impl(self, *args, **kwargs):
        raise NotImplementedError

    def cuda_graphs_impl(self, *args, **kwargs):
        raise NotImplementedError

    def split_batched_state(self, state):
        raise NotImplementedError

    def merge_to_batched_state(self, state_items):
        raise NotImplementedError


class TestMaybeEnableCudaGraphsFallback:
    def test_falls_back_when_conditional_node_check_raises_runtime_error(self):
        # cuda-python's native extension raises a raw RuntimeError (not
        # ImportError/ModuleNotFoundError/EnvironmentError) when
        # cuDriverGetVersion() can't dlopen libcuda.so.1 -- e.g. ROCm/AMD
        # builds of torch, where torch.cuda.is_available() is True (HIP
        # shims the same API) but there is no real NVIDIA driver present.
        computer = _MinimalLabelLoopingComputer(allow_cuda_graphs=True)

        with patch.object(
            label_looping_base,
            "check_cuda_python_cuda_graphs_conditional_nodes_supported",
            side_effect=RuntimeError("Failed to dlopen libcuda.so.1"),
        ):
            computer.maybe_enable_cuda_graphs()

        assert computer.cuda_graphs_mode == computer.CudaGraphsMode.NO_WHILE_LOOPS

    def test_still_falls_back_on_the_previously_handled_error_types(self):
        computer = _MinimalLabelLoopingComputer(allow_cuda_graphs=True)

        with patch.object(
            label_looping_base,
            "check_cuda_python_cuda_graphs_conditional_nodes_supported",
            side_effect=EnvironmentError("CUDA is not available"),
        ):
            computer.maybe_enable_cuda_graphs()

        assert computer.cuda_graphs_mode == computer.CudaGraphsMode.NO_WHILE_LOOPS

    def test_enables_full_graph_mode_when_supported(self):
        computer = _MinimalLabelLoopingComputer(allow_cuda_graphs=True)

        with patch.object(
            label_looping_base,
            "check_cuda_python_cuda_graphs_conditional_nodes_supported",
            return_value=None,
        ):
            computer.maybe_enable_cuda_graphs()

        assert computer.cuda_graphs_mode == computer.CudaGraphsMode.FULL_GRAPH
