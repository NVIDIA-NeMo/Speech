# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import abc
import gc
import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from nemo.utils import logging


class WithOptionalCudaGraphs(abc.ABC):
    """
    Abstract interface for modules with CUDA graphs.
    Allows to enable/disable CUDA graphs on the fly.
    """

    @classmethod
    def disable_cuda_graphs_recursive(cls, module: nn.Module, attribute_path: Optional[str] = None):
        """
        Disable CUDA graphs Enable CUDA graphs, finding submodule recursively.

        Args:
            module: instance of nn.Module
            attribute_path: field containing instance of WithOptionalCudaGraphs
                   E.g., "decoding.decoding" means that "<module>.decoding.decoding" are checked.
                   If None, "<module>" is checked.
        """
        attributes = attribute_path.split(".") if attribute_path else []

        for name, submodule in module.named_modules():
            object_to_check = submodule
            try:
                # recursively get attribute by iterating attribute_path
                for attribute in attributes:
                    object_to_check = getattr(object_to_check, attribute)
            except AttributeError:
                continue  # loop over modules, no attribute

            if isinstance(object_to_check, cls):
                state_changed = object_to_check.disable_cuda_graphs()
                if state_changed:
                    logging.info(f"Disabled CUDA graphs for module {type(submodule)}" + ".".join([name] + attributes))

    @classmethod
    def enable_cuda_graphs_recursive(cls, module: nn.Module, attribute_path: Optional[str] = None):
        """
        Enable CUDA graphs, finding submodule recursively

        Args:
            module: instance of nn.Module
            attribute_path: field containing instance of WithOptionalCudaGraphs
                   E.g., "decoding.decoding" means that "<module>.decoding.decoding" are checked.
                   If None, "<module>" is checked.
        """
        attributes = attribute_path.split(".") if attribute_path else []

        for name, submodule in module.named_modules():
            object_to_check = submodule
            try:
                # recursively get attribute by iterating attribute_path
                for attribute in attributes:
                    object_to_check = getattr(object_to_check, attribute)
            except AttributeError:
                continue  # loop over modules, no attribute

            if isinstance(object_to_check, cls):
                state_changed = object_to_check.maybe_enable_cuda_graphs()
                if state_changed:
                    logging.info(f"Enabled CUDA graphs for module {type(submodule)}" + ".".join([name] + attributes))

    @abc.abstractmethod
    def disable_cuda_graphs(self) -> bool:
        """Disable (maybe temporary) CUDA graphs, return True if state changed"""
        raise NotImplementedError

    @abc.abstractmethod
    def maybe_enable_cuda_graphs(self) -> bool:
        """Enable CUDA graphs if all conditions met, return True if state changed"""
        raise NotImplementedError


def capture_cuda_graph(fn: Callable[[], Any], pool=None) -> tuple[torch.cuda.CUDAGraph, Any]:
    """
    Capture ``fn()`` into a CUDA graph and return the graph with the outputs of the captured call.

    ``fn`` first runs once eagerly on a side stream, so lazy initialization (compilation, autotuning, library
    workspaces) happens outside the capture, and garbage collection is paused during the capture. Every replay of the
    graph rewrites the returned outputs in place.

    Args:
        fn: Callable to capture; it reads its inputs from tensors that callers update before each replay.
        pool: Optional memory pool handle shared with other graphs (``torch.cuda.graph_pool_handle()``).
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    # A garbage collection during the capture can destroy unreachable CUDAGraph objects, and destroying a graph while
    # another one is being captured invalidates the capture.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, pool=pool):
            outputs = fn()
    finally:
        if gc_was_enabled:
            gc.enable()
    return graph, outputs


class SharedGraphBuffers:
    """
    Static input and output buffers shared by CUDA graphs that are never replayed concurrently.

    Every graph reads an input from, or writes an output to, a view of the first elements of one flat buffer per
    ``(slot, dtype, device)``, so a set of graphs needs the memory of its largest tensor per slot instead of the sum.
    A graph that copies its results into such views inside the capture also lets later captures reuse the pool memory
    of its intermediate outputs. A replay's inputs must be copied in right before it, and its outputs read before the
    next replay. A request larger than the buffer allocates a new one; graphs captured earlier keep the old one
    alive, so capturing the largest graphs first allocates once.
    """

    def __init__(self):
        self._buffers = {}

    def view(self, slot: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Return a contiguous ``shape`` view of the shared buffer for ``slot``, ``dtype`` and ``device``.

        Args:
            slot: Name of the tensor's role; tensors one graph uses at the same time need distinct slots.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
            device: Device of the tensor.
        """
        numel = math.prod(shape)
        key = (slot, dtype, torch.device(device))
        buffer = self._buffers.get(key)
        if buffer is None or buffer.numel() < numel:
            buffer = torch.zeros(numel, dtype=dtype, device=device)
            self._buffers[key] = buffer
        return buffer[:numel].view(shape)

    def store(self, slot: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Copy ``tensor`` into the shared buffer for ``slot`` and return the copy.

        Inside a capture, this moves a graph output out of the graph's memory pool, so later captures can reuse the
        pool memory it came from.
        """
        return self.view(slot, tuple(tensor.shape), tensor.dtype, tensor.device).copy_(tensor)


def copy_padded_rows(static: torch.Tensor, rows: torch.Tensor) -> None:
    """
    Copy ``rows`` into the first rows of a graph's static input and fill the remaining rows with copies of the first.

    Padding with a real row keeps every row of the graph on its usual numerical path; callers drop the padded rows'
    outputs.
    """
    n = rows.shape[0]
    static[:n].copy_(rows)
    static[n:].copy_(rows[:1].expand_as(static[n:]))
