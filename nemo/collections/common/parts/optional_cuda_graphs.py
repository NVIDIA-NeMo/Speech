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

import abc
from typing import Optional

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
