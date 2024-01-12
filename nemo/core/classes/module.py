# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager

import torch
from torch.nn import Module

from nemo.core.classes.common import FileIO, Serialization, Typing

__all__ = ['NeuralModule']


class NeuralModule(Module, Typing, Serialization, FileIO):
    """
    Abstract class offering interface shared between all PyTorch Neural Modules.
    """

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of NeuralModule.
        """
        return self._num_weights()

    @torch.jit.ignore
    def _num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    def input_example(self, max_batch=None, max_dim=None):
        """
        Override this method if random inputs won't work
        Returns:
            A tuple sample of valid input data.
        """

        return None

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally unfreezes the module.
        """
        training_mode = self.training
        grad_map = {}
        for pname, param in self.named_parameters():
            grad_map[pname] = param.requires_grad

        self.freeze()
        try:
            yield
        finally:
            self.unfreeze()

            for pname, param in self.named_parameters():
                param.requires_grad = grad_map[pname]

            if training_mode:
                self.train()
            else:
                self.eval()
