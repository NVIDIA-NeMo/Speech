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
from nemo.utils import logging

__all__ = ['NeuralModule', 'freeze', 'unfreeze']


def freeze(module: Module) -> None:
    """Freeze all parameters of ``module`` and snapshot their prior ``requires_grad`` state.

    The snapshot is stored on ``module._frozen_grad_map`` so a later call to ``unfreeze(..., partial=True)``
    can restore the pre-freeze state instead of unconditionally enabling gradients.
    """
    grad_map = {pname: param.requires_grad for pname, param in module.named_parameters()}
    for param in module.parameters():
        param.requires_grad = False
    if not hasattr(module, '_frozen_grad_map'):
        module._frozen_grad_map = grad_map
    else:
        module._frozen_grad_map.update(grad_map)
    module.eval()


def unfreeze(module: Module, partial: bool = False) -> None:
    """Unfreeze parameters of ``module``.

    If ``partial=True``, restore each parameter's ``requires_grad`` from the snapshot recorded by
    ``freeze(module)``; otherwise enable gradients on every parameter. The snapshot is cleared in
    both cases and ``module.train()`` is called.
    """
    if partial and not hasattr(module, '_frozen_grad_map'):
        raise ValueError("Cannot unfreeze partially without first freezing the module with `freeze()`")

    for pname, param in module.named_parameters():
        if not partial:
            param.requires_grad = True
        elif pname in module._frozen_grad_map:
            param.requires_grad = module._frozen_grad_map[pname]
        else:
            logging.warning(
                f"Parameter {pname} not found in list of previously frozen parameters. Unfreezing this parameter."
            )
            param.requires_grad = True

    if hasattr(module, '_frozen_grad_map'):
        delattr(module, '_frozen_grad_map')

    module.train()


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
        r"""Freeze all params for inference. See :func:`freeze` for details."""
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        """Unfreeze parameters for training. See :func:`unfreeze` for details.

        Example:
            ```python
            model.encoder.freeze()              # caller freezes encoder
            model.freeze()                      # freezes everything; encoder snapshot preserved
            model.unfreeze(partial=True)        # decoder unfrozen, encoder stays frozen
            ```
        """
        unfreeze(self, partial=partial)

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally unfreezes the module partially
        to return to original state.

        Allows for either total unfreeze or partial unfreeze (if the module was explicitly frozen
        previously with `freeze()`). The `partial` argument is used to determine whether to unfreeze
        all parameters or only the parameters that were previously unfrozen prior `freeze()`.

        Example:
            with model.as_frozen():  # by default, partial = True
                # Do something with the model
                pass

            # Model's parameters are now back to original state of requires_grad
        """
        training_mode = self.training
        self.freeze()
        try:
            yield
        finally:
            self.unfreeze(partial=True)

            if training_mode:
                self.train()
            else:
                self.eval()
