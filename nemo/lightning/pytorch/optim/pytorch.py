# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable, List, Optional

import lightning.pytorch as pl
import lightning.pytorch as L
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.optim.base import LRSchedulerModule, OptimizerModule


def _param_does_not_have_wd(param_name, param):
    return 'bias' in param_name


def _extract_model_params_for_optim(model, weight_decay=0, no_weight_decay_cond=None):
    params_with_wd, params_without_wd = [], []
    if no_weight_decay_cond is not None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if no_weight_decay_cond(name, param):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
    else:
        params_with_wd = list(filter(lambda x: x.requires_grad, model.parameters()))

    assert max(map(len, (params_with_wd, params_without_wd))) > 0, "Expected at least one optimizer with params"

    return [
        {'params': params, 'weight_decay': wd}
        for params, wd in zip((params_with_wd, params_without_wd), (weight_decay, 0))
    ]


class PytorchOptimizerModule(OptimizerModule):
    """A OptimizerModule for pytorch optimizers.

    Attributes:
        optimizer_fn (Callable[[ParamsT], Optimizer]): Configuration for the optimizer.
        no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
        scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
        lr_mult (float): Learning rate multiplier.

    Example::

        optimizer_fn = run.Partial(
            SGD,
            lr=lr,
            weight_decay=wd,
        )
        lr_scheduler = MyLRSchedulerModule(...)
        optimizer_module = PytorchOptimizerModule(optimizer_fn, lr_scheduler)

    Methods:
        setup(model): Sets up the optimizer.
        optimizers(model): Defines the optimizers.
    """

    def __init__(
        self,
        optimizer_fn: Callable[[ParamsT], Optimizer],
        lr_scheduler: Optional[LRSchedulerModule] = None,
        no_weight_decay_cond: Optional[Callable] = _param_does_not_have_wd,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ):
        """Initializes the PytorchOptimizerModule.

        Args:
            optimizer_fn (Callable[[ParamsT], Optimizer]): Configuration for the optimizer.
            lr_scheduler (Optional[LRSchedulerModule]): The learning rate scheduler module.
            no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
            scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
            lr_mult (float): Learning rate multiplier.
        """

        super().__init__(lr_scheduler=lr_scheduler)
        self.optimizer_fn = optimizer_fn
        self.no_weight_decay_cond = no_weight_decay_cond
        self.scale_lr_cond = scale_lr_cond
        self.lr_mult = lr_mult

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """nooop"""
        # Noop
        pass

    def optimizers(self, model) -> List[Optimizer]:
        """Defines the optimizers.

        Args:
            model (nn.Module): The model for which the optimizers are being defined.

        Returns:
            List[Optimizer]: The list of optimizers.

        Raises:
            ValueError: If the model is an instance of MegatronParallel.
        """

        if isinstance(model, MegatronParallel):
            raise ValueError("Model cannot be an instance of MegatronParallel")

        wd = self.optimizer_fn.keywords.get('weight_decay', 0)
        optim = self.optimizer_fn(_extract_model_params_for_optim(model, wd, self.no_weight_decay_cond))
        self._optimizers = optim
        if not isinstance(optim, list):
            optim = [optim]
        if self.lr_scheduler is None:
            return optim
        else:
            return [self.lr_scheduler.scheduler(model, opt) for opt in optim]

    def connect(self, model: L.LightningModule) -> None:
        """Connects the optimizer module to the model.

        Args:
            model (L.LightningModule): The model to which the optimizer module is being connected.
        """
        model.configure_optimizers = lambda: self.optimizers(model)
