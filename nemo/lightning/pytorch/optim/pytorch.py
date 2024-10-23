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

import pytorch_lightning as pl
from torch.optim import Optimizer

from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.optim.base import LRSchedulerModule, OptimizerModule


def _param_does_not_have_wd(param_name, param):
    return 'bias' in param_name


class PytorchOptimizerModule(OptimizerModule):
    """A OptimizerModule for pytorch optimizers.

    Attributes:
        config (OptimizerConfig): Configuration for the optimizer.
        no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
        scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
        lr_mult (float): Learning rate multiplier.

    Example::

        config = OptimizerConfig(...)
        lr_scheduler = MyLRSchedulerModule(...)
        optimizer_module = PytorchOptimizerModule(config, lr_scheduler)

    Methods:
        setup(model): Sets up the optimizer.
        optimizers(model): Defines the optimizers.
    """

    def __init__(
        self,
        optim_cls,
        config: dict = {'lr': 3e-4},
        lr_scheduler: Optional[LRSchedulerModule] = None,
        no_weight_decay_cond: Optional[Callable] = _param_does_not_have_wd,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ):
        """Initializes the PytorchOptimizerModule.

        Args:
            config (OptimizerConfig): Configuration for the optimizer.
            lr_scheduler (Optional[LRSchedulerModule]): The learning rate scheduler module.
            no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
            scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
            lr_mult (float): Learning rate multiplier.
        """

        super().__init__(lr_scheduler=lr_scheduler)
        self.optim_cls = optim_cls
        self.config = config
        self.no_weight_decay_cond = no_weight_decay_cond
        self.scale_lr_cond = scale_lr_cond
        self.lr_mult = lr_mult
        self.optim_cls = optim_cls

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
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

        params_with_wd, params_without_wd = [], []
        if self.no_weight_decay_cond is not None:
            for name, param in model.named_parameters():
                if self.no_weight_decay_cond(name, param):
                    params_without_wd.append(param)
                else:
                    params_with_wd.append(param)
        else:
            params_with_wd = model.parameters()

        optimizers = []
        if len(params_with_wd) > 0:
            optimizers.append(
                self.optim_cls(
                    params_with_wd,
                    **self.config,
                )
            )

        if len(params_without_wd) > 0:
            wd = self.config.get('weight_decay', None)
            kwargs['weight_decay'] = 0
            optimizers.append(
                self.optim_cls(
                    params_without_wd,
                    **kwargs,
                )
            )
            # restore value
            if wd is not None:
                kwargs['weight_decay'] = wd

        assert len(optimizers) > 0, "Expected at least one optimizer with params"
        return optimizers

    def finalize_model_grads(self, *args, **kwargs):
        # Noop
        pass
