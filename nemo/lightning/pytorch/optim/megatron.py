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

import inspect
from typing import Callable, List, Optional

import pytorch_lightning as pl
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig
from megatron.core.utils import get_model_config
from torch.optim import Optimizer

from nemo.lightning._strategy_lib import setup_megatron_optimizer
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.optim.base import LRSchedulerModule, OptimizerModule


class MegatronOptimizerModule(OptimizerModule):
    """A OptimizerModule for the megatron optimizers.

    Attributes:
        config (OptimizerConfig): Configuration for the optimizer.
        no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
        scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
        lr_mult (float): Learning rate multiplier.

    Example::

        config = OptimizerConfig(...)
        lr_scheduler = MyLRSchedulerModule(...)
        optimizer_module = MegatronOptimizerModule(config, lr_scheduler)

    Methods:
        setup(model): Sets up the optimizer.
        optimizers(model): Defines the optimizers.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        lr_scheduler: Optional[LRSchedulerModule] = None,
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ):
        """Initializes the MegatronOptimizerModule.

        Args:
            config (OptimizerConfig): Configuration for the optimizer.
            lr_scheduler (Optional[LRSchedulerModule]): The learning rate scheduler module.
            no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
            scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
            lr_mult (float): Learning rate multiplier.
        """

        super().__init__(lr_scheduler=lr_scheduler)
        self.config = config
        self.no_weight_decay_cond = no_weight_decay_cond
        self.scale_lr_cond = scale_lr_cond
        self.lr_mult = lr_mult

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """We will add the finalize_model_grads function to the model config.

        Args:
            model: The model for which the optimizer is being set up.
        """

        def finalize_model_grads_func(*args, **kwargs):
            return self.finalize_model_grads(*args, **kwargs)

        get_model_config(pl_module).finalize_model_grads_func = finalize_model_grads_func

    def optimizers(self, model: MegatronParallel) -> List[Optimizer]:
        """Defines the optimizers.

        Args:
            model (MegatronParallel): The model for which the optimizers are being defined.

        Returns:
            List[Optimizer]: The list of optimizers.

        Raises:
            ValueError: If the model is not an instance of MegatronParallel.
        """

        if not isinstance(model, MegatronParallel):
            raise ValueError("Model must be an instance of MegatronParallel")

        optimizer = setup_megatron_optimizer(
            model,
            self.config,
            no_weight_decay_cond=self.no_weight_decay_cond,
            scale_lr_cond=self.scale_lr_cond,
            lr_mult=self.lr_mult,
        )

        return [optimizer]

    def finalize_model_grads(self, *args, **kwargs):
        return finalize_model_grads(*args, **kwargs)
