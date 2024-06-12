from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.utils import get_model_config
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer

if TYPE_CHECKING:
    from nemo.lightning.megatron_parallel import MegatronParallel


@dataclass
class MegatronOptim:
    config: OptimizerConfig
    finalize_model_grads: Callable = finalize_model_grads

    def create_optimizer(
        self,
        megatron_parallel: "MegatronParallel",
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ) -> Optimizer:
        from nemo.core.optim import McoreDistributedOptimizer

        # TODO: Where should we put this?
        get_model_config(megatron_parallel[0]).finalize_model_grads = finalize_model_grads

        mcore_opt = get_megatron_optimizer(
            self.config,
            list(megatron_parallel),
            no_weight_decay_cond=no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
        )

        return McoreDistributedOptimizer(mcore_opt)

    def configure_optimizer(self, megatron_parallel: "MegatronParallel") -> OptimizerLRScheduler:
        from nemo.core.optim.lr_scheduler import CosineAnnealing

        opt = self.create_optimizer(megatron_parallel)

        # TODO: Make this configurable through the dataclass
        lr_scheduler = CosineAnnealing(opt, max_steps=10, warmup_steps=750, constant_steps=80000, min_lr=int(6e-5))

        return {
            "optimizer": opt,
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
        }

    def __call__(self, megatron_parallel: "MegatronParallel") -> OptimizerLRScheduler:
        return self.configure_optimizer(megatron_parallel)
