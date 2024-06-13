from typing import Callable, List, Optional

from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.utils import get_model_config
from torch.optim import Optimizer

from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.opt.base import LRSchedulerModule, OptimizerModule


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

    def setup(self, model):
        """We will add the finalize_model_grads function to the model config.

        Args:
            model: The model for which the optimizer is being set up.
        """

        def finalize_model_grads_func(*args, **kwargs):
            return self.finalize_model_grads(*args, **kwargs)

        get_model_config(model[0]).finalize_model_grads_func = finalize_model_grads_func

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

        from nemo.core.optim import McoreDistributedOptimizer

        mcore_opt = get_megatron_optimizer(
            self.config,
            list(model),
            no_weight_decay_cond=self.no_weight_decay_cond,
            scale_lr_cond=self.scale_lr_cond,
            lr_mult=self.lr_mult,
        )

        return [McoreDistributedOptimizer(mcore_opt)]

    def finalize_model_grads(self, *args, **kwargs):
        return finalize_model_grads(*args, **kwargs)
