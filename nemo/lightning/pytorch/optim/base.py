import types
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional

import pytorch_lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer

from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.megatron_parallel import CallbackMethods


class LRSchedulerModule(L.Callback, CallbackMethods, IOMixin, ABC):
    """A module to standardize the learning rate scheduler setup and configuration.

    This class decouples the learning rate scheduler from the model, similar to how the LightningDataModule
    decouples data handling. It also acts as a Callback to hook into the training loop, which can be useful
    for adding custom all-reduces, logging, early stopping, etc. Next to that standard Lightning callback-event,
    this also supports hooking into the Megatron forward-backward function at a granular level.

    Example::

        class MyLRSchedulerModule(LRSchedulerModule):
            def setup(self, model, optimizer):
                # Custom setup logic
                ...

            def scheduler(self, model, optimizers):
                # Define and return the learning rate scheduler
                ...

    Methods:
        setup(model, optimizer): Sets up the learning rate scheduler.
        scheduler(model, optimizers): Abstract method to define the learning rate scheduler.
        __call__(model, optimizers): Calls the setup and scheduler methods.
    """

    def connect(self, model, optimizer) -> None:
        """Sets up the learning rate scheduler.

        Args:
            model: The model for which the scheduler is being set up.
            optimizer: The optimizer for which the scheduler is being set up.
        """
        ...

    @abstractmethod
    def scheduler(self, model, optimizers) -> OptimizerLRScheduler:
        """Abstract method to define the learning rate scheduler.

        Args:
            model: The model for which the scheduler is being defined.
            optimizers: The optimizers for which the scheduler is being defined.

        Returns:
            OptimizerLRScheduler: The learning rate scheduler.
        """
        raise NotImplementedError("The scheduler method should be implemented by subclasses.")

    def __call__(self, model, optimizers):
        """Calls the setup and scheduler methods.

        Args:
            model: The model for which the scheduler is being called.
            optimizers: The optimizers for which the scheduler is being called.

        Returns:
            OptimizerLRScheduler: The learning rate scheduler.
        """

        self.connect(model, optimizers)

        self._scheduler = self.scheduler(model, optimizers)

        if not isinstance(self._scheduler, (dict, tuple)):
            return optimizers, self._scheduler

        return self._scheduler


class OptimizerModule(L.Callback, CallbackMethods, IOMixin, ABC):
    """A module to standardize the optimizer setup and configuration.

    This class decouples the optimizer from the model, similar to how the LightningDataModule
    decouples data handling. It also acts as a Callback to hook into the training loop, which can be useful
    for adding custom all-reduces, logging, early stopping, etc. Next to that standard Lightning callback-event,
    this also supports hooking into the Megatron forward-backward function at a granular level.

    Attributes:
        lr_scheduler (Optional[LRSchedulerModule]): The learning rate scheduler module.

    Example::

        class MyOptimizerModule(OptimizerModule):
            def __init__(self, lr_scheduler=None):
                super().__init__(lr_scheduler)

            def setup(self, model):
                # Custom setup logic
                ...

            def optimizers(self, model):
                # Define and return the optimizers
                ...

    Methods:
        connect(model, trainer): Connects the optimizer module to the model and trainer.
        setup(model): Sets up the optimizer.
        optimizers(model): Abstract method to define the optimizers.
        __call__(model, megatron_parallel): Calls the setup and optimizers methods.
    """

    def __init__(self, lr_scheduler: Optional[LRSchedulerModule]):
        """Initializes the OptimizerModule.

        Args:
            lr_scheduler (Optional[LRSchedulerModule]): The learning rate scheduler module.
        """
        self.lr_scheduler = lr_scheduler

    def connect(self, model: L.LightningModule) -> None:
        """Connects the optimizer module to the model and trainer.

        Args:
            model (L.LightningModule): The model to which the optimizer module is being connected.
        """

        def custom_configure_optimizers(lightning_module_self, megatron_parallel=None):
            opt = self(lightning_module_self, megatron_parallel=megatron_parallel)
            return opt

        model.configure_optimizers = types.MethodType(custom_configure_optimizers, model)
        model.optim = self

        if hasattr(self, "__io__") and hasattr(model, "__io__"):
            if hasattr(model.__io__, "optim"):
                model.__io__.optim = deepcopy(self.__io__)

    @abstractmethod
    def optimizers(self, model) -> List[Optimizer]:
        """Abstract method to define the optimizers.

        Args:
            model: The model for which the optimizers are being defined.

        Returns:
            List[Optimizer]: The list of optimizers.
        """
        raise NotImplementedError("The optimizers method should be implemented by subclasses.")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self._optimizers is not None:
            lr = self._optimizers[0].param_groups[0]['lr']
            pl_module.log('lr', lr, batch_size=1, prog_bar=True)

    def __call__(self, model: L.LightningModule, megatron_parallel=None) -> OptimizerLRScheduler:
        """Calls the setup and optimizers methods.

        Args:
            model (L.LightningModule): The model for which the optimizers are being called.
            megatron_parallel: Optional parallel model.

        Returns:
            OptimizerLRScheduler: The optimizers and optionally the learning rate scheduler.
        """
        _model = model if megatron_parallel is None else megatron_parallel
        callbacks = _model.trainer.callbacks
        if self not in callbacks:
            callbacks.append(self)
        if self.lr_scheduler is not None and self.lr_scheduler not in callbacks:
            callbacks.append(self.lr_scheduler)

        self._optimizers = self.optimizers(_model)

        _opt = self._optimizers[0] if len(self._optimizers) == 1 else self._optimizers

        if self.lr_scheduler is not None:
            with_scheduler = self.lr_scheduler(_model, _opt)

            return with_scheduler

        return self._optimizers
