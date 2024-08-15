from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import pytorch_lightning as pl
from torch import nn

from nemo.utils import logging


class ModelTransform(pl.Callback):
    """
    A PyTorch Lightning callback that applies a model transformation function at the start of fitting or validation.

    This callback is designed to apply a transformation to the model when fitting or validation begins.
    This design allows for loading the original checkpoint first and then applying the transformation,
    which is particularly useful for techniques like Parameter-Efficient Fine-Tuning (PEFT).

    The transformation function is expected to be defined on the LightningModule
    as an attribute called 'model_transform'.

    Key Features:
    - Applies transformation at the start of fit or validation, not during initialization.
    - Allows loading of original checkpoints before transformation.
    - Supports PEFT and similar techniques that modify model structure.

    Example:
        >>> class MyLightningModule(pl.LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.model = SomeModel()
        ...         self.model_transform = lambda m: SomePEFTMethod()(m)
        ...
        >>> model = MyLightningModule()
        >>> # Load original checkpoint here if needed
        >>> model.load_state_dict(torch.load('original_checkpoint.pth'))
        >>> trainer = pl.Trainer(callbacks=[ModelTransform()])
        >>> # The model will be transformed when trainer.fit() or trainer.validate() is called
        >>> trainer.fit(model)

    Note:
        The transformation is applied only once, at the start of fitting or validation,
        whichever comes first. This ensures that the model structure is modified before
        any forward passes or parameter updates occur, but after the original weights
        have been loaded.
    """

    def __init__(self):
        super().__init__()
        self.model_transform: Optional[Callable[[nn.Module], nn.Module]] = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        logging.info(f"Setting up ModelTransform for stage: {stage}")

        if hasattr(pl_module, 'model_transform'):
            logging.info("Found model_transform attribute on pl_module")
            self.model_transform = _call_counter(pl_module.model_transform)
            pl_module.model_transform = self.model_transform
            logging.info(f"Set model_transform to: {self.model_transform}")
        else:
            logging.info("No model_transform attribute found on pl_module")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._maybe_apply_transform(trainer)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._maybe_apply_transform(trainer)

    def _maybe_apply_transform(self, trainer):
        if self._needs_to_call:
            self.apply_transform(trainer)

    def apply_transform(self, trainer):
        self.model_transform(trainer.model)
        from pytorch_lightning.utilities import model_summary

        logging.info(
            f"After applying model_transform:\n" f"{model_summary.summarize(trainer.lightning_module, max_depth=1)}"
        )

    @property
    def _needs_to_call(self) -> bool:
        return self.model_transform and self.model_transform.__num_calls__ == 0


T = TypeVar('T', bound=Callable[..., Any])


def _call_counter(func: T) -> T:
    """
    A decorator that counts the number of times a function is called.

    This decorator wraps a function and adds a '__num_calls__' attribute to it,
    which is incremented each time the function is called.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function with a call counter.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.__num_calls__ += 1
        return func(*args, **kwargs)

    wrapper.__num_calls__ = 0
    return wrapper  # type: ignore
