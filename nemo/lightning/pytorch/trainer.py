from copy import deepcopy

import fiddle as fdl
import pytorch_lightning as pl
from typing_extensions import Self

from nemo.lightning.io.mixin import IOMixin


class Trainer(pl.Trainer, IOMixin):
    def io_init(self, **kwargs) -> fdl.Config[Self]:
        # Each argument of the trainer can be stateful so we copy them
        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        return fdl.Config(type(self), **cfg_kwargs)
