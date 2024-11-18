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

import warnings
from copy import deepcopy

import fiddle as fdl
import lightning.pytorch as pl
from lightning.pytorch.loops import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher
from typing_extensions import Self

from nemo.lightning.fabric.conversion import to_fabric
from nemo.lightning.fabric.fabric import Fabric
from nemo.lightning.io.mixin import IOMixin, serialization, track_io


class NoValOnRestartTrainingLoop(_TrainingEpochLoop):
    """
    Extend the PTL Epoch loop to skip validation when restarting.
    This happens when resuming a checkpoint that has already run validation, but loading restores
    the training state before validation has run.
    """

    def _should_check_val_fx(self, data_fetcher) -> bool:
        if self.skip_val_on_restart:
            return False
        return super()._should_check_val_fx(data_fetcher)

    def load_state_dict(self, state_dict: dict, prefix: str = "") -> None:
        super().load_state_dict(state_dict, prefix)

        self.skip_val_on_restart = True

    def advance(self, data_fetcher: _DataFetcher) -> None:
        super().advance(data_fetcher)

        self.skip_val_on_restart = False


def configure_no_restart_validation_training_loop(trainer: pl.Trainer) -> None:
    if not isinstance(trainer.fit_loop.epoch_loop, _TrainingEpochLoop):
        warnings.warn("Detected custom epoch loop. Skipping no validation on restart support.", UserWarning)
        return

    ## Pass trainer object to avoid trainer getting overwritten as None
    loop = NoValOnRestartTrainingLoop(trainer, trainer.min_steps, trainer.max_steps)
    trainer.fit_loop.epoch_loop = loop


class Trainer(pl.Trainer, IOMixin):
    def add_io(self, obj):
        """Recurse to the leaves of a container and add io functionality to non-serializable leaves"""
        if isinstance(obj, (dict, list)):
            if isinstance(obj, dict):
                obj = obj.values()
            for item in obj:
                self.add_io(item)
        else:
            if not serialization.find_node_traverser(type(obj)):
                track_io(type(obj))
            return

    def io_init(self, **kwargs) -> fdl.Config[Self]:
        # Each argument of the trainer can be stateful so we copy them
        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        self.add_io(cfg_kwargs)
        return fdl.Config(type(self), **cfg_kwargs)

    def to_fabric(self, callbacks=None, loggers=None) -> Fabric:
        accelerator, devices, strategy, plugins, num_nodes = None, None, None, None, None
        if hasattr(self.__io__, "devices"):
            devices = self.__io__.devices
        if hasattr(self.__io__, "accelerator"):
            accelerator = self.__io__.accelerator
        if hasattr(self.__io__, "strategy"):
            strategy = self.__io__.strategy
            if isinstance(strategy, fdl.Config):
                strategy = fdl.build(strategy)

            strategy = to_fabric(strategy)
        if hasattr(self.__io__, "plugins"):
            plugins = self.__io__.plugins
            if isinstance(plugins, fdl.Config):
                plugins = fdl.build(plugins)
            plugins = to_fabric(plugins)

        if hasattr(self.__io__, "num_nodes"):
            num_nodes = self.__io__.num_nodes

        out = Fabric(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
            num_nodes=num_nodes,
        )

        return out
