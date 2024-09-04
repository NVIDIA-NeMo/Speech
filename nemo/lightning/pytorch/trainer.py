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

from copy import deepcopy

import fiddle as fdl
import pytorch_lightning as pl
from typing_extensions import Self

from nemo.lightning.fabric.conversion import to_fabric
from nemo.lightning.fabric.fabric import Fabric
from nemo.lightning.io.mixin import IOMixin, serialization, track_io


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
        accelerator, devices, strategy, plugins = None, None, None, None
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

        out = Fabric(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )

        return out
