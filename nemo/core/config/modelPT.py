# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from nemo.core import config
from nemo.core.classes.dataset import DatasetConfig
from nemo.utils import exp_manager


@dataclass
class SchedConfig:
    name: str = MISSING
    min_lr: float = 0.0
    last_epoch: int = -1


@dataclass
class OptimConfig:
    name: str = MISSING
    sched: Optional[SchedConfig] = None


@dataclass
class ModelConfig:
    """
    Model component inside ModelPT
    """

    # ...
    train_ds: Optional[DatasetConfig] = None
    validation_ds: Optional[DatasetConfig] = None
    test_ds: Optional[DatasetConfig] = None
    optim: Optional[OptimConfig] = None


@dataclass
class HydraConfig:
    run: Dict[str, Any] = field(default_factory=lambda: {"dir": "."})
    job_logging: Dict[str, Any] = field(default_factory=lambda: {"root": {"handlers": None}})


@dataclass
class NemoConfig:
    name: str = MISSING
    model: ModelConfig = MISSING
    trainer: config.TrainerConfig = field(
        default_factory=lambda: config.TrainerConfig(
            strategy="ddp", enable_checkpointing=False, logger=False, log_every_n_steps=1, accelerator='gpu'
        )
    )
    exp_manager: Optional[Any] = field(default_factory=lambda: exp_manager.ExpManagerConfig())
    hydra: HydraConfig = field(default_factory=lambda: HydraConfig())


class ModelConfigBuilder:
    """Fluent builder for model configuration objects."""

    def __init__(self, model_cfg: ModelConfig):
        """
        Base class for any Model Config Builder.

        A Model Config Builder is a utility class that accepts a ModelConfig dataclass,
        and via a set of utility methods (that are implemented by the subclassed ModelConfigBuilder),
        builds a finalized ModelConfig that can be supplied to a NemoModel dataclass as
        the `model` component.

        Subclasses *must* implement the private method `_finalize_cfg`.
            Inside this method, they must update `self.model_cfg` with all interdependent config
            options that need to be set (either updated by user explicitly or with their default value).

            The updated model config must then be preserved in `self.model_cfg`.

        Example:
            # Create the config builder
            config_builder = <subclass>ModelConfigBuilder()

            # Update the components of the config that are modifiable
            config_builder.set_X(X)
            config_builder.set_Y(Y)

            # Create a "finalized" config dataclass that will contain all the updates
            # that were specified by the builder
            model_config = config_builder.build()

            # Use model config as is (or further update values), then create a new Model
            model = nemo.<domain>.models.<ModelName>Model(cfg=model_config, trainer=Trainer())

        Supported build methods:
        -   set_train_ds: All model configs can accept a subclass of `DatasetConfig` as their
                training config. Subclasses can override this method to enable auto-complete
                by replacing `Optional[DatasetConfig]` with `Optional[<subclass of DatasetConfig>]`.

        -   set_validation_ds: All model configs can accept a subclass of `DatasetConfig` as their
                validation config. Subclasses can override this method to enable auto-complete
                by replacing `Optional[DatasetConfig]` with `Optional[<subclass of DatasetConfig>]`.

        -   set_test_ds: All model configs can accept a subclass of `DatasetConfig` as their
                test config. Subclasses can override this method to enable auto-complete
                by replacing `Optional[DatasetConfig]` with `Optional[<subclass of DatasetConfig>]`.

        -   set_optim: A build method that supports changes to the Optimizer (and optionally,
                the Scheduler) used for training the model. The function accepts two inputs -

                `cfg`: A subclass of `OptimizerParams` - any OptimizerParams subclass can be used,
                    in order to select an appropriate Optimizer. Examples: AdamParams.

                `sched_cfg`: A subclass of `SchedulerParams` - any SchedulerParams subclass can be used,
                    in order to select an appropriate Scheduler. Examples: CosineAnnealingParams.
                    Note that this argument is optional.

        -   build(): The method which should return a "finalized" ModelConfig dataclass.
                Subclasses *should* always override this method, and update the signature
                of this method with the return type of the Dataclass, so that it enables
                autocomplete for the user.

                Example:
                    def build(self) -> EncDecCTCConfig:
                        return super().build()

        Any additional build methods must be added by subclasses of ModelConfigBuilder.

        Args:
            model_cfg:
        """
        self.model_cfg = model_cfg
        self.train_ds_cfg = None
        self.validation_ds_cfg = None
        self.test_ds_cfg = None
        self.optim_cfg = None

    def set_train_ds(self, cfg: Optional[DatasetConfig] = None):
        self.model_cfg.train_ds = cfg

    def set_validation_ds(self, cfg: Optional[DatasetConfig] = None):
        self.model_cfg.validation_ds = cfg

    def set_test_ds(self, cfg: Optional[DatasetConfig] = None):
        self.model_cfg.test_ds = cfg

    def set_optim(self, cfg: config.OptimizerParams, sched_cfg: Optional[config.SchedulerParams] = None):
        """Set optimizer and scheduler configuration."""

        @dataclass
        class WrappedOptimConfig(OptimConfig, cfg.__class__):
            pass

        # Setup optim
        optim_name = cfg.__class__.__name__.replace("Params", "").lower()
        wrapped_cfg = WrappedOptimConfig(name=optim_name, sched=None, **vars(cfg))

        if sched_cfg is not None:

            @dataclass
            class WrappedSchedConfig(SchedConfig, sched_cfg.__class__):
                pass

            # Setup scheduler
            sched_name = sched_cfg.__class__.__name__.replace("Params", "")
            wrapped_sched_cfg = WrappedSchedConfig(name=sched_name, **vars(sched_cfg))

            wrapped_cfg.sched = wrapped_sched_cfg

        self.model_cfg.optim = wrapped_cfg

    def _finalize_cfg(self):
        raise NotImplementedError()

    def build(self) -> ModelConfig:
        # validate config
        self._finalize_cfg()

        return self.model_cfg
