# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional


@dataclass
class SchedulerParams:
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    last_epoch: int = -1


@dataclass
class SquareRootConstantSchedulerParams(SchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    constant_steps: Optional[float] = None
    constant_ratio: Optional[float] = None


@dataclass
class WarmupSchedulerParams(SchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    max_steps: int = 0
    warmup_steps: Optional[float] = None
    warmup_ratio: Optional[float] = None


@dataclass
class WarmupHoldSchedulerParams(WarmupSchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    hold_steps: Optional[float] = None
    hold_ratio: Optional[float] = None
    min_lr: float = 0.0


@dataclass
class WarmupAnnealingHoldSchedulerParams(WarmupSchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    constant_steps: Optional[float] = None
    constant_ratio: Optional[float] = None
    min_lr: float = 0.0


@dataclass
class SquareAnnealingParams(WarmupSchedulerParams):
    """
    Square Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 1e-5


@dataclass
class SquareRootAnnealingParams(WarmupSchedulerParams):
    """
    Square Root Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 0.0


@dataclass
class CosineAnnealingParams(WarmupAnnealingHoldSchedulerParams):
    """
    Cosine Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 0.0


@dataclass
class NoamAnnealingParams(WarmupSchedulerParams):
    """
    Cosine Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 0.0


@dataclass
class NoamHoldAnnealingParams(WarmupHoldSchedulerParams):
    """
    Polynomial Hold Decay Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    decay_rate: float = 0.5


@dataclass
class WarmupAnnealingParams(WarmupSchedulerParams):
    """
    Warmup Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    warmup_ratio: Optional[float] = None


@dataclass
class InverseSquareRootAnnealingParams(WarmupSchedulerParams):
    """
    Inverse Square Root Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """


@dataclass
class PolynomialDecayAnnealingParams(WarmupSchedulerParams):
    """
    Polynomial Decay Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    power: float = 1.0
    cycle: bool = False


@dataclass
class PolynomialHoldDecayAnnealingParams(WarmupSchedulerParams):
    """
    Polynomial Hold Decay Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    power: float = 1.0
    cycle: bool = False


"""
Pytorch Optimizers
"""


@dataclass
class StepLRParams(SchedulerParams):
    """
    Config for StepLR.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    step_size: float = 0.1
    gamma: float = 0.1


@dataclass
class ExponentialLRParams(SchedulerParams):
    """
    Config for ExponentialLR.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    gamma: float = 0.9


@dataclass
class ReduceLROnPlateauParams:
    """
    Config for ReduceLROnPlateau.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    mode: str = 'min'
    factor: float = 0.1
    patience: int = 10
    verbose: bool = False
    threshold: float = 1e-4
    threshold_mode: str = 'rel'
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8


@dataclass
class CyclicLRParams(SchedulerParams):
    """
    Config for CyclicLR.
    NOTE:
    # `scale_fn` is not supported

    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    base_lr: float = 0.001
    max_lr: float = 0.1
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = 'triangular'
    gamma: float = 1.0
    scale_mode: str = 'cycle'
    # scale_fn is not supported
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.9


def register_scheduler_params(name: str, scheduler_params: SchedulerParams):
    """
    Checks if the schduler config name exists in the registry, and if it doesnt, adds it.

    This allows custom schedulers to be added and called by name during instantiation.

    Args:
        name: Name of the optimizer. Will be used as key to retrieve the optimizer.
        scheduler_params: SchedulerParams class
    """
    if name in AVAILABLE_SCHEDULER_PARAMS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_SCHEDULER_PARAMS[name] = scheduler_params


def get_scheduler_config(name: str, **kwargs: Optional[Dict[str, Any]]) -> SchedulerParams:
    """
    Convenience method to obtain a SchedulerParams class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the SchedulerParams in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns:
        a partially instantiated SchedulerParams
    """
    if name not in AVAILABLE_SCHEDULER_PARAMS:
        raise ValueError(
            f"Cannot resolve scheduler parameters '{name}'. Available scheduler parameters are : "
            f"{AVAILABLE_SCHEDULER_PARAMS.keys()}"
        )

    scheduler_params = AVAILABLE_SCHEDULER_PARAMS[name]
    scheduler_params = partial(scheduler_params, **kwargs)
    return scheduler_params


AVAILABLE_SCHEDULER_PARAMS = {
    'SchedulerParams': SchedulerParams,
    'WarmupPolicyParams': WarmupSchedulerParams,
    'WarmupHoldPolicyParams': WarmupHoldSchedulerParams,
    'WarmupAnnealingHoldSchedulerParams': WarmupAnnealingHoldSchedulerParams,
    'SquareAnnealingParams': SquareAnnealingParams,
    'SquareRootAnnealingParams': SquareRootAnnealingParams,
    'InverseSquareRootAnnealingParams': InverseSquareRootAnnealingParams,
    'SquareRootConstantSchedulerParams': SquareRootConstantSchedulerParams,
    'CosineAnnealingParams': CosineAnnealingParams,
    'NoamAnnealingParams': NoamAnnealingParams,
    'NoamHoldAnnealingParams': NoamHoldAnnealingParams,
    'WarmupAnnealingParams': WarmupAnnealingParams,
    'PolynomialDecayAnnealingParams': PolynomialDecayAnnealingParams,
    'PolynomialHoldDecayAnnealingParams': PolynomialHoldDecayAnnealingParams,
    'ReduceLROnPlateauParams': ReduceLROnPlateauParams,
}
