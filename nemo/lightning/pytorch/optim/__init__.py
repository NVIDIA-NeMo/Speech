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

from nemo.lightning.pytorch.optim.base import LRSchedulerModule, OptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import (
    CosineAnnealingScheduler,
    InverseSquareRootAnnealingScheduler,
    NoamAnnealingScheduler,
    NoamHoldAnnealingScheduler,
    PolynomialDecayAnnealingScheduler,
    PolynomialHoldDecayAnnealingScheduler,
    SquareAnnealingScheduler,
    SquareRootAnnealingScheduler,
    T5InverseSquareRootAnnealingScheduler,
    WarmupAnnealingScheduler,
    WarmupHoldPolicyScheduler,
    WarmupPolicyScheduler,
)
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule

__all__ = [
    "OptimizerModule",
    "LRSchedulerModule",
    "MegatronOptimizerModule",
    "WarmupPolicyScheduler",
    "WarmupHoldPolicyScheduler",
    "SquareAnnealingScheduler",
    "SquareRootAnnealingScheduler",
    "NoamAnnealingScheduler",
    "NoamHoldAnnealingScheduler",
    "WarmupAnnealingScheduler",
    "InverseSquareRootAnnealingScheduler",
    "T5InverseSquareRootAnnealingScheduler",
    "PolynomialDecayAnnealingScheduler",
    "PolynomialHoldDecayAnnealingScheduler",
    "CosineAnnealingScheduler",
]
