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

from nemo.core.optim.adafactor import Adafactor
from nemo.core.optim.adan import Adan
from nemo.core.optim.lr_scheduler import (
    CosineAnnealing,
    InverseSquareRootAnnealing,
    NoamAnnealing,
    PolynomialDecayAnnealing,
    PolynomialHoldDecayAnnealing,
    SquareAnnealing,
    SquareRootAnnealing,
    T5InverseSquareRootAnnealing,
    WarmupAnnealing,
    WarmupHoldPolicy,
    WarmupPolicy,
    prepare_lr_scheduler,
)
from nemo.core.optim.mcore_optim import McoreDistributedOptimizer
from nemo.core.optim.novograd import Novograd
from nemo.core.optim.optimizer_with_main_params import MainParamsOptimizerWrapper
from nemo.core.optim.optimizers import get_optimizer, parse_optimizer_args, register_optimizer
