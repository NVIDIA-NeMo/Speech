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

from typing import Optional

import nemo_run as run

from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, PytorchOptimizerModule


@run.cli.factory
def pytorch_sgd_with_cosine_annealing(
    warmup_steps: int = 2000,
    constant_steps: int = 0,
    max_lr: float = 1e-5,
    min_lr: Optional[float] = None,
    wd: float = 1e-4,
) -> run.Config[PytorchOptimizerModule]:
    from torch.optim import SGD

    return run.Config(
        PytorchOptimizerModule,
        optimizer_fn=run.Partial(
            SGD,
            lr=max_lr,
            weight_decay=wd,
        ),
        lr_scheduler=run.Config(
            CosineAnnealingScheduler,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            min_lr=min_lr or (0.1 * max_lr),
        ),
    )


@run.cli.factory
def pytorch_sgd_with_flat_lr(
    lr: float = 1e-5,
    wd: float = 1e-4,
) -> run.Config[PytorchOptimizerModule]:
    from torch.optim import SGD

    return run.Config(
        PytorchOptimizerModule,
        optimizer_fn=run.Partial(
            SGD,
            lr=lr,
            weight_decay=wd,
        ),
    )
