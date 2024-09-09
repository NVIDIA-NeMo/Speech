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


import nemo_run as run
from megatron.core.optimizer import OptimizerConfig

from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule, OptimizerModule


@run.cli.factory
def distributed_fused_adam_with_cosine_annealing(max_lr: float = 1e-4) -> run.Config[OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=max_lr,
        weight_decay=0.1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        clip_grad=1.0,
    )

    sched = run.Config(
        CosineAnnealingScheduler,
        warmup_steps=2000,
        constant_steps=0,
        min_lr=0.1 * max_lr,
    )

    return run.Config(
        MegatronOptimizerModule,
        config=opt_cfg,
        lr_scheduler=sched,
    )
