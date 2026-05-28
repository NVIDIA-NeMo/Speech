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

# Copyright 2018-2020 William Falcon
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

import torch

from .pl_utils import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES


@dataclass(frozen=True)
class LossInput:
    """
    The input for ``nemo.collections.common.metrics.GlobalAverageLossMetric`` metric tests.

    Args:
        loss_sum_or_avg: a one dimensional float tensor which contains losses for averaging. Each element is either a
            sum or mean of several losses depending on the parameter ``take_avg_loss`` of the
            ``nemo.collections.common.metrics.GlobalAverageLossMetric`` class.
        num_measurements: a one dimensional integer tensor which contains number of measurements which sums or average
            values are in ``loss_sum_or_avg``.
    """

    loss_sum_or_avg: torch.Tensor
    num_measurements: torch.Tensor


NO_ZERO_NUM_MEASUREMENTS = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.randint(1, 100, (NUM_BATCHES,)),
)

SOME_NUM_MEASUREMENTS_ARE_ZERO = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.cat(
        (
            torch.randint(1, 100, (NUM_BATCHES // 2,), dtype=torch.int32),
            torch.zeros(NUM_BATCHES - NUM_BATCHES // 2, dtype=torch.int32),
        )
    ),
)

ALL_NUM_MEASUREMENTS_ARE_ZERO = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.zeros(NUM_BATCHES, dtype=torch.int32),
)
