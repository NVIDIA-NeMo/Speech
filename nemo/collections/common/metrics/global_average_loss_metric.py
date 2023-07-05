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

import torch
from torchmetrics import Metric

__all__ = ['GlobalAverageLossMetric']


class GlobalAverageLossMetric(Metric):
    """
    This class is for averaging loss across multiple processes if a distributed backend is used. True average is
    computed not running average. It does not accumulate gradients so the averaged loss cannot be used for optimization.
    If ``take_avg_loss`` is ``True``, the :meth:`update` method ``loss`` argument has to be a mean loss. If
    ``take_avg_loss`` is ``False`` then the :meth:`update` method ``loss`` argument has to be a sum of losses.

    See :doc:`PyTorch Lightning Metrics<pytorch-lightning:metrics>` for the metric usage instruction.

    Args:
        dist_sync_on_step:
            Synchronize metric state across processes at each method :meth:`forward` call before returning the
            value at the step
        process_group:
            Specify the process group on which synchronization is called. default: ``None`` (which selects the entire
            world)
        take_avg_loss:
            If ``True`` values of :meth:`update` method ``loss`` argument has to be a mean loss. If ``False``
            values of :meth:`update` method ``loss`` argument has to be a sum of losses. default: ``True``
    """

    full_state_update = True

    def __init__(self, dist_sync_on_step=False, process_group=None, take_avg_loss=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.add_state("loss_sum", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state("num_measurements", torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')
        self.take_avg_loss = take_avg_loss

    def update(self, loss, num_measurements):
        """
        Updates :attr:`loss_sum` and :attr:`num_measurements`.

        Args:
            loss: A float zero dimensional ``torch.Tensor`` which is either sum or average of losses for processed
                examples. See ``take_avg_loss`` parameter of :meth:`__init__`.
            num_measurements: An integer zero dimensional ``torch.Tensor`` which contains a number of loss measurements.
                The sum or mean of the results of these measurements are in the ``loss`` parameter.
        """
        if self.take_avg_loss:
            self.loss_sum += loss.detach() * num_measurements
        else:
            self.loss_sum += loss.detach()
        self.num_measurements += num_measurements

    def compute(self):
        """
        Returns mean loss.
        """
        if self.num_measurements.eq(0):
            return torch.tensor(float('nan'))
        return self.loss_sum / self.num_measurements
