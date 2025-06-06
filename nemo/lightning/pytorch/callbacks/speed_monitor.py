# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import time
from collections import deque
from typing import Any

import lightning.pytorch as pl
import torch.distributed as dist


class SpeedMonitor(pl.Callback):
    """
    Logs the training throughput and utilization.

    The training throughput is logged on the event once we have reached the `window_size` threshold.

    Example:
        import nemo_run as run
        from nemo.lightning.pytorch.callbacks import SpeedMonitor

        recipe.trainer.callbacks.append(
            run.Config(SpeedMonitor, window_size=100)
        )

    The training throughput is logged by the PTL's logger to the following keys as described below.

    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/batches_per_sec`        | batches) of the number of batches processed per second.   |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/samples_per_sec`        | batches) of the number of samples processed per second.   |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/tokens_per_sec`         | batches) of the number of tokens processed per second.    |
    |                                     | Only logged if dataspec returns tokens per batch.         |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/batches_per_sec` | `throughput/batches_per_sec` divided by world size.       |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/samples_per_sec` | `throughput/samples_per_sec` divided by world size.       |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/tokens_per_sec` divided by world size. Only   |
    | `throughput/device/tokens_per_sec`  | logged if dataspec returns tokens per batch.              |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
        time_unit (str, optional): Time unit to use for `time` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(
        self,
        window_size: int = 100,
        time_unit: str = 'hours',
    ):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.history_samples: deque[int] = deque(maxlen=window_size + 1)
        self.history_tokens: deque[int] = deque(maxlen=window_size + 1)
        self.history_wct: deque[float] = deque(maxlen=window_size + 1)
        self.history_flops: deque[float] = deque(maxlen=window_size + 1)

        self.divider = 1
        if time_unit == 'seconds':
            self.divider = 1
        elif time_unit == 'minutes':
            self.divider = 60
        elif time_unit == 'hours':
            self.divider = 60 * 60
        elif time_unit == 'days':
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".',
            )

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.consumed_samples = 0
        self.consumed_tokens = 0

    def on_train_start(self, trainer, pl_module):
        """ """
        self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """ """
        # Add the new element
        batch_size = trainer.train_dataloader.batch_sampler.global_batch_size
        self.consumed_samples += batch_size
        self.history_samples.append(self.consumed_samples)

        self.consumed_tokens += batch['tokens'].size()[1] * (batch_size)
        self.history_tokens.append(self.consumed_tokens)

        elapsed_time = time.time() - self.start_time
        self.history_wct.append(elapsed_time)

        # Log the throughput
        if len(self.history_wct) == self.history_wct.maxlen:
            world_size = dist.get_world_size()
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = int(self.history_samples[-1]) - int(self.history_samples[0])
            elapsed_tokens = int(self.history_tokens[-1]) - int(self.history_tokens[0])
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            batches_per_sec = elapsed_batches / elapsed_wct
            samples_per_sec = elapsed_samples / elapsed_wct
            dev_batches_per_sec = batches_per_sec / world_size
            dev_samples_per_sec = samples_per_sec / world_size
            metrics = {
                'throughput/batches_per_sec': batches_per_sec,
                'throughput/samples_per_sec': samples_per_sec,
                'throughput/device/batches_per_sec': dev_batches_per_sec,
                'throughput/device/samples_per_sec': dev_samples_per_sec,
                'throughput/micro_batch_size': trainer.train_dataloader.batch_sampler.micro_batch_size,
                'throughput/global_batch_size': batch_size,
            }
            for metric, value in metrics.items():
                self.log(metric, value)
            if elapsed_tokens > 0:
                tokens_per_sec = elapsed_tokens / elapsed_wct
                dev_tokens_per_sec = tokens_per_sec / world_size
                self.log('throughput/tokens_per_sec', tokens_per_sec)
                self.log('throughput/device/tokens_per_sec', dev_tokens_per_sec)
