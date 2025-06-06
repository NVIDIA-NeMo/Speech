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

import math
from typing import Any, Optional, Union

import lightning.pytorch as pl
import torch
import torch.cuda
from torch import distributed


def reduce_value(
    value: Union[int, float],
    reduce_op: str = 'mean',
):
    """
    Reduce a value across distributed processes.

    Args:
        value (Union[int, float]): The value to reduce.
        model_device (torch.device): The device on which the model is located.
        reduce_op (str, optional): The reduction operation to perform. One of 'mean', 'avg', 'sum', 'min', 'max'.
            Defaults to 'mean'.
    """

    tensor_value = torch.tensor(value)

    if reduce_op in ['mean', 'avg', 'sum']:
        op = distributed.ReduceOp.SUM
    elif reduce_op == 'min':
        op = distributed.ReduceOp.MIN
    elif reduce_op == 'max':
        op = distributed.ReduceOp.MAX
    else:
        raise ValueError(f'{reduce_op=} not supported.')

    distributed.all_reduce(tensor_value, op=op)
    if reduce_op in ['mean', 'avg']:
        tensor_value = tensor_value / distributed.get_world_size()

    return tensor_value.item()


class MemoryMonitor(pl.Callback):
    """
    Logs the memory usage of the model.

    This callback calls the torch memory stats API for CUDA and reports different memory statistics.

    Example:
        import nemo_run as run
        from nemo.lightning.pytorch.callbacks import MemoryMonitor

        recipe.trainer.callbacks.append(
            run.Config(MemoryMonitor)
        )

    The memory statistics are logged by the :class:`.Logger` to the following keys as
    described below.

    +--------------------------+-------------------------------------------------------------+
    | Key                      | Logged data                                                 |
    +==========================+=============================================================+
    |                          | Several memory usage statistics                             |
    | ``memory/{statistic}``   | are logged on                                               |
    |                          | :attr:`.Event.AFTER_TRAIN_BATCH` event.                     |
    +--------------------------+-------------------------------------------------------------+

    The following statistics are recorded:

    +------------------------+----------------------------------------------------------------------------------------+
    | Statistic              | Description                                                                            |
    +========================+========================================================================================+
    | current_allocated_mem  | Current amount of allocated memory in gigabytes.                                       |
    +------------------------+----------------------------------------------------------------------------------------+
    | current_active_mem     | Current amount of active memory in gigabytes at the time of recording.                 |
    +------------------------+----------------------------------------------------------------------------------------+
    | current_inactive_mem   | Current amount of inactive, non-releaseable memory in gigabytes.                       |
    +------------------------+----------------------------------------------------------------------------------------+
    | current_reserved_mem   | Current amount of reserved memory in gigabytes at the time of recording.               |
    +------------------------+----------------------------------------------------------------------------------------+
    | peak_allocated_mem     | Peak amount of allocated memory in gigabytes.                                          |
    +------------------------+----------------------------------------------------------------------------------------+
    | peak_active_mem        | Peak amount of active memory in gigabytes at the time of recording.                    |
    +------------------------+----------------------------------------------------------------------------------------+
    | peak_inactive_mem      | Peak amount of inactive, non-releaseable memory in gigabytes at the time of recording. |
    +------------------------+----------------------------------------------------------------------------------------+
    | peak_reserved_mem      | Peak amount of reserved memory in gigabytes at the time of recording.                  |
    +------------------------+----------------------------------------------------------------------------------------+
    | alloc_retries          | Number of failed cudaMalloc calls that result in a cache flush and retry.              |
    +------------------------+----------------------------------------------------------------------------------------+

    Additionally, if `dist_aggregate_batch_interval` is enabled, the `avg`, `min`, and `max` of the
    aformentioned statistics are also logged.

    Args:
        memory_keys (dict[str, str], optional): A dict specifying memory statistics to log. Keys
            are the names of memory statistics to log from `torch.cuda.memory_stats()`, and values
            are the names they will be logged under. If not provided, the above statistics are
            logged. Defaults to None.
        dist_aggregate_batch_interval (int, optional): interval for aggregating memory stats across
            all nodes. Defaults to None (by default the functionality is disabled).
    """

    def __init__(
        self,
        memory_keys: Optional[dict[str, str]] = None,
        dist_aggregate_batch_interval: Optional[int] = None,
    ) -> None:
        self.memory_keys = memory_keys
        self.dist_aggregate_batch_interval = dist_aggregate_batch_interval

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """ """
        memory_report = {}
        memory_report = _get_memory_report(self.memory_keys)
        if self.dist_aggregate_batch_interval:
            dist_memory_report = {}
            for mem_stat, val in memory_report.items():
                dist_memory_report[mem_stat + '_avg'] = reduce_value(val, 'avg')
                dist_memory_report[mem_stat + '_min'] = reduce_value(val, 'min')
                dist_memory_report[mem_stat + '_max'] = reduce_value(val, 'max')
            memory_report.update(dist_memory_report)

        memory_metrics = {f'memory/{mem_stat}': val for (mem_stat, val) in memory_report.items()}
        for metric, value in memory_metrics.items():
            self.log(metric, value)


_MEMORY_KEYS = {
    'allocated_bytes.all.current': 'current_allocated_mem',
    'active_bytes.all.current': 'current_active_mem',
    'inactive_split_bytes.all.current': 'current_inactive_mem',
    'reserved_bytes.all.current': 'current_reserved_mem',
    'allocated_bytes.all.peak': 'peak_allocated_mem',
    'active_bytes.all.peak': 'peak_active_mem',
    'inactive_split_bytes.all.peak': 'peak_inactive_mem',
    'reserved_bytes.all.peak': 'peak_reserved_mem',
    'num_alloc_retries': 'alloc_retries',
}


def _get_memory_report(memory_keys: Optional[dict[str, str]] = None) -> dict[str, Union[int, float]]:
    """
    Returns a dictionary with memory metrics.

    Args:
        memory_keys (Optional[dict[str, str]]): a dict specifying memory statistics to log.

    Retuns:
        dict: memory statistics.
    """

    memory_stats = torch.cuda.memory_stats()
    memory_keys = memory_keys or _MEMORY_KEYS

    # simplify and reformat the memory_stats
    memory_report = {}
    for torch_name, name in memory_keys.items():
        if torch_name in memory_stats:
            # Convert to gigabytes
            if 'bytes' in torch_name:
                gigabytes = memory_stats[torch_name] / 1.0e9
                # Round to preserve 5 significant digits
                if gigabytes != 0:
                    order_of_magnitude = int(math.floor(math.log10(abs(gigabytes))))
                    gigabytes = round(gigabytes, -order_of_magnitude + 4)
                memory_report[name.replace('bytes', 'gigabytes')] = gigabytes
            else:
                memory_report[name] = memory_stats[torch_name]

    return memory_report
