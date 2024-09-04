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

import os

import torch
from pytorch_lightning.callbacks.callback import Callback
from torch.utils.viz._cycles import warn_tensor_cycles

from nemo.lightning import io
from nemo.utils import logging
from nemo.utils.get_rank import get_rank


class MemoryProfileCallback(Callback, io.IOMixin):
    """
    This callback enables recording a timeline of memory allocations during training.
    The generated .pickle profiles can be analyzed at https://pytorch.org/memory_viz

    More info about the profiles can be found [here](https://pytorch.org/blog/understanding-gpu-memory-1/).

    Args:
        dir (Optional[str]): Directory to store the memory profile dump
        warn_cycles (Optional[bool]): Whether to enable [reference cycle detection](https://pytorch.org/blog/understanding-gpu-memory-2/)
        rank (Optional[list[int]]): List of ranks to collect snapshot on, defaults to all if list is empty

    Example:
        >>> callback = MemoryProfileCallback(dir="/mem_profile", ranks=[0])
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self, dir: str = "/mem_profile", warn_cycles=True, ranks=[]):

        self.dir = dir
        self.ranks = ranks

        os.makedirs(self.dir, exist_ok=True)
        logging.info(f"Torch memory profiles will be written to: {self.dir}")

        if warn_cycles:
            logging.info("Enabling reference cycle detector")
            warn_tensor_cycles()

    def enable_on_rank(self) -> bool:
        if not self.ranks:
            return True
        return get_rank() in self.ranks

    def setup(self, trainer, pl_module, stage) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-end
        We use it here to start recording the memory profiler.
        """

        if trainer.max_steps > 1000:
            logging.warning(
                f"Memory profiling creates snapshots during the entire training process, \
            where every iteration increases the size of the snapshot. \
            Try reducing trainer.max_steps to avoid running into issues"
            )

        if torch.distributed.is_initialized() and self.enable_on_rank():
            torch.cuda.memory._record_memory_history(max_entries=100000)

    def on_train_end(self, trainer, pl_module) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-end
        We use it here to finish memory profiling and write the snapshot.
        """

        logging.info(
            f"on_train_batch_end rank: {get_rank()} mem: {torch.cuda.memory_allocated()/1024/1024/1024} / {torch.cuda.max_memory_reserved()/1024/1024/1024}"
        )

        if torch.distributed.is_initialized() and self.enable_on_rank():
            rank = get_rank()
            _snapshot_path = f"{self.dir}/memory_snapshot-rank{rank}.pickle"
            logging.info(f"Writing memory profile snapshot to {_snapshot_path}")
            torch.cuda.memory._dump_snapshot(f"{_snapshot_path}")
            torch.cuda.memory._record_memory_history(enabled=None)
            logging.info(f"Finished writing memory profile snapshot: {_snapshot_path}")
