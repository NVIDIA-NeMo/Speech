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

from typing import List, Optional

import torch
from lightning.pytorch.callbacks.callback import Callback

try:
    from megatron.core.utils import configure_nvtx_profiling

    HAVE_MCORE_UTILS = True
except ImportError:
    HAVE_MCORE_UTILS = False

from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.get_rank import get_rank


def get_current_epoch_step(trainer) -> int:
    """
    Get the value of step within an epoch.
    """
    if hasattr(trainer.strategy, 'current_epoch_step'):
        return trainer.strategy.current_epoch_step
    return max(
        trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
    )


class NsysCallback(Callback):
    """
    A PyTorch Lightning callback for NVIDIA Nsight Systems (Nsys) profiling.

    This callback enables profiling of specific steps during training using NVIDIA Nsys.
    It allows for precise control over when profiling starts and ends, which ranks are profiled,
    and whether to generate detailed shape information.

    More info about nsys can be found [here](https://developer.nvidia.com/nsight-systems).

    Args:
        start_step (int): Global batch to start profiling
        end_step (int): Global batch to end profiling
        ranks (List[int]): Global rank IDs to profile
        gen_shape (bool): Generate model and kernel details including input shapes
        nvtx_ranges (bool): Insert NVTX ranges to categorize execution

    Example:
        >>> callback = NsysCallback(start_step=100, end_step=200, ranks=[0, 1], gen_shape=True, nvtx_ranges=False)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        ranks: List[int] = [0],
        gen_shape: bool = False,
        nvtx_ranges: bool = False,
    ):
        assert type(start_step) is int, f'Nsys start_step must be of type int. Found: {type(start_step)}'
        self._nsys_profile_start_step = start_step

        assert type(end_step) is int, f'Nsys end_step must be of type int. Found: {type(start_step)}'
        self._nsys_profile_end_step = end_step

        assert (
            self._nsys_profile_end_step >= self._nsys_profile_start_step
        ), 'Nsys end_step must be greater than or equal to nsys start_step'

        self._nsys_profile_ranks = ranks
        self._nsys_profile_gen_shape = gen_shape

        app_state = AppState()
        app_state._nvtx_ranges = nvtx_ranges
        if nvtx_ranges and HAVE_MCORE_UTILS:
            configure_nvtx_profiling(True)  # Enable NVTX profiling in MCore

        logging.info(
            f'Nsys profiling setup with start_step: {self._nsys_profile_start_step},'
            f'and end_step: {self._nsys_profile_end_step}'
        )
        self._has_nsys_enabled = False

    def _rank_is_active(self, trainer):
        # TODO(@akoumparouli): is this function cache-able?
        from lightning.pytorch.strategies import SingleDeviceStrategy

        if isinstance(trainer.strategy, SingleDeviceStrategy):
            return True
        if not torch.distributed.is_initialized():
            return True
        return get_rank() in self._nsys_profile_ranks

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> Optional[int]:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
        We use it here to enable nsys profiling.
        """
        if not self._rank_is_active(trainer) or trainer.strategy.root_device.type != 'cuda':
            return

        current_step = get_current_epoch_step(trainer)
        if current_step == self._nsys_profile_start_step and not self._has_nsys_enabled:
            self._has_nsys_enabled = True
            torch.cuda.cudart().cudaProfilerStart()
            if self._nsys_profile_gen_shape:
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
            else:
                torch.autograd.profiler.emit_nvtx().__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
        We use it here to enable nsys profiling.
        """
        if not self._rank_is_active(trainer) or trainer.strategy.root_device.type != 'cuda':
            return

        current_step = get_current_epoch_step(trainer)
        if current_step == self._nsys_profile_end_step and self._has_nsys_enabled:
            torch.cuda.cudart().cudaProfilerStop()
            torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)
            self._has_nsys_enabled = False
