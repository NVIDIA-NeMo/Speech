from typing import List, Optional

import torch
from pytorch_lightning.callbacks.callback import Callback

from nemo.utils import logging
from nemo.utils.get_rank import get_rank


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

    Example:
        >>> callback = NsysCallback(start_step=100, end_step=200, ranks=[0, 1], gen_shape=True)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        ranks: List[int] = [0],
        gen_shape: bool = False,
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

        logging.info(
            f'Nsys profiling setup with start_step: {self._nsys_profile_start_step},'
            f'and end_step: {self._nsys_profile_end_step}'
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> Optional[int]:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
        We use it here to enable nsys profiling.
        """

        device = trainer.strategy.root_device
        current_step = trainer.strategy.current_epoch_step
        if device.type == 'cuda':
            if current_step == self._nsys_profile_start_step and get_rank() in self._nsys_profile_ranks:
                logging.info("====== Start nsys profiling ======")
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

        device = trainer.strategy.root_device
        current_step = trainer.strategy.current_epoch_step
        if device.type == 'cuda':
            if current_step == self._nsys_profile_end_step and get_rank() in self._nsys_profile_ranks:
                logging.info("====== End nsys profiling ======")
                torch.cuda.cudart().cudaProfilerStop()
                torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)
