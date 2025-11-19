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

import atexit
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.strategies.fsdp import _get_sharded_state_dict_context
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.strategies.fsdp import FSDPStrategy as PLFSDPStrategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT

try:
    from megatron.core.transformer.transformer_layer import TransformerLayer

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    TransformerLayer = object
    HAVE_MEGATRON_CORE = False

from torch.distributed.checkpoint.state_dict import (  # get_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
    set_state_dict,
)
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import (
    _destroy_dist_connection,
    ckpt_to_dir,
    create_checkpoint_io,
    fix_progress_bar,
    init_model_parallel,
    mcore_to_pyt_sharded_state_dict,
    pyt_to_mcore_state_dict,
    setup_data_sampler,
    setup_parallel_ranks,
)

_logger = logging.getLogger(__name__)


class FSDPStrategy(PLFSDPStrategy, io.IOMixin):
    """Megatron plugin for Pytorch Lightning.

    This strategy implements Fully-Sharded-Data-Parallel using PyTorch's native FSDP methods.
    Comparing with MegatronStrategy, FSDPStrategy is designed to be more lightweight, with
    minimal modifications over Lightning's FSDPStrategy but preserves necessary features to be
    compatible with nemo and mcore.
    By default, this strategy wraps FSDP per TransformerLayer.

    Note:
        This strategy is designed to work with NVIDIA's Megatron-LM framework and requires
        specific model implementations that are compatible with Megatron's parallelism techniques.
    Note:
        Due to the different optimizer structure (FSDP only uses torch native optimizers),
        MegatronStrategy cannot resume training from checkpoints saved by FSDPStrategy, and vice
        versa. However, the model weights structure is made compatible, so switching strategy is
        possible if users only need the weights not the optimizer states. (E.g. run pretrain with
        megatron 4D parallelism and run SFT with FSDP.)
    """

    def __init__(
        self,
        auto_wrap_policy={TransformerLayer},
        state_dict_type="sharded",
        ckpt_load_optimizer: bool = True,
        ckpt_save_optimizer: bool = True,
        data_sampler=None,
        **kwargs,
    ):
        super().__init__(auto_wrap_policy=auto_wrap_policy, state_dict_type=state_dict_type, **kwargs)

        self.data_sampler = data_sampler
        self.ckpt_load_optimizer = ckpt_load_optimizer
        self.ckpt_save_optimizer = ckpt_save_optimizer
        self.store: Optional[torch.distributed.Store] = None

    @override
    def setup_environment(self) -> None:
        """Initializes rank and process group for communications."""
        setup_parallel_ranks(self)

        self.accelerator.setup_device(self.root_device)

        # Implementation from superclass copied below in order to pass the store to the process group init
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None

        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
        if torch.distributed.is_initialized():
            _logger.debug("torch.distributed is already initialized. Exiting early")
            return

        global_rank = self.cluster_environment.global_rank()
        world_size = self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        _logger.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
        torch.distributed.init_process_group(
            self._process_group_backend, rank=global_rank, world_size=world_size, store=self.store
        )

        if self._process_group_backend == "nccl":
            atexit.register(_destroy_dist_connection)

        # On rank=0 let everyone know training is starting
        rank_zero_info(
            f"{'-' * 100}\n"
            f"distributed_backend={self._process_group_backend}\n"
            f"All distributed processes registered. Starting with {world_size} processes\n"
            f"{'-' * 100}\n"
        )

        # if 'device_mesh' in the `kwargs` is provided as a tuple, update it into the `DeviceMesh` object here
        if isinstance(self.kwargs.get("device_mesh"), tuple):
            from torch.distributed.device_mesh import init_device_mesh

            self.kwargs["device_mesh"] = init_device_mesh("cuda", self.kwargs["device_mesh"])

        init_model_parallel(self.model)

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        """Connect strategy to trainer and handle adjustments before the loop starts."""
        self.trainer = trainer
        setup_data_sampler(self.trainer)
        fix_progress_bar(trainer)
        super().setup(trainer)

    def _get_loss_reduction(self, step_type: str):
        for fn_name in [f"{step_type}_loss_reduction", "loss_reduction"]:
            if hasattr(self.lightning_module, fn_name):
                return getattr(self.lightning_module, fn_name)
        return None

    def _step_proxy(self, step_type, batch, batch_idx=None):
        method_name = f"{step_type}_step"
        if self.model != self.lightning_module:
            loss = self._forward_redirection(self.model, self.lightning_module, method_name, batch, batch_idx)
        else:
            loss = getattr(self.lightning_module, method_name)(batch, batch_idx)

        _loss_reduction = self._get_loss_reduction(step_type)
        if _loss_reduction:
            return _loss_reduction.forward(batch, loss)
        return loss, {'avg': loss}

    @override
    def training_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Run training step and logs results."""
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.train_step_context():
            loss, reduced = self._step_proxy("training", batch, batch_idx)

            self.lightning_module.log(
                'global_step',
                self.trainer.global_step,
                prog_bar=True,
                rank_zero_only=True,
                batch_size=1,
            )

            self.lightning_module.log(
                'step',
                self.trainer.global_step,
            )
            self.lightning_module.log(
                'reduced_train_loss', reduced['avg'], prog_bar=True, rank_zero_only=True, batch_size=1
            )

            # returns unreduced loss for backward
            return loss

    @override
    def validation_step(self, batch, batch_idx=None) -> Any:
        """Run validation step and logs results."""
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            loss, reduced = self._step_proxy("validation", batch, batch_idx)
            self.lightning_module.log('val_loss', reduced['avg'], rank_zero_only=True, batch_size=1)
            return loss

    @override
    def test_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Run test step and logs results."""
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            loss, reduced = self._step_proxy("test", batch, batch_idx)
            self.lightning_module.log('test_loss', reduced['avg'], rank_zero_only=True, batch_size=1)

            return loss

    @override
    def predict_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Run prediction step."""
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            loss, reduced = self._step_proxy("predict", batch, batch_idx)
            return reduced

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Transform dataloader with sampler."""
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        """Get CheckpointIO."""
        if not self._checkpoint_io:
            self._checkpoint_io = create_checkpoint_io()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        """Set CheckpointIO."""
        self._checkpoint_io = io

    @property
    def current_epoch_step(self) -> int:
        """
        Get the value of step within an epoch.
        """
        return max(
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

    @override
    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Delete checkpoint at filepath."""
        # Taken from MegatronStrategy
        ckpt = ckpt_to_dir(filepath)
        if self.is_global_zero:
            if os.path.islink(ckpt):
                os.unlink(ckpt)
            else:
                shutil.rmtree(ckpt)

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Converts PyT checkpoints to MCore format and save using MCore dist ckpt library."""
        checkpoint["sharded_state_dict"] = pyt_to_mcore_state_dict(checkpoint.pop("state_dict"))
        checkpoint["state_dict"] = OrderedDict([])

        if "optimizer_states" in checkpoint and self.trainer.state.fn == TrainerFn.FITTING:
            # Clear the optimizer states. This handles the case where ckpt_save_optimizer=False
            # Ideally, the optimizer state dicts should not be generated in this case
            checkpoint["optimizer_states"] = {}

            # replace unsharded optimizer_states with sharded dict.
            # note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
            # the checkpoint will contain only model weights. Optimizer states will be omitted.
            if self.ckpt_save_optimizer:
                checkpoint['optimizer'] = get_optimizer_state_dict(self.model, self.optimizers)
                pyt_to_mcore_state_dict(checkpoint['optimizer']['state'], prefix="optimizer.state.")

        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    @override
    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """PTL method which we override to integrate distributed checkpoints for FSDP models.
        Different from MegatronStrategy, both model and optimizer states are restore within
        this method.

        The logic here is slightly more complicated:
        1. Obtain PyT state dicts (sharded & unflattened) for model and optim -> torch::ShardedTensor
        2. Convert to MCore state dicts -> mcore::ShardedTensor
        3. Load from checkpoint using MCore dist ckpt API -> torch::Tensor
        4. Convert to PyT state dicts (sharded & unflattened) -> torch::ShardedTensor
        5. Load into model and optim using PyT dist ckpt API
        6. Return the loaded checkpoint for lightning to load other metadata
        """
        path = Path(self.broadcast(checkpoint_path))
        torch.cuda.empty_cache()

        # TODO: the elegant way to load both state dicts. Need pytorch 2.3.1
        # msd, osd = get_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))
        sharded_state_dict = {}
        with _get_sharded_state_dict_context(self.model):
            msd = self.model.state_dict()
            pyt_to_mcore_state_dict(msd)
            sharded_state_dict["sharded_state_dict"] = msd

        if self.ckpt_load_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            osd = get_optimizer_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))
            pyt_to_mcore_state_dict(osd['state'], prefix="optimizer.state.")
            sharded_state_dict["optimizer"] = osd

        checkpoint = self.checkpoint_io.load_checkpoint(path, sharded_state_dict=sharded_state_dict)
        mcore_to_pyt_sharded_state_dict(checkpoint['sharded_state_dict'], msd)

        if self.ckpt_load_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            mcore_to_pyt_sharded_state_dict(checkpoint['optimizer']['state'], osd['state'])

        set_state_dict(
            self.model,
            self.optimizers if self.ckpt_load_optimizer else [],
            model_state_dict=checkpoint['sharded_state_dict'],
            optim_state_dict=checkpoint['optimizer'] if self.ckpt_load_optimizer else None,
        )

        return checkpoint
