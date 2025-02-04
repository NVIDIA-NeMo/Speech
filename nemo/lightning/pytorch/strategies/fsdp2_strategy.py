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
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.strategies.fsdp import _get_sharded_state_dict_context
from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy as PLModelParallelStrategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.distributed.checkpoint.state_dict import (  # get_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
    set_state_dict,
)
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import (
    ckpt_to_dir,
    create_checkpoint_io,
    fix_progress_bar,
    init_model_parallel,
    mcore_to_pyt_sharded_state_dict,
    pyt_to_mcore_state_dict,
    setup_data_sampler,
    setup_parallel_ranks,
)


class FSDP2Strategy(PLModelParallelStrategy, io.IOMixin):
    """Megatron plugin for Pytorch Lightning implementing FSDP 2.

    This strategy utilizes PyTorch's native Fully Sharded Data Parallel (FSDP) version 2 methods.
    Compared to `MegatronStrategy`, `FSDP2Strategy` is designed to be more lightweight while
    maintaining compatibility with NeMo and MCore. By default, this strategy wraps FSDP2 per
    Transformer layer.

    Notes:
        - This strategy is designed for NVIDIA's Megatron-LM framework and requires models
          compatible with Megatron's parallelism techniques.
        - Due to different optimizer structures, training cannot be resumed from checkpoints
          saved with `MegatronStrategy`. However, model weights remain compatible, allowing for
          switching strategies when only weights are needed (e.g., pretraining with Megatron 4D
          parallelism and fine-tuning with FSDP2).
    """

    def __init__(
        self,
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size: Union[Literal["auto"], int] = "auto",
        ckpt_load_optimizer: bool = True,
        ckpt_save_optimizer: bool = True,
        data_sampler=None,
        **kwargs,
    ):
        """Initializes the FSDP2Strategy with specified parallelization settings.

        Args:
            data_parallel_size (Union[Literal["auto"], int]): Number of data-parallel replicas.
            tensor_parallel_size (Union[Literal["auto"], int]): Number of tensor-parallel groups.
            ckpt_load_optimizer (bool): Whether to load optimizer state from checkpoints.
            ckpt_save_optimizer (bool): Whether to save optimizer state in checkpoints.
            data_sampler (optional): Custom data sampler to process dataloaders.
            **kwargs: Additional arguments for base class initialization.
        """
        super().__init__(data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size, **kwargs)

        self.data_sampler = data_sampler
        self.ckpt_load_optimizer = ckpt_load_optimizer
        self.ckpt_save_optimizer = ckpt_save_optimizer

    @override
    def setup_environment(self) -> None:
        """Sets up the parallel environment and initializes model parallelism."""
        setup_parallel_ranks(self)
        super().setup_environment()
        init_model_parallel(self.model)

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        """Configures the strategy within the PyTorch Lightning trainer.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        """
        self.trainer = trainer
        setup_data_sampler(self.trainer)
        fix_progress_bar(trainer)
        super().setup(trainer)

    def _get_loss_reduction(self, step_type: str):
        """Retrieves the loss reduction method for a given step type.

        Args:
            step_type (str): The type of step (e.g., "training", "validation").

        Returns:
            Callable: The loss reduction function, if defined; otherwise, None.
        """
        for fn_name in [f"{step_type}_loss_reduction", "loss_reduction"]:
            if hasattr(self.lightning_module, fn_name):
                return getattr(self.lightning_module, fn_name)
        return None

    def _step_proxy(self, step_type, batch, batch_idx=None):
        """Executes a training, validation, or test step and applies loss reduction if available.

        Args:
            step_type (str): The step type ("training", "validation", "test", "predict").
            batch: The input batch.
            batch_idx (optional): Index of the batch.

        Returns:
            Tuple: The computed loss and a dictionary with reduced loss metrics.
        """
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
        """Defines the training step, logging relevant metrics.

        Args:
            batch: The input batch.
            batch_idx (optional): The index of the batch.

        Returns:
            STEP_OUTPUT: The loss for backpropagation.
        """
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
        """Defines the validation step, logging validation loss.

        Args:
            batch: The input batch.
            batch_idx (optional): The index of the batch.

        Returns:
            Any: The validation loss.
        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            loss, reduced = self._step_proxy("validation", batch, batch_idx)
            self.lightning_module.log('val_loss', reduced['avg'], rank_zero_only=True, batch_size=1)
            return loss

    @override
    def test_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Defines the test step, logging test loss.

        Args:
            batch: The input batch.
            batch_idx (optional): The index of the batch.

        Returns:
            Any: The test loss.
        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            loss, reduced = self._step_proxy("test", batch, batch_idx)
            self.lightning_module.log('test_loss', reduced['avg'], rank_zero_only=True, batch_size=1)

            return loss

    @override
    def predict_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        """Runs one predict step.

        Args:
            batch (dict): the batch to use for pred.
            batch_idx (int, optional): the batch index. Defaults to None.

        Returns:
            STEP_OUTPUT: the reduced loss.
        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            loss, reduced = self._step_proxy("predict", batch, batch_idx)
            return reduced

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Applies data-samples to dataloader"""
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        """CheckpointIO getter

        Returns:
            CheckpointIO: _description_
        """
        if not self._checkpoint_io:
            self._checkpoint_io = create_checkpoint_io()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        """CheckpointIO setter

        Args:
            io (CheckpointIO): the checkpointio to use.
        """
        self._checkpoint_io = io

    @property
    def current_epoch_step(self) -> int:
        """Gets the current step within an epoch.

        Returns:
            int: The step index within the epoch.
        """
        return max(
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

    @override
    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Removes a checkpoint from the filesystem.

        Args:
            filepath (Union[str, Path]): Path to the checkpoint to be removed.
        """
        ckpt = ckpt_to_dir(filepath)
        if self.is_global_zero:
            if os.path.islink(ckpt):
                os.unlink(ckpt)
            elif os.path.exists(ckpt):
                shutil.rmtree(ckpt)

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Converts PyT checkpoints to MCore format and save using MCore dist ckpt library."""

        from nemo.lightning.pytorch.strategies.utils import to_cpu

        module_names = list(checkpoint["state_dict"].keys())
        for name in module_names:
            param = checkpoint["state_dict"].pop(name)
            checkpoint["state_dict"][name] = to_cpu(param)

        if "optimizer_states" in checkpoint and self.trainer.state.fn == TrainerFn.FITTING:
            # Clear the optimizer states. This handles the case where ckpt_save_optimizer=False
            # Ideally, the optimizer state dicts should not be generated in this case
            checkpoint["optimizer_states"] = {}

            # replace unsharded optimizer_states with sharded dict.
            # note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
            # the checkpoint will contain only model weights. Optimizer states will be omitted.
            if self.ckpt_save_optimizer:
                checkpoint['optimizer'] = get_optimizer_state_dict(self.model, self.optimizers)
                pyt_to_mcore_state_dict(
                    checkpoint['optimizer']['state'], prefix="optimizer.state.", device_mesh=self.device_mesh
                )

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
            pyt_to_mcore_state_dict(msd, device_mesh=self.device_mesh)
            sharded_state_dict["sharded_state_dict"] = msd

        if self.ckpt_load_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            osd = get_optimizer_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))
            pyt_to_mcore_state_dict(osd['state'], prefix="optimizer.state.", device_mesh=self.device_mesh)
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
