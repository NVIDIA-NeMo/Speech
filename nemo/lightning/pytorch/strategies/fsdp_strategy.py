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

import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.strategies.fsdp import _get_sharded_state_dict_context
from megatron.core.transformer.transformer_layer import TransformerLayer
from pytorch_lightning.strategies.fsdp import FSDPStrategy as PLFSDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.distributed.checkpoint.state_dict import (  # get_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
    set_state_dict,
)
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import (
    _MegatronBatchProgress,
    ckpt_to_dir,
    create_checkpoint_io,
    fix_progress_bar,
    init_model_parallel,
    mcore_to_pyt_sharded_state_dict,
    pyt_to_mcore_state_dict,
    setup_data_sampler,
    setup_parallel_ranks,
)


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
        overwrite_batch_progress: bool = True,
        **kwargs,
    ):
        super().__init__(auto_wrap_policy=auto_wrap_policy, state_dict_type=state_dict_type, **kwargs)

        self.data_sampler = data_sampler
        self.ckpt_load_optimizer = ckpt_load_optimizer
        self.ckpt_save_optimizer = ckpt_save_optimizer
        self.overwrite_batch_progress = overwrite_batch_progress

    @override
    def setup_environment(self) -> None:
        setup_parallel_ranks(self)
        super().setup_environment()
        init_model_parallel(self.model)

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        self.trainer = trainer
        setup_data_sampler(self.trainer)
        fix_progress_bar(trainer)

        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING and self.overwrite_batch_progress:
            trainer.fit_loop.epoch_loop.batch_progress = _MegatronBatchProgress()

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
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            loss, reduced = self._step_proxy("validation", batch, batch_idx)
            self.lightning_module.log('val_loss', reduced['avg'], rank_zero_only=True, batch_size=1)
            return loss

    @override
    def test_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            loss, reduced = self._step_proxy("test", batch, batch_idx)
            self.lightning_module.log('test_loss', reduced['avg'], rank_zero_only=True, batch_size=1)

            return loss

    @override
    def predict_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            loss, reduced = self._step_proxy("predict", batch, batch_idx)
            return reduced

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        if not self._checkpoint_io:
            self._checkpoint_io = create_checkpoint_io()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
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

        ## replace unsharded optimizer_states with sharded dict.
        ## note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
        ## the checkpoint will contain only model weights. Optimizer states will be omitted.
        if (
            "optimizer_states" in checkpoint
            and self.trainer.state.fn == TrainerFn.FITTING
            and self.ckpt_save_optimizer
        ):
            checkpoint["optimizer_states"] = {}
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
