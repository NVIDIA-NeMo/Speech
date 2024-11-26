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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.base import SaveShardedStrategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.parallel_state import get_data_parallel_group
from torch import nn
from typing_extensions import Self, override

from nemo.lightning.ckpt_utils import WEIGHTS_PATH, ckpt_to_dir
from nemo.lightning.io.capture import IOProtocol
from nemo.lightning.io.mixin import IOMixin

try:
    from nemo.utils.callbacks.dist_ckpt_io import AsyncCompatibleCheckpointIO
except ImportError:
    AsyncCompatibleCheckpointIO = CheckpointIO


log = logging.getLogger(__name__)


LightningModuleT = TypeVar("LightningModuleT", bound=pl.LightningModule)
ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass
class TrainerContext(IOMixin, Generic[LightningModuleT]):
    model: LightningModuleT
    trainer: pl.Trainer
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trainer(cls, trainer: pl.Trainer) -> Self:
        if not hasattr(trainer, "__io__"):
            raise ValueError(f"Trainer must be an instance of {IOProtocol}. Please use the Trainer from nemo.")
        if not hasattr(trainer.lightning_module, "__io__"):
            raise ValueError("LightningModule must extend IOMixin.")

        return cls(trainer=trainer, model=trainer.lightning_module, extra=cls.construct_extra(trainer))

    @classmethod
    def construct_extra(cls, trainer: pl.Trainer) -> Dict[str, Any]:
        extra = {}
        if hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "__io__"):
            extra["datamodule"] = trainer.datamodule.__io__

        return extra


def ckpt_to_weights_subdir(filepath: Union[str, Path], is_saving) -> Path:
    """Given an input checkpoint filepath, clean it using `ckpt_to_dir` and then return the weights subdirectory, if it exists."""
    filepath = ckpt_to_dir(filepath=filepath)
    base_dir = filepath
    assert isinstance(base_dir, Path)
    if base_dir.parts[-1] != WEIGHTS_PATH:
        maybe_base_dir = base_dir / WEIGHTS_PATH
        if maybe_base_dir.is_dir() or is_saving:
            base_dir = maybe_base_dir
    ## handle adapter paths
    if hasattr(base_dir, "base_model_path") and base_dir.base_model_path.parts[-1] != WEIGHTS_PATH:
        maybe_base_model_path = base_dir.base_model_path / WEIGHTS_PATH
        if maybe_base_model_path.is_dir() or is_saving:
            base_dir.base_model_path = base_dir.base_model_path / WEIGHTS_PATH
    if is_saving:
        assert base_dir.parts[-1] == WEIGHTS_PATH
        assert base_dir.parent == Path(filepath)
    return base_dir


class MegatronCheckpointIO(AsyncCompatibleCheckpointIO, IOMixin):
    """CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints respectively,
    common for most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(
        self,
        save_ckpt_format: str = 'torch_dist',
        load_directly_on_device: bool = True,
        async_save: bool = False,
        torch_dist_multiproc: Optional[int] = None,
        assume_constant_structure: bool = False,
        parallel_save: bool = True,
        parallel_save_within_dp: bool = False,
        parallel_load: bool = False,
    ):
        self.save_ckpt_format = save_ckpt_format
        self.load_directly_on_device = load_directly_on_device
        self.async_save = async_save
        self.torch_dist_multiproc = torch_dist_multiproc
        self.assume_constant_structure = assume_constant_structure
        self.parallel_save = parallel_save
        self.parallel_save_within_dp = parallel_save_within_dp
        self.parallel_load = parallel_load

        self._save_sharded_strategy = None
        self.validated_consistency = False

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``

        Raises
        ------
            TypeError:
                If ``storage_options`` arg is passed in

        """
        from megatron.core import dist_checkpointing

        if storage_options is not None and len(storage_options) > 0:
            logging.warning(
                f"{self.__class__.__name__} does not support"
                f" storage_options, but {storage_options=} was provided."
                f" Ignoring given storage_options"
            )
        checkpoint_dir = ckpt_to_weights_subdir(path, is_saving=True)

        fs = get_filesystem(checkpoint_dir)
        fs.makedirs(checkpoint_dir, exist_ok=True)

        validate_sharding_integrity = not (self.validated_consistency and self.assume_constant_structure)
        self.validated_consistency = True

        return dist_checkpointing.save(
            sharded_state_dict=checkpoint,
            checkpoint_dir=checkpoint_dir,
            sharded_strategy=self.save_sharded_strategy,
            validate_access_integrity=validate_sharding_integrity,
            async_sharded_save=self.async_save,
        )

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        sharded_state_dict=None,
        map_location: Optional[Callable] = None,
        strict: Optional['StrictHandling'] | bool = None,
    ) -> Dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
                locations.

        Returns: The loaded checkpoint.

        Raises
        ------
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem

        """
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.validation import StrictHandling

        if map_location is not None:
            raise ValueError("`map_location` argument is not supported for `MegatronCheckpointIO.load_checkpoint`.")

        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if not fs.isdir(path):
            raise ValueError(f"Distributed checkpoints should be a directory. Found: {path}.")

        # Load from ckpt_path/weights (new format) if it exists
        path = ckpt_to_weights_subdir(path, is_saving=False)
        if hasattr(path, "base_model_path") and not path.base_model_path.exists():
            path.base_model_path = path.base_model_path.parent

        if self.save_ckpt_format == 'zarr' and self.load_directly_on_device:
            from megatron.core.dist_checkpointing.strategies.tensorstore import TensorStoreLoadShardedStrategy

            sharded_strategy = TensorStoreLoadShardedStrategy(load_directly_on_device=True)
        else:
            sharded_strategy = None

        if self.parallel_load:
            if sharded_strategy is None:
                sharded_strategy = get_default_load_sharded_strategy(path)
            sharded_strategy = FullyParallelLoadStrategyWrapper(
                sharded_strategy, get_data_parallel_group(with_context_parallel=True)
            )

        if sharded_strategy is not None:
            logging.info(f'Using {sharded_strategy} dist-ckpt load strategy.')

        if isinstance(strict, bool):
            # For backward-compatibility reasons and a bug in MCore (strict check not applied to factories)
            # we must apply a simple strict check here.
            if not strict:
                sharded_state_dict = self.adjust_non_strict_load(path, sharded_state_dict)
            strict = StrictHandling.ASSUME_OK_UNEXPECTED if strict else StrictHandling.LOG_ALL
        if strict is None:
            # Default behavior
            strict = StrictHandling.ASSUME_OK_UNEXPECTED

        checkpoint = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=str(path),
            sharded_strategy=sharded_strategy,
            strict=strict,
        )
        checkpoint = _fix_tensors_device(checkpoint)

        return checkpoint

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f"Removed checkpoint: {path}")

    def _determine_dist_ckpt_save_strategy(self):
        """Determine the saving strategy based on constructor args.

        Relies on the default MCore strategy unless extra PyT Distributed format arguments
        are passed in config or in case of a fully parallel save in which case
        a parallelization wrapper is applied.
        """
        if self.save_ckpt_format == 'zarr':
            logging.warning(
                f'`zarr` distributed checkpoint backend is deprecated.'
                f' Distributed optimizer checkpoint saving might be extremely slow.'
                f' Please switch to PyTorch Distributed format (model.dist_ckpt_format=torch_dist).'
            )

        if self.async_save and self.save_ckpt_format != 'torch_dist':
            raise ValueError('Async dist-ckpt save supported only for torch_dist format')

        torch_dist_kwargs = {} if self.torch_dist_multiproc is None else dict(thread_count=self.torch_dist_multiproc)
        if self.save_ckpt_format == 'torch_dist' and torch_dist_kwargs:
            save_strategy = TorchDistSaveShardedStrategy(self.save_ckpt_format, 1, **torch_dist_kwargs)
        else:
            save_strategy = get_default_save_sharded_strategy(self.save_ckpt_format, 1)

        # MCore v0.8 introduces `use_cached_ckpt_structure` attribute
        if hasattr(save_strategy, 'use_cached_ckpt_structure'):
            save_strategy.use_cached_ckpt_structure = self.assume_constant_structure

        if self.parallel_save:
            parallelization_group = (
                get_data_parallel_group(with_context_parallel=True) if self.parallel_save_within_dp else None
            )
            save_strategy = FullyParallelSaveStrategyWrapper(
                save_strategy, parallelization_group, self.assume_constant_structure
            )

        logging.info(f'Using {save_strategy} dist-ckpt save strategy.')
        return save_strategy

    @property
    def save_sharded_strategy(self) -> 'SaveShardedStrategy':
        if self._save_sharded_strategy is None:
            self._save_sharded_strategy = self._determine_dist_ckpt_save_strategy()
        return self._save_sharded_strategy

    def adjust_non_strict_load(self, path: _PATH, sharded_state_dict: Dict[str, Any]):
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.dict_utils import extract_matching_values
        from megatron.core.dist_checkpointing.mapping import ShardedBase

        ckpt_sharded_metadata = dist_checkpointing.load_tensors_metadata(path)
        loaded_keys = []
        missing_keys = []
        unexpected_keys = []

        def should_remove_missing_sharded_base(x: Any):
            if isinstance(x, ShardedBase):
                if x.key in ckpt_sharded_metadata:
                    loaded_keys.append(x.key)
                    return False
                else:
                    unexpected_keys.append(x.key)
                    return True
            return False

        _, sharded_state_dict = extract_matching_values(sharded_state_dict, should_remove_missing_sharded_base)
        logging.info(f'The following keys are not in the checkpoint and will not be loaded: {unexpected_keys}')

        # TODO: compute missing_keys by:
        #  1. all_gather_object of loaded_keys
        #  2. missing_keys = ckpt_sharded_metadata.keys() - loaded_keys
        return sharded_state_dict


def _fix_tensors_device(ckpt: Dict) -> Dict:
    """Ensure checkpoint tensors are on the correct device."""
    assert torch.cuda.is_initialized(), (torch.cuda.is_available(), torch.cuda.is_initialized())
    cur_dev = torch.device("cuda", index=torch.cuda.current_device())
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace

    def _fix_device(t):
        if isinstance(t, torch.Tensor) and t.is_cuda and t.device != cur_dev:
            t = t.to(cur_dev)
        return t

    return dict_list_map_outplace(_fix_device, ckpt)


def is_distributed_ckpt(path) -> bool:
    """Check if the given path corresponds to a distributed checkpoint directory.

    This function determines if the specified path is a directory that contains a distributed
    checkpoint by checking the directory's metadata.

    Args:
        path (Union[str, Path]): The path to check for being a distributed checkpoint.

    Returns
    -------
        bool: True if the path is a distributed checkpoint directory, False otherwise.

    """
    from megatron.core import dist_checkpointing

    checkpoint_dir = ckpt_to_dir(path)
    fs = get_filesystem(checkpoint_dir)
    if fs.isdir(checkpoint_dir) and dist_checkpointing.check_is_distributed_checkpoint(checkpoint_dir):
        return True

    return False
