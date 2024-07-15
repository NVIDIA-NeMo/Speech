import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
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
        checkpoint_dir = ckpt_to_dir(path)
        fs = get_filesystem(checkpoint_dir)
        if fs.isdir(checkpoint_dir) and dist_checkpointing.check_is_distributed_checkpoint(checkpoint_dir):
            logging.info(f'Distributed checkpoint at path {checkpoint_dir} already exists, skipping saving')
            return
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
        self, path: _PATH, sharded_state_dict=None, map_location: Optional[Callable] = None
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

        if map_location is not None:
            raise ValueError("`map_location` argument is not supported for `MegatronCheckpointIO.load_checkpoint`.")

        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if not fs.isdir(path):
            raise ValueError(f"Distributed checkpoints should be a directory. Found: {path}.")

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

        checkpoint = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict, checkpoint_dir=str(path), sharded_strategy=sharded_strategy
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


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints.
    """
    filepath = Path(filepath)
    if not filepath.suffix == ".ckpt":
        filepath = filepath.with_suffix(filepath.suffix + ".ckpt")

    # adding this assert because we will later remove directories based on the return value of this method
    assert filepath.suffix == ".ckpt", f"filepath: {filepath} must have .ckpt extension"

    # create a new path whose name is the original filepath without the .ckpt extension
    checkpoint_dir = filepath.with_name(filepath.stem)

    return checkpoint_dir


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
