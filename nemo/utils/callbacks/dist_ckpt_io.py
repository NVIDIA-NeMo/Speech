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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from time import time
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning import Callback
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO

from nemo.utils import logging

try:
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.dict_utils import extract_matching_values
    from megatron.core.dist_checkpointing.mapping import ShardedBase
    from megatron.core.dist_checkpointing.serialization import (
        get_default_load_sharded_strategy,
        get_default_save_sharded_strategy,
    )
    from megatron.core.dist_checkpointing.strategies import tensorstore
    from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
    from megatron.core.dist_checkpointing.strategies.base import SaveShardedStrategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
        FullyParallelSaveStrategyWrapper,
    )
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
    from megatron.core.dist_checkpointing.validation import StrictHandling
    from megatron.core.parallel_state import get_data_parallel_group

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    HAVE_MEGATRON_CORE = False
    IMPORT_ERROR = (
        "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        f" Exact error: {e}"
    )


@contextmanager
def _debug_time(name: str):
    """Simple context manager for timing functions/code blocks."""
    start = time()
    try:
        yield
    finally:
        logging.debug(f'{name} took {time() - start:.3f}s')


class AsyncCompatibleCheckpointIO(CheckpointIO, ABC):
    """CheckpointIO that can be used together with async saving.

    Differs from the regular CheckpointIO only by the `save_checkpoint`
    return type. The `save_checkpoint` method itself is synchronous, but returns
    callbacks that can be performed asynchronously.
    """

    @abstractmethod
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None
    ) -> 'AsyncRequest':
        raise NotImplementedError


class AsyncFinalizableCheckpointIO(_WrappingCheckpointIO):
    """CheckpointIO wrapper for async checkpoint saving and synchronous finalization.

    Runs main part of the checkpoint save in a separate process (not thread as the PTL
    AsyncCheckpointIO does). Allows to perform a (synchronous) finalization
    function after all ranks finish checkpoint saving.

    NOTE: for correctness, this plugin must be used together with the
    AsyncFinalizerCallback callback which performs the finalization checks.

    Args:
        checkpoint_io (CheckpointIO): wrapped checkpoint_io object. Must be
            of type AsyncCompatibleCheckpointIO.
    Requires the underlying checkpoint_io.save_checkpoint to return save_fn, save_args, finalize_fn.
    """

    def __init__(self, checkpoint_io: AsyncCompatibleCheckpointIO) -> None:
        if not HAVE_MEGATRON_CORE:
            raise ImportError(IMPORT_ERROR)
        if not isinstance(checkpoint_io, AsyncCompatibleCheckpointIO):
            raise ValueError(f'Incompatible wrapped checkpoint_io type: {type(checkpoint_io)}')

        super().__init__(checkpoint_io)
        self.async_calls_queue = AsyncCallsQueue()

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Executes async request returned from the underlying checkpoint_io asynchronously.

        Requires the underlying checkpoint_io.save_checkpoint to return an AsyncRequest.
        It is then applied with `self.async_calls_queue` asynchronously.

        Args:
            checkpoint (Dict[str, Any]): checkpoint to save. Passed to underlying
                checkpoint_io without modifications.
            path (_PATH): path to save the checkpoint. Passed to underlying
                checkpoint_io without modifications.
            storage_options (Any, optional): storage control modifiers. This class
                consumed the `finalize_fn` parameter (if any), which is expected to be
                a callback and is appended to async finalization functions.

        Applies underlying checkpoint_io finalize callback first, then the external one (postfix order).
        """
        external_finalize_fn = (storage_options or {}).pop('finalize_fn', None)
        assert isinstance(self.checkpoint_io, AsyncCompatibleCheckpointIO), type(self.checkpoint_io)
        async_request = self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options)
        if external_finalize_fn is not None:
            async_request.add_finalize_fn(external_finalize_fn)
        call_idx = self.async_calls_queue.schedule_async_request(async_request)
        logging.debug(f'Scheduled an async call #{call_idx}')

    @_debug_time('AsyncFinalizableCheckpointIO.maybe_finalize_save_checkpoint')
    def maybe_finalize_save_checkpoint(self, blocking: bool = False):
        """Performs checkpoint finalization (if possible).

        Args:
            blocking (bool, optional): if True, waits until all async saves are
                completed. Otherwise, finalizes only those async calls which are
                already done on all ranks. Defaults to False.
        """
        call_idx_finalized = self.async_calls_queue.maybe_finalize_async_calls(blocking)
        if call_idx_finalized:
            logging.debug(f'Finalized async calls: {[f"#{idx}" for idx in call_idx_finalized]}')
        return len(call_idx_finalized) > 0

    def teardown(self) -> None:
        """Warns if there are any pending checkpoint saves."""
        super().teardown()
        if self.async_calls_queue.get_num_unfinalized_calls() > 0:
            # Can't do finalization now because some ranks might be lost
            logging.warning('Some async checkpoint saves might be not finalized properly.')


class AsyncFinalizerCallback(Callback):
    """Callback which finalizes async saves initiated by the AsyncFinalizableCheckpointIO.

    Tries to perform non-blocking finalization on train_batch_end and train_epoch_end.
    On train_end performs a blocking finalization of all pending checkpoints.
    """

    def on_train_batch_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        self._get_checkpoint_io(trainer).maybe_finalize_save_checkpoint(blocking=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        self._get_checkpoint_io(trainer).maybe_finalize_save_checkpoint(blocking=False)

    def on_train_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        checkpoint_io = self._get_checkpoint_io(trainer)
        if checkpoint_io.async_calls_queue.get_num_unfinalized_calls() > 0:
            logging.info('Pending async checkpoint saves. Finalizing them synchronously now')
        self._get_checkpoint_io(trainer).maybe_finalize_save_checkpoint(blocking=True)

    def _get_checkpoint_io(self, trainer) -> AsyncFinalizableCheckpointIO:
        checkpoint_io = trainer.strategy.checkpoint_io
        if not isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
            raise ValueError(f'Async finalizer requires an async compatible CheckpointIO, got: {checkpoint_io}')
        return checkpoint_io


class DistributedCheckpointIO(AsyncCompatibleCheckpointIO):
    """CheckpointIO for a distributed checkpoint format.

    Args:
        save_ckpt_format (str): Distributed checkpoint format to use for checkpoint saving.
        load_directly_on_device (bool, optional): if True, loads the weights directly
            on GPU. Has effect only for `zarr` based checkpoints (PyT Distributed
            always loads on device). Defaults to True.
        load_strictness (StrictHandling, optional): defines loading strictness.
            If not None, overwrites the `strict` flag passed to `load_checkpoint`.
            Defaults to None.
        async_save (bool): whether to save asynchronously. Should be set to True if
            this class will be wrapped with AsyncFinalizableCheckpointIO.
        torch_dist_multiproc (int, optional): number of extra processes per rank
            used during ckpt save with PyTorch distributed format. Defaults, to None
            which means using an MCore default (2).
        parallel_save (bool): parallelizes the save across ranks. Defaults to True
        parallel_load (bool): parallelizes the load across ranks (followed by params all gather).
            Defaults to False due to some extra memory usage requirement.
    """

    def __init__(
        self,
        save_ckpt_format: str,
        load_directly_on_device: bool = True,
        load_strictness: Optional['StrictHandling'] = None,
        async_save: bool = False,
        torch_dist_multiproc: Optional[int] = None,
        assume_constant_structure: bool = False,
        parallel_save: bool = False,
        parallel_save_within_dp: bool = False,
        parallel_load: bool = False,
    ):
        super().__init__()
        if not HAVE_MEGATRON_CORE:
            raise ImportError(IMPORT_ERROR)

        self.save_ckpt_format = save_ckpt_format
        self.load_directly_on_device = load_directly_on_device
        self.load_strictness = load_strictness
        self.async_save = async_save
        self.torch_dist_multiproc = torch_dist_multiproc
        self.assume_constant_structure = assume_constant_structure
        self.parallel_save = parallel_save
        self.parallel_save_within_dp = parallel_save_within_dp
        self.parallel_load = parallel_load

        self._save_sharded_strategy = None
        self.validated_consistency = False

    @classmethod
    def from_config(cls, model_cfg: dict, async_save: bool = False):
        """Instantiates a DistributedCheckpointIO from a config dict.

        Args:
            model_cfg (dict): model config dict. Most of the configuration
                is extracted from this config.
            async_save (bool, optional): async_save flag is not part of the model config,
                it should be provided separately. Defaults to False.
        """
        return cls(
            save_ckpt_format=model_cfg.get('dist_ckpt_format', 'zarr'),
            load_directly_on_device=model_cfg.get('dist_ckpt_load_on_device', True),
            load_strictness=model_cfg.get('dist_ckpt_load_strictness', None),
            async_save=async_save,
            torch_dist_multiproc=model_cfg.get('dist_ckpt_torch_dist_multiproc', None),
            parallel_save=model_cfg.get('dist_ckpt_parallel_save', False),
            parallel_save_within_dp=model_cfg.get('dist_ckpt_parallel_save_within_dp', False),
            parallel_load=model_cfg.get('dist_ckpt_parallel_load', False),
        )

    @_debug_time('DistributedCheckpointIO.save_checkpoint')
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None
    ) -> Optional['AsyncRequest']:
        """Saves a distributed checkpoint. Creates the checkpoint root directory if doesn't exist.

        Args:
            checkpoint (Dict[str, Any]): sharded state dict to save
            path (_PATH): checkpoint directory
            storage_options (Any, optional): Optional parameters when saving the checkpoint
        """
        fs = get_filesystem(path)
        fs.makedirs(path, exist_ok=True)

        validate_sharding_integrity = not (self.validated_consistency and self.assume_constant_structure)
        self.validated_consistency = True
        return dist_checkpointing.save(
            sharded_state_dict=checkpoint,
            checkpoint_dir=path,
            sharded_strategy=self.save_sharded_strategy,
            validate_access_integrity=validate_sharding_integrity,
            async_sharded_save=self.async_save,
        )

    @_debug_time('DistributedCheckpointIO.load_checkpoint')
    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Any] = None,
        sharded_state_dict: Dict[str, Any] = None,
        strict: Union[None, bool, 'StrictHandling'] = None,
        validate_access_integrity: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """Loads a distributed checkpoint.

        Args:
            path (_PATH): checkpoint directory
            map_location (Any, optional): required to be None in this implementation
            sharded_state_dict (Dict[str, Any], optional): state dict which
                defines the loading procedure for the distributed checkpoint.
                Defaults to None to comply with the CheckpointIO interface,
                but it's a required argument.
            strict (bool, StrictHandling, optional): adjust load strictness. bool value
                is translated to StrictHandling instance. Gets overwritten by
                `self.load_strictness`. Defaults to None. If `self.load_strictness`
                is also None, strict becomes StrictHandling.ASSUME_OK_UNEXPECTED.

        Returns:
            Dist[str, Any]: loaded checkpoint.
        """
        if sharded_state_dict is None:
            raise ValueError('DistributedCheckpointIO requires passing sharded_state_dict argument to load_checkpoint')
        if map_location is not None:
            raise ValueError('DistributedCheckpointIO doesnt handle map_location argument')

        if self.save_ckpt_format == 'zarr' and self.load_directly_on_device:
            sharded_strategy = tensorstore.TensorStoreLoadShardedStrategy(load_directly_on_device=True)
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
        if self.load_strictness is not None:
            # Overwrites function argument
            strict = self.load_strictness
        if strict is None:
            # Default behavior
            strict = StrictHandling.ASSUME_OK_UNEXPECTED

        logging.debug(f'Dist ckpt load strictness: {strict}')

        return dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=path,
            sharded_strategy=sharded_strategy,
            validate_access_integrity=validate_access_integrity,
            strict=strict,
        )

    def adjust_non_strict_load(self, path: _PATH, sharded_state_dict: Dict[str, Any]):
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

    @_debug_time('DistributedCheckpointIO.remove_checkpoint')
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove a distributed checkpoint.

        Due to potentially large number of files, the implementation remove the whole directory at once.
        """
        shutil.rmtree(path, ignore_errors=True)

    @property
    def save_sharded_strategy(self) -> 'SaveShardedStrategy':
        if self._save_sharded_strategy is None:
            self._save_sharded_strategy = self._determine_dist_ckpt_save_strategy()
        return self._save_sharded_strategy

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
