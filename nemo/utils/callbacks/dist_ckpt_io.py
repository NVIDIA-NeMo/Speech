# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from abc import ABC, abstractmethod
from contextlib import contextmanager
from time import time
from typing import Any, Dict, Optional

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import Callback
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO

from nemo.utils import logging

try:
    from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    HAVE_MEGATRON_CORE = False
    IMPORT_ERROR = f"Async checkpoint saving requires the optional megatron-core async utilities. Exact error: {e}"


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
        """Interface to implement save_checkpoint and return an AsyncRequest"""
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

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
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
        if self.async_calls_queue.get_num_unfinalized_calls() == 0:
            return False

        start_time = time()
        call_idx_finalized = self.async_calls_queue.maybe_finalize_async_calls(blocking)
        if call_idx_finalized:
            logging.debug(f'Finalized async calls: {[f"#{idx}" for idx in call_idx_finalized]}')
        end_time = time()
        logging.info(f"Async finalization time took {end_time - start_time:.3f} s")
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
        """Override hook to finalize pending checkpoint(s) if they exist."""
        self._get_checkpoint_io(trainer).maybe_finalize_save_checkpoint(blocking=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        """Override hook to finalize pending checkpoint(s) if they exist."""
        self._get_checkpoint_io(trainer).maybe_finalize_save_checkpoint(blocking=False)

    def on_train_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        """Override hook to finalize pending checkpoint(s) if they exist."""
        checkpoint_io = self._get_checkpoint_io(trainer)
        if checkpoint_io.async_calls_queue.get_num_unfinalized_calls() > 0:
            logging.info('Pending async checkpoint saves. Finalizing them synchronously now')
        self._get_checkpoint_io(trainer).maybe_finalize_save_checkpoint(blocking=True)

    def _get_checkpoint_io(self, trainer) -> AsyncFinalizableCheckpointIO:
        checkpoint_io = trainer.strategy.checkpoint_io
        if not isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
            raise ValueError(
                f'Async finalizer requires an async compatible CheckpointIO, got: {checkpoint_io.__class__}'
            )
        return checkpoint_io
