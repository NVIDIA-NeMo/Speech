# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Helpers for working with PyTorch Lightning's ``training_step``."""
from typing import Any, Iterator, Tuple

import lightning.pytorch as pl


def read_batch(dataloader_iter: Iterator, model: pl.LightningModule) -> Tuple[Any, int]:
    """Pull the next batch from a Lightning ``dataloader_iter`` and apply the
    device/precision conversions that ``_PrefetchDataFetcher`` would have
    applied for the default ``training_step(batch, batch_idx)`` signature.

    Use this from a ``training_step(self, dataloader_iter)``-style step. That
    signature makes Lightning select ``_DataLoaderIterDataFetcher`` (no
    prefetch), which is required for bit-identical checkpoint resumption with
    a stateful dataloader: the default ``_PrefetchDataFetcher`` re-primes one
    batch on every iter init (including on resume), advancing the stateful
    dataloader past the saved snapshot point and giving the resumed run a
    one-batch drift versus the continuous run.

    Also checks shutdown conditions before pulling the next batch. Lightning
    still calls timer and preemption callbacks in normal ``dataloader_iter``
    runs, but checking here closes the deadline/preemption window before user
    code advances a stateful iterator. If the time budget is already exhausted
    or preemption was already signaled, the helper saves ``last.ckpt`` and
    exits before another sample is consumed.

    Args:
        dataloader_iter: The iterator passed by Lightning into a
            ``training_step(self, dataloader_iter)`` (an instance of
            ``_DataFetcherWrapper``). Yields ``(batch, batch_idx, dataloader_idx)``.
        model: The ``LightningModule`` whose ``trainer`` carries the precision
            plugin and strategy used to move the batch to device.

    Returns:
        ``(batch, batch_idx)`` — batch is already converted to the right
        precision and moved to the model's device, ready for forward.
    """
    trainer = model.trainer
    _check_shutdown_before_next_batch(trainer)
    batch, batch_idx, dataloader_idx = next(dataloader_iter)
    batch = trainer.precision_plugin.convert_input(batch)
    batch = model._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
    batch = trainer.strategy.batch_to_device(batch, dataloader_idx=dataloader_idx)
    return batch, batch_idx


def _check_shutdown_before_next_batch(trainer: pl.Trainer) -> None:
    """Handle pending shutdown before advancing a stateful ``dataloader_iter``."""
    _log_read_batch_shutdown_guards_once(trainer)
    _force_fire_preemption_callback(trainer)
    _save_and_exit_if_lightning_received_sigterm(trainer)
    _force_fire_stateless_timer(trainer)


def _log_read_batch_shutdown_guards_once(trainer: pl.Trainer) -> None:
    """Log the active ``read_batch`` shutdown guards once per trainer."""
    if getattr(trainer, "_nemo_read_batch_shutdown_guards_logged", False):
        return
    setattr(trainer, "_nemo_read_batch_shutdown_guards_logged", True)

    try:
        has_preemption = any(getattr(cb, "preemption_enabled", False) for cb in trainer.callbacks)
        has_sigterm = hasattr(trainer, "received_sigterm")
        has_timer = _has_stateless_timer(trainer)
        from nemo.utils import logging

        logging.info(
            "read_batch shutdown guards active: "
            f"stateless_timer={has_timer} preemption_callback={has_preemption} "
            f"lightning_sigterm_state={has_sigterm}"
        )
    except Exception:
        # This is observability only; never let it affect the training path.
        return


def _force_fire_preemption_callback(trainer: pl.Trainer) -> None:
    """Save and exit if NeMo's preemption callback has observed SIGTERM.

    ``PreemptionCallback.on_train_batch_end`` still handles the normal
    post-batch case. This pre-fetch check covers the ``training_step(
    dataloader_iter)`` path where user code is responsible for advancing the
    stateful iterator and can otherwise enter ``next(dataloader_iter)`` after
    rank 0 already received the preemption signal.
    """
    for cb in trainer.callbacks:
        if not getattr(cb, "preemption_enabled", False):
            continue
        if cb.interrupted:
            from nemo.utils.exp_manager import _save_last_checkpoint_and_exit

            _save_last_checkpoint_and_exit(
                trainer,
                "read_batch observed a pending preemption signal before consuming the next batch",
            )


def _save_and_exit_if_lightning_received_sigterm(trainer: pl.Trainer) -> None:
    """Handle Lightning's own SIGTERM state before consuming a stateful batch."""
    if not getattr(trainer, "received_sigterm", False):
        return

    from nemo.utils.exp_manager import _save_last_checkpoint_and_exit

    _save_last_checkpoint_and_exit(
        trainer,
        "read_batch observed trainer.received_sigterm before consuming the next batch",
    )


def _has_stateless_timer(trainer: pl.Trainer) -> bool:
    """Return whether trainer has NeMo's StatelessTimer callback."""
    from nemo.utils.exp_manager import StatelessTimer

    return any(isinstance(cb, StatelessTimer) for cb in trainer.callbacks)


def _force_fire_stateless_timer(trainer: pl.Trainer) -> None:
    """Invoke ``StatelessTimer._check_time_remaining`` directly.

    Defensive deadline check for Lightning's ``dataloader_iter`` step flavor.
    The standard callback path checks the timer after a batch. This pre-fetch
    check prevents a resumed stateful iterator from being advanced when the
    deadline has already expired before the next batch is requested.

    Idempotent on the time-not-yet-up case (cheap: one ``time_elapsed()``
    check + one comparison). On the time-up case, ``StatelessTimer`` saves a
    ``-last.ckpt`` via ``NeMoModelCheckpoint._save_last_checkpoint`` and
    raises ``_TunerExitException`` to exit Lightning gracefully — that
    exception propagates up through ``read_batch`` → ``training_step`` →
    Lightning's epoch loop, which Lightning treats as a clean stop.
    """
    # Local import to avoid a circular import at module load time
    # (exp_manager imports from various nemo submodules).
    from nemo.utils.exp_manager import StatelessTimer

    for cb in trainer.callbacks:
        if isinstance(cb, StatelessTimer):
            cb._check_time_remaining(trainer)
            return
