# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""
Checkpointable randomness for Lhotse datasets.

Per-example randomness in a dataset's ``__getitem__`` (random crops, context slicing, text augmentation, ...)
must be part of the dataloader state; otherwise a run resumed from a ``StatefulDataLoader`` checkpoint reads the
same cuts as the uninterrupted run but draws different random numbers for them. The global ``random`` and
``np.random`` generators are not checkpointed, so datasets use :class:`StatefulRandomMixin` instead.

Checkpointed randomness is opt-in: it is enabled when the dataset is used with ``use_stateful_dataloader=True``
(see :func:`configure_dataset_checkpointed_rng`). Otherwise ``self.rng`` is the global ``random`` module, so
existing configurations keep drawing exactly the same random numbers as before.

How the state reaches the checkpoint:

* Map-style dataloading (the sampler lives in the training process): ``torchdata``'s ``StatefulDataLoader`` calls
  ``dataset.state_dict()`` in every worker (or in the main process when ``num_workers=0``) and restores it with
  ``dataset.load_state_dict()`` before the worker resumes.
* Iterable-style dataloading: NeMo wraps the dataset in :class:`StatefulIterableDatasetWrapper`, which adds the
  wrapped dataset's state to the state of Lhotse's ``IterableDatasetWrapper`` (which only stores the sampler).
"""
import os
import random
from types import ModuleType
from typing import Any, Union

import torch
from lhotse.dataset import IterableDatasetWrapper
from lhotse.dataset.dataloading import get_rank


class StatefulRandomMixin:
    """
    Mixin that gives a dataset a private :class:`random.Random` (``self.rng``) whose state is saved and restored
    by ``torchdata``'s ``StatefulDataLoader``.

    Until :meth:`enable_checkpointed_rng` is called, ``self.rng`` is the global ``random`` module (the legacy
    behaviour) and the saved state is empty. Once enabled, the generator is created lazily on first use in the
    process that uses it (a dataloader worker, or the training process when ``num_workers=0``) with
    :func:`process_seeded_rng`, i.e. from ``torch.initial_seed()``, which PyTorch and Lhotse's ``worker_init_fn``
    set per worker from the dataloader seed, combined with the data-parallel rank. A generator created in another
    process (e.g. by a ``state_dict()`` call before the dataset was copied into dataloader workers) is replaced,
    so workers never share a sequence. The draws are deterministic given the dataloader seed and the position in
    the data, and are restored exactly on resume. ``load_state_dict`` restores the generator in place, so objects
    holding a reference to ``self.rng`` keep drawing from the restored sequence.

    Example::

        >>> class MyDataset(StatefulRandomMixin, torch.utils.data.Dataset):
        ...     def __getitem__(self, cuts):
        ...         offsets = [self.rng.uniform(0, c.duration) for c in cuts]
    """

    def enable_checkpointed_rng(self) -> None:
        """Switch ``self.rng`` from the global ``random`` module to a private, checkpointed generator."""
        self._checkpointed_rng_enabled = True

    @property
    def checkpointed_rng_enabled(self) -> bool:
        """Whether ``self.rng`` is a private, checkpointed generator (see :meth:`enable_checkpointed_rng`)."""
        return self.__dict__.get("_checkpointed_rng_enabled", False)

    @property
    def rng(self) -> Union[random.Random, ModuleType]:
        """The private generator of the current process when enabled, otherwise the global ``random`` module."""
        if not self.checkpointed_rng_enabled:
            return random
        pid = os.getpid()
        rng = self.__dict__.get("_stateful_rng")
        if rng is None or self.__dict__.get("_stateful_rng_pid") != pid:
            rng = process_seeded_rng()
            self._stateful_rng = rng
            self._stateful_rng_pid = pid
        return rng

    def state_dict(self) -> dict[str, Any]:
        """Return the generator state using only plain Python types (``None`` when not enabled)."""
        if not self.checkpointed_rng_enabled:
            return {"rng_state": None}
        version, internal_state, gauss_next = self.rng.getstate()
        return {"rng_state": {"version": version, "internal_state": list(internal_state), "gauss_next": gauss_next}}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore, in place, the generator state saved by :meth:`state_dict`."""
        state = state_dict["rng_state"]
        if state is None:
            return
        if not self.checkpointed_rng_enabled:
            raise RuntimeError(
                f"The checkpoint contains a random generator state for {type(self).__name__}, but checkpointed "
                "randomness is not enabled for it (it is enabled by use_stateful_dataloader=True)."
            )
        # self.rng returns the generator owned by this process (creating it if needed); restore it in place.
        self.rng.setstate((state["version"], tuple(state["internal_state"]), state["gauss_next"]))


class StatefulIterableDatasetWrapper(IterableDatasetWrapper):
    """
    Lhotse's :class:`~lhotse.dataset.iterable_dataset.IterableDatasetWrapper` that also checkpoints the state of
    the wrapped dataset (e.g. the random generator of a :class:`StatefulRandomMixin` dataset).

    Lhotse's wrapper only saves the epoch and the sampler state, so without this the wrapped dataset's random
    draws after a resume would differ from the uninterrupted run.
    """

    def state_dict(self) -> dict:
        state = super().state_dict()
        if _is_stateful(self.dataset):
            state["dataset_state"] = self.dataset.state_dict()
        return state

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        # Checkpoints written before the dataset state was saved don't have the key; the dataset then keeps
        # its freshly seeded state, which matches the behaviour those checkpoints were saved with.
        if "dataset_state" in sd:
            if not _is_stateful(self.dataset):
                raise RuntimeError(
                    f"The dataloader checkpoint contains a dataset state, but the dataset "
                    f"{type(self.dataset).__name__} does not implement load_state_dict()."
                )
            self.dataset.load_state_dict(sd["dataset_state"])


def configure_dataset_checkpointed_rng(dataset: Any, enabled: bool) -> Any:
    """
    Enable checkpointed randomness on a :class:`StatefulRandomMixin` dataset when ``enabled``. Wrappers that keep
    the wrapped dataset in ``.dataset`` (e.g. ``FallbackDataset``) are looked through.
    """
    target = dataset
    while enabled and target is not None:
        if isinstance(target, StatefulRandomMixin):
            target.enable_checkpointed_rng()
            break
        target = getattr(target, "dataset", None)
    return dataset


def process_seeded_rng(namespace: str = "") -> random.Random:
    """
    Create a :class:`random.Random` seeded for the current process.

    The seed combines ``namespace``, ``torch.initial_seed()`` (which PyTorch and Lhotse's ``worker_init_fn`` set
    per dataloader worker from the dataloader seed) and the data-parallel rank, so it is deterministic given the
    seeds and differs between workers and ranks. It never touches the global ``random`` state.
    """
    return random.Random(f"{namespace}{torch.initial_seed()}:{get_rank()}")


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))
