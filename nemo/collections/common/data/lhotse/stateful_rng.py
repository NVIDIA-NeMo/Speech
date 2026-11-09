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

How the state reaches the checkpoint:

* Map-style dataloading (the sampler lives in the training process): ``torchdata``'s ``StatefulDataLoader`` calls
  ``dataset.state_dict()`` in every worker (or in the main process when ``num_workers=0``) and restores it with
  ``dataset.load_state_dict()`` before the worker resumes.
* Iterable-style dataloading: NeMo wraps the dataset in :class:`StatefulIterableDatasetWrapper`, which adds the
  wrapped dataset's state to the state of Lhotse's ``IterableDatasetWrapper`` (which only stores the sampler).
"""
import random
from typing import Any

import torch
from lhotse.dataset import IterableDatasetWrapper
from lhotse.dataset.dataloading import get_rank


class StatefulRandomMixin:
    """
    Mixin that gives a dataset a private :class:`random.Random` (``self.rng``) whose state is saved and restored
    by ``torchdata``'s ``StatefulDataLoader``.

    The generator is created lazily on first use in the process that uses it (a dataloader worker, or the
    training process when ``num_workers=0``). It is seeded from ``torch.initial_seed()``, which PyTorch and
    Lhotse's ``worker_init_fn`` set per worker from the dataloader seed, combined with the data-parallel rank.
    The draws are therefore deterministic given the dataloader seed and the position in the data, and are
    restored exactly on resume.

    Example::

        >>> class MyDataset(StatefulRandomMixin, torch.utils.data.Dataset):
        ...     def __getitem__(self, cuts):
        ...         offsets = [self.rng.uniform(0, c.duration) for c in cuts]
    """

    @property
    def rng(self) -> random.Random:
        """The dataset's private random generator for the current process."""
        rng = self.__dict__.get("_stateful_rng")
        if rng is None:
            rng = random.Random(f"{torch.initial_seed()}:{get_rank()}")
            self._stateful_rng = rng
        return rng

    def state_dict(self) -> dict[str, Any]:
        """Return the random generator state, using only plain Python types."""
        version, internal_state, gauss_next = self.rng.getstate()
        return {"rng_state": {"version": version, "internal_state": list(internal_state), "gauss_next": gauss_next}}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the random generator state saved by :meth:`state_dict`."""
        state = state_dict["rng_state"]
        rng = random.Random()
        rng.setstate((state["version"], tuple(state["internal_state"]), state["gauss_next"]))
        self._stateful_rng = rng


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


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))
