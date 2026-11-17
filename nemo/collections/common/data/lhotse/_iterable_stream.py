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
"""Dataloader pieces that keep an iterable Lhotse stream alive across Lightning epochs.

Shared by the plain loader in ``dataloader.py`` and the stateful loader in ``_stateful_dataloader.py``. This module
depends only on torch, so both can import it without importing each other.
"""

import torch
from torch.utils.data.dataloader import _DatasetKind, _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter


class _ContinueIterableStreamMixin:
    """Make ``iter(loader)`` continue the live iterator of an iterable-style dataset.

    Lightning calls ``iter()`` on the train loader at every epoch boundary, including the boundaries set by
    ``limit_train_batches`` (speechlm2 uses them as an epoch length over an infinite Lhotse stream). A stock
    ``DataLoader`` with persistent workers then resets its iterator, which discards the
    ``prefetch_factor * num_workers`` batches the workers had already read from the stream, so that data is
    skipped at every boundary. Here ``iter()`` returns the existing iterator until the consumer has seen it raise
    ``StopIteration`` (the iterator's ``_finished`` flag, which torchdata's iterators set and
    :class:`_LhotseDataLoader`'s iterators mirror). Only then, i.e. at the end of a finite stream, does the next
    ``iter()`` start a new pass. Workers that have reported the end of their stream are not enough: the iterator
    may still hold their buffered batches. Map-style datasets keep the stock behavior: each ``iter()`` starts a
    new epoch of the sampler.
    """

    def __iter__(self):
        iterator = getattr(self, "_iterator", None)
        if (
            self._dataset_kind == _DatasetKind.Iterable
            and iterator is not None
            # StatefulDataLoader created this iterator in state_dict(); its own __iter__ hands it out.
            and not getattr(self, "_initial_iter_for_state_dict", False)
            and not iterator._finished
        ):
            return iterator
        return super().__iter__()


class _LhotseDataLoader(_ContinueIterableStreamMixin, torch.utils.data.DataLoader):
    """``torch.utils.data.DataLoader`` used for Lhotse data when ``use_stateful_dataloader=False``."""

    def _get_iterator(self):
        if self.num_workers == 0:
            return _FinishTrackingSingleProcessIter(self)
        self.check_worker_number_rationality()
        return _FinishTrackingMultiProcessingIter(self)


class _FinishTrackingIterMixin:
    """Record in ``_finished`` that the consumer saw ``StopIteration``, like torchdata's iterators do."""

    _finished = False

    def _reset(self, *args, **kwargs):
        super()._reset(*args, **kwargs)
        self._finished = False

    def __next__(self):
        try:
            return super().__next__()
        except StopIteration:
            self._finished = True
            raise


class _FinishTrackingSingleProcessIter(_FinishTrackingIterMixin, _SingleProcessDataLoaderIter):
    pass


class _FinishTrackingMultiProcessingIter(_FinishTrackingIterMixin, _MultiProcessingDataLoaderIter):
    pass
