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
"""``StatefulDataLoader`` subclass for Lhotse samplers.

This module imports torchdata, so ``nemo.collections.common.data.lhotse.dataloader`` imports it lazily, only when
``use_stateful_dataloader=True``. The class lives at module scope so that loaders (and models holding them) can be
pickled, e.g. by spawn-based launchers.
"""

from lhotse.dataset.sampling.base import CutSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo.collections.common.data.lhotse.dataloader import _ContinueIterableStreamMixin


class LhotseStatefulDataLoader(_ContinueIterableStreamMixin, StatefulDataLoader):
    """``StatefulDataLoader`` that leaves a restored Lhotse sampler in a normal iterating state.

    Iterable-style datasets keep one live iterator across ``iter()`` calls; see
    :class:`~nemo.collections.common.data.lhotse.dataloader._ContinueIterableStreamMixin`.
    ``load_state_dict()`` drops that iterator, so the next ``iter()`` resumes from the loaded state.

    When torchdata restores a map-style loader, it loads the saved state into the index
    sampler, calls ``iter()`` on it, and then loads the same state again into the sampler
    iterator. A Lhotse ``CutSampler`` is its own iterator, so the second load re-arms the
    deferred restore (``_needs_fast_forward``) of a sampler that is already positioned at the
    checkpoint. In lhotse<=2.0.0a6 that flag is only consumed by the next ``iter()``. Until
    then ``set_epoch()`` is ignored, so at the next epoch boundary the sampler rewinds to the
    checkpoint instead of starting the new epoch, and the resumed run diverges from the
    uninterrupted one.

    Once the restored iterator exists, the restore has been applied, so we clear the pending
    restore state with the public ``CutSampler.allow_iter_to_reset_state()``. Iterable-style
    loaders keep their sampler inside ``IterableDatasetWrapper`` (``self.sampler`` is not a
    ``CutSampler``), so they are left alone.
    """

    def __iter__(self):
        restoring = self.next_iter_state is not None
        iterator = super().__iter__()
        if restoring and isinstance(self.sampler, CutSampler):
            self.sampler.allow_iter_to_reset_state()
        return iterator
