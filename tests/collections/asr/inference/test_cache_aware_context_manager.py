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

import pytest
import torch

from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext, CacheAwareContextManager


class _FakeModel:
    """Cache-aware model stub that creates its initial cache on the CPU."""

    def get_initial_cache_state(self, batch_size):
        return torch.zeros(2, batch_size, 3, 4), torch.zeros(2, batch_size, 4, 5), torch.zeros(batch_size)


def _context_of(value_per_stream):
    """A context in batch order whose every cache entry of stream i is filled with value_per_stream[i]."""
    values = torch.tensor(value_per_stream, dtype=torch.float32)
    return CacheAwareContext(
        cache_last_channel=values.view(1, -1, 1, 1).expand(2, -1, 3, 4).clone(),
        cache_last_time=values.view(1, -1, 1, 1).expand(2, -1, 4, 5).clone(),
        cache_last_channel_len=values.clone(),
    )


@pytest.mark.unit
def test_cache_is_kept_per_stream_across_reordered_batches():
    manager = CacheAwareContextManager(_FakeModel(), num_slots=4)
    manager.get_context([10, 11, 12])
    manager.update_cache([10, 11, 12], _context_of([1.0, 2.0, 3.0]))

    # the same streams in another order, as the active batch changes between chunks
    context = manager.get_context([12, 10])
    assert context.cache_last_channel_len.tolist() == [3.0, 1.0]
    assert context.cache_last_channel[:, 0].eq(3.0).all()
    assert context.cache_last_time[:, 1].eq(1.0).all()


@pytest.mark.unit
def test_update_cache_for_other_streams_than_the_last_get_context():
    manager = CacheAwareContextManager(_FakeModel(), num_slots=4)
    manager.get_context([10, 11, 12])
    manager.get_context([11])
    # the slots of the last get_context must not be reused for streams it did not gather
    manager.update_cache([10, 12], _context_of([5.0, 7.0]))

    assert manager.get_context([10, 11, 12]).cache_last_channel_len.tolist() == [5.0, 0.0, 7.0]


@pytest.mark.unit
def test_finished_stream_is_zeroed_and_its_slot_reused():
    manager = CacheAwareContextManager(_FakeModel(), num_slots=2)
    manager.get_context([10, 11])
    manager.update_cache([10, 11], _context_of([1.0, 2.0]))
    manager.reset_slots([10, 11], [True, False])

    # the new stream takes the freed slot and starts from an empty cache; the other one keeps its state
    context = manager.get_context([20, 11])
    assert context.cache_last_channel_len.tolist() == [0.0, 2.0]
    assert context.cache_last_channel[:, 0].eq(0.0).all()
    assert context.cache_last_time[:, 0].eq(0.0).all()
    assert context.cache_last_time[:, 1].eq(2.0).all()


@pytest.mark.unit
def test_no_free_slot_raises():
    manager = CacheAwareContextManager(_FakeModel(), num_slots=1)
    manager.get_context([10])
    with pytest.raises(RuntimeError, match="No free slots"):
        manager.get_context([11])
