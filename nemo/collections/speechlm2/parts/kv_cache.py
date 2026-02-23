# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""KV cache management with attention sinks + sliding window for StreamingSALM."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


def maybe_evict_cache(
    cache: DynamicCache | tuple,
    cache_pos: int,
    sink_size: int,
    window_size: int,
) -> tuple[DynamicCache | tuple, int]:
    """
    Evict old KV cache entries beyond the sink + window limit.

    Preserves:
    - First ``sink_size`` entries (attention sinks / prompt)
    - Last ``window_size`` entries (recent context)

    Evicts everything in between.

    Args:
        cache: ``DynamicCache`` or legacy tuple of (key, value) pairs per layer,
               each with shape (B, num_heads, seq_len, head_dim).
        cache_pos: current number of entries in the cache
        sink_size: number of protected prefix entries
        window_size: maximum number of recent entries to keep

    Returns:
        new_cache: evicted cache (``DynamicCache`` when eviction occurs,
                   original type when no eviction is needed)
        new_pos: updated cache position
    """
    max_size = sink_size + window_size
    if cache_pos <= max_size:
        return cache, cache_pos

    # Number of entries to evict
    evict_count = cache_pos - max_size

    # Convert plain tuple to DynamicCache if needed
    if not isinstance(cache, DynamicCache):
        dc = DynamicCache()
        for key, value in cache:
            dc.update(key, value, layer_idx=len(dc))
        cache = dc

    # Evict in-place on the DynamicCache layers
    for layer in cache.layers:
        key = layer.keys
        value = layer.values
        layer.keys = torch.cat(
            [key[:, :, :sink_size, :], key[:, :, sink_size + evict_count :, :]], dim=2
        )
        layer.values = torch.cat(
            [value[:, :, :sink_size, :], value[:, :, sink_size + evict_count :, :]], dim=2
        )

    return cache, max_size
