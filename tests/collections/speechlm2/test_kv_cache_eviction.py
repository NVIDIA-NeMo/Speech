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
"""Test KV cache eviction with attention sinks + sliding window."""

import torch
from transformers.cache_utils import DynamicCache

from nemo.collections.speechlm2.parts.kv_cache import maybe_evict_cache


def make_fake_cache(num_layers, batch_size, num_heads, seq_len, head_dim):
    """Create synthetic KV cache for testing."""
    cache = []
    for _ in range(num_layers):
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.append((key, value))
    return tuple(cache)


def make_dynamic_cache(num_layers, batch_size, num_heads, seq_len, head_dim):
    """Create synthetic DynamicCache (the production type used by HF Transformers)."""
    dc = DynamicCache()
    for layer_idx in range(num_layers):
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        dc.update(key, value, layer_idx=layer_idx)
    return dc


class TestKVCacheEviction:
    def test_no_eviction_when_below_limit(self):
        """Cache within limit should be returned unchanged."""
        cache = make_fake_cache(2, 1, 4, 50, 16)
        new_cache, new_pos = maybe_evict_cache(cache, 50, sink_size=10, window_size=100)
        assert new_pos == 50
        for i in range(len(cache)):
            assert torch.equal(new_cache[i][0], cache[i][0])
            assert torch.equal(new_cache[i][1], cache[i][1])

    def test_eviction_preserves_sinks(self):
        """After eviction, first sink_size entries should be preserved."""
        sink = 10
        window = 20
        total = 50
        cache = make_fake_cache(2, 1, 4, total, 16)
        new_cache, new_pos = maybe_evict_cache(cache, total, sink, window)

        assert new_pos == sink + window
        for layer_orig, layer_new in zip(cache, new_cache):
            # Sink entries preserved
            assert torch.equal(layer_new[0][:, :, :sink, :], layer_orig[0][:, :, :sink, :])
            assert torch.equal(layer_new[1][:, :, :sink, :], layer_orig[1][:, :, :sink, :])

    def test_eviction_preserves_window(self):
        """After eviction, last window_size entries should be preserved."""
        sink = 10
        window = 20
        total = 50
        cache = make_fake_cache(2, 1, 4, total, 16)
        new_cache, new_pos = maybe_evict_cache(cache, total, sink, window)

        for layer_orig, layer_new in zip(cache, new_cache):
            # Window entries preserved (last `window` entries)
            assert torch.equal(
                layer_new[0][:, :, sink:, :],
                layer_orig[0][:, :, total - window :, :],
            )

    def test_eviction_correct_final_size(self):
        """After eviction, cache seq_len = sink_size + window_size."""
        sink = 5
        window = 15
        total = 100
        cache = make_fake_cache(3, 2, 8, total, 32)
        new_cache, new_pos = maybe_evict_cache(cache, total, sink, window)

        assert new_pos == sink + window
        for k, v in new_cache:
            assert k.shape[2] == sink + window
            assert v.shape[2] == sink + window

    def test_eviction_at_exact_limit(self):
        """Cache at exactly sink+window should not be evicted."""
        sink = 10
        window = 20
        total = 30  # exactly sink + window
        cache = make_fake_cache(2, 1, 4, total, 16)
        new_cache, new_pos = maybe_evict_cache(cache, total, sink, window)
        assert new_pos == total  # no change
        for i in range(len(cache)):
            assert torch.equal(new_cache[i][0], cache[i][0])

    def test_eviction_multiple_rounds(self):
        """Repeated eviction should maintain invariants."""
        sink = 4
        window = 8
        cache = make_fake_cache(2, 1, 4, sink + window, 16)
        pos = sink + window

        # Simulate adding 20 more entries with periodic eviction
        for _ in range(20):
            # Add one entry
            new_kv = []
            for k, v in cache:
                new_k = torch.cat([k, torch.randn(1, 4, 1, 16)], dim=2)
                new_v = torch.cat([v, torch.randn(1, 4, 1, 16)], dim=2)
                new_kv.append((new_k, new_v))
            cache = tuple(new_kv)
            pos += 1
            cache, pos = maybe_evict_cache(cache, pos, sink, window)
            # Should always be at most sink + window + 1 (just added)
            # but maybe_evict_cache brings it to sink + window
            assert pos <= sink + window + 1
            for k, v in cache:
                assert k.shape[2] <= sink + window + 1

    def test_single_layer_cache(self):
        """Works correctly with a single-layer cache."""
        cache = make_fake_cache(1, 1, 2, 50, 8)
        new_cache, new_pos = maybe_evict_cache(cache, 50, sink_size=5, window_size=10)
        assert new_pos == 15
        assert len(new_cache) == 1
        assert new_cache[0][0].shape[2] == 15

    def test_batched_cache(self):
        """Eviction works correctly with batched KV cache."""
        B = 4
        cache = make_fake_cache(2, B, 4, 60, 16)
        new_cache, new_pos = maybe_evict_cache(cache, 60, sink_size=8, window_size=12)
        assert new_pos == 20
        for k, v in new_cache:
            assert k.shape[0] == B
            assert k.shape[2] == 20

    # ---- New tests: DynamicCache input (production code path) ----

    def test_dynamic_cache_input_no_eviction(self):
        """DynamicCache within limit should be returned unchanged."""
        dc = make_dynamic_cache(2, 1, 4, 50, 16)
        original_keys = [layer.keys.clone() for layer in dc.layers]
        new_cache, new_pos = maybe_evict_cache(dc, 50, sink_size=10, window_size=100)
        assert new_pos == 50
        assert isinstance(new_cache, DynamicCache)
        for i, layer in enumerate(new_cache.layers):
            assert torch.equal(layer.keys, original_keys[i])

    def test_dynamic_cache_input_eviction(self):
        """DynamicCache should be evicted correctly (the actual production path)."""
        sink = 8
        window = 12
        total = 40
        dc = make_dynamic_cache(2, 1, 4, total, 16)
        original_sink_keys = [layer.keys[:, :, :sink, :].clone() for layer in dc.layers]
        original_window_keys = [layer.keys[:, :, total - window:, :].clone() for layer in dc.layers]

        new_cache, new_pos = maybe_evict_cache(dc, total, sink, window)

        assert new_pos == sink + window
        assert isinstance(new_cache, DynamicCache)
        for i, layer in enumerate(new_cache.layers):
            assert layer.keys.shape[2] == sink + window
            # Verify sinks preserved
            assert torch.equal(layer.keys[:, :, :sink, :], original_sink_keys[i])
            # Verify window preserved
            assert torch.equal(layer.keys[:, :, sink:, :], original_window_keys[i])

    def test_dynamic_cache_get_seq_length_after_eviction(self):
        """After eviction, DynamicCache.get_seq_length() returns correct value."""
        sink = 5
        window = 10
        dc = make_dynamic_cache(2, 1, 4, 30, 16)
        new_cache, new_pos = maybe_evict_cache(dc, 30, sink, window)
        assert new_cache.get_seq_length() == sink + window
        assert new_pos == sink + window
