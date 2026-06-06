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

import pytest
import torch

from nemo.collections.asr.inference.utils.cache import CastCache

# Cache tensors are [L, B, T, D] with the slot dimension on dim 1.
_L, _B, _T, _D = 2, 4, 3, 8

# Reconstruction tolerance (max abs error) per storage dtype.
_CAST_BOUND = {torch.float32: 0.0, torch.float16: 0.05, torch.bfloat16: 0.05}


def _make_x(seed: int = 0) -> torch.Tensor:
    """Random fp32 cache tensor of shape [L, B, T, D]."""
    torch.manual_seed(seed)
    return torch.randn(_L, _B, _T, _D)


def _rel_err(out: torch.Tensor, ref: torch.Tensor, vec_axis: int = 3) -> float:
    """Max per-vector reconstruction error relative to the vector's absmax."""
    denom = ref.abs().amax(dim=vec_axis, keepdim=True).clamp_min(1e-9)
    return ((out - ref).abs() / denom).max().item()


def _assert_slot_ops(cache, src: torch.Tensor, bound: float) -> None:
    """Shared assertions for update_slots / reset_slots / gather_slots ordering.

    `cache` must start zero-valued with _B slots. Copies src slots [0, 1] into cache
    slots [1, 2], checks reconstruction, then resets slot 1 and verifies slot 2 survives.
    """
    dst_ids = torch.tensor([1, 2])
    src_ids = torch.tensor([0, 1])
    cache.update_slots(dst_ids, src, src_ids)

    out = cache.gather_slots([1, 2])
    ref = src.index_select(1, src_ids)
    assert out.shape == ref.shape
    assert _rel_err(out, ref) <= bound + 1e-4

    # Slot 0 was never written, so it stays at its zero init.
    assert torch.count_nonzero(cache.gather_slots([0])) == 0

    cache.reset_slots(torch.tensor([1]))
    assert torch.count_nonzero(cache.gather_slots([1])) == 0
    # Slot 2 (holding src slot 1) must be untouched by resetting slot 1.
    assert torch.count_nonzero(cache.gather_slots([2])) > 0


def _assert_gather_out_matches(cache, slot_ids: list[int]) -> None:
    """`gather_slots(out=buf)` must write into `buf` and match the freshly-allocated gather."""
    ref = cache.gather_slots(slot_ids)
    buf = torch.empty_like(ref)
    got = cache.gather_slots(slot_ids, out=buf)
    assert got.data_ptr() == buf.data_ptr()  # result was written into the provided buffer
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


class TestCastCache:

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_round_trip(self, dtype):
        x = _make_x()
        cache = CastCache(x, dtype)
        out = cache.gather_slots(list(range(_B)))
        assert out.shape == x.shape
        assert out.dtype == x.dtype  # gather returns the source dtype
        assert (out - x).abs().max().item() <= _CAST_BOUND[dtype]

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_empty_is_zero(self, dtype):
        cache = CastCache.empty(
            (_L, _B, _T, _D), source_dtype=torch.float32, storage_dtype=dtype, device=torch.device("cpu")
        )
        assert torch.count_nonzero(cache.gather_slots(list(range(_B)))) == 0

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_slot_ops(self, dtype):
        cache = CastCache.empty(
            (_L, _B, _T, _D), source_dtype=torch.float32, storage_dtype=dtype, device=torch.device("cpu")
        )
        _assert_slot_ops(cache, _make_x(seed=2), bound=_CAST_BOUND[dtype])

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype,bytes_per_elem", [(torch.float32, 4), (torch.float16, 2), (torch.bfloat16, 2)])
    def test_storage_nbytes(self, dtype, bytes_per_elem):
        cache = CastCache(_make_x(), dtype)
        assert cache.storage_nbytes() == bytes_per_elem * (_L * _B * _T * _D)

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_gather_out_buffer(self, dtype):
        cache = CastCache(_make_x(), dtype)
        _assert_gather_out_matches(cache, [2, 0, 1])

    @pytest.mark.unit
    def test_source_dtype_overrides_seed(self):
        # bf16 working dtype + bf16 storage is a lossless, no-cast round trip.
        x = _make_x()  # fp32 seed
        cache = CastCache(x, torch.bfloat16, source_dtype=torch.bfloat16)
        out = cache.gather_slots(list(range(_B)))
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, x.to(torch.bfloat16))
