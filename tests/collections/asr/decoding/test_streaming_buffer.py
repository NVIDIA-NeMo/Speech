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

"""Unit tests for `StreamingBatchedAudioBuffer` and accompanying helper
classes defined in
`nemo.collections.asr.parts.utils.streaming_utils`.
"""

from __future__ import annotations

import math

import pytest
import torch

from nemo.collections.asr.parts.utils.streaming_utils import (
    ContextSize,
    ContextSizeBatch,
    DynamicLengthTensor,
    StreamingBatchedAudioBuffer,
)

# -----------------------------------------------------------------------------
# Helper constants / fixtures
# -----------------------------------------------------------------------------

DEVICES: list[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _create_audio_batch(batch_size: int, length: int, device: torch.device, dtype: torch.dtype = torch.float32):
    """Create a dummy audio batch of shape (batch_size, length)."""
    # Use a simple ramp signal to ease debugging.
    vals = torch.arange(batch_size * length, device=device, dtype=dtype)
    return vals.view(batch_size, length)


def _make_chunk(batch_size: int, length: int, channels: int, start: float, device: torch.device) -> torch.Tensor:
    """Create a deterministic chunk of shape (batch_size, length, channels)."""
    n = batch_size * length * channels
    return (start + torch.arange(n, device=device, dtype=torch.float32)).view(batch_size, length, channels)


def _make_ndim_chunk(
    batch_size: int, length: int, dim_shape: list[int], start: float, device: torch.device
) -> torch.Tensor:
    """Create a deterministic chunk of shape (batch_size, length, *dim_shape)."""
    n = batch_size * length * math.prod(dim_shape)
    return (start + torch.arange(n, device=device, dtype=torch.float32)).view(batch_size, length, *dim_shape)


# -----------------------------------------------------------------------------
# Tests for ContextSize and ContextSizeBatch
# -----------------------------------------------------------------------------


class TestContextSize:
    @pytest.mark.unit
    def test_context_size_total_and_subsample(self):
        ctx = ContextSize(left=4, chunk=2, right=1)
        assert ctx.total() == 7

        half_ctx = ctx.subsample(factor=2)
        assert isinstance(half_ctx, ContextSize)
        assert half_ctx.left == 2 and half_ctx.chunk == 1 and half_ctx.right == 0
        assert half_ctx.total() == math.floor(7 / 2)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_context_size_batch_total_and_subsample(self, device: torch.device):
        left = torch.tensor([4, 4], dtype=torch.long, device=device)
        chunk = torch.tensor([2, 2], dtype=torch.long, device=device)
        right = torch.tensor([2, 2], dtype=torch.long, device=device)
        batch_ctx = ContextSizeBatch(left=left, chunk=chunk, right=right)

        # total() should equal element-wise sum
        expected_total = left + chunk + right
        assert torch.equal(batch_ctx.total(), expected_total)

        # After subsampling by 2 each component should be halved (floor division)
        half_ctx = batch_ctx.subsample(2)
        assert torch.equal(half_ctx.left, left // 2)
        assert torch.equal(half_ctx.chunk, chunk // 2)
        assert torch.equal(half_ctx.right, right // 2)


# -----------------------------------------------------------------------------
# Tests for StreamingBatchedAudioBuffer
# -----------------------------------------------------------------------------


class TestStreamingBatchedAudioBuffer:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_streaming_batched_audio_buffer(self, device: torch.device):
        batch_size = 2
        expected_ctx = ContextSize(left=4, chunk=2, right=1)  # total = 7
        buffer = StreamingBatchedAudioBuffer(
            batch_size=batch_size,
            context_samples=expected_ctx,
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------------------
        # First add : chunk + right (filling initial buffer)
        # ------------------------------------------------------------------
        first_len = expected_ctx.chunk + expected_ctx.right  # 3
        audio_batch = _create_audio_batch(batch_size, first_len, device)
        audio_lens = torch.full(
            [
                batch_size,
            ],
            first_len,
            dtype=torch.long,
            device=device,
        )
        buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lens,
            is_last_chunk=False,
            is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )

        # Validate context sizes
        assert buffer.context_size.left == 0
        assert buffer.context_size.chunk == expected_ctx.chunk
        assert buffer.context_size.right == expected_ctx.right
        assert buffer.samples.shape[1] == first_len  # No truncation yet

        # ------------------------------------------------------------------
        # Second add : only chunk length
        # ------------------------------------------------------------------
        chunk_len = expected_ctx.chunk  # 2
        audio_batch = _create_audio_batch(batch_size, chunk_len, device)
        audio_lens.fill_(chunk_len)
        buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lens,
            is_last_chunk=False,
            is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )

        # After second add, left should have grown by previous chunk (2)
        assert buffer.context_size.left == 2
        assert buffer.context_size.chunk == expected_ctx.chunk
        assert buffer.context_size.right == expected_ctx.right
        assert buffer.samples.shape[1] == 5  # 2 (left) + 2 (chunk) + 1 (right)

        # ------------------------------------------------------------------
        # Third add : another chunk, buffer should now reach full capacity (7)
        # ------------------------------------------------------------------
        buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lens,
            is_last_chunk=False,
            is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )

        assert buffer.samples.shape[1] == expected_ctx.total()
        assert buffer.context_size.total() == expected_ctx.total()

        # ------------------------------------------------------------------
        # Fourth add : buffer overflows by 2 samples; implementation should
        # drop the excess from the left context.
        # ------------------------------------------------------------------
        buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lens,
            is_last_chunk=False,
            is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )

        # Buffer length remains constant (total context size)
        assert buffer.samples.shape[1] == expected_ctx.total()
        assert buffer.context_size.total() == expected_ctx.total()

        # Left context should have been clipped by 2 samples (from 6 to 4)
        assert buffer.context_size.left == expected_ctx.left  # 4

        # ------------------------------------------------------------------
        # Final add : mark last chunk with shorter length; right context
        # should go to 0 afterwards.
        # ------------------------------------------------------------------
        last_len = 1
        audio_batch = _create_audio_batch(batch_size, last_len, device)
        audio_lens.fill_(last_len)
        buffer.add_audio_batch_(
            audio_batch=audio_batch,
            audio_lengths=audio_lens,
            is_last_chunk=True,
            is_last_chunk_batch=torch.ones(batch_size, dtype=torch.bool, device=device),
        )

        # After last chunk, right context must be zero and total size preserved
        assert buffer.context_size.right == 0
        assert buffer.context_size.total() == expected_ctx.total()
        assert buffer.samples.shape[1] == expected_ctx.total()

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_streaming_batched_audio_buffer_raises_on_too_long_chunk(self, device: torch.device):
        """`add_audio_batch_` should raise if provided chunk is larger than chunk + right."""

        expected_ctx = ContextSize(left=0, chunk=2, right=1)
        buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=expected_ctx,
            dtype=torch.float32,
            device=device,
        )

        # Attempt to add a chunk that is too long (4 > 3)
        too_long_chunk_size = expected_ctx.chunk + expected_ctx.right + 1
        audio = _create_audio_batch(1, too_long_chunk_size, device)
        audio_lens = torch.tensor([too_long_chunk_size], dtype=torch.long, device=device)

        with pytest.raises(ValueError):
            buffer.add_audio_batch_(
                audio_batch=audio,
                audio_lengths=audio_lens,
                is_last_chunk=False,
                is_last_chunk_batch=torch.tensor([False], dtype=torch.bool, device=device),
            )


# -----------------------------------------------------------------------------
# Tests for DynamicLengthTensor
# -----------------------------------------------------------------------------


class TestDynamicLengthTensor:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "dim_shape, expected_dim_shape",
        [
            (None, []),
            (3, [3]),
            ([4, 5], [4, 5]),
        ],
    )
    def test_init(self, device, dim_shape, expected_dim_shape):
        batch_size, init_length = 2, 5
        t = DynamicLengthTensor(
            batch_size=batch_size,
            init_length=init_length,
            dim_shape=dim_shape,
            device=device,
            dtype=torch.float32,
        )

        assert t.dim_shape == expected_dim_shape
        assert list(t.data.shape) == [batch_size, init_length, *expected_dim_shape]
        assert list(t.lengths.shape) == [batch_size]
        assert t.lengths.dtype == torch.long
        assert t.data.dtype == torch.float32
        # Freshly created storage is zeroed and reports no content.
        assert torch.count_nonzero(t.lengths) == 0
        assert torch.count_nonzero(t.data) == 0

    @pytest.mark.unit
    def test_init_minimum_length(self):
        """`init_length` is clamped to at least 1 so doubling-based growth works."""
        t = DynamicLengthTensor(batch_size=2, init_length=0, dim_shape=1)
        assert t._max_length == 1
        assert t.data.shape[1] == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_append_with_lengths(self, device):
        """Per-batch `lengths` control how many frames become valid for each item."""
        t = DynamicLengthTensor(batch_size=2, init_length=4, dim_shape=1, device=device, dtype=torch.float32)

        # First chunk: batch item 0 keeps 2 frames, item 1 keeps 1 frame.
        chunk1 = _make_chunk(batch_size=2, length=2, channels=1, start=10.0, device=device)
        # chunk1 == [[[10], [11]], [[12], [13]]]
        t.append_(data=chunk1, lengths=torch.tensor([2, 1], device=device))
        assert t.lengths.tolist() == [2, 1]

        # Second chunk: item 0 keeps 1 frame, item 1 keeps 2 frames. Item 1 should
        # overwrite the previously written "garbage" frame at position 1.
        chunk2 = _make_chunk(batch_size=2, length=2, channels=1, start=30.0, device=device)
        # chunk2 == [[[30], [31]], [[32], [33]]]
        t.append_(data=chunk2, lengths=torch.tensor([1, 2], device=device))
        assert t.lengths.tolist() == [3, 3]

        # Valid frames are everything up to the per-item length.
        item0 = t.data[0, : t.lengths[0], 0].tolist()
        item1 = t.data[1, : t.lengths[1], 0].tolist()
        assert item0 == [10.0, 11.0, 30.0]
        assert item1 == [12.0, 32.0, 33.0]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_append_without_lengths(self, device):
        """Without `lengths`, every frame in the chunk is appended for all items."""
        t = DynamicLengthTensor(batch_size=2, init_length=2, dim_shape=1, device=device, dtype=torch.float32)

        chunk = _make_chunk(batch_size=2, length=3, channels=1, start=0.0, device=device)
        t.append_(data=chunk)

        assert t.lengths.tolist() == [3, 3]
        assert torch.equal(t.data[:, :3], chunk)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dim_shape", [[], [3], [2, 3], [2, 3, 4]])
    def test_append_without_lengths_multidim(self, device, dim_shape):
        """Append must scatter the whole feature vector for arbitrary trailing `dim_shape`."""
        batch_size = 2
        # init_length < appended length so this also exercises the growth path with multi-dim shapes.
        t = DynamicLengthTensor(
            batch_size=batch_size, init_length=2, dim_shape=dim_shape, device=device, dtype=torch.float32
        )

        chunk = _make_ndim_chunk(batch_size, length=3, dim_shape=dim_shape, start=0.0, device=device)
        t.append_(data=chunk)

        assert t.lengths.tolist() == [3, 3]
        assert torch.equal(t.data[:, :3], chunk)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_append_with_lengths_multidim(self, device):
        """Per-batch `lengths` must place the full multi-dim feature vectors at the right offsets."""
        dim_shape = [2, 3]
        t = DynamicLengthTensor(batch_size=2, init_length=4, dim_shape=dim_shape, device=device, dtype=torch.float32)

        # First chunk: item 0 keeps 2 frames, item 1 keeps 1 frame.
        chunk1 = _make_ndim_chunk(2, length=2, dim_shape=dim_shape, start=10.0, device=device)
        t.append_(data=chunk1, lengths=torch.tensor([2, 1], device=device))
        assert t.lengths.tolist() == [2, 1]

        # Second chunk: item 0 keeps 1 frame, item 1 keeps 2 frames. Item 1's second frame
        # must overwrite the previously written "garbage" frame at position 1.
        chunk2 = _make_ndim_chunk(2, length=2, dim_shape=dim_shape, start=100.0, device=device)
        t.append_(data=chunk2, lengths=torch.tensor([1, 2], device=device))
        assert t.lengths.tolist() == [3, 3]

        # Item 0: chunk1[0, 0], chunk1[0, 1], chunk2[0, 0]; each is a full (2, 3) feature vector.
        assert torch.equal(t.data[0, 0], chunk1[0, 0])
        assert torch.equal(t.data[0, 1], chunk1[0, 1])
        assert torch.equal(t.data[0, 2], chunk2[0, 0])
        # Item 1: chunk1[1, 0], chunk2[1, 0] (overwrites garbage at pos 1), chunk2[1, 1].
        assert torch.equal(t.data[1, 0], chunk1[1, 0])
        assert torch.equal(t.data[1, 1], chunk2[1, 0])
        assert torch.equal(t.data[1, 2], chunk2[1, 1])

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_growth_preserves_data(self, device):
        """Appending more than the initial capacity reallocates and keeps content."""
        t = DynamicLengthTensor(batch_size=1, init_length=2, dim_shape=1, device=device, dtype=torch.float32)
        initial_capacity = t._max_length

        big_len = 10
        chunk = _make_chunk(batch_size=1, length=big_len, channels=1, start=0.0, device=device)
        t.append_(data=chunk, lengths=torch.tensor([big_len], device=device))

        assert t._max_length > initial_capacity
        assert t._max_length >= big_len
        assert t.lengths.tolist() == [big_len]
        assert t.data[0, :big_len, 0].tolist() == list(range(big_len))

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_incremental_appends_double_capacity(self, device):
        """Repeated single-frame appends grow capacity geometrically (amortized O(1))."""
        n_appends = 9
        t = DynamicLengthTensor(batch_size=1, init_length=1, dim_shape=1, device=device, dtype=torch.float32)

        capacities = []
        for i in range(n_appends):
            frame = torch.full((1, 1, 1), float(i), device=device)
            t.append_(data=frame)
            capacities.append(t._max_length)

        # Everything that was appended is retained, in order.
        assert t.lengths.tolist() == [n_appends]
        assert t.data[0, :n_appends, 0].tolist() == [float(i) for i in range(n_appends)]
        # Capacity is always at least what is stored, and grew far less than linearly.
        assert t._max_length >= n_appends
        assert len(set(capacities)) < n_appends  # capacity reused across appends, not bumped every time

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_clear(self, device):
        """`clear_` resets both lengths and storage to zero while keeping capacity."""
        t = DynamicLengthTensor(batch_size=2, init_length=4, dim_shape=1, device=device, dtype=torch.float32)
        t.append_(data=_make_chunk(2, 3, 1, start=1.0, device=device), lengths=torch.tensor([3, 3], device=device))

        capacity_before = t._max_length
        t.clear_()

        assert t.lengths.tolist() == [0, 0]
        assert torch.count_nonzero(t.data) == 0
        assert t._max_length == capacity_before

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_merge(self, device):
        """`merge_` concatenates another tensor's content along the length dim."""
        a = DynamicLengthTensor(batch_size=2, init_length=2, dim_shape=1, device=device, dtype=torch.float32)
        a.append_(data=_make_chunk(2, 2, 1, start=1.0, device=device), lengths=torch.tensor([2, 2], device=device))

        b = DynamicLengthTensor(batch_size=2, init_length=2, dim_shape=1, device=device, dtype=torch.float32)
        b.append_(data=_make_chunk(2, 2, 1, start=100.0, device=device), lengths=torch.tensor([1, 2], device=device))

        a_item0_before = a.data[0, : a.lengths[0], 0].tolist()
        a_item1_before = a.data[1, : a.lengths[1], 0].tolist()
        b_item0 = b.data[0, : b.lengths[0], 0].tolist()
        b_item1 = b.data[1, : b.lengths[1], 0].tolist()

        ret = a.merge_(b)
        assert ret is a  # in-place, returns self
        assert a.lengths.tolist() == [3, 4]
        assert a.data[0, : a.lengths[0], 0].tolist() == a_item0_before + b_item0
        assert a.data[1, : a.lengths[1], 0].tolist() == a_item1_before + b_item1

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_clone_is_independent(self, device):
        """`clone` returns a deep copy: same data/lengths, but independent storage."""
        t = DynamicLengthTensor(batch_size=2, init_length=4, dim_shape=3, device=device, dtype=torch.float32)
        t.append_(data=_make_chunk(2, 2, 3, start=1.0, device=device), lengths=torch.tensor([2, 1], device=device))

        clone = t.clone()
        assert clone is not t
        assert clone.dim_shape == t.dim_shape
        assert clone.data.shape == t.data.shape
        assert torch.equal(clone.lengths, t.lengths)
        assert torch.equal(clone.data, t.data)

        # Mutating the clone must not affect the original.
        clone.append_(
            data=_make_chunk(2, 1, 3, start=50.0, device=device), lengths=torch.tensor([1, 1], device=device)
        )
        assert clone.lengths.tolist() == [3, 2]
        assert t.lengths.tolist() == [2, 1]

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to verify cross-device move")
    def test_to_device(self):
        """`to_device` moves the underlying storage (not just the bookkeeping attr)."""
        t = DynamicLengthTensor(
            batch_size=2, init_length=4, dim_shape=1, device=torch.device("cpu"), dtype=torch.float32
        )
        t.append_(data=_make_chunk(2, 2, 1, start=1.0, device=torch.device("cpu")), lengths=torch.tensor([2, 2]))

        ret = t.to_device("cuda:0")
        assert ret is t
        assert t.device == "cuda:0"
        assert t.data.device.type == "cuda"
        assert t.lengths.device.type == "cuda"
        # Content survives the move.
        assert t.data[0, :2, 0].tolist() == [1.0, 2.0]
