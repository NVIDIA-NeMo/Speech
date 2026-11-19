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

"""Attention over packed sequences: the document block mask and the Transformer-XL attention against padded."""

import pytest
import torch
import torch._dynamo
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, packed_sequence_ids
from nemo.collections.asr.parts.submodules import packed_attention as packed_attention_module
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.packed_attention import (
    PackedAttentionContext,
    document_block_mask,
    packed_attention,
)


def _valid_frames(padded, lengths):
    """Rows of a padded (B, T, D) batch inside the lengths, in packing order."""
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    return padded[mask]


@pytest.fixture(autouse=True)
def reset_dynamo_after_each_test():
    """Leave no compiled state behind for the suites that run after this module.

    Dynamo keeps the entries of a compiled function on its code object and counts them against the recompile limit;
    the kernels these tests compile would otherwise stay in the process until a later compile hits that limit
    and silently runs eagerly (the online CUDA-graph test of the parallel expert encoder then saw dense FlexAttention).
    """
    yield
    torch._dynamo.reset()


class TestDocumentBlockMask:
    @pytest.mark.unit
    @pytest.mark.parametrize("lengths", [[5, 3, 7], [300, 1, 129, 128, 40], [257]])
    def test_matches_dense_document_mask(self, lengths):
        lengths = torch.tensor(lengths)
        packed = pack_encoder_output(torch.zeros(len(lengths), int(lengths.max()), 1), lengths)
        sequence_ids = packed_sequence_ids(packed)
        total = packed.total_tokens
        block_mask = document_block_mask(sequence_ids, packed.cu_seqlens, total)

        padded_ids = torch.full((-(-total // 128) * 128,), -1)
        padded_ids[:total] = sequence_ids

        def mask_mod(b, h, q_idx, kv_idx):
            return padded_ids[q_idx] == padded_ids[kv_idx]

        reference = create_block_mask(mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device="cpu")
        torch.testing.assert_close(block_mask.to_dense(), reference.to_dense())
        assert block_mask.full_kv_num_blocks is not None
        if max(lengths.tolist()) >= 256:
            assert int(block_mask.full_kv_num_blocks.sum()) > 0

    @pytest.mark.unit
    def test_padded_token_axis_keeps_the_tail_masked(self):
        """The partially filled last block stays partial, so pad keys never reach the last utterance's queries."""
        lengths = torch.tensor([300, 7, 150, 1, 260])
        packed = pack_encoder_output(torch.zeros(len(lengths), 300, 1), lengths)
        sequence_ids = packed_sequence_ids(packed)
        total, padded_total = packed.total_tokens, 1024
        block_mask = document_block_mask(sequence_ids, packed.cu_seqlens, total, padded_total_tokens=padded_total)
        padded_ids = torch.full((padded_total,), -1)
        padded_ids[:total] = sequence_ids

        def mask_mod(b, h, q_idx, kv_idx):
            # the tail queries attend nothing, as the skipped tail blocks of the built mask
            return (padded_ids[q_idx] == padded_ids[kv_idx]) & (padded_ids[q_idx] >= 0)

        reference = create_block_mask(mask_mod, B=None, H=None, Q_LEN=padded_total, KV_LEN=padded_total, device="cpu")
        torch.testing.assert_close(block_mask.to_dense(), reference.to_dense())
        last_real_block = -(-total // 128) - 1
        assert int(block_mask.full_kv_num_blocks[..., last_real_block].sum()) == 0

    @pytest.mark.unit
    def test_context_from_lengths(self):
        context = PackedAttentionContext.from_lengths(torch.tensor([3, 0, 2]))
        assert context.total_tokens == 5 and context.max_seqlen == 3
        assert context.cu_seqlens.tolist() == [0, 3, 3, 5]
        assert context.position_ids.tolist() == [0, 1, 2, 0, 1]
        assert context.sequence_slices == ((0, 3), (3, 3), (3, 5))


def _key_window(name, max_seqlen, device="cpu"):
    """Key windows by query position: the first and the last key position, None for a side without a limit."""
    positions = torch.arange(max_seqlen, device=device)
    chunk_start = positions // 100 * 100
    return {
        "band": (positions - 40, positions + 13),
        "wide_band": (positions - 300, positions + 300),
        "left_only": (positions - 70, None),
        "right_only": (None, positions + 5),
        "chunks": (chunk_start - 200, chunk_start + 99),
        "own_token": (positions, positions),
    }[name]


def _dense_window_attention(q, k, v, lengths, key_window, rel_pos_bias=None):
    """Attention one query at a time over the keys of its window, written out with loops."""
    first_key, last_key = key_window
    outputs, offset = [], 0
    scale = q.shape[-1] ** -0.5
    for length in lengths.tolist():
        for i in range(length):
            lo = 0 if first_key is None else max(int(first_key[i]), 0)
            hi = length - 1 if last_key is None else min(int(last_key[i]), length - 1)
            keys = slice(offset + lo, offset + hi + 1)
            scores = torch.einsum("hd,jhd->hj", q[offset + i], k[keys]) * scale
            if rel_pos_bias is not None:
                center = (rel_pos_bias.shape[-1] - 1) // 2
                rel = center - i + torch.arange(lo, hi + 1, device=q.device)
                scores = scores + rel_pos_bias[offset + i][:, rel] * scale
            outputs.append(torch.einsum("hj,jhd->hd", torch.softmax(scores, dim=-1), v[keys]))
        offset += length
    return torch.stack(outputs)


class TestPackedAttentionKeyWindow:
    """A limited attention context: the keys of every token narrowed to a window inside its utterance."""

    @pytest.mark.unit
    def test_key_spans_stay_inside_the_utterance_and_hold_the_token(self):
        lengths = torch.tensor([5, 0, 3])
        packed = pack_encoder_output(torch.zeros(3, 5, 1), lengths)
        positions = torch.arange(5)
        context = PackedAttentionContext(packed, key_window=(positions - 1, positions + 2))
        start, end = context.key_spans()
        assert start.tolist() == [0, 0, 1, 2, 3, 5, 5, 6]
        assert end.tolist() == [3, 4, 5, 5, 5, 8, 8, 8]
        one_sided = PackedAttentionContext(packed, key_window=(None, positions))
        assert [t.tolist() for t in one_sided.key_spans()] == [[0, 0, 0, 0, 0, 5, 5, 5], [1, 2, 3, 4, 5, 6, 7, 8]]
        # a window that leaves its own token out is widened to it
        beside = PackedAttentionContext(packed, key_window=(positions + 1, positions + 1))
        assert [t.tolist() for t in beside.key_spans()] == [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 5, 7, 8, 8]]
        whole = PackedAttentionContext(packed)
        assert whole.key_window is None
        assert [t.tolist() for t in whole.key_spans()] == [[0, 0, 0, 0, 0, 5, 5, 5], [5, 5, 5, 5, 5, 8, 8, 8]]
        assert PackedAttentionContext(packed, key_window=(None, None)).key_window is None
        with pytest.raises(ValueError, match="key_window bounds must have shape"):
            PackedAttentionContext(packed, key_window=(torch.arange(8), None))
        with pytest.raises(TypeError, match="integer tensors"):
            PackedAttentionContext(packed, key_window=(positions.double(), None))

    @pytest.mark.unit
    def test_bias_narrower_than_the_utterances_needs_a_two_sided_window(self):
        packed = pack_encoder_output(torch.zeros(2, 60, 1), torch.tensor([60, 20]))
        positions = torch.arange(60)
        q = torch.randn(80, 2, 4)
        bias = torch.randn(80, 2, 7)
        for key_window in (None, (positions - 3, None), (None, positions + 3)):
            with pytest.raises(ValueError, match="rel_pos_bias covers 7 relative positions"):
                packed_attention(q, q, q, PackedAttentionContext(packed, key_window=key_window), rel_pos_bias=bias)
        context = PackedAttentionContext(packed, key_window=(positions - 3, positions + 3))
        assert packed_attention(q, q, q, context, rel_pos_bias=bias).shape == q.shape

    @pytest.mark.unit
    def test_local_attention_checks_its_window_and_position_table(self):
        attention = RelPositionMultiHeadAttentionLongformer(
            n_head=2, n_feat=8, dropout_rate=0.0, pos_bias_u=None, pos_bias_v=None, att_context_size=[4, 4]
        )
        packed = pack_encoder_output(torch.randn(2, 40, 8), torch.tensor([40, 9]))
        positions = torch.arange(40)
        context = PackedAttentionContext(packed, key_window=(positions - 4, positions + 4))
        assert attention.forward_packed(packed.data, context, pos_emb=torch.randn(1, 9, 8)).shape == (49, 8)
        with pytest.raises(ValueError, match="needs pos_emb of 9 relative positions, got 13"):
            attention.forward_packed(packed.data, context, pos_emb=torch.randn(1, 13, 8))
        with pytest.raises(ValueError, match="needs the key window"):
            attention.forward_packed(packed.data, PackedAttentionContext(packed), pos_emb=torch.randn(1, 9, 8))
        with pytest.raises(ValueError, match="needs the key window"):
            one_sided = PackedAttentionContext(packed, key_window=(positions - 4, None))
            attention.forward_packed(packed.data, one_sided, pos_emb=torch.randn(1, 9, 8))

    @pytest.mark.unit
    @pytest.mark.parametrize("window", ["band", "wide_band", "left_only", "right_only", "chunks", "own_token"])
    @pytest.mark.parametrize("lengths", [[1000, 1, 129, 0, 128, 40], [257]])
    def test_block_mask_matches_dense_window_mask(self, lengths, window):
        lengths = torch.tensor(lengths)
        packed = pack_encoder_output(torch.zeros(len(lengths), int(lengths.max()), 1), lengths)
        context = PackedAttentionContext(packed, key_window=_key_window(window, packed.max_seqlen))
        total, padded_total = context.total_tokens, context.padded_total_tokens
        start, end = context.key_spans()
        starts = torch.full((padded_total,), padded_total)
        starts[:total] = start
        ends = torch.zeros(padded_total, dtype=torch.int64)
        ends[:total] = end

        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])

        reference = create_block_mask(mask_mod, B=None, H=None, Q_LEN=padded_total, KV_LEN=padded_total, device="cpu")
        block_mask = context.block_mask()
        torch.testing.assert_close(block_mask.to_dense(), reference.to_dense())

        document = PackedAttentionContext(packed).block_mask()
        blocks = int(block_mask.kv_num_blocks.sum() + block_mask.full_kv_num_blocks.sum())
        document_blocks = int(document.kv_num_blocks.sum() + document.full_kv_num_blocks.sum())
        assert blocks <= document_blocks
        if window in ("band", "own_token") and max(lengths.tolist()) >= 1000:
            # the blocks outside the window are skipped, not evaluated
            assert blocks < document_blocks / 2
        if window == "wide_band" and max(lengths.tolist()) >= 1000:
            assert int(block_mask.full_kv_num_blocks.sum()) > 0

    @pytest.mark.unit
    @pytest.mark.parametrize("window", ["band", "left_only", "right_only", "chunks", "own_token"])
    @pytest.mark.parametrize("bias_width", [None, "utterance", "window"])
    def test_reference_path_attends_inside_the_window(self, window, bias_width):
        """Outputs and gradients against attention written out per query; a bias as wide as the window suffices."""
        if bias_width == "window" and window != "band":
            pytest.skip("a bias narrower than the utterance needs a window bounded on both sides")
        torch.manual_seed(1)
        lengths = torch.tensor([150, 1, 0, 60])
        packed = pack_encoder_output(torch.zeros(4, 150, 1), lengths)
        key_window = _key_window(window, packed.max_seqlen)
        context = PackedAttentionContext(packed, key_window=key_window)
        total = context.total_tokens
        q, k, v = (torch.randn(total, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(3))
        bias = None
        if bias_width is not None:
            reach = 150 if bias_width == "utterance" else 41
            bias = torch.randn(total, 2, 2 * reach - 1, dtype=torch.float64, requires_grad=True)
        inputs = [t for t in (q, k, v, bias) if t is not None]

        out = packed_attention(q, k, v, context, rel_pos_bias=bias)
        grads = torch.autograd.grad(out.square().sum(), inputs)
        reference = _dense_window_attention(q, k, v, lengths, key_window, rel_pos_bias=bias)
        reference_grads = torch.autograd.grad(reference.square().sum(), inputs)
        torch.testing.assert_close(out, reference)
        for grad, reference_grad in zip(grads, reference_grads):
            torch.testing.assert_close(grad, reference_grad)

    @pytest.mark.unit
    def test_empty_batch_keeps_gradients(self):
        packed = pack_encoder_output(torch.zeros(2, 0, 1), torch.tensor([0, 0]))
        empty = torch.empty(0, dtype=torch.int64)
        context = PackedAttentionContext(packed, key_window=(empty, empty))
        q = torch.randn(0, 2, 4, requires_grad=True)
        out = packed_attention(q, q, q, context)
        out.sum().backward()
        assert out.shape == (0, 2, 4) and q.grad is not None

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    @pytest.mark.parametrize("window", ["band", "wide_band", "left_only", "right_only", "chunks", "own_token"])
    @pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
    def test_flex_attention_matches_reference_on_cuda(self, window, with_bias):
        """Utterances over several blocks of 128 tokens, of one token and of none; outputs and all gradients."""
        torch.manual_seed(0)
        lengths = torch.tensor([1000, 1, 300, 0, 129], device="cuda")
        packed = pack_encoder_output(torch.zeros(5, 1000, 1, device="cuda"), lengths)
        context = PackedAttentionContext(packed, key_window=_key_window(window, packed.max_seqlen, device="cuda"))
        total, heads, head_dim = packed.total_tokens, 2, 32
        inputs = [torch.randn(total, heads, head_dim, device="cuda", requires_grad=True) for _ in range(3)]
        bias = None
        if with_bias:
            bias = torch.randn(total, heads, 2 * packed.max_seqlen - 1, device="cuda", requires_grad=True)
            inputs.append(bias)

        out = packed_attention(*inputs[:3], context, rel_pos_bias=bias)
        grads = torch.autograd.grad(out.square().sum(), inputs)
        reference = packed_attention_module._reference_packed_attention(
            *inputs[:3], context, rel_pos_bias=bias, dropout_p=0.0, scale=head_dim**-0.5
        )
        reference_grads = torch.autograd.grad(reference.square().sum(), inputs)
        torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)
        for grad, reference_grad in zip(grads, reference_grads):
            torch.testing.assert_close(grad, reference_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    @pytest.mark.parametrize("dtype, tolerance", [(torch.float16, 4e-3), (torch.bfloat16, 2e-2)], ids=["fp16", "bf16"])
    def test_half_precision_window_runs_flex_attention_not_flash(self, dtype, tolerance, monkeypatch):
        """Variable-length FlashAttention knows no window, so bias-free half precision takes FlexAttention too."""

        def no_flash_attention(queries):
            raise AssertionError("variable-length FlashAttention cannot express a key window")

        monkeypatch.setattr(packed_attention_module, "_flash_attention_varlen", no_flash_attention)
        torch.manual_seed(0)
        lengths = torch.tensor([127, 0, 128, 129, 300, 1], device="cuda")
        packed = pack_encoder_output(torch.zeros(6, 300, 1, device="cuda"), lengths)
        context = PackedAttentionContext(packed, key_window=_key_window("band", packed.max_seqlen, device="cuda"))
        q, k, v = (torch.randn(context.total_tokens, 2, 64, device="cuda", dtype=dtype) for _ in range(3))
        upstream = torch.randn(context.total_tokens, 2, 64, device="cuda", dtype=dtype)

        inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        out = packed_attention(*inputs, context)
        out.backward(upstream)
        reference_inputs = [t.detach().float().requires_grad_() for t in (q, k, v)]
        reference = packed_attention_module._reference_packed_attention(
            *reference_inputs, context, rel_pos_bias=None, dropout_p=0.0, scale=64**-0.5
        )
        reference.backward(upstream.float())
        torch.testing.assert_close(out.float(), reference, atol=tolerance, rtol=tolerance)
        for value, reference_value in zip(inputs, reference_inputs):
            torch.testing.assert_close(value.grad.float(), reference_value.grad, atol=tolerance, rtol=tolerance)

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    def test_dropout_with_a_window_raises_on_cuda(self):
        lengths = torch.tensor([40, 20], device="cuda")
        packed = pack_encoder_output(torch.zeros(2, 40, 1, device="cuda"), lengths)
        context = PackedAttentionContext(packed, key_window=_key_window("band", 40, device="cuda"))
        q = torch.randn(60, 2, 64, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="limited attention context"):
            packed_attention(q, q, q, context, dropout_p=0.1)


class TestPackedAttentionKernelShapes:
    """The shapes the compiled kernels see: powers of two from the padding floors, few per run."""

    @pytest.mark.unit
    def test_padded_sizes_are_powers_of_two_from_the_floors(self):
        contexts = [PackedAttentionContext.from_lengths(torch.tensor([n])) for n in (1, 4096, 4097, 70000)]
        assert [context.padded_total_tokens for context in contexts] == [4096, 4096, 8192, 131072]
        contexts = [PackedAttentionContext.from_lengths(torch.tensor([n])) for n in (1, 128, 129, 500)]
        assert [context.padded_max_seqlen for context in contexts] == [128, 128, 256, 512]
        context = PackedAttentionContext.from_lengths(torch.tensor([300, 7]))
        assert context.bias_rows().tolist() == list(range(307)) + [306] * (4096 - 307)

    @pytest.mark.unit
    def test_kernel_shapes_of_a_run_stay_below_the_recompile_limit(self):
        """Batches of any size up to 150,000 tokens with utterances up to 100,000 frames share few shapes."""
        shapes = set()
        for total in range(1, 150_000, 1_009):
            longest = min(total, 100_000)
            context = PackedAttentionContext.from_lengths(torch.tensor([longest, total - longest]))
            shapes.add((context.padded_total_tokens, context.padded_max_seqlen))
        assert len(shapes) < packed_attention_module.RECOMPILE_LIMIT // 2

    # Dynamo settings are changed through the config module's own `patch`, not through monkeypatch:
    # pytest's undo deletes the entry from torch's config module (torch 2.12 and 2.14),
    # and every later lookup of `recompile_limit` in the process fails.

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    def test_recompile_limit_miss_raises_instead_of_running_eagerly(self, monkeypatch):
        """Past its recompile limit Dynamo would run flex_attention eagerly over dense scores; the miss must fail."""
        monkeypatch.setattr(packed_attention_module, "RECOMPILE_LIMIT", 1)
        limit_name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
        torch._dynamo.reset()

        def attend(lengths):
            context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
            q, k, v = (torch.randn(context.total_tokens, 2, 16, device="cuda") for _ in range(3))
            bias = torch.randn(context.total_tokens, 2, 2 * context.padded_max_seqlen - 1, device="cuda")
            return packed_attention(q, k, v, context, rel_pos_bias=bias)

        with torch._dynamo.config.patch(**{limit_name: 1}):
            attend([40, 30])
            with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
                attend([5000, 30])

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    def test_token_total_does_not_recompile(self):
        """Batches with equal padded shapes but different token totals share one kernel.

        The relative-position bias enters the kernel with its unpadded token axis;
        without the dynamic mark on that axis the static compile specializes on every distinct total.
        """
        torch._dynamo.reset()

        def attend(lengths):
            context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
            q, k, v = (torch.randn(context.total_tokens, 2, 16, device="cuda") for _ in range(3))
            bias = torch.randn(context.total_tokens, 2, 2 * context.padded_max_seqlen - 1, device="cuda")
            return packed_attention(q, k, v, context, rel_pos_bias=bias)

        attend([40, 30])
        with torch._dynamo.config.patch(error_on_recompile=True):
            attend([40, 35])
            attend([45, 35, 20])

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    def test_kernels_have_a_dynamo_cache_of_their_own(self):
        """The packed kernels leave the code object of flex_attention itself alone.

        The Transformer encoders compile flex_attention itself with dynamic shapes;
        entries of the packed kernels on that code object would count against that compile's recompile limit.
        """
        torch._dynamo.reset()

        def attend(lengths):
            context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
            q, k, v = (torch.randn(context.total_tokens, 2, 16, device="cuda") for _ in range(3))
            bias = torch.randn(context.total_tokens, 2, 2 * context.padded_max_seqlen - 1, device="cuda")
            return packed_attention(q, k, v, context, rel_pos_bias=bias)

        attend([40, 30])
        attend([5000, 30])  # a second kernel shape, a second cache entry
        limit_name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
        # with a shared cache the two entries above would already be over the limit of this compile
        with torch._dynamo.config.patch(**{limit_name: 1, "fail_on_recompile_limit_hit": True}):
            compiled_itself = torch.compile(flex_attention, dynamic=True)
            q = torch.randn(1, 2, 64, 16, device="cuda")
            assert compiled_itself(q, q, q).shape == q.shape


class TestPackedRelPositionAttention:
    @pytest.mark.unit
    def test_matches_padded(self):
        torch.manual_seed(2)
        attention = RelPositionMultiHeadAttention(
            n_head=2, n_feat=8, dropout_rate=0.0, pos_bias_u=None, pos_bias_v=None
        )
        attention = attention.double()
        torch.nn.init.normal_(attention.pos_bias_u)
        torch.nn.init.normal_(attention.pos_bias_v)
        pos_enc = RelPositionalEncoding(d_model=8, dropout_rate=0.0, max_len=64).double()
        pos_enc.extend_pe(64, "cpu", torch.float64)
        lengths = torch.tensor([7, 3, 11])
        x = torch.randn(3, 11, 8, dtype=torch.float64, requires_grad=True)
        _, pos_emb = pos_enc(x)
        pad = torch.arange(11)[None, :] >= lengths[:, None]
        mask = pad[:, None, :] | pad[:, :, None]
        padded_out = attention(query=x, key=x, value=x, mask=mask, pos_emb=pos_emb)

        packed_x = x.detach().clone().requires_grad_()
        packed = pack_encoder_output(packed_x, lengths)
        packed_out = attention.forward_packed(packed.data, PackedAttentionContext(packed), pos_emb=pos_emb)
        torch.testing.assert_close(packed_out, _valid_frames(padded_out, lengths))

        _valid_frames(padded_out, lengths).square().sum().backward()
        reference_grads = {name: p.grad.clone() for name, p in attention.named_parameters()}
        attention.zero_grad()
        packed_out.square().sum().backward()
        torch.testing.assert_close(_valid_frames(packed_x.grad, lengths), _valid_frames(x.grad, lengths))
        for name, parameter in attention.named_parameters():
            torch.testing.assert_close(parameter.grad, reference_grads[name], msg=name)

    @pytest.mark.unit
    def test_empty_batch_keeps_gradients(self):
        q = torch.randn(0, 2, 4, requires_grad=True)
        bias = torch.randn(0, 2, 1, requires_grad=True)
        context = PackedAttentionContext.from_lengths(torch.tensor([0, 0]))
        out = packed_attention(q, q, q, context, rel_pos_bias=bias)
        out.sum().backward()
        assert out.shape == (0, 2, 4) and q.grad is not None and bias.grad is not None

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    @pytest.mark.parametrize(
        "lengths, head_dim",
        [([300, 7, 150, 1, 260], 32), ([1000, 513, 2], 32), ([300, 7, 150], 8), ([300, 7, 150], 44)],
        ids=["five_utterances", "long_utterances", "head_dim_8", "head_dim_44"],
    )
    def test_flex_attention_matches_reference_on_cuda(self, lengths, head_dim):
        """Also a head width below the kernel minimum of 16 and a non-power-of-two width, both zero-padded.

        Width 44 is the head of the released FastConformer Small; without the padding to a power of two
        it fell to a generic kernel configuration whose shared memory exceeds consumer Blackwell's (torch 2.12).
        """
        torch.manual_seed(0)
        lengths = torch.tensor(lengths, device="cuda")
        heads = 4
        states = torch.randn(len(lengths), int(lengths.max()), heads * head_dim, device="cuda")
        packed = pack_encoder_output(states, lengths)
        context = PackedAttentionContext(packed)
        total, max_len = packed.total_tokens, packed.max_seqlen
        q, k, v = (torch.randn(total, heads, head_dim, device="cuda", requires_grad=True) for _ in range(3))
        bias = torch.randn(total, heads, 2 * max_len - 1, device="cuda", requires_grad=True)

        out = packed_attention(q, k, v, context, rel_pos_bias=bias)
        out.square().sum().backward()
        grads = [t.grad.clone() for t in (q, k, v, bias)]
        for t in (q, k, v, bias):
            t.grad = None
        reference = packed_attention_module._reference_packed_attention(
            q, k, v, context, rel_pos_bias=bias, dropout_p=0.0, scale=head_dim**-0.5
        )
        reference.square().sum().backward()

        torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)
        for grad, t in zip(grads, (q, k, v, bias)):
            torch.testing.assert_close(grad, t.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPackedAttentionBackends:
    """Bias-free attention through variable-length FlashAttention and through FlexAttention, against a float32 reference.

    The two CUDA backends see the same half-precision queries, keys and values and the same nonuniform upstream
    gradient; the dispatch is asserted, and outputs and input gradients are compared with the dense per-utterance
    reference in float32 and with each other.
    The lengths hold an empty utterance and utterances around the 128-token block boundary.
    """

    @staticmethod
    def _flash_attention_available():
        from nemo.collections.asr.modules import transformer_encoder_utils

        return transformer_encoder_utils._get_flash_attention_varlen_for_device(torch.device("cuda")) is not None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "dtype, tolerance",
        [(torch.float16, 4e-3), (torch.bfloat16, 2e-2)],
        ids=["fp16", "bf16"],
    )
    def test_flash_and_flex_match_the_dense_reference(self, dtype, tolerance, monkeypatch):
        if not self._flash_attention_available():
            pytest.skip("requires a variable-length FlashAttention provider")
        torch.manual_seed(0)
        lengths = torch.tensor([127, 0, 128, 129, 300, 1], device="cuda")
        context = PackedAttentionContext.from_lengths(lengths)
        heads, head_dim = 2, 64
        q, k, v = (torch.randn(context.total_tokens, heads, head_dim, device="cuda", dtype=dtype) for _ in range(3))
        ramp = torch.linspace(0.1, 2.0, context.total_tokens, device="cuda", dtype=dtype)
        upstream = torch.randn(context.total_tokens, heads, head_dim, device="cuda", dtype=dtype) * ramp[:, None, None]
        flash_provider = packed_attention_module._flash_attention_varlen

        def run(backend):
            dispatched = []

            def select_provider(queries):
                provider = None if backend == "flex" else flash_provider(queries)
                dispatched.append("flex" if provider is None else "flash")
                return provider

            inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(packed_attention_module, "_flash_attention_varlen", select_provider)
                out = packed_attention(*inputs, context)
            out.backward(upstream)
            assert dispatched == [backend]
            return out.float(), [t.grad.float() for t in inputs]

        flash_out, flash_grads = run("flash")
        flex_out, flex_grads = run("flex")
        inputs = [t.detach().float().requires_grad_() for t in (q, k, v)]
        reference = packed_attention_module._reference_packed_attention(
            *inputs, context, rel_pos_bias=None, dropout_p=0.0, scale=head_dim**-0.5
        )
        reference.backward(upstream.float())
        reference_grads = [t.grad for t in inputs]

        for name, flash_value, flex_value, reference_value in zip(
            ("output", "dq", "dk", "dv"),
            [flash_out, *flash_grads],
            [flex_out, *flex_grads],
            [reference, *reference_grads],
        ):
            torch.testing.assert_close(flash_value, reference_value, atol=tolerance, rtol=tolerance, msg=name)
            torch.testing.assert_close(flex_value, reference_value, atol=tolerance, rtol=tolerance, msg=name)
            torch.testing.assert_close(flash_value, flex_value, atol=tolerance, rtol=tolerance, msg=name)
