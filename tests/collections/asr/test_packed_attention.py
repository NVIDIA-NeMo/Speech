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

"""Attention over packed sequences: the document block mask, the Triton kernel and the Transformer-XL attention."""

import pytest
import torch
import torch._dynamo
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, packed_sequence_ids
from nemo.collections.asr.parts.submodules import packed_attention as packed_attention_module
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.submodules.packed_attention import (
    PackedAttentionContext,
    document_block_mask,
    packed_attention,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton import rel_pos_attention as kernel_module

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="requires CUDA and Triton"
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


@pytest.fixture
def flex_attention_only(monkeypatch):
    """Route rel_pos attention to FlexAttention, the path without the Triton kernel."""
    monkeypatch.setattr(packed_attention_module, "TRITON_AVAILABLE", False)


def _rel_pos_inputs(context, heads, head_dim, device="cuda", dtype=torch.float32, pos_scale=1.0):
    """Random q, k, v, pos_query and pos_emb for the context, as leaves."""
    total, max_len = context.total_tokens, context.rel_pos_max_seqlen
    q, k, v, pos_query = (
        torch.randn(total, heads, head_dim, device=device, dtype=dtype, requires_grad=True) for _ in range(4)
    )
    pos_emb = (torch.randn(2 * max_len - 1, heads, head_dim, device=device, dtype=dtype) * pos_scale).requires_grad_()
    return q, k, v, pos_query, pos_emb


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
        assert context.seq_starts.tolist() == [0, 3, 3] and context.seq_lens.tolist() == [3, 0, 2]
        assert context.seq_starts.dtype == torch.int32 and context.seq_lens.dtype == torch.int32


class TestPackedAttentionKernelShapes:
    """The shapes the compiled FlexAttention kernels see: powers of two from the padding floors, few per run."""

    @pytest.mark.unit
    def test_padded_sizes_are_powers_of_two_from_the_floors(self):
        contexts = [PackedAttentionContext.from_lengths(torch.tensor([n])) for n in (1, 4096, 4097, 70000)]
        assert [context.padded_total_tokens for context in contexts] == [4096, 4096, 8192, 131072]
        contexts = [PackedAttentionContext.from_lengths(torch.tensor([n])) for n in (1, 128, 129, 500)]
        assert [context.padded_max_seqlen for context in contexts] == [128, 128, 256, 512]
        context = PackedAttentionContext.from_lengths(torch.tensor([300, 7]))
        assert context.bias_rows().tolist() == list(range(307)) + [306] * (4096 - 307)

    @pytest.mark.unit
    def test_rel_pos_table_length_follows_the_backend(self, monkeypatch):
        """FlexAttention needs the rounded length, the Triton kernel the longest utterance; the CPU runs neither."""
        context = PackedAttentionContext.from_lengths(torch.tensor([300, 7]))
        assert not context.rel_pos_kernel and context.rel_pos_max_seqlen == 512
        if torch.cuda.is_available():
            context = PackedAttentionContext.from_lengths(torch.tensor([300, 7], device="cuda"))
            assert context.rel_pos_kernel == TRITON_AVAILABLE
            assert context.rel_pos_max_seqlen == (300 if TRITON_AVAILABLE else 512)
            monkeypatch.setattr(packed_attention_module, "TRITON_AVAILABLE", False)
            context = PackedAttentionContext.from_lengths(torch.tensor([300, 7], device="cuda"))
            assert not context.rel_pos_kernel and context.rel_pos_max_seqlen == 512
            # an empty batch keeps a one-row table for the kernel
            monkeypatch.setattr(packed_attention_module, "TRITON_AVAILABLE", True)
            context = PackedAttentionContext.from_lengths(torch.tensor([0, 0], device="cuda"))
            assert context.rel_pos_kernel and context.rel_pos_max_seqlen == 1

    @pytest.mark.unit
    def test_kernel_shapes_of_a_run_stay_below_the_recompile_limit(self):
        """Batches of any size up to 150,000 tokens with utterances up to 100,000 frames share few shapes."""
        shapes = set()
        for total in range(1, 150_000, 1_009):
            longest = min(total, 100_000)
            context = PackedAttentionContext.from_lengths(torch.tensor([longest, total - longest]))
            shapes.add((context.padded_total_tokens, context.padded_max_seqlen))
        assert len(shapes) < packed_attention_module.RECOMPILE_LIMIT // 2

    @staticmethod
    def _attend_with_flex(lengths):
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        assert not context.rel_pos_kernel
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=2, head_dim=16)
        return packed_attention(q, k, v, context, pos_query=pos_query, pos_emb=pos_emb)

    @pytest.mark.unit
    @requires_cuda
    def test_recompile_limit_miss_raises_instead_of_running_eagerly(self, monkeypatch, flex_attention_only):
        """Past its recompile limit Dynamo would run flex_attention eagerly over dense scores; the miss must fail."""
        monkeypatch.setattr(packed_attention_module, "RECOMPILE_LIMIT", 1)
        limit_name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
        monkeypatch.setattr(torch._dynamo.config, limit_name, 1)
        torch._dynamo.reset()
        self._attend_with_flex([40, 30])
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            self._attend_with_flex([5000, 30])

    @pytest.mark.unit
    @requires_cuda
    def test_token_total_does_not_recompile(self, monkeypatch, flex_attention_only):
        """Batches with equal padded shapes but different token totals share one kernel.

        The relative-position bias enters the kernel with its unpadded token axis;
        without the dynamic mark on that axis the static compile specializes on every distinct total.
        """
        torch._dynamo.reset()
        self._attend_with_flex([40, 30])
        monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
        self._attend_with_flex([40, 35])
        self._attend_with_flex([45, 35, 20])

    @pytest.mark.unit
    @requires_cuda
    def test_kernels_have_a_dynamo_cache_of_their_own(self, monkeypatch, flex_attention_only):
        """The packed kernels leave the code object of flex_attention itself alone.

        The Transformer encoders compile flex_attention itself with dynamic shapes;
        entries of the packed kernels on that code object would count against that compile's recompile limit.
        """
        torch._dynamo.reset()
        self._attend_with_flex([40, 30])
        self._attend_with_flex([5000, 30])  # a second kernel shape, a second cache entry
        limit_name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
        monkeypatch.setattr(torch._dynamo.config, limit_name, 1)
        monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
        # with a shared cache the two entries above would already be over the limit of this compile
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
        pos_emb = torch.randn(1, 2, 4, requires_grad=True)
        context = PackedAttentionContext.from_lengths(torch.tensor([0, 0]))
        out = packed_attention(q, q, q, context, pos_query=q, pos_emb=pos_emb)
        out.sum().backward()
        assert out.shape == (0, 2, 4) and q.grad is not None and pos_emb.grad is not None

    @pytest.mark.unit
    def test_position_factors_come_together(self):
        q = torch.randn(5, 2, 4)
        context = PackedAttentionContext.from_lengths(torch.tensor([5]))
        with pytest.raises(ValueError, match="both or neither"):
            packed_attention(q, q, q, context, pos_query=q)

    @pytest.mark.unit
    @requires_cuda
    @pytest.mark.parametrize(
        "lengths, head_dim",
        [([300, 7, 150, 1, 260], 32), ([1000, 513, 2], 32), ([300, 7, 150], 8), ([300, 7, 150], 44)],
        ids=["five_utterances", "long_utterances", "head_dim_8", "head_dim_44"],
    )
    def test_flex_attention_matches_reference_on_cuda(self, lengths, head_dim, flex_attention_only):
        """Also a head width below the kernel minimum of 16 (zero-padded) and a non-power-of-two width."""
        torch.manual_seed(0)
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        assert not context.rel_pos_kernel
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=4, head_dim=head_dim)
        leaves = (q, k, v, pos_query, pos_emb)

        out = packed_attention(q, k, v, context, pos_query=pos_query, pos_emb=pos_emb)
        out.square().sum().backward()
        grads = [t.grad.clone() for t in leaves]
        for t in leaves:
            t.grad = None
        reference = packed_attention_module._reference_packed_attention(
            q, k, v, context, pos_query=pos_query, pos_emb=pos_emb, dropout_p=0.0, scale=head_dim**-0.5
        )
        reference.square().sum().backward()

        torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)
        for grad, t in zip(grads, leaves):
            torch.testing.assert_close(grad, t.grad, atol=1e-3, rtol=1e-3)


def _dense_reference(q, k, v, pos_query, pos_emb, starts, lens, scale, keep=None, dropout_p=0.0):
    """Dense attention one utterance at a time in the dtype of the inputs, for any packed layout.

    ``keep`` is the kept-weight mask ``(T_total, H, max_seqlen)`` of the kernel's dropout, if any.
    """
    out = torch.zeros_like(q)
    heads = q.shape[1]
    center = (pos_emb.shape[0] - 1) // 2
    for start, length in zip(starts, lens):
        if length == 0:
            continue
        rows = slice(start, start + length)
        scores = torch.einsum("ihd,jhd->hij", q[rows], k[rows]) * scale
        positions = torch.arange(length, device=q.device)
        relative = center + positions[None, :] - positions[:, None]
        bias = torch.einsum("ihd,rhd->hir", pos_query[rows], pos_emb)
        scores = scores + bias.gather(2, relative.expand(heads, -1, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        if keep is not None:
            weights = weights * keep[rows, :, :length].permute(1, 0, 2).to(weights.dtype) / (1.0 - dropout_p)
        out[rows] = torch.einsum("hij,jhd->ihd", weights, v[rows])
    return out


def _lengths_starts(lengths):
    """Starts of a dense packing of these lengths."""
    return [sum(lengths[:i]) for i in range(len(lengths))]


def _max_relative_error(actual, reference):
    """Largest absolute difference relative to the largest magnitude of the reference, in float64."""
    reference = reference.double()
    return float((actual.double() - reference).abs().max() / reference.abs().max().clamp_min(1e-12))


def _relative_norm_error(actual, reference):
    reference = reference.double()
    return float((actual.double() - reference).norm() / reference.norm().clamp_min(1e-12))


@requires_kernel
class TestPackedRelPosAttentionKernel:
    """The Triton kernel against the dense per-utterance reference in float64, on the kernel's own layout."""

    @staticmethod
    def _run(q, k, v, pos_query, pos_emb, starts, lens, max_seqlen, scale, *, dropout_p=0.0, seed=None):
        """Forward and the five gradients of the kernel, for a random upstream gradient of fixed seed."""
        leaves = (q, k, v, pos_query, pos_emb)
        out = kernel_module.packed_rel_pos_attention(
            q, k, v, pos_query, pos_emb, starts, lens, max_seqlen, dropout_p=dropout_p, dropout_seed=seed, scale=scale
        )
        d_out = torch.randn(out.shape, generator=torch.Generator(device="cuda").manual_seed(99), device="cuda")
        d_out = d_out.to(out.dtype)
        grads = torch.autograd.grad(out, leaves, d_out)
        return out, grads, d_out

    @staticmethod
    def _reference(q, k, v, pos_query, pos_emb, starts, lens, scale, d_out, *, keep=None, dropout_p=0.0):
        """The float64 reference outputs and gradients for the same (rounded) inputs."""
        leaves = [t.detach().double().requires_grad_() for t in (q, k, v, pos_query, pos_emb)]
        out = _dense_reference(*leaves, starts, lens, scale, keep=keep, dropout_p=dropout_p)
        grads = torch.autograd.grad(out, leaves, d_out.double())
        return out, grads

    @staticmethod
    def _check(kernel, reference, tolerance, norm=False):
        names = ("out", "d_q", "d_k", "d_v", "d_pos_query", "d_pos_emb")
        for name, actual, expected in zip(names, (kernel[0], *kernel[1]), (reference[0], *reference[1])):
            assert torch.isfinite(actual).all(), name
            error = _relative_norm_error(actual, expected) if norm else _max_relative_error(actual, expected)
            assert error <= tolerance, (name, error)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "lengths, head_dim",
        [([300, 7, 150, 1, 260], 64), ([257, 33], 128), ([5, 0, 1, 3], 64), ([300, 7, 150], 44)],
        ids=["five_utterances", "odd_widths", "empty_and_single_frame", "head_dim_44"],
    )
    def test_matches_float64_reference(self, lengths, head_dim):
        """Outputs and all five gradients in float32, against the reference in float64 and in float32."""
        torch.manual_seed(0)
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=4, head_dim=head_dim, pos_scale=0.5)
        starts, lens = context.seq_starts, context.seq_lens
        scale = head_dim**-0.5
        kernel = self._run(q, k, v, pos_query, pos_emb, starts, lens, context.max_seqlen, scale)
        reference64 = self._reference(q, k, v, pos_query, pos_emb, _lengths_starts(lengths), lengths, scale, kernel[2])
        self._check(kernel, reference64, tolerance=2e-5)
        # the tolerance is at the float32 level: the float32 reference meets it against the float64 one as well
        reference32 = _dense_reference(
            *(t.detach() for t in (q, k, v, pos_query, pos_emb)), _lengths_starts(lengths), lengths, scale
        )
        assert _max_relative_error(reference32, reference64[0]) <= 2e-5

    @pytest.mark.unit
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_matches_float64_reference_in_bfloat16(self, head_dim):
        torch.manual_seed(1)
        lengths = [300, 7, 150, 1, 260]
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(
            context, heads=4, head_dim=head_dim, dtype=torch.bfloat16, pos_scale=0.5
        )
        scale = head_dim**-0.5
        kernel = self._run(
            q, k, v, pos_query, pos_emb, context.seq_starts, context.seq_lens, context.max_seqlen, scale
        )
        reference64 = self._reference(q, k, v, pos_query, pos_emb, _lengths_starts(lengths), lengths, scale, kernel[2])
        assert kernel[0].dtype == torch.bfloat16 and all(grad.dtype == torch.bfloat16 for grad in kernel[1])
        self._check(kernel, reference64, tolerance=3e-2, norm=True)

    @pytest.mark.unit
    def test_gaps_between_utterances_and_junk_rows(self):
        """Any layout: the rows between the utterances are never read and come back as zeros."""
        torch.manual_seed(2)
        lens = [40, 0, 25, 9]
        starts = [8, 60, 61, 100]
        total, heads, head_dim = 128, 2, 32
        q, k, v, pos_query = (torch.randn(total, heads, head_dim, device="cuda") * 100 for _ in range(4))
        for start, length in zip(starts, lens):
            for tensor in (q, k, v, pos_query):
                tensor[start : start + length] = torch.randn(length, heads, head_dim, device="cuda")
        leaves = [t.requires_grad_() for t in (q, k, v, pos_query)]
        pos_emb = (torch.randn(2 * 40 - 1, heads, head_dim, device="cuda") * 0.5).requires_grad_()
        starts_t = torch.tensor(starts, device="cuda", dtype=torch.int32)
        lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
        scale = head_dim**-0.5
        kernel = self._run(*leaves, pos_emb, starts_t, lens_t, 40, scale)
        reference64 = self._reference(*leaves, pos_emb, starts, lens, scale, kernel[2])
        self._check(kernel, reference64, tolerance=2e-5)
        valid = torch.zeros(total, dtype=torch.bool, device="cuda")
        for start, length in zip(starts, lens):
            valid[start : start + length] = True
        assert (kernel[0][~valid] == 0).all()
        for grad in kernel[1][:4]:
            assert (grad[~valid] == 0).all()

    @pytest.mark.unit
    def test_wider_position_table_is_sliced_to_the_utterances(self):
        """A table beyond the longest utterance adds nothing; its outer rows get no gradient."""
        torch.manual_seed(3)
        lengths = [30, 12]
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        q, k, v, pos_query, _ = _rel_pos_inputs(context, heads=2, head_dim=32)
        pos_emb = torch.randn(2 * 100 - 1, 2, 32, device="cuda", requires_grad=True)
        scale = 32**-0.5
        kernel = self._run(q, k, v, pos_query, pos_emb, context.seq_starts, context.seq_lens, 30, scale)
        reference64 = self._reference(q, k, v, pos_query, pos_emb, [0, 30], lengths, scale, kernel[2])
        self._check(kernel, reference64, tolerance=2e-5)
        d_pos_emb = kernel[1][4]
        assert (d_pos_emb[:70] == 0).all() and (d_pos_emb[129:] == 0).all() and d_pos_emb[70:129].abs().sum() > 0

    @pytest.mark.unit
    def test_dropout_mask_statistics_and_exact_parity(self):
        """The kept weights follow the probability, and the output equals the reference with the kernel's mask."""
        torch.manual_seed(4)
        lengths = [300, 7, 150, 260]
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        heads, head_dim, dropout_p = 4, 64, 0.3
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=heads, head_dim=head_dim, pos_scale=0.5)
        seed = torch.tensor([12345], dtype=torch.int32, device="cuda")
        keep = kernel_module._keep_mask(
            context.total_tokens, heads, context.max_seqlen, dropout_p=dropout_p, dropout_seed=seed, device="cuda"
        )
        assert keep.shape == (context.total_tokens, heads, context.max_seqlen)
        valid_pairs = sum(length * length for length in lengths) * heads
        kept = sum(
            int(keep[start : start + length, :, :length].sum()) for start, length in zip([0, 300, 307, 457], lengths)
        )
        assert abs(kept / valid_pairs - (1.0 - dropout_p)) < 0.01
        scale = head_dim**-0.5
        kernel = self._run(
            q,
            k,
            v,
            pos_query,
            pos_emb,
            context.seq_starts,
            context.seq_lens,
            context.max_seqlen,
            scale,
            dropout_p=dropout_p,
            seed=seed,
        )
        reference64 = self._reference(
            q, k, v, pos_query, pos_emb, [0, 300, 307, 457], lengths, scale, kernel[2], keep=keep, dropout_p=dropout_p
        )
        self._check(kernel, reference64, tolerance=2e-5)

    @pytest.mark.unit
    def test_dropout_seed_determines_the_mask(self):
        torch.manual_seed(5)
        context = PackedAttentionContext.from_lengths(torch.tensor([64, 30], device="cuda"))
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=2, head_dim=32)
        args = (q, k, v, pos_query, pos_emb, context.seq_starts, context.seq_lens, context.max_seqlen)

        def attend(seed):
            return kernel_module.packed_rel_pos_attention(*args, dropout_p=0.5, dropout_seed=seed).detach()

        seed_a = torch.tensor([7], dtype=torch.int32, device="cuda")
        torch.testing.assert_close(attend(seed_a), attend(seed_a), atol=0.0, rtol=0.0)
        assert not torch.equal(attend(seed_a), attend(torch.tensor([8], dtype=torch.int32, device="cuda")))
        # without a seed the kernel draws one from the device generator: fresh per call, reproducible per seed
        torch.manual_seed(11)
        first = attend(None)
        second = attend(None)
        assert not torch.equal(first, second)
        torch.manual_seed(11)
        torch.testing.assert_close(attend(None), first, atol=0.0, rtol=0.0)
        without_dropout = kernel_module.packed_rel_pos_attention(*args).detach()
        assert not torch.equal(first, without_dropout)

    @pytest.mark.unit
    def test_cuda_graph_replay_follows_new_contents_and_lengths(self):
        """Forward and backward captured once; replays see new tensors, new lengths within the bound, the seed."""
        torch.manual_seed(6)
        heads, head_dim, capacity, bound = 2, 32, 160, 64
        static = {
            name: torch.randn(capacity, heads, head_dim, device="cuda", requires_grad=True)
            for name in ("q", "k", "v", "pos_query")
        }
        static["pos_emb"] = torch.randn(2 * bound - 1, heads, head_dim, device="cuda", requires_grad=True)
        d_out = torch.randn(capacity, heads, head_dim, device="cuda")
        starts = torch.tensor([0, 60, 100], dtype=torch.int32, device="cuda")
        lens = torch.tensor([60, 40, 60], dtype=torch.int32, device="cuda")
        seed = torch.tensor([3], dtype=torch.int32, device="cuda")
        leaves = tuple(static.values())

        def step():
            out = kernel_module.packed_rel_pos_attention(
                *leaves, starts, lens, bound, dropout_p=0.2, dropout_seed=seed, scale=head_dim**-0.5
            )
            return (out, *torch.autograd.grad(out, leaves, d_out))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()

        def eager():
            return [t.detach().clone() for t in step()]

        for new_lens, new_seed in (([30, 0, 55], 21), ([60, 37, 2], 22)):
            for tensor in leaves:
                tensor.data.copy_(torch.randn_like(tensor))
            lens.copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
            seed.fill_(new_seed)
            graph.replay()
            replayed = [t.clone() for t in captured]
            for actual, expected in zip(replayed, eager()):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.unit
    def test_cuda_graph_replays_draw_fresh_dropout_masks(self):
        """Without a given seed the capture holds a device draw, so every replay has another mask."""
        torch.manual_seed(7)
        context = PackedAttentionContext.from_lengths(torch.tensor([64, 30], device="cuda"))
        q, k, v, pos_query, pos_emb = (t.detach() for t in _rel_pos_inputs(context, heads=2, head_dim=32))
        args = (q, k, v, pos_query, pos_emb, context.seq_starts, context.seq_lens, context.max_seqlen)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                kernel_module.packed_rel_pos_attention(*args, dropout_p=0.5)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = kernel_module.packed_rel_pos_attention(*args, dropout_p=0.5)
        graph.replay()
        first = out.clone()
        graph.replay()
        second = out.clone()
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        assert not torch.equal(first, second)

    @pytest.mark.unit
    def test_production_shape_stays_below_the_int32_bound(self):
        """Eight heads of 128 over a packed batch of the training size: finite, and one call equals two halves.

        The position term is addressed as ``h * T_total * R + i * R + r`` in 64 bits (R padded to eight);
        at this shape the product stays below 2**31, so a 32-bit regression would not show here,
        which test_offsets_beyond_the_int32_bound covers.
        """
        torch.manual_seed(8)
        generator = torch.Generator().manual_seed(8)
        lengths = []
        while sum(lengths) < 83_000:
            lengths.append(int(torch.randint(50, 668, (1,), generator=generator)))
        heads, head_dim = 8, 128
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        assert heads * context.total_tokens * (2 * context.max_seqlen - 1 + 8) < 2**31
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=heads, head_dim=head_dim, dtype=torch.bfloat16)
        self._full_equals_halves(q, k, v, pos_query, pos_emb, lengths, head_dim**-0.5)

    @pytest.mark.unit
    def test_offsets_beyond_the_int32_bound(self):
        """The position term of 96 utterances of 2,000 frames with four heads exceeds 2**31 elements."""
        if torch.cuda.get_device_properties(0).total_memory < 64 * 1024**3:
            pytest.skip("needs a 64 GiB GPU for the position term and its gradient")
        torch.manual_seed(9)
        heads, head_dim, seq_len, n_seqs = 4, 64, 2000, 96
        lengths = [seq_len] * n_seqs
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
        assert heads * context.total_tokens * (2 * seq_len - 1) > 2**31
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=heads, head_dim=head_dim, pos_scale=0.1)
        self._full_equals_halves(q, k, v, pos_query, pos_emb, lengths, head_dim**-0.5)

    def _full_equals_halves(self, q, k, v, pos_query, pos_emb, lengths, scale):
        """One call over all utterances against two calls over the halves of the batch, outputs and gradients."""
        max_seqlen = max(lengths)
        starts = _lengths_starts(lengths)
        d_out = torch.randn(q.shape, generator=torch.Generator(device="cuda").manual_seed(99), device="cuda")
        d_out = d_out.to(q.dtype)

        def run(first, last):
            begin, end = starts[first], starts[last - 1] + lengths[last - 1]
            leaves = [t[begin:end].detach().clone().requires_grad_() for t in (q, k, v, pos_query)]
            table = pos_emb.detach().clone().requires_grad_()
            starts_t = torch.tensor([s - begin for s in starts[first:last]], dtype=torch.int32, device="cuda")
            lens_t = torch.tensor(lengths[first:last], dtype=torch.int32, device="cuda")
            out = kernel_module.packed_rel_pos_attention(*leaves, table, starts_t, lens_t, max_seqlen, scale=scale)
            return [out.detach(), *torch.autograd.grad(out, (*leaves, table), d_out[begin:end])]

        half = len(lengths) // 2
        full, first, second = run(0, len(lengths)), run(0, half), run(half, len(lengths))
        tolerance = 1e-2 if q.dtype == torch.bfloat16 else 1e-4
        for name, whole, a, b in zip(("out", "d_q", "d_k", "d_v", "d_pos_query", "d_pos_emb"), full, first, second):
            halves = a + b if name == "d_pos_emb" else torch.cat([a, b])
            assert torch.isfinite(whole).all(), name
            error = _relative_norm_error(whole, halves)
            assert error <= tolerance, (name, error)

    @pytest.mark.unit
    def test_rejects_what_the_kernels_cannot_take(self):
        context = PackedAttentionContext.from_lengths(torch.tensor([20, 5], device="cuda"))
        q, k, v, pos_query, pos_emb = (t.detach() for t in _rel_pos_inputs(context, heads=2, head_dim=32))
        starts, lens, max_seqlen = context.seq_starts, context.seq_lens, context.max_seqlen
        attend = kernel_module.packed_rel_pos_attention
        with pytest.raises(TypeError, match="float32, bfloat16 or float16"):
            attend(q.double(), k.double(), v.double(), pos_query.double(), pos_emb.double(), starts, lens, max_seqlen)
        with pytest.raises(TypeError, match="pos_emb must be"):
            attend(q, k, v, pos_query, pos_emb.half(), starts, lens, max_seqlen)
        with pytest.raises(ValueError, match="odd number of rows"):
            attend(q, k, v, pos_query, pos_emb[1:], starts, lens, max_seqlen)
        with pytest.raises(ValueError, match="covers no utterance"):
            attend(q, k, v, pos_query, pos_emb[2:-2], starts, lens, max_seqlen)
        with pytest.raises(ValueError, match="dropout_p"):
            attend(q, k, v, pos_query, pos_emb, starts, lens, max_seqlen, dropout_p=1.0)
        with pytest.raises(ValueError, match="dropout_seed"):
            attend(q, k, v, pos_query, pos_emb, starts, lens, max_seqlen, dropout_p=0.1, dropout_seed=starts)
        with pytest.raises(ValueError, match="shape of q"):
            attend(q, k[:-1], v, pos_query, pos_emb, starts, lens, max_seqlen)


@requires_kernel
class TestPackedRelPosAttentionDispatch:
    """What packed_attention routes to the kernel, and what it keeps for FlexAttention and the reference."""

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_kernel_is_the_default_for_rel_pos_on_cuda(self, monkeypatch, dtype):
        calls = []
        original = packed_attention_module.packed_rel_pos_attention
        monkeypatch.setattr(
            packed_attention_module, "packed_rel_pos_attention", lambda *a, **kw: calls.append(1) or original(*a, **kw)
        )
        monkeypatch.setattr(packed_attention_module, "_flex_packed_attention", _fail)
        context = PackedAttentionContext.from_lengths(torch.tensor([30, 7], device="cuda"))
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=2, head_dim=32, dtype=dtype)
        out = packed_attention(q, k, v, context, pos_query=pos_query, pos_emb=pos_emb, dropout_p=0.1)
        assert calls == [1] and out.dtype == dtype and out.shape == q.shape

    @pytest.mark.unit
    def test_float64_runs_the_reference_on_cuda(self, monkeypatch):
        """Neither the kernel nor FlexAttention's Inductor template takes float64; the dense reference does."""
        calls = []
        original = packed_attention_module._reference_packed_attention
        monkeypatch.setattr(
            packed_attention_module,
            "_reference_packed_attention",
            lambda *a, **kw: calls.append(1) or original(*a, **kw),
        )
        monkeypatch.setattr(packed_attention_module, "_flex_packed_attention", _fail)
        context = PackedAttentionContext.from_lengths(torch.tensor([30, 7], device="cuda"))
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=2, head_dim=32, dtype=torch.float64)
        out = packed_attention(q, k, v, context, pos_query=pos_query, pos_emb=pos_emb, dropout_p=0.1)
        assert calls == [1] and out.dtype == torch.float64 and out.shape == q.shape

    @pytest.mark.unit
    def test_dropout_is_refused_only_without_the_kernel(self, flex_attention_only):
        context = PackedAttentionContext.from_lengths(torch.tensor([30, 7], device="cuda"))
        q, k, v, pos_query, pos_emb = _rel_pos_inputs(context, heads=2, head_dim=32)
        with pytest.raises(NotImplementedError, match="dropout_att"):
            packed_attention(q, k, v, context, pos_query=pos_query, pos_emb=pos_emb, dropout_p=0.1)

    @pytest.mark.unit
    def test_rel_position_attention_module_matches_padded_without_a_padded_copy(self, monkeypatch):
        """The Transformer-XL module through the kernel equals its padded forward; no pad and no FlexAttention run."""
        torch.manual_seed(2)
        attention = RelPositionMultiHeadAttention(
            n_head=4, n_feat=256, dropout_rate=0.0, pos_bias_u=None, pos_bias_v=None
        )
        torch.nn.init.normal_(attention.pos_bias_u)
        torch.nn.init.normal_(attention.pos_bias_v)
        attention = attention.cuda()
        pos_enc = RelPositionalEncoding(d_model=256, dropout_rate=0.0, max_len=400).cuda()
        pos_enc.extend_pe(400, "cuda", torch.float32)
        lengths = torch.tensor([300, 7, 150], device="cuda")
        x = torch.randn(3, 300, 256, device="cuda", requires_grad=True)
        _, pos_emb = pos_enc(x)
        pad = torch.arange(300, device="cuda")[None, :] >= lengths[:, None]
        mask = pad[:, None, :] | pad[:, :, None]
        padded_out = attention(query=x, key=x, value=x, mask=mask, pos_emb=pos_emb)
        _valid_frames(padded_out, lengths).square().sum().backward()
        reference_grads = {name: p.grad.clone() for name, p in attention.named_parameters()}
        attention.zero_grad()

        # the padded forward pads inside rel_shift; the packed one must need neither pad nor FlexAttention
        monkeypatch.setattr(packed_attention_module, "_flex_packed_attention", _fail)
        monkeypatch.setattr(torch.nn.functional, "pad", _fail)
        packed_x = x.detach().clone().requires_grad_()
        packed = pack_encoder_output(packed_x, lengths)
        context = PackedAttentionContext(packed)
        assert context.rel_pos_kernel
        packed_out = attention.forward_packed(packed.data, context, pos_emb=pos_emb)
        torch.testing.assert_close(packed_out, _valid_frames(padded_out, lengths), atol=1e-4, rtol=1e-4)
        packed_out.square().sum().backward()
        torch.testing.assert_close(
            _valid_frames(packed_x.grad, lengths), _valid_frames(x.grad, lengths), atol=1e-3, rtol=1e-3
        )
        for name, parameter in attention.named_parameters():
            torch.testing.assert_close(parameter.grad, reference_grads[name], atol=1e-3, rtol=1e-3, msg=name)


def _fail(*args, **kwargs):
    raise AssertionError("this path must not run")
