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
from torch.nn.attention.flex_attention import create_block_mask

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


def _valid_frames(padded, lengths):
    """Rows of a padded (B, T, D) batch inside the lengths, in packing order."""
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    return padded[mask]


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

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    def test_recompile_limit_miss_raises_instead_of_running_eagerly(self, monkeypatch):
        """Past its recompile limit Dynamo would run flex_attention eagerly over dense scores; the miss must fail."""
        monkeypatch.setattr(packed_attention_module, "RECOMPILE_LIMIT", 1)
        monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 1)
        torch._dynamo.reset()

        def attend(lengths):
            context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device="cuda"))
            q, k, v = (torch.randn(context.total_tokens, 2, 16, device="cuda") for _ in range(3))
            bias = torch.randn(context.total_tokens, 2, 2 * context.padded_max_seqlen - 1, device="cuda")
            return packed_attention(q, k, v, context, rel_pos_bias=bias)

        attend([40, 30])
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            attend([5000, 30])


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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
    @pytest.mark.parametrize("lengths", [[300, 7, 150, 1, 260], [1000, 513, 2]])
    def test_flex_attention_matches_reference_on_cuda(self, lengths):
        torch.manual_seed(0)
        lengths = torch.tensor(lengths, device="cuda")
        heads, head_dim = 4, 32
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
