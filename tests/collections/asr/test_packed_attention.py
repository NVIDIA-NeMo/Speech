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


_CPU_AND_CUDA = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")],
    ),
]


class TestPackedAttentionArguments:
    """Every backend of ``packed_attention`` accepts and rejects the same calls, before the dispatch."""

    @staticmethod
    def _inputs(device, dtype, lengths, with_bias):
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device=device))
        q, k, v = (torch.randn(context.total_tokens, 2, 16, device=device, dtype=dtype) for _ in range(3))
        bias = None
        if with_bias:
            bias = torch.randn(context.total_tokens, 2, max(2 * context.max_seqlen - 1, 1), device=device, dtype=dtype)
        return q, k, v, context, bias

    # float32 without a bias runs FlexAttention on CUDA, the half types variable-length FlashAttention where available
    @pytest.mark.unit
    @pytest.mark.parametrize("device", _CPU_AND_CUDA)
    @pytest.mark.parametrize(
        "dtype, with_bias",
        [(torch.float32, False), (torch.bfloat16, False), (torch.float16, False), (torch.float32, True)],
        ids=["fp32", "bf16", "fp16", "fp32_rel_pos_bias"],
    )
    @pytest.mark.parametrize("lengths", [[5, 0, 3], [0, 0]], ids=["tokens", "empty"])
    @pytest.mark.parametrize(
        "argument, match",
        [
            ({"dropout_p": 1.0}, "dropout_p must be in"),
            ({"dropout_p": -0.1}, "dropout_p must be in"),
            ({"dropout_p": 1.5}, "dropout_p must be in"),
            ({"dropout_p": float("nan")}, "dropout_p must be in"),
            ({"scale": 0.0}, "scale must be positive"),
            ({"scale": -0.25}, "scale must be positive"),
            ({"scale": float("inf")}, "scale must be positive"),
            ({"scale": float("nan")}, "scale must be positive"),
        ],
        ids=lambda value: "_".join(f"{k}_{v}" for k, v in value.items()) if isinstance(value, dict) else "",
    )
    def test_scalar_arguments_are_checked_before_the_dispatch(
        self, device, dtype, with_bias, lengths, argument, match
    ):
        q, k, v, context, bias = self._inputs(device, dtype, lengths, with_bias)
        with pytest.raises(ValueError, match=match):
            packed_attention(q, k, v, context, rel_pos_bias=bias, **argument)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", _CPU_AND_CUDA)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    @pytest.mark.parametrize("lengths", [[5, 0, 3], [0, 0]], ids=["tokens", "empty"])
    @pytest.mark.parametrize(
        "name, shape, match",
        [
            ("q", (9, 2, 16), "q, k and v must share one shape"),
            ("k", (9, 2, 16), "q, k and v must share one shape"),
            ("k", (7, 2, 16), "q, k and v must share one shape"),
            ("k", (8, 3, 16), "q, k and v must share one shape"),
            ("v", (8, 2, 8), "q, k and v must share one shape"),
            ("rel_pos_bias", (9, 2, 9), "rel_pos_bias must have shape"),
            ("rel_pos_bias", (8, 3, 9), "rel_pos_bias must have shape"),
            ("rel_pos_bias", (8, 2, 8), "rel_pos_bias must have shape"),
            ("rel_pos_bias", (8, 2, 7), "rel_pos_bias must have shape"),
        ],
        ids=lambda value: "x".join(map(str, value)) if isinstance(value, tuple) else str(value).split(" must")[0],
    )
    def test_shapes_must_match_the_context(self, device, dtype, lengths, name, shape, match):
        q, k, v, context, bias = self._inputs(device, dtype, lengths, with_bias=True)
        tensors = {"q": q, "k": k, "v": v, "rel_pos_bias": bias}
        tensors[name] = torch.zeros(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match=match):
            packed_attention(tensors["q"], tensors["k"], tensors["v"], context, rel_pos_bias=tensors["rel_pos_bias"])

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize("devices", [("cpu", "cuda"), ("cuda", "cpu")], ids=["cuda_context", "cpu_context"])
    @pytest.mark.parametrize("lengths", [[5, 0, 3], [0, 0]], ids=["tokens", "empty"])
    def test_context_must_be_on_the_device_of_the_queries(self, devices, lengths):
        q, k, v, _, _ = self._inputs(devices[0], torch.float32, lengths, with_bias=False)
        context = PackedAttentionContext.from_lengths(torch.tensor(lengths, device=devices[1]))
        with pytest.raises(ValueError, match="the context on"):
            packed_attention(q, k, v, context)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", _CPU_AND_CUDA)
    @pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "rel_pos_bias"])
    @pytest.mark.parametrize("scale", [0.07, 1, torch.tensor(0.6)], ids=["0.07", "int_1", "tensor_0.6"])
    def test_explicit_scale_matches_dense_attention(self, device, with_bias, scale):
        """An accepted scale reaches the scores and the bias alike; float32 runs FlexAttention on CUDA."""
        torch.manual_seed(0)
        lengths = [5, 0, 3]
        q, k, v, context, bias = self._inputs(device, torch.float32, lengths, with_bias)

        def dense(q, k, v, bias):
            outputs = []
            for start, end in context.sequence_slices:
                scores = torch.einsum("ihd,jhd->hij", q[start:end], k[start:end])
                if bias is not None:
                    positions = torch.arange(end - start, device=q.device)
                    index = (bias.shape[-1] - 1) // 2 - positions[:, None] + positions[None, :]
                    scores = scores + bias[start:end].gather(2, index[:, None, :].expand(-1, 2, -1)).transpose(0, 1)
                weights = torch.softmax(scores * float(scale), dim=-1)
                outputs.append(torch.einsum("hij,jhd->ihd", weights, v[start:end]))
            return torch.cat(outputs)

        def run(function, dtype):
            inputs = [t.detach().to(dtype).requires_grad_() for t in (q, k, v)]
            bias_input = None if bias is None else bias.detach().to(dtype).requires_grad_()
            out = function(*inputs, bias_input)
            out.square().sum().backward()
            grads = [t.grad for t in inputs] + ([] if bias is None else [bias_input.grad])
            return [out] + grads

        actual = run(lambda q, k, v, bias: packed_attention(q, k, v, context, rel_pos_bias=bias, scale=scale), q.dtype)
        expected = run(dense, torch.float64)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value.double(), reference, rtol=1e-4, atol=1e-5)

    @pytest.mark.unit
    def test_reference_path_stays_finite_in_float16(self):
        """Products of queries and keys beyond the float16 range whose scaled scores are in range.

        The softmax saturates on such scores, so this checks that nothing overflows, not the scale.
        """
        torch.manual_seed(0)
        context = PackedAttentionContext.from_lengths(torch.tensor([5, 0, 3]))
        q = torch.randn(context.total_tokens, 2, 16, dtype=torch.float64) * 72.0
        v = torch.randn(context.total_tokens, 2, 16, dtype=torch.float64)
        largest_product = q.square().sum(-1).max()
        assert largest_product > torch.finfo(torch.float16).max > 0.25 * largest_product + 100

        def run(dtype):
            inputs = [t.to(dtype).requires_grad_() for t in (q, v)]
            # keys equal to the queries: every token's product with itself is a sum of squares
            out = packed_attention(inputs[0], inputs[0], inputs[1], context)
            out.float().sum().backward()
            return out, [t.grad for t in inputs]

        out, grads = run(torch.float16)
        reference, _ = run(torch.float64)
        assert out.dtype == torch.float16 and all(torch.isfinite(grad).all() for grad in grads)
        torch.testing.assert_close(out.double(), reference, rtol=2e-3, atol=2e-3)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", _CPU_AND_CUDA)
    def test_context_from_lengths_accepts_integer_lengths(self, device):
        reference = PackedAttentionContext.from_lengths(torch.tensor([3, 0, 2], device=device))
        context = PackedAttentionContext.from_lengths(torch.tensor([3, 0, 2], dtype=torch.int32, device=device))
        assert context.lengths.dtype == torch.int64 and torch.equal(context.cu_seqlens, reference.cu_seqlens)
        empty = PackedAttentionContext.from_lengths(torch.zeros(0, dtype=torch.int64, device=device))
        assert empty.total_tokens == 0 and empty.max_seqlen == 0 and empty.cu_seqlens.tolist() == [0]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", _CPU_AND_CUDA)
    @pytest.mark.parametrize(
        "lengths, error, match",
        [
            ([2.7, 3.2], TypeError, "integer dtype"),
            ([True, False], TypeError, "integer dtype"),
            ([3, -1, 2], ValueError, "must not be negative"),
            ([[3, 2]], ValueError, "must have shape"),
            ([2**31 - 1, 1], ValueError, "at most"),
        ],
        ids=["float", "bool", "negative", "two_dims", "beyond_int32"],
    )
    def test_context_from_lengths_rejects_other_lengths(self, device, lengths, error, match):
        with pytest.raises(error, match=match):
            PackedAttentionContext.from_lengths(torch.tensor(lengths, device=device))


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
