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

"""The Triton depthwise convolution over packed utterances against a dense per-utterance reference."""

import pytest
import torch
import torch.nn.functional as F

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton import packed_depthwise_conv as kernel_module

requires_kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="requires CUDA and Triton"
)

# (kernel size, left padding, channels): the Conformer's kernel 32 over 1,024 channels (left 15, right 16),
# the odd kernel 31, a short kernel with an odd channel count, a causal kernel, and a channel count below a block
SHAPES = [(32, 15, 1024), (31, 15, 64), (5, 2, 44), (4, 3, 16), (8, 3, 6)]
LENGTHS = [37, 0, 1, 64, 5, 100, 33]


def _lengths_starts(lengths, device="cuda"):
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    starts = torch.cumsum(lengths, 0) - lengths
    return lengths, starts.to(torch.int32)


def _inputs(lengths, channels, width, *, has_bias, dtype, device="cuda"):
    """Random leaves: the packed activations (total, C), the filter (C, K) and the bias (C,)."""
    total = int(sum(lengths))
    x = torch.randn(total, channels, device=device, dtype=dtype, requires_grad=True)
    weight = (torch.randn(channels, width, device=device, dtype=dtype) * width**-0.5).requires_grad_()
    bias = torch.randn(channels, device=device, dtype=dtype, requires_grad=True) if has_bias else None
    return x, weight, bias


def _dense_reference(x, weight, bias, lengths, pad_left):
    """``F.conv1d`` of every utterance with the explicit left and right padding, concatenated in order."""
    width = weight.shape[1]
    outputs = []
    offset = 0
    for length in lengths:
        if length > 0:
            segment = F.pad(x[offset : offset + length].t().unsqueeze(0), (pad_left, width - 1 - pad_left))
            output = F.conv1d(segment, weight.unsqueeze(1), bias, groups=weight.shape[0])
            outputs.append(output.squeeze(0).t())
        offset += length
    return torch.cat(outputs) if outputs else x[:0]


def _relative_norm_error(actual, reference):
    reference = reference.double()
    return float((actual.double() - reference).norm() / reference.norm().clamp_min(1e-30))


def _run(fn, leaves, d_out):
    out = fn()
    grads = torch.autograd.grad(out, [leaf for leaf in leaves if leaf is not None], d_out.to(out.dtype))
    return out, grads


def _compare(kernel, reference, *, dtype):
    out, grads = kernel
    ref_out, ref_grads = reference
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref_out.to(dtype), rtol=1e-5, atol=1e-5)
        for grad, ref_grad in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(grad, ref_grad.to(dtype), rtol=1e-5, atol=1e-5)
    else:
        assert _relative_norm_error(out, ref_out) < 2e-2
        for grad, ref_grad in zip(grads, ref_grads, strict=True):
            assert _relative_norm_error(grad, ref_grad) < 2e-2


@requires_kernel
class TestPackedDepthwiseConvKernel:
    """Outputs and the gradients for the input, the filter and the bias against the float64 reference."""

    @pytest.mark.unit
    @pytest.mark.parametrize("width,pad_left,channels", SHAPES)
    @pytest.mark.parametrize("has_bias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_float64_reference(self, width, pad_left, channels, has_bias, dtype):
        torch.manual_seed(0)
        lengths, starts = _lengths_starts(LENGTHS)
        x, weight, bias = _inputs(LENGTHS, channels, width, has_bias=has_bias, dtype=dtype)
        d_out = torch.randn_like(x)
        leaves = [x, weight, bias]
        kernel = _run(
            lambda: kernel_module.packed_depthwise_conv1d(
                x, weight, bias, starts, lengths, max(LENGTHS), pad_left=pad_left
            ),
            leaves,
            d_out,
        )
        ref_leaves = [leaf.detach().double().requires_grad_() if leaf is not None else None for leaf in leaves]
        reference = _run(
            lambda: _dense_reference(ref_leaves[0], ref_leaves[1], ref_leaves[2], LENGTHS, pad_left),
            ref_leaves,
            d_out,
        )
        _compare(kernel, reference, dtype=dtype)

    @pytest.mark.unit
    def test_matches_padded_conv1d_at_a_production_batch(self):
        """Many utterances of unequal length over 1,024 channels with 32 taps in bfloat16, about 20,000 frames."""
        torch.manual_seed(1)
        lengths = torch.randint(1, 667, (60,)).tolist()
        length_tensor, starts = _lengths_starts(lengths)
        x, weight, bias = _inputs(lengths, 1024, 32, has_bias=True, dtype=torch.bfloat16)
        d_out = torch.randn_like(x)
        kernel = _run(
            lambda: kernel_module.packed_depthwise_conv1d(
                x, weight, bias, starts, length_tensor, max(lengths), pad_left=15
            ),
            [x, weight, bias],
            d_out,
        )
        ref_leaves = [leaf.detach().double().requires_grad_() for leaf in (x, weight, bias)]
        reference = _run(lambda: _dense_reference(*ref_leaves, lengths, 15), ref_leaves, d_out)
        _compare(kernel, reference, dtype=torch.bfloat16)
        # the filter gradient reduces over fixed row splits without atomics: a second run is bitwise the same
        again = _run(
            lambda: kernel_module.packed_depthwise_conv1d(
                x, weight, bias, starts, length_tensor, max(lengths), pad_left=15
            ),
            [x, weight, bias],
            d_out,
        )
        for first, second in zip(kernel[1], again[1], strict=True):
            assert torch.equal(first, second)

    @pytest.mark.unit
    def test_float16_matches_float64_reference(self):
        torch.manual_seed(5)
        lengths, starts = _lengths_starts(LENGTHS)
        x, weight, bias = _inputs(LENGTHS, 64, 32, has_bias=True, dtype=torch.float16)
        d_out = torch.randn_like(x)
        kernel = _run(
            lambda: kernel_module.packed_depthwise_conv1d(x, weight, bias, starts, lengths, max(LENGTHS), pad_left=15),
            [x, weight, bias],
            d_out,
        )
        ref_leaves = [leaf.detach().double().requires_grad_() for leaf in (x, weight, bias)]
        reference = _run(lambda: _dense_reference(*ref_leaves, LENGTHS, 15), ref_leaves, d_out)
        _compare(kernel, reference, dtype=torch.float16)

    @pytest.mark.unit
    def test_utterances_do_not_see_their_neighbours(self):
        """Changing one utterance changes neither the outputs nor the input gradients of the others."""
        torch.manual_seed(2)
        lengths = [20, 3, 40]
        length_tensor, starts = _lengths_starts(lengths)
        x, weight, bias = _inputs(lengths, 32, 7, has_bias=True, dtype=torch.float32)
        d_out = torch.randn_like(x)

        def run(x):
            out = kernel_module.packed_depthwise_conv1d(x, weight, bias, starts, length_tensor, 40, pad_left=3)
            (dx,) = torch.autograd.grad(out, [x], d_out)
            return out, dx

        out, dx = run(x)
        changed = x.detach().clone()
        changed[20:23] = torch.randn(3, 32, device="cuda")
        changed.requires_grad_()
        out_changed, dx_changed = run(changed)
        rows = torch.ones(63, dtype=torch.bool, device="cuda")
        rows[20:23] = False
        torch.testing.assert_close(out_changed[rows], out[rows])
        torch.testing.assert_close(dx_changed[rows], dx[rows])
        assert not torch.allclose(out_changed[20:23], out[20:23])

    @pytest.mark.unit
    def test_rows_outside_every_utterance_are_zero(self):
        """A capacity buffer with rows past the last utterance (and a gap between utterances) leaves them zero."""
        torch.manual_seed(3)
        channels, width = 16, 4
        x = torch.randn(50, channels, device="cuda", requires_grad=True)
        weight = torch.randn(channels, width, device="cuda", requires_grad=True)
        starts = torch.tensor([0, 15, 25], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([10, 7, 20], dtype=torch.int32, device="cuda")
        bias = torch.randn(channels, device="cuda", requires_grad=True)
        out = kernel_module.packed_depthwise_conv1d(x, weight, bias, starts, lengths, 20, pad_left=1)
        # junk in the rows outside every utterance must not enter the filter and bias gradients
        d_out = torch.randn_like(out)
        dx, dw, db = torch.autograd.grad(out, [x, weight, bias], d_out)
        inside = torch.zeros(50, dtype=torch.bool, device="cuda")
        inside[0:10] = inside[15:22] = inside[25:45] = True
        assert torch.equal(out[~inside], torch.zeros_like(out[~inside]))
        assert torch.equal(dx[~inside], torch.zeros_like(dx[~inside]))
        ref_weight = weight.detach().double().requires_grad_()
        ref_bias = bias.detach().double().requires_grad_()
        ref_dw = torch.zeros_like(ref_weight)
        ref_db = torch.zeros_like(ref_bias)
        for start, length in ((0, 10), (15, 7), (25, 20)):
            segment = x[start : start + length].detach().double().requires_grad_()
            reference = _dense_reference(segment, ref_weight, ref_bias, [length], 1)
            torch.testing.assert_close(out[start : start + length], reference.float(), rtol=1e-5, atol=1e-5)
            grads = torch.autograd.grad(
                reference, [segment, ref_weight, ref_bias], d_out[start : start + length].double()
            )
            torch.testing.assert_close(dx[start : start + length], grads[0].float(), rtol=1e-5, atol=1e-5)
            ref_dw += grads[1]
            ref_db += grads[2]
        torch.testing.assert_close(dw, ref_dw.float(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(db, ref_db.float(), rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    def test_cuda_graph_replay_follows_new_contents_and_lengths(self):
        """Forward and backward captured once, replayed with new activations and new utterance lengths."""
        torch.manual_seed(4)
        channels, width, pad_left = 24, 5, 2
        static_lengths, static_starts = _lengths_starts([30, 0, 55])
        x = torch.randn(85, channels, device="cuda", requires_grad=True)
        weight = torch.randn(channels, width, device="cuda", requires_grad=True)
        bias = torch.randn(channels, device="cuda", requires_grad=True)
        d_out = torch.randn(85, channels, device="cuda")

        def step():
            out = kernel_module.packed_depthwise_conv1d(
                x, weight, bias, static_starts, static_lengths, 60, pad_left=pad_left
            )
            return (out,) + torch.autograd.grad(out, [x, weight, bias], d_out)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = step()

        for lengths in ([30, 0, 55], [60, 20, 5], [1, 59, 25]):
            new_lengths, new_starts = _lengths_starts(lengths)
            static_lengths.copy_(new_lengths)
            static_starts.copy_(new_starts)
            with torch.no_grad():
                x.copy_(torch.randn_like(x))
                weight.copy_(torch.randn_like(weight))
                d_out.copy_(torch.randn_like(d_out))
            graph.replay()
            torch.cuda.synchronize()
            replayed = [tensor.clone() for tensor in static_outputs]
            eager = step()
            for name, got, want in zip(("out", "dx", "dw", "db"), replayed, eager, strict=True):
                torch.testing.assert_close(got, want, msg=name)

    @pytest.mark.unit
    def test_empty_input_keeps_the_gradient_edges(self):
        x = torch.zeros(0, 8, device="cuda", requires_grad=True)
        weight = torch.randn(8, 3, device="cuda", requires_grad=True)
        bias = torch.randn(8, device="cuda", requires_grad=True)
        starts = torch.zeros(2, dtype=torch.int32, device="cuda")
        lengths = torch.zeros(2, dtype=torch.int32, device="cuda")
        out = kernel_module.packed_depthwise_conv1d(x, weight, bias, starts, lengths, 0, pad_left=1)
        assert out.shape == (0, 8)
        grads = torch.autograd.grad(out.sum(), [x, weight, bias])
        assert all(not grad.any() for grad in grads)

    @pytest.mark.unit
    def test_rejects_what_the_kernels_cannot_take(self):
        lengths, starts = _lengths_starts([4, 6])
        x, weight, bias = _inputs([4, 6], 8, 3, has_bias=True, dtype=torch.float32)
        call = kernel_module.packed_depthwise_conv1d
        with pytest.raises(ValueError, match="CUDA"):
            call(x.cpu(), weight.cpu(), bias.cpu(), starts.cpu(), lengths.cpu(), 6, pad_left=1)
        with pytest.raises(ValueError, match="float32, bfloat16 or float16"):
            call(x.double(), weight.double(), bias.double(), starts, lengths, 6, pad_left=1)
        with pytest.raises(ValueError, match="dtype and device of x"):
            call(x, weight.half(), bias, starts, lengths, 6, pad_left=1)
        with pytest.raises(ValueError, match="shape \\(C, K\\)"):
            call(x, weight[:4], bias, starts, lengths, 6, pad_left=1)
        with pytest.raises(ValueError, match="pad_left"):
            call(x, weight, bias, starts, lengths, 6, pad_left=3)
        with pytest.raises(ValueError, match="int32 or int64"):
            call(x, weight, bias, starts.float(), lengths, 6, pad_left=1)
        with pytest.raises(ValueError, match="same shape"):
            call(x, weight, bias, starts[:1], lengths, 6, pad_left=1)
