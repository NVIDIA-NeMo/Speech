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

"""The fused activation and dropout kernel of the feed-forward hidden."""

import pytest
import torch

from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import SquaredReLU
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if not TRITON_AVAILABLE:
    pytest.skip("requires Triton", allow_module_level=True)

from nemo.collections.asr.parts.triton.activation_dropout import (  # noqa: E402
    FUSED_ACTIVATIONS,
    activation_dropout,
    activation_dropout_linear,
    activation_dropout_mask,
    fused_activation_name,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

REFERENCE = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "squared_relu": lambda x: torch.relu(x) ** 2,
    "silu": torch.nn.functional.silu,
    "gelu": torch.nn.functional.gelu,
}


def _seed(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int32, device="cuda")


class TestFusedActivationName:
    @pytest.mark.unit
    def test_registry_activations(self):
        assert fused_activation_name(torch.nn.Identity()) == "identity"
        assert fused_activation_name(torch.nn.ReLU()) == "relu"
        assert fused_activation_name(SquaredReLU()) == "squared_relu"
        assert fused_activation_name(torch.nn.SiLU()) == "silu"
        assert fused_activation_name(Swish()) == "silu"
        assert fused_activation_name(torch.nn.GELU()) == "gelu"
        assert set(FUSED_ACTIVATIONS) == {"identity", "relu", "squared_relu", "silu", "gelu"}

    @pytest.mark.unit
    def test_other_activations_are_not_fused(self):
        assert fused_activation_name(torch.nn.GELU(approximate="tanh")) is None
        assert fused_activation_name(torch.nn.Tanh()) is None
        assert fused_activation_name(torch.nn.Hardtanh()) is None

        class ReLUVariant(torch.nn.ReLU):
            def forward(self, x):
                return super().forward(x) * 2

        assert fused_activation_name(ReLUVariant()) is None


@requires_cuda
class TestActivationDropoutKernel:
    @pytest.mark.unit
    @pytest.mark.parametrize("activation", sorted(FUSED_ACTIVATIONS))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_matches_reference_under_its_mask(self, activation, dtype):
        torch.manual_seed(0)
        x = (torch.randn(37, 1031, device="cuda") * 3).to(dtype).requires_grad_()
        d_out = torch.randn(37, 1031, device="cuda").to(dtype)
        p = 0.3
        out = activation_dropout(x, activation, p, seed=_seed(123))
        out.backward(d_out)
        keep = activation_dropout_mask(x.shape, p, _seed(123), x.device)
        reference = x.detach().double().requires_grad_()
        expected = torch.where(keep, REFERENCE[activation](reference) / (1 - p), 0.0)
        expected.backward(d_out.double())
        tolerance = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
        assert out.dtype == dtype and x.grad.dtype == dtype
        torch.testing.assert_close(out.double(), expected, **tolerance)
        torch.testing.assert_close(x.grad.double(), reference.grad, **tolerance)
        assert 0.6 < keep.float().mean().item() < 0.8

    @pytest.mark.unit
    def test_without_dropout_is_the_activation(self):
        x = torch.randn(5, 7, 64, device="cuda", requires_grad=True)
        out = activation_dropout(x, "silu", 0.0)
        out.sum().backward()
        reference = x.detach().requires_grad_()
        expected = torch.nn.functional.silu(reference)
        expected.sum().backward()
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(x.grad, reference.grad, rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_mask_statistics(self):
        x = torch.ones(4_000_000, device="cuda")
        p = 0.35
        out = activation_dropout(x, "identity", p)
        kept = (out > 0).float().mean().item()
        # four standard deviations of the kept fraction of 4,000,000 Bernoulli draws
        assert abs(kept - (1 - p)) < 4 * ((1 - p) * p / 4_000_000) ** 0.5
        torch.testing.assert_close(out[out > 0], torch.full_like(out[out > 0], 1 / (1 - p)))
        # unbiased: the mean of the dropped-out activation is the mean of the activation
        assert abs(out.mean().item() - 1.0) < 2e-3

    @pytest.mark.unit
    def test_seed_determines_the_mask_and_calls_draw_fresh_ones(self):
        x = torch.randn(3, 1000, device="cuda")
        same = [activation_dropout(x, "relu", 0.5, seed=_seed(7)) for _ in range(2)]
        torch.testing.assert_close(same[0], same[1], rtol=0.0, atol=0.0)
        assert not torch.equal(same[0], activation_dropout(x, "relu", 0.5, seed=_seed(8)))
        torch.manual_seed(11)
        fresh = [activation_dropout(x, "relu", 0.5) for _ in range(2)]
        assert not torch.equal(fresh[0], fresh[1])
        torch.manual_seed(11)
        torch.testing.assert_close(activation_dropout(x, "relu", 0.5), fresh[0], rtol=0.0, atol=0.0)
        # a 64-bit seed is taken modulo 2**31
        wide = torch.tensor([2**31 + 7], dtype=torch.int64, device="cuda")
        torch.testing.assert_close(activation_dropout(x, "relu", 0.5, seed=wide), same[0], rtol=0.0, atol=0.0)

    @pytest.mark.unit
    def test_saves_the_input_and_the_seed_only(self):
        x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, t.numel())) or t, lambda t: t):
            out = activation_dropout(x, "squared_relu", 0.1)
        assert sorted(saved, key=str) == [(torch.bfloat16, x.numel()), (torch.int32, 1)]
        out.sum().backward()
        assert x.grad.shape == x.shape

    @pytest.mark.unit
    def test_cuda_graph_replays_draw_fresh_masks_and_follow_new_contents(self):
        torch.manual_seed(3)
        x = torch.randn(8, 512, device="cuda", requires_grad=True)
        d_out = torch.randn(8, 512, device="cuda")
        static_grad = torch.empty_like(x)

        def step():
            x.grad = None
            out = activation_dropout(x, "gelu", 0.25)
            out.backward(d_out)
            static_grad.copy_(x.grad)
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = step()
        outputs = []
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            outputs.append(static_out.clone())
            kept = static_out != 0
            # the kept entries hold the activation scaled by 1 / (1 - p), the gradient follows the same mask
            expected = torch.nn.functional.gelu(x.detach()) / 0.75
            torch.testing.assert_close(static_out[kept], expected[kept], rtol=1e-5, atol=1e-5)
            assert torch.equal(static_grad != 0, kept | (d_out == 0))
        assert not torch.equal(outputs[0] != 0, outputs[1] != 0)
        with torch.no_grad():
            x.copy_(torch.randn_like(x) * 2)
        graph.replay()
        torch.cuda.synchronize()
        kept = static_out != 0
        torch.testing.assert_close(
            static_out[kept], (torch.nn.functional.gelu(x.detach()) / 0.75)[kept], rtol=1e-5, atol=1e-5
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("bias", [True, False])
    def test_with_the_linear_layer_matches_the_separate_modules(self, dtype, bias):
        torch.manual_seed(0)
        linear = torch.nn.Linear(96, 24, bias=bias).cuda().to(dtype)
        x = torch.randn(3, 11, 96, device="cuda", dtype=dtype, requires_grad=True)
        d_out = torch.randn(3, 11, 24, device="cuda", dtype=dtype)
        p = 0.4
        out = activation_dropout_linear(x, "squared_relu", p, linear.weight, linear.bias, seed=_seed(5))
        out.backward(d_out)
        keep = activation_dropout_mask(x.shape, p, _seed(5), x.device)
        reference = torch.nn.Linear(96, 24, bias=bias).cuda().double()
        reference.load_state_dict(linear.state_dict())
        x_ref = x.detach().double().requires_grad_()
        expected = reference(torch.where(keep, torch.relu(x_ref) ** 2 / (1 - p), 0.0))
        expected.backward(d_out.double())
        assert out.dtype == dtype

        def close(actual, reference_value):
            # the absolute tolerance follows the tensor's magnitude: a sum of rounded products cancels at times
            relative = 1e-4 if dtype == torch.float32 else 2e-2
            scale = float(reference_value.detach().abs().max())
            torch.testing.assert_close(actual.double(), reference_value, rtol=relative, atol=relative * scale)

        close(out, expected)
        close(x.grad, x_ref.grad)
        close(linear.weight.grad, reference.weight.grad)
        if bias:
            close(linear.bias.grad, reference.bias.grad)

    @pytest.mark.unit
    def test_with_the_linear_layer_under_autocast(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 16).cuda()
        x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        saved = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                out = activation_dropout_linear(x, "silu", 0.2, linear.weight, linear.bias, seed=_seed(9))
        assert out.dtype == torch.bfloat16
        # the pre-activation, the seed and the weight; no activation output, mask or linear input
        assert {(str(t.dtype), t.numel()) for t in saved} == {
            ("torch.bfloat16", x.numel()),
            ("torch.float32", linear.weight.numel()),
            ("torch.int32", 1),
        }
        out.float().sum().backward()
        keep = activation_dropout_mask(x.shape, 0.2, _seed(9), x.device)
        hidden = torch.where(keep, torch.nn.functional.silu(x.detach().float()) / 0.8, 0.0)
        torch.testing.assert_close(linear.weight.grad, hidden.sum(0).expand(16, 64), rtol=2e-2, atol=2e-2)
        assert linear.bias.grad.dtype == torch.float32
        torch.testing.assert_close(linear.bias.grad, torch.full((16,), 5.0, device="cuda"))
        assert x.grad.dtype == torch.bfloat16

    @pytest.mark.unit
    def test_rejects_what_the_kernels_cannot_take(self):
        x = torch.randn(4, 8, device="cuda")
        with pytest.raises(RuntimeError, match="CUDA"):
            activation_dropout(x.cpu(), "relu", 0.1)
        with pytest.raises(TypeError, match="float32, bfloat16 or float16"):
            activation_dropout(x.double(), "relu", 0.1)
        with pytest.raises(ValueError, match="unknown activation 'tanh'"):
            activation_dropout(x, "tanh", 0.1)
        with pytest.raises(ValueError, match=r"p must be in \[0, 1\)"):
            activation_dropout(x, "relu", 1.0)
        with pytest.raises(ValueError, match="seed must be one integer"):
            activation_dropout(x, "relu", 0.1, seed=torch.tensor([1, 2], device="cuda"))
        assert activation_dropout(x[:0], "relu", 0.1).shape == (0, 8)
        with pytest.raises(ValueError, match="broadcast mask needs an axis"):
            activation_dropout(x[0, 0], "relu", 0.1, broadcast=True)
        assert activation_dropout(x[:0], "relu", 0.1, broadcast=True).shape == (0, 8)


class TestBroadcastActivationDropout:
    """One mask over the last axis, shared by every row: the kernels read it from a vector instead of hashing."""

    # widths below, at and above the kernel's tile of 1,024, so a tile wraps rows, matches them, or crosses one row end
    WIDTHS = [48, 1024, 1536]

    @pytest.mark.unit
    @pytest.mark.parametrize("activation", sorted(FUSED_ACTIVATIONS))
    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_reference_under_its_mask(self, activation, width, dtype):
        torch.manual_seed(0)
        x = (torch.randn(37, width, device="cuda") * 3).to(dtype).requires_grad_()
        d_out = torch.randn(37, width, device="cuda").to(dtype)
        p = 0.3
        out = activation_dropout(x, activation, p, broadcast=True, seed=_seed(123))
        out.backward(d_out)
        keep = activation_dropout_mask((width,), p, _seed(123), x.device)
        reference = x.detach().double().requires_grad_()
        expected = torch.where(keep, REFERENCE[activation](reference) / (1 - p), 0.0)
        expected.backward(d_out.double())
        tolerance = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
        assert out.dtype == dtype and x.grad.dtype == dtype
        torch.testing.assert_close(out.double(), expected, **tolerance)
        torch.testing.assert_close(x.grad.double(), reference.grad, **tolerance)
        # the same features are dropped in every row
        assert torch.equal(out[:, ~keep], torch.zeros_like(out[:, ~keep]))
        assert 0.5 < keep.float().mean().item() < 0.9

    @pytest.mark.unit
    def test_mask_is_shared_over_the_leading_axes_and_fresh_per_call(self):
        torch.manual_seed(0)
        x = torch.ones(4, 16, 1000, device="cuda")
        p = 0.35
        out = activation_dropout(x, "identity", p, broadcast=True)
        kept = out != 0
        assert torch.equal(kept, kept[:1, :1].expand_as(kept))
        torch.testing.assert_close(out[kept], torch.full_like(out[kept], 1 / (1 - p)))
        # a thousand Bernoulli draws of the feature mask: four standard deviations of the kept fraction
        assert abs(kept[0, 0].float().mean().item() - (1 - p)) < 4 * ((1 - p) * p / 1000) ** 0.5
        assert not torch.equal(kept, activation_dropout(x, "identity", p, broadcast=True) != 0)
        same = [activation_dropout(x, "identity", p, broadcast=True, seed=_seed(3)) for _ in range(2)]
        torch.testing.assert_close(same[0], same[1], rtol=0.0, atol=0.0)

    @pytest.mark.unit
    def test_identity_saves_the_mask_alone_and_drops_the_gradient(self):
        x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, t.numel())) or t, lambda t: t):
            out = activation_dropout(x, "identity", 0.1, broadcast=True)
        assert sorted(saved, key=str) == [(torch.bool, 4096), (torch.int32, 1)]
        d_out = torch.randn_like(out)
        out.backward(d_out)
        kept = out != 0
        torch.testing.assert_close(x.grad[kept], (d_out / 0.9)[kept], rtol=1e-2, atol=1e-2)
        assert torch.equal(x.grad[~kept], torch.zeros_like(x.grad[~kept]))
        # without the broadcast the identity saves the seed alone
        saved.clear()
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, t.numel())) or t, lambda t: t):
            activation_dropout(x, "identity", 0.1).sum().backward()
        assert saved == [(torch.int32, 1)]

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_with_the_linear_layer_matches_the_separate_modules(self, dtype):
        torch.manual_seed(0)
        linear = torch.nn.Linear(96, 24).cuda().to(dtype)
        x = torch.randn(3, 11, 96, device="cuda", dtype=dtype, requires_grad=True)
        d_out = torch.randn(3, 11, 24, device="cuda", dtype=dtype)
        p = 0.4
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            out = activation_dropout_linear(
                x, "squared_relu", p, linear.weight, linear.bias, broadcast=True, seed=_seed(5)
            )
        # the pre-activation, the seed, the 96 kept features and the weight
        assert sorted((str(t.dtype), t.numel()) for t in saved) == sorted(
            [(str(dtype), x.numel()), ("torch.int32", 1), ("torch.bool", 96), (str(dtype), linear.weight.numel())]
        )
        out.backward(d_out)
        keep = activation_dropout_mask((96,), p, _seed(5), x.device)
        reference = torch.nn.Linear(96, 24).cuda().double()
        reference.load_state_dict(linear.state_dict())
        x_ref = x.detach().double().requires_grad_()
        expected = reference(torch.where(keep, torch.relu(x_ref) ** 2 / (1 - p), 0.0))
        expected.backward(d_out.double())

        def close(actual, reference_value):
            relative = 1e-4 if dtype == torch.float32 else 2e-2
            scale = float(reference_value.detach().abs().max())
            torch.testing.assert_close(actual.double(), reference_value, rtol=relative, atol=relative * scale)

        close(out, expected)
        close(x.grad, x_ref.grad)
        close(linear.weight.grad, reference.weight.grad)
        close(linear.bias.grad, reference.bias.grad)

    @pytest.mark.unit
    def test_cuda_graph_replays_draw_fresh_feature_masks(self):
        torch.manual_seed(3)
        x = torch.randn(8, 512, device="cuda", requires_grad=True)

        def step():
            x.grad = None
            out = activation_dropout(x, "relu", 0.25, broadcast=True)
            out.sum().backward()
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = step()
        masks = []
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            kept = static_out != 0
            masks.append(kept.clone())
            # one feature mask for every row: a kept feature is nonzero wherever the activation is, so is its gradient
            assert torch.equal(kept, kept.any(0)[None] & (x.detach() > 0))
            torch.testing.assert_close(static_out[kept], (torch.relu(x.detach()) / 0.75)[kept], rtol=1e-5, atol=1e-5)
            assert torch.equal(x.grad != 0, kept)
        assert not torch.equal(masks[0], masks[1])
