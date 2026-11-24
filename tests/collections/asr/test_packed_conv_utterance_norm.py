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
"""Instance and group normalization of the Conformer convolution module over sequence-packed input."""

import copy
import dataclasses
import functools

import pytest
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, packed_slot_layout
from nemo.collections.asr.parts.submodules.conformer_modules import (
    ConformerConvolution,
    ConformerFeedForward,
    packed_depthwise_conv_kernel_available,
)
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

D_MODEL = 6
KERNEL_SIZE = 5
# instance norm and one group per channel normalize every channel alone, group_norm1 all channels together
NORM_TYPES = ["instance_norm", "group_norm1", "group_norm2", f"group_norm{D_MODEL}"]


def _module(norm_type: str, d_model: int = D_MODEL) -> ConformerConvolution:
    module = ConformerConvolution(d_model=d_model, kernel_size=KERNEL_SIZE, norm_type=norm_type)
    if norm_type.startswith("group_norm"):
        # the default affine parameters are the identity, which would hide a wrong channel order
        torch.nn.init.normal_(module.batch_norm.weight)
        torch.nn.init.normal_(module.batch_norm.bias)
    return module


def _varied(batch: torch.Tensor) -> torch.Tensor:
    """Give every utterance its own scale and offset, so that statistics over the whole batch fit none of them."""
    index = torch.arange(batch.shape[0], device=batch.device, dtype=batch.dtype).view(-1, 1, 1)
    return (batch * (0.5 + 0.5 * index) + index - 1.0).requires_grad_()


def _packed_forward(module: ConformerConvolution, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    packed = pack_encoder_output(padded, lengths)
    return module.forward_packed(packed.data, packed_slot_layout(packed, gap=KERNEL_SIZE - 1))


def _alone_forward(module: ConformerConvolution, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The padded forward of every utterance as a batch of its own, the valid frames in packing order."""
    outputs = [module(padded[index : index + 1, :length]) for index, length in enumerate(lengths.tolist()) if length]
    return torch.cat(outputs, dim=1).squeeze(0)


def _gradients(module: torch.nn.Module, inputs: torch.Tensor, loss: torch.Tensor) -> dict[str, torch.Tensor]:
    names, parameters = zip(*module.named_parameters())
    gradients = torch.autograd.grad(loss, (inputs,) + parameters)
    return dict(zip(("input",) + names, gradients))


def _valid(padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return padded[torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]]


def _encoder(norm_type: str, **overrides) -> ConformerEncoder:
    """A small Conformer with the strided subsampling block and the given norm in its convolution modules."""
    options = dict(
        feat_in=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        subsampling_factor=6,
        xscaling=False,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
        pre_encode=functools.partial(
            ConvSubsampling,
            subsampling="striding",
            subsampling_factor=6,
            conv_channels=[2, 3, 3],
            conv_strides=[1, 3, 2],
            freq_pools=[2, 1, 1],
        ),
        feed_forward=functools.partial(ConformerFeedForward, activation="squared_relu"),
        conv=functools.partial(ConformerConvolution, kernel_size=4, norm_type=norm_type),
    )
    options.update(overrides)
    return ConformerEncoder(**options)


def _encoder_alone(encoder: ConformerEncoder, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The padded encoder on every utterance as a batch of its own, the output frames in packing order."""
    outputs = []
    for index, length in enumerate(lengths.tolist()):
        output, _ = encoder(audio_signal=features[index : index + 1, :, :length], length=lengths[index : index + 1])
        outputs.append(output.transpose(1, 2).squeeze(0))
    return torch.cat(outputs)


class TestPackedConvUtteranceNorm:
    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    @pytest.mark.parametrize("training", [True, False])
    def test_matches_every_utterance_alone(self, norm_type, training):
        """Outputs and all gradients equal the padded forward of every utterance as a batch of its own."""
        torch.manual_seed(1)
        module = _module(norm_type).double().train(training)
        lengths = torch.tensor([9, 2, 0, 40])
        padded = _varied(torch.randn(4, 40, D_MODEL, dtype=torch.float64))
        packed_out = _packed_forward(module, padded, lengths)
        alone_out = _alone_forward(module, padded, lengths)
        torch.testing.assert_close(packed_out, alone_out)
        weights = torch.randn_like(packed_out)
        packed_grads = _gradients(module, padded, (packed_out * weights).sum())
        alone_grads = _gradients(module, padded, (alone_out * weights).sum())
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, alone_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    def test_matches_padded_batch_without_padding(self, norm_type):
        """The padded forward of a batch agrees where it has no padded frame to count into the statistics."""
        torch.manual_seed(2)
        module = _module(norm_type).double()
        padded = torch.randn(3, 11, D_MODEL, dtype=torch.float64, requires_grad=True)
        even = torch.tensor([11, 11, 11])
        packed_out = _packed_forward(module, padded, even)
        padded_out = module(padded, pad_mask=torch.zeros(3, 11, dtype=torch.bool)).flatten(0, 1)
        torch.testing.assert_close(packed_out, padded_out)
        packed_grads = _gradients(module, padded, packed_out.square().sum())
        padded_grads = _gradients(module, padded, padded_out.square().sum())
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

        # with padding the padded forward counts the padded frames of the short utterance into its statistics
        uneven = torch.tensor([11, 4, 11])
        pad_mask = torch.arange(11)[None, :] >= uneven[:, None]
        difference = _packed_forward(module, padded, uneven) - _valid(module(padded, pad_mask=pad_mask), uneven)
        assert difference.abs().max() > 1e-3

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", f"group_norm{D_MODEL}"])
    def test_single_value_group_normalizes_to_the_bias(self, norm_type):
        """One frame of a one-channel group has no variance; the other utterances of the batch are unaffected."""
        torch.manual_seed(3)
        module = _module(norm_type).double()
        lengths = torch.tensor([1, 7])
        padded = torch.randn(2, 7, D_MODEL, dtype=torch.float64, requires_grad=True)
        packed_out = _packed_forward(module, padded, lengths)
        bias = module.batch_norm.bias if norm_type != "instance_norm" else torch.zeros(D_MODEL, dtype=torch.float64)
        expected = module.pointwise_conv2(module.activation(bias)[None, :, None]).flatten()
        torch.testing.assert_close(packed_out[0], expected)
        torch.testing.assert_close(packed_out[1:], module(padded[1:2]).squeeze(0))
        gradients = _gradients(module, padded, packed_out.square().sum())
        assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
        assert gradients["input"][0, 0].abs().max() == 0

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm2"])
    @pytest.mark.parametrize("lengths", [[], [0, 0]])
    def test_batch_without_frames_keeps_gradient_edges(self, norm_type, lengths):
        module = _module(norm_type).double()
        lengths = torch.tensor(lengths, dtype=torch.int64)
        padded = torch.zeros(lengths.numel(), 0, D_MODEL, dtype=torch.float64, requires_grad=True)
        packed_out = _packed_forward(module, padded, lengths)
        assert packed_out.shape == (0, D_MODEL)
        gradients = _gradients(module, padded, packed_out.sum())
        assert all(gradient.abs().sum() == 0 for gradient in gradients.values())

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm2"])
    def test_utterances_without_frames_in_a_layout_without_gap(self, norm_type):
        """A kernel of one frame needs no gap, so the utterances without frames share their slot start with the next."""
        torch.manual_seed(9)
        module = ConformerConvolution(d_model=D_MODEL, kernel_size=1, norm_type=norm_type).double()
        lengths = torch.tensor([0, 0, 5, 0, 3, 0])
        padded = _varied(torch.randn(6, 5, D_MODEL, dtype=torch.float64))
        packed = pack_encoder_output(padded, lengths)
        packed_out = module.forward_packed(packed.data, packed_slot_layout(packed, gap=0))
        alone_out = _alone_forward(module, padded, lengths)
        torch.testing.assert_close(packed_out, alone_out)
        packed_grads = _gradients(module, padded, packed_out.square().sum())
        alone_grads = _gradients(module, padded, alone_out.square().sum())
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, alone_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm2"])
    @pytest.mark.parametrize("gap", [0, KERNEL_SIZE - 1])
    def test_layout_without_lengths_gives_the_same_result(self, norm_type, gap):
        """``packed_slot_layout`` carries the lengths and the utterance indices; a layout built without them
        has them derived from its slots, also where utterances without frames share a slot start."""
        torch.manual_seed(12)
        kernel_size = 1 if gap == 0 else KERNEL_SIZE
        module = ConformerConvolution(d_model=D_MODEL, kernel_size=kernel_size, norm_type=norm_type).double()
        lengths = torch.tensor([0, 4, 0, 0, 7, 1, 0])
        packed = pack_encoder_output(_varied(torch.randn(7, 7, D_MODEL, dtype=torch.float64)), lengths)
        layout = packed_slot_layout(packed, gap=gap)
        assert torch.equal(layout.lengths, lengths) and layout.sequence_ids.tolist() == [1] * 4 + [4] * 7 + [5]
        bare = dataclasses.replace(layout, lengths=None, sequence_ids=None)
        outputs = module.forward_packed(packed.data, layout)
        torch.testing.assert_close(module.forward_packed(packed.data, bare), outputs)

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type, lengths", [("instance_norm", [50, 13, 31]), ("group_norm4", [50, 3, 31])])
    @pytest.mark.parametrize("self_attention_model", ["rel_pos", "abs_pos"])
    def test_encoder_matches_every_utterance_alone(self, norm_type, lengths, self_attention_model):
        """Through ``forward_sequence_packed`` in training mode; three input frames subsample to one encoder frame."""
        torch.manual_seed(4)
        encoder = _encoder(norm_type, self_attention_model=self_attention_model).double().train()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor(lengths)
        features = _varied(torch.randn(3, 8, 50, dtype=torch.float64))
        packed = encoder.forward_sequence_packed(features, lengths)
        alone = _encoder_alone(encoder, features, lengths)
        torch.testing.assert_close(packed.data, alone)
        packed_grads = _gradients(encoder, features, packed.data.square().sum())
        alone_grads = _gradients(encoder, features, alone.square().sum())
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, alone_grads[name], msg=name)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPackedConvUtteranceNormCuda:
    @pytest.fixture(autouse=True)
    def _float32_arithmetic(self):
        """TF32 off, which another test of the process may have left on: the float32 bounds are those of float32."""
        cudnn, matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cuda.matmul.allow_tf32 = matmul

    @staticmethod
    def _reference(module: ConformerConvolution, padded: torch.Tensor, lengths: torch.Tensor):
        """Outputs and gradients of the padded forward of every utterance alone, in float64."""
        module = copy.deepcopy(module).double()
        padded = padded.detach().double().requires_grad_()
        outputs = _alone_forward(module, padded, lengths)
        return outputs, _gradients(module, padded, outputs.square().mean())

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    def test_float32_matches_the_float64_reference(self, norm_type):
        torch.manual_seed(5)
        module = _module(norm_type, d_model=24).cuda()
        lengths = torch.tensor([300, 2, 0, 150], device="cuda")
        padded = _varied(torch.randn(4, 300, 24, device="cuda"))
        reference, reference_grads = self._reference(module, padded, lengths)
        outputs = _packed_forward(module, padded, lengths)
        assert outputs.dtype == torch.float32
        torch.testing.assert_close(outputs.double(), reference, atol=1e-4, rtol=1e-4)
        for name, gradient in _gradients(module, padded, outputs.square().mean()).items():
            torch.testing.assert_close(gradient.double(), reference_grads[name], atol=1e-5, rtol=1e-3, msg=name)

    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    def test_kernel_path_takes_the_boundaries_from_the_context(self, norm_type):
        """Where the depthwise convolution runs its Triton kernel the encoder builds no gapped layout:
        the normalization takes the utterance lengths and indices from the attention context,
        matches the float64 reference as the layout path does, and reads no value back from the device."""
        torch.manual_seed(5)
        module = _module(norm_type, d_model=24).cuda()
        lengths = torch.tensor([300, 2, 0, 150], device="cuda")
        padded = _varied(torch.randn(4, 300, 24, device="cuda"))
        reference, reference_grads = self._reference(module, padded, lengths)
        packed = pack_encoder_output(padded, lengths)
        assert packed_depthwise_conv_kernel_available(packed.data)
        context = PackedAttentionContext(packed)
        outputs = module.forward_packed(packed.data, context=context)
        torch.testing.assert_close(outputs.double(), reference, atol=1e-4, rtol=1e-4)
        for name, gradient in _gradients(module, padded, outputs.square().mean()).items():
            torch.testing.assert_close(gradient.double(), reference_grads[name], atol=1e-5, rtol=1e-3, msg=name)
        # a leaf: the backward of the packing itself, a boolean mask, is not what is tested
        data = packed.data.detach().requires_grad_()
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            module.forward_packed(data, context=context).square().sum().backward()
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        assert data.grad is not None and torch.isfinite(data.grad).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_half_precision_autocast_matches_the_float64_reference(self, norm_type, dtype):
        """Forward and backward under autocast; half precision rounds, so the gradients are compared in norm.

        Measured over twelve seeds: outputs within 1 percent (bfloat16) and 0.13 percent (float16),
        gradients within 9 percent and 1.2 percent; the bounds are about twice that for bfloat16, four times for float16.
        A normalization per channel removes the bias of the depthwise convolution,
        so its gradient is zero in exact arithmetic and 8e-5 in bfloat16, hence the absolute floor.
        """
        torch.manual_seed(6)
        module = _module(norm_type, d_model=24).cuda()
        lengths = torch.tensor([300, 2, 0, 150], device="cuda")
        padded = _varied(torch.randn(4, 300, 24, device="cuda"))
        reference, reference_grads = self._reference(module, padded, lengths)
        with torch.autocast("cuda", dtype=dtype):
            outputs = _packed_forward(module, padded, lengths)
        assert outputs.dtype == dtype
        relative = 2e-2 if dtype == torch.bfloat16 else 5e-3
        assert (outputs.double() - reference).norm() <= relative * reference.norm()
        for name, gradient in _gradients(module, padded, outputs.float().square().mean()).items():
            error = (gradient.double() - reference_grads[name]).norm()
            assert error <= 10 * relative * reference_grads[name].norm() + 5e-4, name

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm2"])
    def test_forward_and_backward_are_reproducible(self, norm_type):
        """The sums over an utterance run in a fixed order, forward and backward, as the other norm types do."""
        torch.manual_seed(10)
        module = _module(norm_type, d_model=64).cuda()
        lengths = torch.tensor([2900, 0, 1, 700, 1500], device="cuda")
        padded = _varied(torch.randn(5, 2900, 64, device="cuda"))
        results = []
        for _ in range(4):
            outputs = _packed_forward(module, padded, lengths)
            results.append((outputs, _gradients(module, padded, outputs.square().sum())))
        for outputs, gradients in results[1:]:
            assert torch.equal(outputs, results[0][0])
            assert all(torch.equal(gradient, results[0][1][name]) for name, gradient in gradients.items())

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm2", "layer_norm"])
    @pytest.mark.parametrize("with_lengths", [True, False])
    def test_forward_and_backward_do_not_synchronize(self, norm_type, with_lengths):
        """With a layout built before, the module reads no value back from the device, as with layer norm:
        the lengths and the utterance indices are those of the layout, or derived from its slots on the device."""
        torch.manual_seed(13)
        module = _module(norm_type, d_model=24).cuda()
        lengths = torch.tensor([16, 0, 7], device="cuda")
        packed = pack_encoder_output(_varied(torch.randn(3, 16, 24, device="cuda")), lengths)
        layout = packed_slot_layout(packed, gap=KERNEL_SIZE - 1)
        if not with_lengths:
            layout = dataclasses.replace(layout, lengths=None, sequence_ids=None)
        # a leaf: the backward of the packing itself, a boolean mask, is not what is tested
        data = packed.data.detach().requires_grad_()
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            module.forward_packed(data, layout).square().sum().backward()
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        assert data.grad is not None and torch.isfinite(data.grad).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm2"])
    def test_float16_module_statistics_do_not_overflow(self, norm_type):
        """A float16 module without autocast: activations of magnitude 3000 square beyond float16.

        Statistics in float16 give a relative error near one here; in float32 the measured error is 5e-4.
        """
        torch.manual_seed(7)
        module = _module(norm_type, d_model=24).cuda().half()
        lengths = torch.tensor([64, 9], device="cuda")
        padded = (3000.0 * torch.randn(2, 64, 24, device="cuda")).half().requires_grad_()
        reference, _ = self._reference(module, padded, lengths)
        outputs = _packed_forward(module, padded, lengths)
        assert outputs.dtype == torch.float16
        assert (outputs.double() - reference).norm() <= 5e-3 * reference.norm()
        # the gradients at this scale are below the float16 resolution, so only their finiteness is checked
        gradients = _gradients(module, padded, outputs.float().square().mean())
        assert all(torch.isfinite(gradient).all() for gradient in gradients.values())

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["instance_norm", "group_norm4"])
    def test_encoder_bf16_autocast_matches_every_utterance_alone(self, norm_type):
        """The training path through ``forward_sequence_packed`` in the precision the trainings use.

        Measured over twelve seeds: outputs within 7.2 percent with instance norm, whose padded forward
        has the same bfloat16 error, and 0.6 percent with group norm;
        gradients within 1 percent of their reference norm plus the absolute floor,
        which covers the tensors of the subsampling whose gradient norm is below 1e-5.
        """
        torch.manual_seed(8)
        encoder = _encoder(norm_type, d_model=32).cuda().train()
        lengths = torch.tensor([300, 13, 150], device="cuda")
        features = _varied(torch.randn(3, 8, 300, device="cuda"))
        reference_encoder = copy.deepcopy(encoder).double()
        reference_features = features.detach().double().requires_grad_()
        reference = _encoder_alone(reference_encoder, reference_features, lengths)
        reference_grads = _gradients(reference_encoder, reference_features, reference.square().mean())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed = encoder.forward_sequence_packed(features, lengths)
        assert (packed.data.double() - reference).norm() <= 0.15 * reference.norm()
        for name, gradient in _gradients(encoder, features, packed.data.float().square().mean()).items():
            error = (gradient.double() - reference_grads[name]).norm()
            assert error <= 0.05 * reference_grads[name].norm() + 1e-5, name
