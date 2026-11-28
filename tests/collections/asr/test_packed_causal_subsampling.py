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
"""The causal ``ConvSubsampling`` stacks over sequence-packed input, against the padded forward."""

import copy

import pytest
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling

FEAT_IN = 8
FEAT_OUT = 12
CAUSAL_STACKS = [(subsampling, factor) for subsampling in ("striding", "dw_striding") for factor in (2, 4, 8, 16)]
CUDA_CAUSAL_STACKS = [("striding", 4), ("dw_striding", 8)]
# empty, one-frame and odd utterances and multiples of the factors, whose slots have no spare frame, between long ones;
# a causal stack gives an empty utterance one output frame
LENGTHS = [37, 1, 0, 12, 2, 3, 36, 5, 16, 7, 32, 8, 4, 64]


def _causal(subsampling: str, factor: int) -> ConvSubsampling:
    return ConvSubsampling(subsampling, factor, FEAT_IN, FEAT_OUT, conv_channels=4, is_causal=True)


def _valid(padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Rows of a padded (B, T, D) batch inside the lengths, in packing order."""
    return padded[torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]]


def _gradients(module: torch.nn.Module, inputs: torch.Tensor, loss: torch.Tensor) -> dict[str, torch.Tensor]:
    names, parameters = zip(*module.named_parameters())
    gradients = torch.autograd.grad(loss, (inputs,) + parameters)
    return dict(zip(("input",) + names, gradients))


def _weighted_sum(outputs: torch.Tensor) -> torch.Tensor:
    """A loss with a gradient into every layer; the sum of squares behind the final layer norm is nearly constant."""
    weights = torch.linspace(-1.0, 1.0, outputs.numel(), dtype=outputs.dtype, device=outputs.device)
    return (outputs * torch.sin(37.0 * weights).view_as(outputs)).sum()


def _encoder(subsampling: str, **overrides) -> ConformerEncoder:
    """A small Conformer with causal subsampling and causal convolutions, the layout of the streaming models."""
    options = dict(
        feat_in=FEAT_IN,
        n_layers=2,
        d_model=16,
        n_heads=2,
        subsampling=subsampling,
        subsampling_factor=4,
        subsampling_conv_channels=4,
        causal_downsampling=True,
        conv_kernel_size=5,
        conv_context_size="causal",
        conv_norm_type="layer_norm",
        xscaling=False,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
    )
    options.update(overrides)
    return ConformerEncoder(**options)


def _encoder_padded(encoder: ConformerEncoder, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    outputs, out_lengths = encoder(audio_signal=features, length=lengths)
    return _valid(outputs.transpose(1, 2), out_lengths)


class TestPackedCausalConvSubsampling:
    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling, factor", CAUSAL_STACKS)
    def test_matches_padded_outputs_lengths_and_gradients(self, subsampling, factor):
        torch.manual_seed(1)
        module = _causal(subsampling, factor).double()
        lengths = torch.tensor(LENGTHS)
        features = torch.randn(len(LENGTHS), max(LENGTHS), FEAT_IN, dtype=torch.float64, requires_grad=True)
        padded_out, padded_lengths = module(x=features, lengths=lengths)
        packed = module.forward_packed(pack_encoder_output(features, lengths))

        assert packed.lengths.tolist() == padded_lengths.tolist()
        assert padded_lengths[LENGTHS.index(0)] == 1
        padded_valid = _valid(padded_out, padded_lengths)
        torch.testing.assert_close(packed.data, padded_valid)
        weights = torch.randn_like(padded_valid)
        packed_grads = _gradients(module, features, (packed.data * weights).sum())
        padded_grads = _gradients(module, features, (padded_valid * weights).sum())
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling, factor", CAUSAL_STACKS)
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_utterance_does_not_depend_on_its_neighbours(self, subsampling, factor, value):
        """A non-finite utterance stays alone: the left padding of a causal stack reads the gap, not the neighbour.

        The lengths are multiples of the factor, so the slots hold the gap and no further frame.
        """
        torch.manual_seed(2)
        module = _causal(subsampling, factor).double()
        lengths = torch.tensor([2 * factor, factor, 3 * factor])
        features = torch.randn(3, 3 * factor, FEAT_IN, dtype=torch.float64)
        packed = module.forward_packed(pack_encoder_output(features, lengths))
        features[1] = value
        changed = module.forward_packed(pack_encoder_output(features, lengths))
        middle = slice(int(packed.cu_seqlens[1]), int(packed.cu_seqlens[2]))
        assert not torch.isfinite(changed.data[middle]).any()
        torch.testing.assert_close(changed.data[: middle.start], packed.data[: middle.start], rtol=0, atol=0)
        torch.testing.assert_close(changed.data[middle.stop :], packed.data[middle.stop :], rtol=0, atol=0)

    @pytest.mark.unit
    def test_sequence_groups_match_one_group(self):
        torch.manual_seed(3)
        module = _causal("dw_striding", 8).double()
        lengths = torch.tensor(LENGTHS)
        packed = pack_encoder_output(torch.randn(len(LENGTHS), max(LENGTHS), FEAT_IN, dtype=torch.float64), lengths)
        whole = module.forward_packed(packed)
        module.subsampling_conv_chunking_factor = 4
        grouped = module.forward_packed(packed)
        torch.testing.assert_close(grouped.data, whole.data)
        assert grouped.lengths.tolist() == whole.lengths.tolist()

    @pytest.mark.unit
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_factor_that_is_no_power_of_two_is_refused(self, is_causal):
        """Factor 6 builds two stages, which subsample by 4, while the slots would be aligned to 6."""
        module = ConvSubsampling("striding", 6, FEAT_IN, FEAT_OUT, conv_channels=4, is_causal=is_causal)
        packed = pack_encoder_output(torch.randn(2, 20, FEAT_IN), torch.tensor([20, 7]))
        with pytest.raises(NotImplementedError, match="subsampling_factor=6 is not a power of two"):
            module.forward_packed(packed)

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    @pytest.mark.parametrize("self_attention_model", ["rel_pos", "abs_pos"])
    def test_encoder_matches_padded_outputs_and_gradients(self, subsampling, self_attention_model):
        """Through ``forward_sequence_packed`` in training mode, with utterances of a single frame, without frames,
        and of multiples of the factor, whose slots hold the gap and no further frame."""
        torch.manual_seed(4)
        encoder = _encoder(subsampling, self_attention_model=self_attention_model).double().train()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([48, 1, 4, 0, 8, 31])
        features = torch.randn(6, FEAT_IN, 48, dtype=torch.float64, requires_grad=True)
        packed = encoder.forward_sequence_packed(features, lengths)
        padded = _encoder_padded(encoder, features, lengths)
        torch.testing.assert_close(packed.data, padded)
        packed_grads = _gradients(encoder, features, _weighted_sum(packed.data))
        padded_grads = _gradients(encoder, features, _weighted_sum(padded))
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    def test_chunked_training_with_first_chunk_padding_is_refused(self, subsampling):
        """In training with ``chunked_limited`` attention the padded forward pads every utterance with silence
        before the causal subsampling and drops the frames of that padding, which the packed forward does not do,
        so it names that; in evaluation mode the padded forward does neither and the two forms agree."""
        torch.manual_seed(5)
        encoder = _encoder(subsampling, att_context_style="chunked_limited", att_context_size=[4, 3]).double()
        lengths = torch.tensor([48, 31, 8])
        features = torch.randn(3, FEAT_IN, 48, dtype=torch.float64)
        encoder.train()
        assert "pads the first chunk" in encoder.sequence_packed_unsupported_reason()
        with pytest.raises(NotImplementedError, match="pads the first chunk"):
            encoder.forward_sequence_packed(features, lengths)
        encoder.eval()
        assert encoder.supports_sequence_packed_output
        packed = encoder.forward_sequence_packed(features, lengths)
        torch.testing.assert_close(packed.data, _encoder_padded(encoder, features, lengths))
        # without the limited context the padded forward of the training mode does not pad either
        assert _encoder(subsampling).train().supports_sequence_packed_output

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    def test_encoder_with_a_limited_regular_context_matches_padded(self, subsampling):
        """Training mode with a limited attention context of the ``regular`` style, which the padded forward
        runs without the padding of the first chunk: outputs and gradients against the padded forward."""
        torch.manual_seed(6)
        encoder = _encoder(subsampling, att_context_size=[3, 1]).double().train()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([48, 1, 0, 31])
        features = torch.randn(4, FEAT_IN, 48, dtype=torch.float64, requires_grad=True)
        packed = encoder.forward_sequence_packed(features, lengths)
        padded = _encoder_padded(encoder, features, lengths)
        torch.testing.assert_close(packed.data, padded)
        packed_grads = _gradients(encoder, features, _weighted_sum(packed.data))
        padded_grads = _gradients(encoder, features, _weighted_sum(padded))
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPackedCausalConvSubsamplingCuda:
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
    def _reference(module: ConvSubsampling, features: torch.Tensor, lengths: torch.Tensor):
        """Outputs, lengths and gradients of the padded forward in float64."""
        module = copy.deepcopy(module).double()
        features = features.detach().double().requires_grad_()
        outputs, out_lengths = module(x=features, lengths=lengths)
        outputs = _valid(outputs, out_lengths)
        return outputs, out_lengths, _gradients(module, features, outputs.square().sum())

    @staticmethod
    def _gradient_errors(gradients: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, float]:
        """Error norm of every gradient relative to its reference norm, floored at 1 percent of the largest one."""
        floor = 0.01 * max(gradient.norm() for gradient in reference.values())
        return {
            name: float((gradient.double() - reference[name]).norm() / reference[name].norm().clamp(min=floor))
            for name, gradient in gradients.items()
        }

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling, factor", CUDA_CAUSAL_STACKS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_the_float64_padded_reference(self, subsampling, factor, dtype):
        """Forward and backward, in float32 and under bfloat16 autocast, compared in norm.

        Measured over twelve seeds: outputs within 7e-8 and 0.4 percent, gradients within 6e-7 and 6 percent.
        With TF32 in the matrix products the float32 outputs are within 3e-4 and the gradients within 2e-3.
        With the gap of the non-causal stacks (the factor) the outputs are off by up to 5 percent
        and the gradients by up to 24 percent, depending on the seed.
        """
        torch.manual_seed(5)
        module = _causal(subsampling, factor).cuda()
        lengths = torch.tensor([300, 1, 0, 150, 7, 16, 32], device="cuda")
        features = torch.randn(7, 300, FEAT_IN, device="cuda", requires_grad=True)
        reference, reference_lengths, reference_grads = self._reference(module, features, lengths)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            packed = module.forward_packed(pack_encoder_output(features, lengths))
        assert packed.data.dtype == dtype
        assert packed.lengths.tolist() == reference_lengths.tolist()
        relative = 1e-6 if dtype == torch.float32 else 1e-2
        assert (packed.data.double() - reference).norm() <= relative * reference.norm()
        gradients = _gradients(module, features, packed.data.float().square().sum())
        for name, error in self._gradient_errors(gradients, reference_grads).items():
            assert error <= (1e-5 if dtype == torch.float32 else 0.12), name

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    def test_encoder_bf16_autocast_matches_the_float64_padded_reference(self, subsampling):
        """The training path through ``forward_sequence_packed`` in the precision the trainings use.

        Short utterances, so that a frame read across a boundary would move a large share of the output.
        Measured over twelve seeds: outputs within 0.8 percent,
        and up to 8 percent with a gap of half the factor between the utterances, which is what the output bound catches.
        The gradients of the small convolution stack are noisy in bfloat16 on these few frames, within 54 percent,
        so their bound only excludes gross errors.
        """
        torch.manual_seed(6)
        encoder = _encoder(subsampling, d_model=32).cuda().train()
        lengths = torch.tensor([40, 1, 16, 8, 4, 32, 24], device="cuda")
        features = torch.randn(7, FEAT_IN, 40, device="cuda", requires_grad=True)
        reference_encoder = copy.deepcopy(encoder).double()
        reference_features = features.detach().double().requires_grad_()
        reference = _encoder_padded(reference_encoder, reference_features, lengths)
        reference_grads = _gradients(reference_encoder, reference_features, _weighted_sum(reference))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed = encoder.forward_sequence_packed(features, lengths)
        assert (packed.data.double() - reference).norm() <= 2e-2 * reference.norm()
        gradients = _gradients(encoder, features, _weighted_sum(packed.data.float()))
        for name, error in self._gradient_errors(gradients, reference_grads).items():
            assert error <= 0.9, name
