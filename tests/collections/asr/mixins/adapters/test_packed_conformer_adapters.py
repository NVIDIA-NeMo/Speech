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
"""Adapters of the Conformer encoder over sequence-packed input, against the padded forward."""

import copy
import functools

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.modules import transformer_encoder_utils
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoderAdapter
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.asr.parts.submodules.adapters import multi_head_attention_adapter_module as mha_adapters
from nemo.collections.asr.parts.submodules.adapters import (
    transformer_multi_head_attention_adapter_module as transformer_mha_adapters,
)
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution, ConformerLayer
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

D_MODEL = 16
# adapter kind, attention model of the encoder, adapter options
ADAPTER_CASES = [
    ("linear", "rel_pos", dict(norm_position="pre")),
    ("linear", "abs_pos", dict(norm_position="post")),
    ("linear", "rope", dict(norm_position="pre")),
    ("mha", "abs_pos", dict()),
    ("mha", "abs_pos", dict(proj_dim=4)),
    ("mha", "rel_pos", dict()),
    ("relmha", "rel_pos", dict()),
    ("relmha", "rel_pos", dict(proj_dim=-1)),
]
# the adapters with the head shape of the encoder's own attention, so the CUDA tests compile no further kernel
CUDA_ADAPTER_CASES = [ADAPTER_CASES[0], ADAPTER_CASES[3], ADAPTER_CASES[6]]


def _adapter_cfg(kind: str, **options):
    if kind == "linear":
        cfg = adapter_modules.LinearAdapterConfig(in_features=D_MODEL, dim=5, **options)
    elif kind == "mha":
        cfg = mha_adapters.MultiHeadAttentionAdapterConfig(n_head=2, n_feat=D_MODEL, **options)
    elif kind == "relmha":
        cfg = mha_adapters.RelPositionMultiHeadAttentionAdapterConfig(n_head=2, n_feat=D_MODEL, **options)
    else:
        raise ValueError(kind)
    return OmegaConf.structured(cfg)


def _encoder(self_attention_model: str = "rel_pos", **overrides) -> ConformerEncoderAdapter:
    """A small Conformer without dropout; layer norm in the convolution, so both modes compare with the padded form."""
    options = dict(
        feat_in=8,
        n_layers=2,
        d_model=D_MODEL,
        n_heads=2,
        subsampling_factor=6,
        self_attention_model=self_attention_model,
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
        conv=functools.partial(ConformerConvolution, kernel_size=4, norm_type="layer_norm"),
    )
    options.update(overrides)
    return ConformerEncoderAdapter(**options)


def _add_adapter(encoder: ConformerEncoderAdapter, name: str, kind: str, **options) -> None:
    """Add an adapter with random parameters: the default initialization makes every adapter the identity."""
    encoder.add_adapter(name, _adapter_cfg(kind, **options))
    for layer in encoder.layers:
        for parameter in layer.adapter_layer[name].parameters():
            torch.nn.init.normal_(parameter, std=0.3)


class _OtherInputStrategy(mha_adapters.MHAResidualAddAdapterStrategy):
    """A strategy of the adapter interface that gives the adapter one attention input other than its value."""

    def __init__(self, other_input: str):
        super().__init__()
        self.other_input = other_input

    def forward(self, input: dict, adapter: torch.nn.Module, *, module):
        other = input[self.other_input].flip(-1)
        return super().forward(dict(input, **{self.other_input: other}), adapter, module=module)


def _valid_frames(padded_bdt: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Rows of a padded (B, D, T) encoder output inside the lengths, in packing order."""
    padded = padded_bdt.transpose(1, 2)
    return padded[torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]]


def _padded_forward(encoder: ConformerEncoderAdapter, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    outputs, out_lengths = encoder(audio_signal=features, length=lengths)
    return _valid_frames(outputs, out_lengths)


def _weighted_sum(outputs: torch.Tensor) -> torch.Tensor:
    """A loss with a gradient into every layer; the sum of squares behind the final layer norm is nearly constant."""
    weights = torch.linspace(-1.0, 1.0, outputs.numel(), dtype=outputs.dtype, device=outputs.device)
    return (outputs * torch.sin(37.0 * weights).view_as(outputs)).sum()


def _gradient_errors(gradients: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, float]:
    """Error norm of every gradient relative to its reference norm, floored at 1 percent of the largest one."""
    floor = 0.01 * max(gradient.norm() for gradient in reference.values())
    return {
        name: float((gradient.double() - reference[name]).norm() / reference[name].norm().clamp(min=floor))
        for name, gradient in gradients.items()
    }


def _gradients(module: torch.nn.Module, inputs: torch.Tensor, loss: torch.Tensor) -> dict[str, torch.Tensor]:
    names, parameters = zip(*module.named_parameters())
    gradients = torch.autograd.grad(loss, (inputs,) + parameters)
    return dict(zip(("input",) + names, gradients))


class TestPackedConformerAdapters:
    @pytest.mark.unit
    @pytest.mark.parametrize("kind, self_attention_model, options", ADAPTER_CASES)
    @pytest.mark.parametrize("training", [True, False])
    def test_matches_padded_outputs_and_gradients(self, kind, self_attention_model, options, training):
        """One utterance of the batch subsamples to a single encoder frame."""
        torch.manual_seed(1)
        encoder = _encoder(self_attention_model).double().train(training)
        lengths = torch.tensor([50, 3, 31])
        features = torch.randn(3, 8, 50, dtype=torch.float64, requires_grad=True)
        without_adapter = encoder.forward_sequence_packed(features, lengths).data.detach()
        _add_adapter(encoder, "adapter_0", kind, **options)
        encoder.double()
        assert encoder.supports_sequence_packed_output

        packed = encoder.forward_sequence_packed(features, lengths).data
        padded = _padded_forward(encoder, features, lengths)
        torch.testing.assert_close(packed, padded)
        # a linear adapter is skipped where the attention adapters run and the other way round, never both
        assert (packed - without_adapter).abs().max() > 1e-3
        packed_grads = _gradients(encoder, features, packed.square().sum())
        padded_grads = _gradients(encoder, features, padded.square().sum())
        assert all(gradient.abs().max() > 0 for name, gradient in packed_grads.items() if "adapter_layer" in name)
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("kind, attention", [("mha", "abs_pos"), ("mha", "rel_pos"), ("relmha", "rel_pos")])
    @pytest.mark.parametrize("style, size", [("regular", [3, 1]), ("chunked_limited", [4, 1])])
    def test_attention_adapter_with_a_limited_attention_context(self, kind, attention, style, size):
        """The adapter attends under the attention context of the layer, as it does under the padded mask."""
        torch.manual_seed(11)
        encoder = _encoder(attention, att_context_style=style, att_context_size=size)
        _add_adapter(encoder, "adapter_0", kind)
        encoder.double().train()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([50, 3, 31])
        features = torch.randn(3, 8, 50, dtype=torch.float64, requires_grad=True)
        packed = encoder.forward_sequence_packed(features, lengths).data
        padded = _padded_forward(encoder, features, lengths)
        torch.testing.assert_close(packed, padded)
        packed_grads = _gradients(encoder, features, _weighted_sum(packed))
        padded_grads = _gradients(encoder, features, _weighted_sum(padded))
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("kind, self_attention_model", [("mha", "abs_pos"), ("relmha", "rel_pos")])
    def test_attention_adapter_with_another_strategy(self, kind, self_attention_model):
        """The packed call is the adapter's own, so every strategy that calls the adapter reaches it."""
        torch.manual_seed(8)
        encoder = _encoder(self_attention_model).double().train()
        strategy = adapter_mixin_strategies.ReturnResultAdapterStrategyConfig()
        _add_adapter(encoder, "adapter_0", kind, adapter_strategy=strategy)
        encoder.double()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([50, 3, 31])
        features = torch.randn(3, 8, 50, dtype=torch.float64, requires_grad=True)
        packed = encoder.forward_sequence_packed(features, lengths).data
        padded = _padded_forward(encoder, features, lengths)
        torch.testing.assert_close(packed, padded)
        packed_grads = _gradients(encoder, features, _weighted_sum(packed))
        padded_grads = _gradients(encoder, features, _weighted_sum(padded))
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("kind, self_attention_model", [("mha", "abs_pos"), ("relmha", "rel_pos")])
    @pytest.mark.parametrize("other_input", ["query", "key"])
    def test_strategy_with_distinct_attention_inputs_is_refused(self, kind, self_attention_model, other_input):
        """A strategy may give the adapter a query or key other than its value, which the padded attention reads.
        The packed attention is self-attention over one tensor, so the packed call refuses them
        where taking the value alone would return another result than the padded forward."""
        torch.manual_seed(8)
        encoder = _encoder(self_attention_model).double().train()
        _add_adapter(encoder, "adapter_0", kind)
        encoder.double()
        lengths = torch.tensor([50, 3, 31])
        features = torch.randn(3, 8, 50, dtype=torch.float64)
        same_inputs = _padded_forward(encoder, features, lengths)
        torch.testing.assert_close(encoder.forward_sequence_packed(features, lengths).data, same_inputs)
        for layer in encoder.layers:
            layer.adapter_layer["adapter_0"].adapter_strategy = _OtherInputStrategy(other_input)
        distinct_inputs = _padded_forward(encoder, features, lengths)
        assert (distinct_inputs - same_inputs).abs().max() > 1e-3
        with pytest.raises(ValueError, match="query, key and value must be the same tensor"):
            encoder.forward_sequence_packed(features, lengths)

    @pytest.mark.unit
    def test_linear_and_attention_adapters_together(self):
        torch.manual_seed(2)
        encoder = _encoder("rel_pos").double().train()
        _add_adapter(encoder, "linear_0", "linear")
        _add_adapter(encoder, "attention_0", "relmha")
        encoder.double()
        lengths = torch.tensor([50, 3, 31])
        features = torch.randn(3, 8, 50, dtype=torch.float64, requires_grad=True)
        packed = encoder.forward_sequence_packed(features, lengths).data
        padded = _padded_forward(encoder, features, lengths)
        torch.testing.assert_close(packed, padded)
        packed_grads = _gradients(encoder, features, packed.square().sum())
        padded_grads = _gradients(encoder, features, padded.square().sum())
        for name, gradient in packed_grads.items():
            torch.testing.assert_close(gradient, padded_grads[name], msg=name)

        # one of the two disabled: the other still runs, as in the padded forward
        encoder.set_enabled_adapters("attention_0", enabled=False)
        only_linear = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(only_linear, _padded_forward(encoder, features, lengths))
        assert (only_linear - packed).abs().max() > 1e-3

    @pytest.mark.unit
    @pytest.mark.parametrize("kind", ["linear", "relmha"])
    def test_disabled_adapter_does_not_run(self, kind):
        torch.manual_seed(3)
        encoder = _encoder("rel_pos").double().train()
        lengths = torch.tensor([50, 3, 31])
        features = torch.randn(3, 8, 50, dtype=torch.float64)
        without_adapter = encoder.forward_sequence_packed(features, lengths).data
        # a disabled adapter with a strategy the packed form refuses is no reason either
        _add_adapter(encoder, "adapter_0", kind)
        for layer in encoder.layers:
            layer.adapter_layer["adapter_0"].adapter_strategy.l2_lambda = 0.1
        encoder.double()
        encoder.set_enabled_adapters(enabled=False)
        assert encoder.is_adapter_available()
        assert encoder.supports_sequence_packed_output
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, without_adapter, rtol=0, atol=0)
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths))

    @pytest.mark.unit
    @pytest.mark.parametrize("kind", ["linear", "relmha"])
    def test_strategy_stochastic_depth(self, kind):
        """With probability one the strategy drops the adapter output in training mode, as in the padded forward."""
        torch.manual_seed(4)
        encoder = _encoder("rel_pos").double().train()
        lengths = torch.tensor([50, 31])
        features = torch.randn(2, 8, 50, dtype=torch.float64)
        without_adapter = encoder.forward_sequence_packed(features, lengths).data
        _add_adapter(encoder, "adapter_0", kind)
        for layer in encoder.layers:
            layer.adapter_layer["adapter_0"].adapter_strategy.stochastic_depth = 1.0
        encoder.double()
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, without_adapter)
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths))
        encoder.eval()
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths))
        assert (packed - without_adapter).abs().max() > 1e-3

    @pytest.mark.unit
    @pytest.mark.parametrize("kind", ["linear", "relmha"])
    def test_auxiliary_l2_loss_is_refused_in_training(self, kind):
        """The loss averages over padded batch rows; evaluation mode does not compute it and runs."""
        torch.manual_seed(5)
        encoder = _encoder("rel_pos").double()
        _add_adapter(encoder, "adapter_0", kind)
        for layer in encoder.layers:
            layer.adapter_layer["adapter_0"].adapter_strategy.l2_lambda = 0.1
        encoder.double()
        lengths = torch.tensor([50, 31])
        features = torch.randn(2, 8, 50, dtype=torch.float64)
        encoder.train()
        assert not encoder.supports_sequence_packed_output
        assert "l2_lambda=0.1" in encoder.sequence_packed_unsupported_reason()
        with pytest.raises(NotImplementedError, match="adapter 'adapter_0'.*l2_lambda"):
            encoder.forward_sequence_packed(features, lengths)
        states = torch.randn(7, D_MODEL, dtype=torch.float64)
        context = PackedAttentionContext(pack_encoder_output(states.unsqueeze(0), torch.tensor([7])))
        with pytest.raises(NotImplementedError, match="l2_lambda"):
            encoder.layers[0].forward_packed(states, attention_context=context, conv_layout=None)
        encoder.eval()
        assert encoder.supports_sequence_packed_output
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths))

    @pytest.mark.unit
    def test_reason_reads_every_enabled_adapter(self):
        """A refused adapter behind one that runs is found; mask and cache have no packed meaning for an adapter."""
        encoder = _encoder("rel_pos").double().train()
        _add_adapter(encoder, "linear_0", "linear")
        _add_adapter(encoder, "attention_0", "relmha")
        assert encoder.supports_sequence_packed_output
        for layer in encoder.layers:
            layer.adapter_layer["attention_0"].adapter_strategy.l2_lambda = 0.1
        assert "'attention_0'" in encoder.sequence_packed_unsupported_reason()

        adapter = encoder.layers[0].adapter_layer["attention_0"].double()
        states = torch.randn(7, D_MODEL, dtype=torch.float64)
        context = PackedAttentionContext(pack_encoder_output(states.unsqueeze(0), torch.tensor([7])))
        with pytest.raises(ValueError, match="mask and cache"):
            adapter(states, states, states, None, None, cache=states.unsqueeze(0), attention_context=context)

    @pytest.mark.unit
    def test_transformer_attention_adapter_is_refused(self):
        layer = ConformerLayer(d_model=D_MODEL, d_ff=32, n_heads=2, conv_kernel_size=3).double()
        cfg = transformer_mha_adapters.TransformerMultiHeadAttentionAdapterConfig(
            hidden_size=D_MODEL, num_attention_heads=2
        )
        layer.add_adapter("adapter_0", OmegaConf.structured(cfg))
        states = torch.randn(7, D_MODEL, dtype=torch.float64)
        context = PackedAttentionContext(pack_encoder_output(states.unsqueeze(0), torch.tensor([7])))
        with pytest.raises(NotImplementedError, match="'adapter_0' .TransformerMultiHeadAttentionAdapter."):
            layer.forward_packed(states, attention_context=context, conv_layout=None)
        layer.set_enabled_adapters(enabled=False)
        assert layer.sequence_packed_adapters_unsupported_reason() is None

    @pytest.mark.unit
    @pytest.mark.parametrize("attention, context", [("rope", [-1, -1]), ("rel_pos_local_attn", [4, 4])])
    def test_attention_adapter_needs_rel_pos_or_abs_pos_attention(self, attention, context):
        """The adapter call of the layer knows ``rel_pos`` and ``abs_pos`` attention only and raises otherwise,
        padded and packed, so the encoder names the adapter instead of running into that; linear adapters run."""
        torch.manual_seed(14)
        encoder = _encoder(attention, att_context_size=context)
        _add_adapter(encoder, "adapter_0", "mha")
        encoder.double().eval()
        lengths = torch.tensor([50, 31])
        features = torch.randn(2, 8, 50, dtype=torch.float64)
        assert "'adapter_0' runs with rel_pos and abs_pos" in encoder.sequence_packed_unsupported_reason()
        with pytest.raises(ValueError, match="Unsupported value of self_attention_model"):
            _padded_forward(encoder, features, lengths)
        encoder.set_enabled_adapters(enabled=False)
        _add_adapter(encoder, "linear_0", "linear")
        encoder.double()
        assert encoder.supports_sequence_packed_output
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths))

    @pytest.mark.unit
    @pytest.mark.parametrize("kind", ["linear", "relmha"])
    @pytest.mark.parametrize("lengths", [[], [0, 0]])
    def test_batch_without_frames_keeps_gradient_edges(self, kind, lengths):
        torch.manual_seed(6)
        encoder = _encoder("rel_pos").double().train()
        _add_adapter(encoder, "adapter_0", kind)
        encoder.double()
        lengths = torch.tensor(lengths, dtype=torch.int64)
        features = torch.zeros(lengths.numel(), 8, 0, dtype=torch.float64, requires_grad=True)
        packed = encoder.forward_sequence_packed(features, lengths)
        assert packed.data.shape == (0, D_MODEL)
        gradients = _gradients(encoder, features, packed.data.sum())
        assert all(gradient.abs().sum() == 0 for gradient in gradients.values())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPackedConformerAdaptersCuda:
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
    def _reference(encoder: ConformerEncoderAdapter, features: torch.Tensor, lengths: torch.Tensor):
        """Outputs and gradients of the padded forward in float64."""
        encoder = copy.deepcopy(encoder).double()
        features = features.detach().double().requires_grad_()
        outputs = _padded_forward(encoder, features, lengths)
        return outputs, _gradients(encoder, features, _weighted_sum(outputs))

    @staticmethod
    def _cuda_encoder(kind: str, self_attention_model: str, options: dict, **overrides) -> ConformerEncoderAdapter:
        torch.manual_seed(7)
        encoder = _encoder(self_attention_model, **overrides)
        _add_adapter(encoder, "adapter_0", kind, **options)
        return encoder.cuda()

    @pytest.mark.unit
    @pytest.mark.parametrize("kind, self_attention_model, options", CUDA_ADAPTER_CASES)
    def test_float32_matches_the_float64_padded_reference(self, kind, self_attention_model, options):
        encoder = self._cuda_encoder(kind, self_attention_model, options).train()
        lengths = torch.tensor([300, 3, 150, 1], device="cuda")
        features = torch.randn(4, 8, 300, device="cuda", requires_grad=True)
        reference, reference_grads = self._reference(encoder, features, lengths)
        packed = encoder.forward_sequence_packed(features, lengths).data
        # measured over eight seeds: outputs within 2.2e-7, gradients within 6.3e-5
        assert (packed.double() - reference).norm() <= 1e-6 * reference.norm()
        gradients = _gradients(encoder, features, _weighted_sum(packed))
        for name, error in _gradient_errors(gradients, reference_grads).items():
            assert error <= 3e-4, name

    @pytest.mark.unit
    @pytest.mark.parametrize("kind, self_attention_model, options", CUDA_ADAPTER_CASES)
    def test_bf16_autocast_matches_the_float64_padded_reference(self, kind, self_attention_model, options):
        """The training path, forward and backward; bfloat16 rounds, so the comparison is in norm.

        Measured over eight seeds: outputs within 0.7 percent;
        the gradients of this small model are noisy in bfloat16, within 47 percent for the worst tensor.
        """
        encoder = self._cuda_encoder(kind, self_attention_model, options).train()
        lengths = torch.tensor([300, 3, 150, 1], device="cuda")
        features = torch.randn(4, 8, 300, device="cuda", requires_grad=True)
        reference, reference_grads = self._reference(encoder, features, lengths)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed = encoder.forward_sequence_packed(features, lengths).data
        assert (packed.double() - reference).norm() <= 2e-2 * reference.norm()
        gradients = _gradients(encoder, features, _weighted_sum(packed.float()))
        for name, error in _gradient_errors(gradients, reference_grads).items():
            assert error <= 0.9, name

    @pytest.mark.unit
    def test_attention_adapter_dropout_follows_the_attention_dropout_rule(self):
        """FlexAttention has no dropout, so with ``packed_rel_pos_attention: flex``
        a ``rel_pos`` adapter with dropout has no packed form in training."""
        options = dict(dropout_rate=0.1)
        encoder = self._cuda_encoder("relmha", "rel_pos", options, packed_rel_pos_attention="flex")
        lengths = torch.tensor([50, 31], device="cuda")
        features = torch.randn(2, 8, 50, device="cuda")
        encoder.train()
        assert "attention adapters" in encoder.sequence_packed_unsupported_reason()
        with pytest.raises(NotImplementedError, match="has no dropout on the attention weights"):
            encoder.forward_sequence_packed(features, lengths)
        encoder.set_enabled_adapters(enabled=False)
        assert encoder.supports_sequence_packed_output
        encoder.set_enabled_adapters(enabled=True)
        encoder.eval()
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths), atol=1e-4, rtol=1e-4)

    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_rel_pos_adapter_keeps_its_dropout_through_the_kernel(self):
        """With the Triton kernel of ``rel_pos`` attention, the default backend,
        a ``rel_pos`` adapter with dropout runs packed in training and draws a fresh mask per call."""
        encoder = self._cuda_encoder("relmha", "rel_pos", dict(dropout_rate=0.1)).train()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([50, 31], device="cuda")
        features = torch.randn(2, 8, 50, device="cuda")
        packed = encoder.forward_sequence_packed(features, lengths).data
        again = encoder.forward_sequence_packed(features, lengths).data
        assert torch.isfinite(packed).all() and not torch.equal(packed, again)
        packed.square().mean().backward()
        adapter = encoder.layers[0].adapter_layer["adapter_0"]
        assert adapter.linear_pos.weight.grad is not None

    @pytest.mark.unit
    def test_dropout_rule_follows_the_mode_of_the_adapter(self):
        """A frozen encoder in evaluation mode with its adapters in training mode, and the reverse."""
        options = dict(dropout_rate=0.1)
        encoder = self._cuda_encoder("relmha", "rel_pos", options, packed_rel_pos_attention="flex")
        adapters = [layer.adapter_layer["adapter_0"] for layer in encoder.layers]
        encoder.eval()
        for adapter in adapters:
            adapter.train()
        assert "attention adapters" in encoder.sequence_packed_unsupported_reason()
        encoder.train()
        for adapter in adapters:
            adapter.eval()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([50, 31], device="cuda")
        features = torch.randn(2, 8, 50, device="cuda")
        packed = encoder.forward_sequence_packed(features, lengths).data
        torch.testing.assert_close(packed, _padded_forward(encoder, features, lengths), atol=1e-4, rtol=1e-4)

    @pytest.mark.unit
    def test_dropout_rule_reads_every_attention_module(self):
        """Under bfloat16 autocast an adapter without a position bias keeps its dropout through FlashAttention,
        in a ``rel_pos`` encoder too, unless its own head width rules that out (``proj_dim`` 4 gives a width of 2)."""
        if transformer_encoder_utils._get_flash_attention_varlen_for_device(torch.device("cuda")) is None:
            pytest.skip("requires a variable-length FlashAttention provider")
        lengths = torch.tensor([50, 31], device="cuda")
        features = torch.randn(2, 8, 50, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoder = self._cuda_encoder("mha", "rel_pos", dict(dropout_rate=0.1)).train()
            assert encoder.supports_sequence_packed_output
            assert torch.isfinite(encoder.forward_sequence_packed(features, lengths).data).all()
            narrow = self._cuda_encoder("mha", "abs_pos", dict(dropout_rate=0.1, proj_dim=4)).train()
            assert "head width" in narrow.sequence_packed_unsupported_reason()
            with pytest.raises(NotImplementedError):
                narrow.forward_sequence_packed(features, lengths)
            # one module that cannot apply its dropout is enough, whatever the others can
            options = dict(dropout_rate=0.1, proj_dim=4)
            mixed_width = self._cuda_encoder("mha", "abs_pos", options, dropout_att=0.1).train()
            assert "head width" in mixed_width.sequence_packed_unsupported_reason()
            options = dict(dropout_rate=0.1)
            mixed_bias = self._cuda_encoder(
                "mha", "rel_pos", options, dropout_att=0.1, packed_rel_pos_attention="flex"
            ).train()
            assert "has no dropout on the attention weights" in mixed_bias.sequence_packed_unsupported_reason()
