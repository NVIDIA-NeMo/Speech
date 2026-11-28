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

"""Transformer decoder modules: configurable blocks (embedding, norm, self- and cross-attention, feed-forward)."""

import functools
import math

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_modules import (
    FixedPositionalEncoding,
    GatedPositionWiseFF,
    MultiHeadAttention,
    PositionWiseFF,
    RMSNorm,
    TransformerEmbedding,
)
from nemo.collections.asr.modules.transformer.transformer_utils import get_nemo_transformer
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext
from nemo.collections.common.parts.transformer_utils import transformer_weights_init
from nemo.core.optim.muon import Muon

_FUSED_BLOCKS = dict(
    self_attn=functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0, fused_qkv=True),
    cross_attn=functools.partial(MultiHeadAttention, fused_kv=True),
)


def _composed_decoder(**overrides):
    options = dict(
        vocab_size=11,
        hidden_size=16,
        num_layers=2,
        inner_size=24,
        num_attention_heads=2,
        max_sequence_length=32,
        num_token_types=0,
        pre_ln=True,
        embedding=functools.partial(
            TransformerEmbedding,
            scale_embeddings=True,
            positional_encoding_layout="tensor2tensor",
            use_layer_norm=False,
        ),
        norm=functools.partial(RMSNorm, eps=1e-6),
        self_attn=functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0),
        feed_forward=functools.partial(GatedPositionWiseFF, hidden_act="swish"),
    )
    options.update(overrides)
    return TransformerDecoderNM(**options).eval()


class TestTransformerDecoderBlockComposition:
    @pytest.mark.unit
    def test_blocks_from_partials(self):
        decoder = _composed_decoder()
        block = decoder.decoder.layers[0]
        assert isinstance(block.layer_norm_1, RMSNorm) and block.layer_norm_1.eps == 1e-6
        assert isinstance(decoder.decoder.final_layer_norm, RMSNorm)
        assert block.first_sub_layer.query_net.bias is None and block.first_sub_layer.rotary_pos_enc is not None
        # the cross-attention keeps the default block
        assert block.second_sub_layer.key_net.bias is not None and block.second_sub_layer.rotary_pos_enc is None
        assert block.third_sub_layer.dense_in.weight.shape == (48, 16) and block.third_sub_layer.dense_in.bias is None
        assert decoder.embedding.layer_norm is None and decoder.embedding.embedding_scale == math.sqrt(16)
        assert decoder.embedding.token_embedding.num_embeddings == 11

    @pytest.mark.unit
    @pytest.mark.parametrize("rotary", [True, False])
    def test_incremental_decoding_matches_full_pass(self, rotary):
        torch.manual_seed(0)
        overrides = {} if rotary else dict(self_attn=functools.partial(MultiHeadAttention, use_bias=False))
        decoder = _composed_decoder(**overrides)
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.3)
        input_ids = torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0]])
        encoder_states = torch.randn(2, 9, 16)
        encoder_mask = torch.ones(2, 9)
        with torch.no_grad():
            full, _ = decoder.decoder(
                decoder.embedding(input_ids), torch.ones_like(input_ids), encoder_states, encoder_mask
            )
            mems = None
            for step in range(input_ids.shape[1]):
                embedded = decoder.embedding(input_ids[:, step : step + 1], start_pos=step)
                mems, _ = decoder.decoder(
                    embedded, torch.ones(2, 1), encoder_states, encoder_mask, mems, return_mems=True
                )
        torch.testing.assert_close(mems[-1], full, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("pre_ln", [True, False])
    @pytest.mark.parametrize("autocast_dtype", [None, torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_fused_projections_match_separate_ones_with_the_same_weights(self, pre_ln, autocast_dtype, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(0)
        separate = _composed_decoder(pre_ln=pre_ln).to(device)
        for parameter in separate.parameters():
            parameter.data.normal_(0, 0.3)
        fused = _composed_decoder(pre_ln=pre_ln, **_FUSED_BLOCKS).to(device)
        fused.load_state_dict(separate.state_dict())
        input_ids = torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0]], device=device)
        decoder_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=device)
        encoder_states = torch.randn(2, 9, 16, device=device)
        encoder_mask = (torch.arange(9, device=device) < torch.tensor([[9], [6]], device=device)).long()

        def run(decoder):
            # train mode without dropout, as a training step runs it
            decoder.train()
            with torch.autocast(device, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                states = decoder(
                    input_ids=input_ids,
                    decoder_mask=decoder_mask,
                    encoder_embeddings=encoder_states,
                    encoder_mask=encoder_mask,
                )
            states.float().square().sum().backward()
            return states, {name: parameter.grad for name, parameter in decoder.named_parameters()}

        states, grads = run(fused)
        expected_states, separate_grads = run(separate)
        torch.testing.assert_close(states, expected_states)
        # the gradient of a fused matrix is the gradients of its parts, stacked in the order query, key, value
        assert len(grads) < len(separate_grads)
        for name, grad in grads.items():
            if ".qkv_net." in name:
                parts = [name.replace("qkv_net", part) for part in ("query_net", "key_net", "value_net")]
            elif ".kv_net." in name:
                parts = [name.replace("kv_net", part) for part in ("key_net", "value_net")]
            else:
                parts = [name]
            expected = torch.cat([separate_grads[part] for part in parts])
            torch.testing.assert_close(grad, expected, msg=lambda text: f"{name}: {text}")

    @pytest.mark.unit
    @pytest.mark.parametrize("source", [dict(), dict(fused_qkv=True), dict(fused_kv=True)])
    @pytest.mark.parametrize("target", [dict(), dict(fused_qkv=True), dict(fused_kv=True)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_projection_layouts_load_each_other_and_agree(self, source, target, use_bias):
        torch.manual_seed(0)
        reference = MultiHeadAttention(16, 2, use_bias=use_bias, **source)
        attention = MultiHeadAttention(16, 2, use_bias=use_bias, **target)
        result = attention.load_state_dict(reference.state_dict(), strict=False)
        assert not result.missing_keys and not result.unexpected_keys
        # three different tensors: each takes its own rows of a fused matrix
        queries, keys, values = torch.randn(3, 5, 16), torch.randn(3, 7, 16), torch.randn(3, 7, 16)
        mask = torch.zeros(3, 1, 5, 7)
        mask[1, :, :, 4:] = -10000.0
        output, _ = attention(queries, keys, values, mask)
        expected, _ = reference(queries, keys, values, mask)
        torch.testing.assert_close(output, expected)

    @pytest.mark.unit
    def test_fused_projections_are_single_matrices_for_muon(self):
        block = _composed_decoder(**_FUSED_BLOCKS).decoder.layers[0]
        assert {name: tuple(p.shape) for name, p in block.first_sub_layer.named_parameters()} == {
            "qkv_net.weight": (48, 16),
            "out_projection.weight": (16, 16),
        }
        assert {name: tuple(p.shape) for name, p in block.second_sub_layer.named_parameters()} == {
            "query_net.weight": (16, 16),
            "query_net.bias": (16,),
            "kv_net.weight": (32, 16),
            "kv_net.bias": (32,),
            "out_projection.weight": (16, 16),
            "out_projection.bias": (16,),
        }
        for fused in (block.first_sub_layer.qkv_net, block.second_sub_layer.kv_net):
            assert Muon.selects_muon(fused.weight, use_muon=None, adam_dim_threshold=8192)
        # an initializer that depends on the shape sees the fused one
        block.apply(transformer_weights_init)
        assert block.first_sub_layer.qkv_net.weight.abs().max() <= math.sqrt(6 / (48 + 16))
        assert block.first_sub_layer.qkv_net.weight.abs().max() > 0.8 * math.sqrt(6 / (48 + 16))

    @pytest.mark.unit
    @pytest.mark.parametrize("pre_ln", [True, False])
    def test_fused_projections_incremental_decoding_matches_full_pass(self, pre_ln):
        torch.manual_seed(0)
        decoder = _composed_decoder(pre_ln=pre_ln, **_FUSED_BLOCKS)
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.3)
        input_ids = torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0]])
        encoder_states = torch.randn(2, 9, 16)
        encoder_mask = torch.ones(2, 9)
        with torch.no_grad():
            full, _ = decoder.decoder(
                decoder.embedding(input_ids), torch.ones_like(input_ids), encoder_states, encoder_mask
            )
            mems = None
            for step in range(input_ids.shape[1]):
                embedded = decoder.embedding(input_ids[:, step : step + 1], start_pos=step)
                mems, _ = decoder.decoder(
                    embedded, torch.ones(2, 1), encoder_states, encoder_mask, mems, return_mems=True
                )
        torch.testing.assert_close(mems[-1], full, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_fused_projections_checkpoints(self, tmp_path):
        torch.manual_seed(0)
        separate = _composed_decoder()
        # the default layout keeps its names and their order
        assert [name for name in separate.state_dict() if "layers.0.first_sub_layer" in name] == [
            f"_decoder.layers.0.first_sub_layer.{name}.weight"
            for name in ("query_net", "key_net", "value_net", "out_projection")
        ]
        inputs = dict(
            input_ids=torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0]]),
            decoder_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            encoder_embeddings=torch.randn(2, 9, 16),
            encoder_mask=torch.ones(2, 9, dtype=torch.long),
        )
        with torch.no_grad():
            expected = separate(**inputs)

        # a checkpoint with separate projections restores a fused decoder, strictly
        torch.save(separate.state_dict(), tmp_path / "separate.pt")
        fused = _composed_decoder(**_FUSED_BLOCKS)
        fused.load_state_dict(torch.load(tmp_path / "separate.pt"))
        with torch.no_grad():
            torch.testing.assert_close(fused(**inputs), expected)

        # and the fused checkpoint restores both layouts
        torch.save(fused.state_dict(), tmp_path / "fused.pt")
        assert "_decoder.layers.0.first_sub_layer.qkv_net.weight" in torch.load(tmp_path / "fused.pt")
        for options in (dict(), _FUSED_BLOCKS):
            restored = _composed_decoder(**options)
            restored.load_state_dict(torch.load(tmp_path / "fused.pt"))
            with torch.no_grad():
                torch.testing.assert_close(restored(**inputs), expected)

    @pytest.mark.unit
    def test_fused_projections_exclude_each_other(self):
        with pytest.raises(ValueError, match="fused_kv=True cannot be set"):
            MultiHeadAttention(16, 2, fused_qkv=True, fused_kv=True)

    @pytest.mark.unit
    def test_blocks_from_hydra_sections(self):
        config = DictConfig(
            {
                "vocab_size": 11,
                "hidden_size": 16,
                "num_layers": 1,
                "inner_size": 24,
                "num_attention_heads": 2,
                "max_sequence_length": 32,
                "num_token_types": 0,
                "pre_ln": True,
                "norm": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.RMSNorm",
                    "_partial_": True,
                    "eps": 1e-6,
                },
                "self_attn": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.MultiHeadAttention",
                    "_partial_": True,
                    "use_bias": False,
                    "rotary_base": 10000.0,
                },
                "feed_forward": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.GatedPositionWiseFF",
                    "_partial_": True,
                    "hidden_act": "swish",
                },
            }
        )
        decoder = get_nemo_transformer(config_dict=config, encoder=False)
        block = decoder.decoder.layers[0]
        assert isinstance(block.layer_norm_1, RMSNorm) and block.layer_norm_1.eps == 1e-6
        assert block.first_sub_layer.query_net.bias is None and block.first_sub_layer.rotary_pos_enc is not None
        assert type(block.third_sub_layer) is GatedPositionWiseFF
        assert block.third_sub_layer.dense_in.weight.shape == (48, 16) and block.third_sub_layer.dense_in.bias is None
        assert isinstance(block.third_sub_layer.act_fn, torch.nn.SiLU)
        assert isinstance(decoder.embedding.layer_norm, torch.nn.LayerNorm)

    @pytest.mark.unit
    def test_fused_projections_from_hydra_sections(self):
        attention = "nemo.collections.asr.modules.transformer.transformer_modules.MultiHeadAttention"
        config = DictConfig(
            {
                "vocab_size": 11,
                "hidden_size": 16,
                "num_layers": 1,
                "inner_size": 24,
                "num_attention_heads": 2,
                "self_attn": {"_target_": attention, "_partial_": True, "fused_qkv": True},
                "cross_attn": {"_target_": attention, "_partial_": True, "fused_kv": True},
            }
        )
        block = get_nemo_transformer(config_dict=config, encoder=False).decoder.layers[0]
        assert block.first_sub_layer.qkv_net.weight.shape == (48, 16) and block.first_sub_layer.query_net is None
        assert block.second_sub_layer.kv_net.weight.shape == (32, 16) and block.second_sub_layer.key_net is None
        assert block.second_sub_layer.query_net.weight.shape == (16, 16)
        assert block.second_sub_layer.return_xatt_scores

    @pytest.mark.unit
    def test_rms_norm(self):
        torch.manual_seed(0)
        norm = RMSNorm(8, eps=1e-6)
        norm.weight.data.normal_()
        x = torch.randn(3, 8)
        torch.testing.assert_close(norm(x), torch.nn.functional.rms_norm(x, (8,), norm.weight, eps=1e-6))
        assert list(norm.state_dict()) == ["weight"]

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rms_norm_half_precision(self, dtype):
        # the mean square must not overflow: 1000 squared is outside the float16 range
        norm = RMSNorm(8, eps=1e-6).to(dtype)
        large = torch.full((2, 8), 1000.0, dtype=dtype)
        torch.testing.assert_close(norm(large), torch.ones_like(large))
        # forward and backward against the float32 computation; mixed signs, a zero row, a non-unit weight
        torch.manual_seed(0)
        x32 = torch.randn(5, 8) * 300.0
        x32[2] = 0.0
        w32 = torch.randn(8)
        norm.weight.data = w32.to(dtype)
        x = x32.to(dtype).requires_grad_()
        out = norm(x)
        assert out.dtype == dtype
        out.sum().backward()
        reference = RMSNorm(8, eps=1e-6)
        reference.weight.data = w32.clone()
        x_ref = x32.clone().requires_grad_()
        expected = reference(x_ref)
        expected.sum().backward()
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(x.grad.float(), x_ref.grad, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(norm.weight.grad.float(), reference.weight.grad, rtol=2e-2, atol=5e-2)

    @pytest.mark.unit
    def test_export_covers_rotary_positions_beyond_the_example(self, tmp_path):
        # the exported graph holds the rotary table as a constant,
        # so it must cover every position the decoder supports, not only the example's key length
        decoder = _composed_decoder(max_sequence_length=8)
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.3)

        def example(length):
            torch.manual_seed(length)
            input_ids = torch.randint(0, 11, (1, length))
            decoder_mask = torch.ones(1, length, dtype=torch.int64)
            encoder_states = torch.randn(1, 9, 16)
            encoder_mask = torch.ones(1, 9, dtype=torch.int64)
            decoder_mems = torch.randn(1, decoder.num_states, length - 1, 16)
            return input_ids, decoder_mask, encoder_states, encoder_mask, decoder_mems

        path = str(tmp_path / "decoder.ts")
        decoder.export(path, input_example=example(4), check_trace=False)
        exported = torch.jit.load(path)
        for length in (4, 5, 8):
            input_ids, decoder_mask, encoder_states, encoder_mask, decoder_mems = example(length)
            with torch.no_grad():
                # export switched the module to returning the states of every layer, as the graph does
                expected = decoder(
                    input_ids=input_ids,
                    decoder_mask=decoder_mask,
                    encoder_embeddings=encoder_states,
                    encoder_mask=encoder_mask,
                    decoder_mems=decoder_mems,
                )
            torch.testing.assert_close(
                exported(input_ids, decoder_mask, encoder_states, encoder_mask, decoder_mems), expected
            )

    @pytest.mark.unit
    def test_gated_feed_forward(self):
        torch.manual_seed(0)
        ff = GatedPositionWiseFF(16, 24)
        assert ff.dense_in.bias is None and ff.dense_in.weight.shape == (48, 16)
        x = torch.randn(3, 16)
        activated, gate = ff.dense_in(x).chunk(2, dim=-1)
        torch.testing.assert_close(ff(x), ff.dense_out(torch.nn.functional.silu(activated) * gate))
        # both activations are free: GEGLU-style first half with a gated second half, from registry names
        geglu = GatedPositionWiseFF(16, 24, hidden_act="gelu", gate_act="relu", use_bias=True)
        assert isinstance(geglu.act_fn, torch.nn.GELU) and isinstance(geglu.gate_act_fn, torch.nn.ReLU)
        assert geglu.dense_in.bias is not None
        # the plain network takes its activation from the registry as well
        assert isinstance(PositionWiseFF(16, 24, hidden_act="gelu").act_fn, torch.nn.GELU)
        with pytest.raises(ValueError, match="activation"):
            PositionWiseFF(16, 24, hidden_act="tanh")

    @pytest.mark.unit
    def test_tensor2tensor_positional_table(self):
        table = FixedPositionalEncoding(8, max_sequence_length=4, layout="tensor2tensor", scale=1.0).pos_enc
        positions = torch.arange(4.0)[:, None]
        frequencies = torch.exp(-math.log(10000.0) / 3 * torch.arange(4.0))
        torch.testing.assert_close(table[:, :4], torch.sin(positions * frequencies))
        torch.testing.assert_close(table[:, 4:], torch.cos(positions * frequencies))
        default = FixedPositionalEncoding(8, max_sequence_length=4).pos_enc
        torch.testing.assert_close(default[1, 0], torch.sin(torch.tensor(1.0)) / math.sqrt(8))

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        decoder = TransformerDecoderNM(
            vocab_size=8, hidden_size=16, num_layers=1, inner_size=24, num_attention_heads=2
        )
        block = decoder.decoder.layers[0]
        assert type(block.layer_norm_1) is torch.nn.LayerNorm and block.layer_norm_1.eps == 1e-5
        assert block.first_sub_layer.query_net.bias is not None and block.first_sub_layer.rotary_pos_enc is None
        assert type(block.third_sub_layer) is PositionWiseFF
        assert isinstance(block.third_sub_layer.act_fn, torch.nn.ReLU)
        assert block.third_sub_layer.dense_in.weight.shape == (24, 16)
        assert (
            isinstance(decoder.embedding.layer_norm, torch.nn.LayerNorm) and decoder.embedding.embedding_scale is None
        )
        assert decoder.decoder.final_layer_norm is None


def _packed_inputs(input_ids, input_lengths, encoder_states, encoder_lengths):
    """Token-flat ids and encoder states with their packing metadata, in batch order."""
    valid = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < input_lengths[:, None]
    packed_encoder = pack_encoder_output(encoder_states, encoder_lengths)
    return (
        input_ids[valid],
        PackedAttentionContext.from_lengths(input_lengths),
        packed_encoder.data,
        PackedAttentionContext(packed_encoder),
    )


def _valid_rows(padded, lengths):
    return padded[torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]]


class TestPackedTransformerDecoder:
    """Sequence-packed teacher forcing against the padded forward: outputs at valid positions and gradients."""

    @pytest.mark.unit
    @pytest.mark.parametrize("pre_ln", [True, False])
    def test_matches_padded_forward(self, pre_ln):
        torch.manual_seed(0)
        if pre_ln:
            decoder = _composed_decoder().double()
        else:
            options = dict(vocab_size=11, hidden_size=16, num_layers=2, inner_size=24, num_attention_heads=2)
            decoder = TransformerDecoderNM(num_token_types=0, **options).double().eval()
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.3)
        input_ids = torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0], [10, 6, 0, 0, 0]])
        input_lengths = torch.tensor([5, 3, 2])
        encoder_lengths = torch.tensor([9, 4, 7])
        encoder_states = torch.randn(3, 9, 16, dtype=torch.float64, requires_grad=True)
        decoder_mask = (torch.arange(5)[None, :] < input_lengths[:, None]).to(torch.int64)
        encoder_mask = (torch.arange(9)[None, :] < encoder_lengths[:, None]).to(torch.int64)

        padded = decoder(
            input_ids=input_ids,
            decoder_mask=decoder_mask,
            encoder_embeddings=encoder_states,
            encoder_mask=encoder_mask,
        )
        packed_encoder_states = encoder_states.detach().clone().requires_grad_()
        ids, decoder_context, states, encoder_context = _packed_inputs(
            input_ids, input_lengths, packed_encoder_states, encoder_lengths
        )
        packed = decoder.forward_packed(ids, decoder_context, states, encoder_context)

        torch.testing.assert_close(packed, _valid_rows(padded, input_lengths))
        _valid_rows(padded, input_lengths).square().sum().backward()
        reference_grads = {name: p.grad.clone() for name, p in decoder.named_parameters() if p.grad is not None}
        decoder.zero_grad()
        packed.square().sum().backward()
        torch.testing.assert_close(
            _valid_rows(packed_encoder_states.grad, encoder_lengths), _valid_rows(encoder_states.grad, encoder_lengths)
        )
        for name, parameter in decoder.named_parameters():
            if name in reference_grads:
                torch.testing.assert_close(parameter.grad, reference_grads[name], msg=name)

    @pytest.mark.unit
    def test_fused_projections_match_separate_ones(self):
        torch.manual_seed(0)
        separate = _composed_decoder().double()
        for parameter in separate.parameters():
            parameter.data.normal_(0, 0.3)
        fused = _composed_decoder(**_FUSED_BLOCKS).double()
        fused.load_state_dict(separate.state_dict())
        input_ids = torch.tensor([[10, 1, 5, 7, 2], [10, 3, 4, 0, 0], [10, 6, 0, 0, 0]])
        encoder_states = torch.randn(3, 9, 16, dtype=torch.float64)
        inputs = _packed_inputs(input_ids, torch.tensor([5, 3, 2]), encoder_states, torch.tensor([9, 4, 7]))

        packed = fused.forward_packed(*inputs)
        expected = separate.forward_packed(*inputs)
        torch.testing.assert_close(packed, expected)
        packed.square().sum().backward()
        expected.square().sum().backward()
        grads = dict(separate.named_parameters())
        for name, parameter in fused.named_parameters():
            if ".qkv_net." in name:
                parts = [name.replace("qkv_net", part) for part in ("query_net", "key_net", "value_net")]
            elif ".kv_net." in name:
                parts = [name.replace("kv_net", part) for part in ("key_net", "value_net")]
            else:
                parts = [name]
            torch.testing.assert_close(parameter.grad, torch.cat([grads[part].grad for part in parts]), msg=name)

    @pytest.mark.unit
    def test_half_precision_attention_matches_padded_on_cpu(self):
        """Rows of 40 through identity projections: the padded attention pre-scales, the packed one must stay finite too."""
        attention = MultiHeadAttention(64, 1, use_bias=False).half().eval()
        for net in (attention.query_net, attention.key_net, attention.value_net, attention.out_projection):
            net.weight.data.copy_(torch.eye(64, dtype=torch.float16))
        x = torch.full((1, 2, 64), 40.0, dtype=torch.float16)
        padded, _ = attention(x, x, x, None)
        context = PackedAttentionContext.from_lengths(torch.tensor([2]))
        packed, _ = attention.forward_packed(x[0], x[0], x[0], context, context)
        assert torch.isfinite(packed).all()
        torch.testing.assert_close(packed, padded[0])

    @pytest.mark.unit
    def test_rejects_sequences_beyond_the_learned_positions(self):
        decoder = _composed_decoder(max_sequence_length=4, learn_positional_encodings=True)
        ids, decoder_context, states, encoder_context = _packed_inputs(
            torch.tensor([[10, 1, 5, 7, 2]]), torch.tensor([5]), torch.randn(1, 3, 16), torch.tensor([3])
        )
        with pytest.raises(ValueError, match="longer than maximum"):
            decoder.forward_packed(ids, decoder_context, states, encoder_context)

    @pytest.mark.unit
    def test_rejects_other_diagonals_and_memories(self):
        decoder = _composed_decoder()
        ids, decoder_context, states, encoder_context = _packed_inputs(
            torch.tensor([[10, 1]]), torch.tensor([2]), torch.randn(1, 3, 16), torch.tensor([3])
        )
        decoder.decoder.diagonal = 1
        with pytest.raises(NotImplementedError, match="diagonal"):
            decoder.forward_packed(ids, decoder_context, states, encoder_context)
        decoder.decoder.diagonal = 0
        decoder.return_mems = True
        with pytest.raises(NotImplementedError, match="memories"):
            decoder.forward_packed(ids, decoder_context, states, encoder_context)

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize("pre_ln", [True, False])
    def test_bf16_training_step_matches_padded_on_cuda(self, pre_ln):
        """Forward and backward under bfloat16 autocast through the projections, residual adds and norms.

        Both paths run in eval mode, so that dropout does not separate them, with a shared random upstream gradient.
        Gradients are compared by L2 norm: a relative part for the bfloat16 noise of two kernel orders and an absolute
        floor from the largest parameter gradient, since a key bias cancels in the softmax and its gradient is noise
        around zero on both paths.
        """
        torch.manual_seed(1)
        sizes = dict(vocab_size=11, hidden_size=64, num_layers=2, inner_size=96, num_attention_heads=2)
        if pre_ln:
            decoder = _composed_decoder(**sizes).cuda()
        else:
            decoder = TransformerDecoderNM(num_token_types=0, **sizes).cuda().eval()
        for parameter in decoder.parameters():
            parameter.data.normal_(0, 0.1)
        input_ids = torch.randint(1, 11, (3, 30), device="cuda")
        input_lengths = torch.tensor([30, 17, 2], device="cuda")
        encoder_lengths = torch.tensor([200, 45, 130], device="cuda")
        encoder_states = torch.randn(3, 200, 64, device="cuda", requires_grad=True)
        decoder_mask = (torch.arange(30, device="cuda")[None, :] < input_lengths[:, None]).to(torch.int64)
        encoder_mask = (torch.arange(200, device="cuda")[None, :] < encoder_lengths[:, None]).to(torch.int64)
        upstream = torch.randn(int(input_lengths.sum()), 64, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            padded = decoder(
                input_ids=input_ids,
                decoder_mask=decoder_mask,
                encoder_embeddings=encoder_states,
                encoder_mask=encoder_mask,
            )
        padded = _valid_rows(padded, input_lengths)
        (padded.float() * upstream).sum().backward()
        reference_grads = {name: p.grad.clone() for name, p in decoder.named_parameters() if p.grad is not None}
        reference_encoder_grad = _valid_rows(encoder_states.grad, encoder_lengths).clone()
        decoder.zero_grad()
        encoder_states.grad = None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed = decoder.forward_packed(*_packed_inputs(input_ids, input_lengths, encoder_states, encoder_lengths))
        (packed.float() * upstream).sum().backward()

        torch.testing.assert_close(packed.float(), padded.float(), atol=5e-2, rtol=5e-2)
        encoder_grad = _valid_rows(encoder_states.grad, encoder_lengths)
        assert (encoder_grad - reference_encoder_grad).norm() <= 5e-2 * reference_encoder_grad.norm()
        largest = max(grad.norm() for grad in reference_grads.values())
        for name, parameter in decoder.named_parameters():
            if name not in reference_grads:
                continue
            assert parameter.grad is not None, name
            reference = reference_grads[name]
            difference = (parameter.grad.float() - reference.float()).norm()
            assert difference <= 0.15 * reference.norm() + 0.02 * largest, (
                name,
                difference,
                reference.norm(),
                largest,
            )
