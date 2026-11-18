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
