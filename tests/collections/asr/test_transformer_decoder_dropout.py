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

"""Dropout of the Transformer decoder: where it acts, and whether a mask is per element or shared over positions."""

import functools

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules.transformer.transformer import NeMoTransformerDecoderConfig, TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_modules import (
    GatedPositionWiseFF,
    MultiHeadAttention,
    PositionWiseFF,
    RMSNorm,
    TransformerEmbedding,
)
from nemo.collections.asr.modules.transformer.transformer_utils import get_nemo_transformer
from nemo.collections.asr.parts.submodules.broadcast_dropout import BroadcastDropout

HIDDEN = 16
INNER = 24
RATE = 0.5


def _decoder(**overrides):
    """The composed decoder of the tests, every dropout at ``RATE`` except what ``overrides`` names."""
    options = dict(
        vocab_size=11,
        hidden_size=HIDDEN,
        num_layers=2,
        inner_size=INNER,
        num_attention_heads=2,
        max_sequence_length=32,
        num_token_types=0,
        embedding_dropout=RATE,
        ffn_dropout=RATE,
        attn_score_dropout=RATE,
        attn_layer_dropout=RATE,
        ffn_hidden_dropout=RATE,
        hidden_act="swish",
        pre_ln=True,
        embedding=functools.partial(TransformerEmbedding, scale_embeddings=True, use_layer_norm=False),
        norm=functools.partial(RMSNorm, eps=1e-6),
        self_attn=functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0),
        feed_forward=GatedPositionWiseFF,
    )
    options.update(overrides)
    decoder = TransformerDecoderNM(**options)
    for parameter in decoder.parameters():
        parameter.data.normal_(0, 0.1)
    return decoder


def _inputs():
    input_ids = torch.tensor([[10, 1, 5, 7, 2, 4], [10, 3, 4, 6, 0, 0], [10, 8, 0, 0, 0, 0]])
    decoder_mask = (input_ids != 0).long()
    encoder_states = torch.randn(3, 9, HIDDEN)
    encoder_mask = torch.ones(3, 9, dtype=torch.long)
    return dict(
        input_ids=input_ids, decoder_mask=decoder_mask, encoder_embeddings=encoder_states, encoder_mask=encoder_mask
    )


def _record_dropouts(decoder):
    """Hooks on every dropout of the first block and of the embedding; returns name -> (input, output).

    The tensors are copied: the blocks add the residual into the dropout output in place.
    """
    block = decoder.decoder.layers[0]
    modules = {
        "embedding": decoder.embedding.dropout,
        "self_attn_scores": block.first_sub_layer.attn_dropout,
        "self_attn_output": block.first_sub_layer.layer_dropout,
        "cross_attn_scores": block.second_sub_layer.attn_dropout,
        "cross_attn_output": block.second_sub_layer.layer_dropout,
        "ffn_hidden": block.third_sub_layer.hidden_dropout,
        "ffn_output": block.third_sub_layer.layer_dropout,
    }
    records = {}

    def record(name, _module, args, output):
        records[name] = (args[0].detach().clone(), output.detach().clone())

    for name, module in modules.items():
        module.register_forward_hook(functools.partial(record, name))
    return records


def _kept(record):
    """The keep mask of a recorded dropout call, shape (positions, features).

    Positions with an exact zero in the input are left out, since the mask is not visible there:
    a query whose attention weights were all dropped has a zero attention output.
    """
    dropped_input, output = (tensor.flatten(0, -2) for tensor in record)
    positions = (dropped_input != 0).all(dim=-1)
    assert positions.sum() >= positions.numel() // 2
    dropped_input, output = dropped_input[positions].float(), output[positions].float()
    kept = output != 0
    # what is kept is scaled, nothing else changes; the tolerance covers bfloat16
    torch.testing.assert_close(output[kept], dropped_input[kept] / (1.0 - RATE), rtol=1e-2, atol=0)
    return kept


def _is_shared_over_positions(kept):
    return bool((kept == kept[0]).all())


BRANCH_SITES = ["self_attn_output", "cross_attn_output", "ffn_hidden", "ffn_output"]


class TestBroadcastDropout:
    @pytest.mark.unit
    @pytest.mark.parametrize("shape", [(7,), (5, 7), (3, 4, 7)])
    def test_one_mask_over_the_last_axis(self, shape):
        torch.manual_seed(0)
        dropout = BroadcastDropout(0.5).train()
        x = torch.randn(shape).abs() + 1.0
        out = dropout(x)
        kept = (out != 0).reshape(-1, 7)
        assert _is_shared_over_positions(kept)
        assert 0 < kept[0].sum() < 7
        torch.testing.assert_close(out, x * kept[0] * 2.0)
        # a new mask per call
        masks = {tuple((dropout(x) != 0).reshape(-1, 7)[0].tolist()) for _ in range(8)}
        assert len(masks) > 1

    @pytest.mark.unit
    def test_drop_rate(self):
        torch.manual_seed(0)
        dropout = BroadcastDropout(0.25).train()
        kept = dropout(torch.ones(2, 20000)) != 0
        assert abs(kept.float().mean().item() - 0.75) < 0.02

    @pytest.mark.unit
    def test_identity_in_evaluation_and_without_dropout(self):
        x = torch.randn(3, 4, 7)
        assert BroadcastDropout(0.5).eval()(x) is x
        assert BroadcastDropout(0.0).train()(x) is x

    @pytest.mark.unit
    @pytest.mark.parametrize("p", [1.5, -0.1])
    def test_rate_range_of_torch_dropout(self, p):
        with pytest.raises(ValueError, match="dropout probability"):
            BroadcastDropout(p)
        with pytest.raises(ValueError, match="dropout probability"):
            torch.nn.Dropout(p)

    @pytest.mark.unit
    def test_everything_dropped_at_rate_one(self):
        out = BroadcastDropout(1.0).train()(torch.randn(3, 7))
        assert (out == 0).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_dtype_and_gradient(self, dtype):
        torch.manual_seed(0)
        dropout = BroadcastDropout(0.5).train()
        x = (torch.randn(3, 4, 7, dtype=torch.float64).abs() + 1.0).to(dtype).requires_grad_()
        out = dropout(x)
        assert out.dtype == dtype
        out.sum().backward()
        kept = (out != 0).to(dtype)
        torch.testing.assert_close(x.grad, kept * 2.0)
        torch.testing.assert_close(out, x.detach() * kept * 2.0)


class TestTransformerDecoderDropout:
    @pytest.mark.unit
    @pytest.mark.parametrize("pre_ln", [True, False])
    @pytest.mark.parametrize("feed_forward", [GatedPositionWiseFF, PositionWiseFF])
    def test_broadcast_masks_on_the_branches_only(self, pre_ln, feed_forward):
        torch.manual_seed(0)
        decoder = _decoder(feed_forward=feed_forward, pre_ln=pre_ln, dropout_broadcast=True).train()
        records = _record_dropouts(decoder)
        decoder(**_inputs())
        for site in BRANCH_SITES:
            kept = _kept(records[site])
            assert _is_shared_over_positions(kept), site
            assert 0 < kept[0].sum() < kept.shape[1], site
        assert records["ffn_hidden"][0].shape[-1] == INNER and records["ffn_output"][0].shape[-1] == HIDDEN
        # the embedding and the attention weights keep one draw per element
        assert not _is_shared_over_positions(_kept(records["embedding"]))
        for site in ("self_attn_scores", "cross_attn_scores"):
            _, output = records[site]
            assert not _is_shared_over_positions((output != 0).flatten(0, -2)), site

    @pytest.mark.unit
    def test_masks_per_element_by_default(self):
        torch.manual_seed(0)
        decoder = _decoder().train()
        records = _record_dropouts(decoder)
        decoder(**_inputs())
        for site in BRANCH_SITES + ["embedding"]:
            assert not _is_shared_over_positions(_kept(records[site])), site

    @pytest.mark.unit
    @pytest.mark.parametrize("feed_forward", [GatedPositionWiseFF, PositionWiseFF])
    @pytest.mark.parametrize("dropout_broadcast", [True, False])
    def test_hidden_dropout_sits_between_activation_and_second_projection(self, feed_forward, dropout_broadcast):
        torch.manual_seed(0)
        ff = feed_forward(
            HIDDEN, INNER, hidden_act="swish", ffn_hidden_dropout=RATE, dropout_broadcast=dropout_broadcast
        ).train()
        records = {}
        ff.hidden_dropout.register_forward_hook(lambda _module, args, output: records.update(hidden=(args[0], output)))
        x = torch.randn(3, 5, HIDDEN)
        out = ff(x)
        activated, dropped = records["hidden"]
        first = ff.dense_in(x)
        if feed_forward is GatedPositionWiseFF:
            first, gate = first.chunk(2, dim=-1)
            expected = torch.nn.functional.silu(first) * gate
        else:
            expected = torch.nn.functional.silu(first)
        torch.testing.assert_close(activated, expected)
        assert _is_shared_over_positions(_kept((activated, dropped))) == dropout_broadcast
        torch.testing.assert_close(out, ff.dense_out(dropped))

    @pytest.mark.unit
    @pytest.mark.parametrize("feed_forward", [GatedPositionWiseFF, PositionWiseFF])
    def test_dropped_hidden_features_get_no_gradient(self, feed_forward):
        # forward and backward through the whole decoder: with a shared mask a dropped hidden feature
        # is dropped at every position, so its column of the second projection receives no gradient
        torch.manual_seed(0)
        overrides = dict(embedding_dropout=0.0, ffn_dropout=0.0, attn_score_dropout=0.0, attn_layer_dropout=0.0)
        decoder = _decoder(feed_forward=feed_forward, num_layers=1, dropout_broadcast=True, **overrides).train()
        records = _record_dropouts(decoder)
        out = decoder(**_inputs())
        out.square().sum().backward()
        kept = _kept(records["ffn_hidden"])[0]
        gradient = decoder.decoder.layers[0].third_sub_layer.dense_out.weight.grad
        assert (gradient[:, ~kept] == 0).all()
        assert (gradient[:, kept].abs().sum(dim=0) > 0).all()
        assert all(torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())

    @pytest.mark.unit
    def test_evaluation_ignores_the_options(self):
        torch.manual_seed(0)
        with_options = _decoder(dropout_broadcast=True).eval()
        reference = _decoder().eval()
        no_dropout = dict(embedding_dropout=0.0, ffn_dropout=0.0, attn_score_dropout=0.0, attn_layer_dropout=0.0)
        plain = _decoder(ffn_hidden_dropout=0.0, **no_dropout).eval()
        # the options add no parameter or buffer
        reference.load_state_dict(with_options.state_dict(), strict=True)
        plain.load_state_dict(with_options.state_dict(), strict=True)
        inputs = _inputs()
        with torch.no_grad():
            expected = plain(**inputs)
            torch.testing.assert_close(with_options(**inputs), expected, rtol=0, atol=0)
            torch.testing.assert_close(reference(**inputs), expected, rtol=0, atol=0)

    @pytest.mark.unit
    def test_default_modules(self):
        decoder = TransformerDecoderNM(
            vocab_size=11, hidden_size=HIDDEN, num_layers=1, inner_size=INNER, num_attention_heads=2, ffn_dropout=RATE
        )
        block = decoder.decoder.layers[0]
        for dropout in (
            block.first_sub_layer.layer_dropout,
            block.second_sub_layer.layer_dropout,
            block.third_sub_layer.layer_dropout,
            block.third_sub_layer.hidden_dropout,
        ):
            assert type(dropout) is torch.nn.Dropout
        assert block.third_sub_layer.hidden_dropout.p == 0.0 and block.third_sub_layer.layer_dropout.p == RATE

    @pytest.mark.unit
    @pytest.mark.parametrize("feed_forward", [GatedPositionWiseFF, PositionWiseFF])
    def test_default_feed_forward_training_output_unchanged(self, feed_forward):
        # without the options the feed-forward draws one mask per element, on its output, and nothing else
        torch.manual_seed(0)
        ff = feed_forward(HIDDEN, INNER, ffn_dropout=RATE, hidden_act="swish").train()
        x = torch.randn(3, 5, HIDDEN)
        hidden = ff.dense_in(x)
        if feed_forward is GatedPositionWiseFF:
            hidden, gate = hidden.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(hidden) * gate
        else:
            hidden = torch.nn.functional.silu(hidden)
        torch.manual_seed(1)
        out = ff(x)
        torch.manual_seed(1)
        expected = torch.nn.functional.dropout(ff.dense_out(hidden), RATE)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @pytest.mark.unit
    def test_options_from_config(self):
        config = DictConfig(
            {
                "vocab_size": 11,
                "hidden_size": HIDDEN,
                "num_layers": 1,
                "inner_size": INNER,
                "num_attention_heads": 2,
                "num_token_types": 0,
                "ffn_dropout": 0.1,
                "attn_layer_dropout": 0.2,
                "ffn_hidden_dropout": 0.3,
                "dropout_broadcast": True,
                "feed_forward": {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_modules.GatedPositionWiseFF",
                    "_partial_": True,
                },
            }
        )
        # both options are fields of the structured decoder config
        structured = OmegaConf.structured(NeMoTransformerDecoderConfig)
        assert structured.ffn_hidden_dropout == 0.0 and structured.dropout_broadcast is False
        structured = OmegaConf.merge(structured, {"ffn_hidden_dropout": 0.3, "dropout_broadcast": True})
        assert structured.ffn_hidden_dropout == 0.3 and structured.dropout_broadcast is True
        block = get_nemo_transformer(config_dict=config, encoder=False).decoder.layers[0]
        rates = {
            block.first_sub_layer.layer_dropout: 0.2,
            block.second_sub_layer.layer_dropout: 0.2,
            block.third_sub_layer.hidden_dropout: 0.3,
            block.third_sub_layer.layer_dropout: 0.1,
        }
        for dropout, rate in rates.items():
            assert type(dropout) is BroadcastDropout and dropout.p == rate
        assert type(block.first_sub_layer.attn_dropout) is torch.nn.Dropout

    @pytest.mark.unit
    def test_block_without_the_options(self):
        class OwnFeedForward(torch.nn.Module):
            def __init__(self, hidden_size, inner_size, ffn_dropout, hidden_act):
                super().__init__()
                self.dense = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, hidden_states):
                return self.dense(hidden_states)

        options = dict(vocab_size=11, hidden_size=HIDDEN, num_layers=1, inner_size=INNER, num_attention_heads=2)
        TransformerDecoderNM(feed_forward=OwnFeedForward, **options)
        # a block that lacks an option is not built with the option silently dropped
        with pytest.raises(TypeError, match="OwnFeedForward.*dropout_broadcast"):
            TransformerDecoderNM(feed_forward=OwnFeedForward, dropout_broadcast=True, **options)
        with pytest.raises(TypeError, match="OwnFeedForward.*ffn_hidden_dropout"):
            TransformerDecoderNM(feed_forward=OwnFeedForward, ffn_hidden_dropout=RATE, **options)

    @pytest.mark.unit
    @pytest.mark.parametrize("dropout_broadcast", [True, False])
    def test_attention_output_dropout(self, dropout_broadcast):
        torch.manual_seed(0)
        attention = MultiHeadAttention(
            HIDDEN, 2, attn_score_dropout=RATE, attn_layer_dropout=RATE, dropout_broadcast=dropout_broadcast
        ).train()
        assert type(attention.attn_dropout) is torch.nn.Dropout
        records = {}
        attention.layer_dropout.register_forward_hook(
            lambda _module, args, output: records.update(output=(args[0].detach(), output.detach()))
        )
        x = torch.randn(3, 5, HIDDEN, requires_grad=True)
        out, _ = attention(x, x, x, None)
        out.sum().backward()
        assert _is_shared_over_positions(_kept(records["output"])) == dropout_broadcast
        assert torch.isfinite(x.grad).all()

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("pre_ln", [True, False])
    def test_training_step_under_bfloat16_autocast(self, pre_ln):
        torch.manual_seed(0)
        decoder = _decoder(pre_ln=pre_ln, dropout_broadcast=True).cuda().train()
        records = _record_dropouts(decoder)
        inputs = {name: tensor.cuda() for name, tensor in _inputs().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = decoder(**inputs)
        out.float().square().sum().backward()
        for site in BRANCH_SITES:
            dropped_input, output = records[site]
            assert output.dtype == dropped_input.dtype == torch.bfloat16, site
            assert _is_shared_over_positions(_kept(records[site])), site
        assert all(torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())
