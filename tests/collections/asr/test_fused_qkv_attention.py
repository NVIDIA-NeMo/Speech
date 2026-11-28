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
import functools

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules import transformer_encoder_utils
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RoPEMultiHeadAttention,
)
from nemo.core.optim import Muon

ATTENTION_CLASSES = {
    "rel_pos": RelPositionMultiHeadAttention,
    "abs_pos": MultiHeadAttention,
    "rope": RoPEMultiHeadAttention,
}
ATTENTION_MODELS = list(ATTENTION_CLASSES)
PROJECTION_NAMES = ("linear_q", "linear_k", "linear_v")


def _encoder(self_attention_model, fused_qkv, **overrides):
    options = dict(
        feat_in=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        subsampling_conv_channels=4,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
        self_attention_model=self_attention_model,
    )
    if fused_qkv:
        options["self_attn"] = functools.partial(ATTENTION_CLASSES[self_attention_model], fused_qkv=True)
    options.update(overrides)
    return ConformerEncoder(**options)


def _encoder_pair(self_attention_model, dtype=torch.float64, device="cpu", **overrides):
    """A separate and a fused encoder with the same weights, the fused one loaded from the separate state dict."""
    separate = _encoder(self_attention_model, fused_qkv=False, **overrides).to(device=device, dtype=dtype).eval()
    fused = _encoder(self_attention_model, fused_qkv=True, **overrides).to(device=device, dtype=dtype).eval()
    fused.load_state_dict(separate.state_dict(), strict=True)
    return separate, fused


def _features(lengths, dtype=torch.float64, device="cpu"):
    return torch.randn(len(lengths), 8, int(max(lengths)), dtype=dtype, device=device), lengths.to(device)


def _gradient_pairs(separate, fused):
    """Per parameter of the separate module: its name, its gradient and the matching gradient of the fused module.

    A separate projection is paired with its row block of the fused one,
    so each of query, key and value is compared on its own scale.
    """
    fused_grads = {name: parameter.grad for name, parameter in fused.named_parameters()}
    for name, parameter in separate.named_parameters():
        part = next((part for part in PROJECTION_NAMES if f"{part}." in name), None)
        if part is None:
            yield name, parameter.grad, fused_grads[name]
        else:
            block = fused_grads[name.replace(part, "linear_qkv")].chunk(3)[PROJECTION_NAMES.index(part)]
            yield name, parameter.grad, block


def _assert_gradients_match(separate, fused):
    """In float64: each gradient to 1e-9 of its own largest entry, zero gradients to 1e-12 of the largest of all."""
    pairs = list(_gradient_pairs(separate, fused))
    largest = max(reference.abs().max() for _, reference, _ in pairs)
    assert largest > 0
    for name, reference, gradient in pairs:
        assert reference.abs().max() > 0 or name.endswith("linear_k.bias"), name
        tolerance = 1e-9 * reference.abs().max() + 1e-12 * largest
        assert (gradient - reference).abs().max() <= tolerance, name


class TestFusedQKVAttention:
    """`fused_qkv=True` against the three separate projections with the same weights."""

    @pytest.mark.unit
    @pytest.mark.parametrize("use_pytorch_sdpa", [False, True])
    @pytest.mark.parametrize("self_attention_model", ATTENTION_MODELS)
    def test_padded_forward_and_gradients_match_separate(self, self_attention_model, use_pytorch_sdpa):
        torch.manual_seed(0)
        separate, fused = _encoder_pair(self_attention_model, use_pytorch_sdpa=use_pytorch_sdpa)
        features, lengths = _features(torch.tensor([90, 13, 57, 4]))
        reference, reference_lengths = separate(audio_signal=features, length=lengths)
        output, output_lengths = fused(audio_signal=features, length=lengths)
        assert output_lengths.tolist() == reference_lengths.tolist()
        torch.testing.assert_close(output, reference)
        reference.square().sum().backward()
        output.square().sum().backward()
        _assert_gradients_match(separate, fused)

    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ATTENTION_MODELS)
    def test_packed_forward_and_gradients_match_separate(self, self_attention_model):
        torch.manual_seed(1)
        separate, fused = _encoder_pair(self_attention_model)
        features, lengths = _features(torch.tensor([90, 13, 57, 4]))
        reference = separate.forward_sequence_packed(features, lengths)
        output = fused.forward_sequence_packed(features, lengths)
        assert output.lengths.tolist() == reference.lengths.tolist()
        torch.testing.assert_close(output.data, reference.data)
        reference.data.square().sum().backward()
        output.data.square().sum().backward()
        _assert_gradients_match(separate, fused)

    @pytest.mark.unit
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_distinct_query_key_and_value_inputs(self, use_bias):
        """Inputs that are not one tensor, as with a streaming cache, take their rows of the fused projection."""
        torch.manual_seed(2)
        separate = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, use_bias=use_bias).double()
        fused = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, use_bias=use_bias, fused_qkv=True).double()
        fused.load_state_dict(separate.state_dict(), strict=True)
        query, key, value = (torch.randn(3, length, 6, dtype=torch.float64) for length in (5, 7, 7))
        reference = separate(query, key, value, mask=None)
        output = fused(query, key, value, mask=None)
        torch.testing.assert_close(output, reference)
        reference.square().sum().backward()
        output.square().sum().backward()
        _assert_gradients_match(separate, fused)

    @pytest.mark.unit
    def test_streaming_cache_matches_separate(self):
        torch.manual_seed(3)
        separate, fused = _encoder_pair("rel_pos", att_context_size=[8, 0], att_context_style="chunked_limited")
        features, lengths = _features(torch.tensor([64, 40]))
        outputs = []
        for encoder in (separate, fused):
            cache = encoder.get_initial_cache_state(batch_size=2, dtype=torch.float64)
            outputs.append(
                encoder.cache_aware_stream_step(
                    processed_signal=features,
                    processed_signal_length=lengths,
                    cache_last_channel=cache[0],
                    cache_last_time=cache[1],
                    cache_last_channel_len=cache[2],
                )
            )
        for output, reference in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(output, reference)

    @pytest.mark.unit
    def test_default_keeps_the_separate_projections(self):
        attention = RelPositionMultiHeadAttention(
            n_head=2, n_feat=6, dropout_rate=0.0, pos_bias_u=None, pos_bias_v=None
        )
        assert attention.linear_qkv is None
        names = set(attention.state_dict())
        assert {f"{part}.{name}" for part in PROJECTION_NAMES for name in ("weight", "bias")} <= names
        assert not any("linear_qkv" in name for name in names)

    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ATTENTION_MODELS)
    def test_optimizer_sees_one_projection_matrix(self, self_attention_model):
        separate, fused = _encoder_pair(self_attention_model)
        separate_shapes = {name: tuple(parameter.shape) for name, parameter in separate.named_parameters()}
        fused_shapes = {name: tuple(parameter.shape) for name, parameter in fused.named_parameters()}
        for layer in range(2):
            prefix = f"layers.{layer}.self_attn."
            assert fused_shapes[prefix + "linear_qkv.weight"] == (48, 16)
            assert fused_shapes[prefix + "linear_qkv.bias"] == (48,)
            assert all(separate_shapes[prefix + part + ".weight"] == (16, 16) for part in PROJECTION_NAMES)
        # nothing but the projections differs, and no separate projection is left
        assert {name for name in fused_shapes if "linear_qkv" not in name} == {
            name for name in separate_shapes if not any(f".{part}." in name for part in PROJECTION_NAMES)
        }
        # Muon takes the fused projection as one matrix
        optimizer = Muon(fused.parameters(), lr=0.01)
        group = optimizer.param_groups[0]
        muon_shapes = [
            tuple(parameter.shape)
            for parameter in group["params"]
            if Muon.selects_muon(parameter, use_muon=group["use_muon"], adam_dim_threshold=group["adam_dim_threshold"])
        ]
        assert muon_shapes.count((48, 16)) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_state_dict_converts_in_both_directions(self, use_bias):
        torch.manual_seed(4)
        separate = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, use_bias=use_bias)
        fused = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, use_bias=use_bias, fused_qkv=True)
        fused.load_state_dict(separate.state_dict(), strict=True)
        torch.testing.assert_close(
            fused.linear_qkv.weight,
            torch.cat([separate.linear_q.weight, separate.linear_k.weight, separate.linear_v.weight]),
        )
        if use_bias:
            torch.testing.assert_close(
                fused.linear_qkv.bias,
                torch.cat([separate.linear_q.bias, separate.linear_k.bias, separate.linear_v.bias]),
            )
        restored = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, use_bias=use_bias)
        restored.load_state_dict(fused.state_dict(), strict=True)
        for name, parameter in separate.named_parameters():
            torch.testing.assert_close(restored.get_parameter(name), parameter, msg=name)

    @pytest.mark.unit
    @pytest.mark.parametrize("fused_qkv", [True, False])
    def test_incomplete_or_ambiguous_state_dict_is_rejected(self, fused_qkv):
        separate = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0)
        fused = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, fused_qkv=True)
        target = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, fused_qkv=fused_qkv)
        # both forms at once
        with pytest.raises(RuntimeError, match="Unexpected key"):
            target.load_state_dict({**separate.state_dict(), **fused.state_dict()}, strict=True)
        # a separate form without its key projection
        incomplete = {name: value for name, value in separate.state_dict().items() if "linear_k" not in name}
        with pytest.raises(RuntimeError, match="Missing key"):
            target.load_state_dict(incomplete, strict=True)

    @pytest.mark.unit
    def test_separate_projections_of_unequal_shapes_are_not_fused(self):
        fused = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, fused_qkv=True)
        state = MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0).state_dict()
        # 18 rows in total, as the fused weight has
        state["linear_q.weight"] = torch.zeros(8, 6)
        state["linear_k.weight"] = torch.zeros(4, 6)
        with pytest.raises(RuntimeError, match="Missing key"):
            fused.load_state_dict(state, strict=True)

    @pytest.mark.unit
    @pytest.mark.parametrize("value", ["false", 1, None])
    def test_option_must_be_a_bool(self, value):
        with pytest.raises(TypeError, match="fused_qkv must be a bool"):
            MultiHeadAttention(n_head=2, n_feat=6, dropout_rate=0.0, fused_qkv=value)

    @pytest.mark.unit
    def test_hydra_section_round_trip_and_change_attention_model(self):
        torch.manual_seed(5)
        section = {
            "_target_": "nemo.collections.asr.parts.submodules.multi_head_attention.RelPositionMultiHeadAttention",
            "_partial_": True,
            "fused_qkv": True,
        }
        config = {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": 8,
            "n_layers": 2,
            "d_model": 16,
            "n_heads": 2,
            "subsampling_conv_channels": 4,
            "dropout": 0.0,
            "dropout_att": 0.0,
            "dropout_emb": 0.0,
            "dropout_pre_encoder": 0.0,
            "pos_emb_max_len": 64,
            "untie_biases": True,
        }
        separate = ConformerEncoder.from_config_dict(DictConfig(config)).eval()
        features, lengths = _features(torch.tensor([40, 25]), dtype=torch.float32)
        reference, _ = separate(audio_signal=features, length=lengths)

        # a model built with separate projections converts in place and keeps its outputs
        separate.change_attention_model(self_attn=section)
        assert all(
            layer.self_attn.linear_qkv is not None and layer.self_attn.linear_q is None for layer in separate.layers
        )
        converted, _ = separate(audio_signal=features, length=lengths)
        torch.testing.assert_close(converted, reference, atol=1e-5, rtol=1e-5)

        # the saved config rebuilds the fused encoder, and the fused state dict fits it
        saved = separate.to_config_dict()
        assert saved["self_attn"]["fused_qkv"] is True
        rebuilt = ConformerEncoder.from_config_dict(saved).eval()
        rebuilt.load_state_dict(separate.state_dict(), strict=True)
        torch.testing.assert_close(rebuilt(audio_signal=features, length=lengths)[0], converted)

    @pytest.mark.unit
    def test_local_attention_rejects_the_option(self):
        with pytest.raises(TypeError, match="fused_qkv"):
            _encoder(
                "rel_pos_local_attn",
                fused_qkv=False,
                att_context_size=[4, 4],
                self_attn={
                    "_target_": "nemo.collections.asr.parts.submodules.multi_head_attention."
                    "RelPositionMultiHeadAttentionLongformer",
                    "_partial_": True,
                    "fused_qkv": True,
                },
            )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestFusedQKVAttentionCuda:
    """The training path in bfloat16 autocast: forward and backward through the encoder, padded and packed."""

    @pytest.mark.unit
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("self_attention_model", ATTENTION_MODELS)
    def test_bf16_autocast_training_step_matches_separate(self, self_attention_model, packed):
        flash_attention = transformer_encoder_utils._get_flash_attention_varlen_for_device(torch.device("cuda"))
        if packed and self_attention_model != "rel_pos" and flash_attention is None:
            pytest.skip("requires a variable-length FlashAttention provider")
        torch.manual_seed(6)
        separate, fused = _encoder_pair(self_attention_model, dtype=torch.float32, device="cuda", d_model=32)
        separate.train()
        fused.train()
        features, lengths = _features(torch.tensor([300, 7, 150, 1, 260]), dtype=torch.float32, device="cuda")
        outputs = []
        for encoder in (separate, fused):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if packed:
                    output = encoder.forward_sequence_packed(features, lengths).data
                else:
                    output = encoder(audio_signal=features, length=lengths)[0]
            output.float().square().sum().backward()
            outputs.append(output.float())
        torch.testing.assert_close(outputs[1], outputs[0], atol=1e-2, rtol=1e-2)
        for name, reference, gradient in _gradient_pairs(separate, fused):
            # zero in exact arithmetic: the key bias cancels in the softmax, the conv bias in the batch norm
            if name.endswith(("linear_k.bias", "depthwise_conv.bias")):
                continue
            # the backward of one fused product rounds to bfloat16 once, that of three separate ones three times
            assert (gradient - reference).norm() <= 5e-2 * reference.norm(), name
