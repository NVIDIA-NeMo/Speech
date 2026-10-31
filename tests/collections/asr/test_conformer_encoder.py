# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.conformer_modules import (
    ConformerConvolution,
    ConformerFeedForward,
    ConformerLayer,
)
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling


class TestStochasticDepth:
    """Testing stochastic depth functionality."""

    def test_stochastic_depth_model_creation(self):
        """Testing basic model creation and the drop probs are correctly assigned."""
        n_layers = 4
        model = ConformerEncoder(feat_in=10, n_layers=n_layers, d_model=4, feat_out=8)

        # checking that by default SD is disabled
        assert model.layer_drop_probs == [0.0] * n_layers

        # linear mode
        for drop_prob in [0.3, 0.5, 0.9]:
            for start_layer in [1, 3]:
                model = ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_drop_prob=drop_prob,
                    stochastic_depth_start_layer=start_layer,
                )
                L = n_layers - start_layer
                assert model.layer_drop_probs == [0.0] * start_layer + [drop_prob * l / L for l in range(1, L + 1)]

        # uniform mode
        for drop_prob in [0.3, 0.5, 0.9]:
            model = ConformerEncoder(
                feat_in=10,
                n_layers=n_layers,
                d_model=4,
                feat_out=8,
                stochastic_depth_drop_prob=drop_prob,
                stochastic_depth_mode="uniform",
                stochastic_depth_start_layer=start_layer,
            )
            L = n_layers - start_layer
            assert model.layer_drop_probs == [0.0] * start_layer + [drop_prob] * L

        # checking for errors
        for drop_prob in [-1.0, 1.0]:
            with pytest.raises(ValueError, match="stochastic_depth_drop_prob has to be in"):
                ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_drop_prob=drop_prob,
                    stochastic_depth_mode="uniform",
                )

        with pytest.raises(ValueError, match="stochastic_depth_mode has to be one of"):
            ConformerEncoder(feat_in=10, n_layers=n_layers, d_model=4, feat_out=8, stochastic_depth_mode="weird")

        for start_layer in [-1, 0, 5]:
            with pytest.raises(ValueError, match="stochastic_depth_start_layer has to be in"):
                ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_start_layer=start_layer,
                )

    @pytest.mark.pleasefixme
    def test_stochastic_depth_forward(self):
        """Testing that forward works and we get randomness during training, but not during eval."""
        random_input = torch.rand((1, 2, 2))
        random_length = torch.tensor([2], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=2,
            n_layers=3,
            d_model=4,
            feat_out=4,
            stochastic_depth_drop_prob=0.8,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        model.train()
        outputs = [None] * 5
        for i in range(5):
            outputs[i] = model(audio_signal=random_input, length=random_length)[0]
        # checking that not all outputs are the same
        num_diff = 0
        for i in range(1, 5):
            if not torch.allclose(outputs[i], outputs[0]):
                num_diff += 1
        assert num_diff > 0

        model.eval()
        outputs = [None] * 5
        for i in range(5):
            outputs[i] = model(audio_signal=random_input, length=random_length)[0]
        # checking that not all outputs are the same
        num_diff = 0
        for i in range(1, 5):
            if not torch.allclose(outputs[i], outputs[0]):
                num_diff += 1
        assert num_diff == 0


class TestBypassPreEncode:
    """Testing bypass pre-encode functionality."""

    def test_bypass_pre_encode_forward(self):
        """Testing that forward works with "bypass pre-encode" mode."""
        # For pre-encoded embeddings, the shape is (batch_size, n_frames, emb_dim)
        batch_size = 2
        n_frames, emb_dim, feat_out = 17, 16, 8
        random_input = torch.rand((batch_size, n_frames, emb_dim))
        random_length = torch.tensor([n_frames], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=10,
            n_layers=3,
            d_model=emb_dim,
            feat_out=feat_out,
            stochastic_depth_drop_prob=0.0,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        model.train()
        fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

    def test_error_shape_invalid_bypass_pre_encode_forward(self):
        """
        Testing that error messages are correctly triggered regarding "bypass pre-encode" mode.
        Both correct samples and wrongs samples are tested.

        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)
        """
        batch_size = 2
        n_frames, emb_dim, feat_in, feat_out = 17, 16, 10, 8

        pre_encode_input = torch.rand((batch_size, n_frames, emb_dim))
        feat_input = torch.rand((batch_size, feat_in, n_frames))
        input_length = torch.tensor([n_frames], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=feat_in,
            n_layers=3,
            d_model=emb_dim,
            feat_out=feat_out,
            stochastic_depth_drop_prob=0.0,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        sub_sampled_n_frames = np.ceil(n_frames / model.subsampling_factor)

        # Test with bypass_pre_encode = True, should be pre_encode_input but given feat_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        # Test with bypass_pre_encode = True, given the correct input pre_encode_input.
        model.train()
        fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        # Test with bypass_pre_encode = False, should be feat_input but given pre_encode_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        # Test with bypass_pre_encode = False, given the correct input feat_input.
        model.train()
        fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)


def _composed_encoder(**overrides):
    options = dict(
        feat_in=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        subsampling_factor=8,  # overridden by the block's own factor
        xscaling=False,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        pre_encode=functools.partial(
            ConvSubsampling, subsampling="dw_striding", subsampling_factor=4, conv_channels=4
        ),
        feed_forward=functools.partial(ConformerFeedForward, activation="squared_relu", use_bias=False),
        conv=functools.partial(ConformerConvolution, kernel_size=4, norm_eps=1e-3),
        norm=functools.partial(torch.nn.LayerNorm, eps=1e-6),
    )
    options.update(overrides)
    return ConformerEncoder(**options).eval()


class TestConformerBlockComposition:
    """Configurable blocks: subsampling, feed-forward, convolution, attention and norm."""

    @pytest.mark.unit
    def test_blocks_from_partials(self):
        encoder = _composed_encoder()
        assert isinstance(encoder.pre_encode, ConvSubsampling) and encoder.subsampling_factor == 4
        assert encoder.pre_encode._feat_in == 8 and encoder.pre_encode._feat_out == 16
        layer = encoder.layers[0]
        assert layer.feed_forward1.linear1.bias is None and layer.feed_forward2.linear2.bias is None
        assert type(layer.feed_forward1.activation).__name__ == "SquaredReLU"
        assert layer.conv.depthwise_conv.kernel_size == (4,)
        assert (layer.conv.depthwise_conv._left_padding, layer.conv.depthwise_conv._right_padding) == (1, 2)
        assert layer.conv.batch_norm.eps == 1e-3
        assert layer.norm_out.eps == 1e-6 and layer.norm_feed_forward1.eps == 1e-6 and layer.norm_conv.eps == 1e-6
        # the encoder follows the conv block's kernel for its streaming context
        assert encoder.conv_kernel_size == 4 and list(encoder.conv_context_size) == [1, 2]
        # attention keeps the bias: use_bias is the layer's default, the blocks only change their own arguments
        assert layer.self_attn.linear_q.bias is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("kernel_size", [4, 32])
    def test_even_kernel_padding_invariance(self, kernel_size):
        torch.manual_seed(0)
        encoder = _composed_encoder(conv=functools.partial(ConformerConvolution, kernel_size=kernel_size))
        features = torch.randn(2, 8, 40)
        lengths = torch.tensor([40, 25])
        with torch.no_grad():
            states, out_lengths = encoder(audio_signal=features, length=lengths)
            alone, alone_lengths = encoder(audio_signal=features[1:, :, :25], length=lengths[1:])
        assert states.shape == (2, 16, 10) and out_lengths.tolist() == [10, 7] and alone_lengths.tolist() == [7]
        torch.testing.assert_close(states[1, :, :7], alone[0, :, :7], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_blocks_from_hydra_sections(self):
        config = DictConfig(
            {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 8,
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 2,
                "subsampling_conv_channels": 4,
                "feed_forward": {
                    "_target_": "nemo.collections.asr.parts.submodules.conformer_modules.ConformerFeedForward",
                    "_partial_": True,
                    "activation": "gelu",
                },
                "conv": {
                    "_target_": "nemo.collections.asr.parts.submodules.conformer_modules.ConformerConvolution",
                    "_partial_": True,
                    "kernel_size": 8,
                    "norm_type": "layer_norm",
                },
                "norm": {"_target_": "torch.nn.LayerNorm", "_partial_": True, "eps": 1e-6},
            }
        )
        encoder = ConformerEncoder.from_config_dict(config)
        layer = encoder.layers[0]
        assert isinstance(layer.feed_forward1.activation, torch.nn.GELU)
        assert layer.conv.depthwise_conv.kernel_size == (8,) and isinstance(layer.conv.batch_norm, torch.nn.LayerNorm)
        assert (layer.conv.depthwise_conv._left_padding, layer.conv.depthwise_conv._right_padding) == (3, 4)
        assert layer.norm_out.eps == 1e-6 and list(encoder.conv_context_size) == [3, 4]
        # the section's partial survives the round trip through the module's config
        assert encoder.to_config_dict()["conv"]["kernel_size"] == 8

    @pytest.mark.unit
    def test_attention_block(self):
        encoder = _composed_encoder(self_attn=functools.partial(RelPositionMultiHeadAttention, use_bias=False))
        assert encoder.layers[0].self_attn.linear_q.bias is None
        assert encoder.layers[0].self_attn.h == 2

    @pytest.mark.unit
    def test_conv_block_context_reaches_the_encoder(self):
        # a block that sets its own context: the encoder's streaming caches follow it, not the kernel option
        encoder = _composed_encoder(
            conv=functools.partial(ConformerConvolution, kernel_size=4, conv_context_size=[3, 0])
        )
        assert encoder.conv_kernel_size == 4 and list(encoder.conv_context_size) == [3, 0]
        assert list(encoder.layers[0].conv.conv_context_size) == [3, 0]
        _, cache_last_time, _ = encoder.get_initial_cache_state(batch_size=1)
        assert cache_last_time.shape == (2, 1, 16, 3)
        causal = _composed_encoder(
            conv=functools.partial(ConformerConvolution, kernel_size=4, conv_context_size="causal")
        )
        assert list(causal.conv_context_size) == [3, 0]
        with pytest.raises(ValueError, match="conv_context_size"):
            ConformerConvolution(d_model=16, kernel_size=4, conv_context_size=[1, 1])

    @pytest.mark.unit
    def test_callable_conv_block_context_reaches_the_encoder(self):
        # a plain callable binds nothing visibly, so the encoder reads the context from the module it built
        def causal4(**kwargs):
            kwargs.update(kernel_size=4, conv_context_size=[3, 0])
            return ConformerConvolution(**kwargs)

        encoder = _composed_encoder(conv=causal4, att_context_size=[2, 0])
        assert encoder.conv_kernel_size == 4 and list(encoder.conv_context_size) == [3, 0]
        cache_last_channel, cache_last_time, cache_len = encoder.get_initial_cache_state(batch_size=1)
        assert cache_last_time.shape == (2, 1, 16, 3)
        out, out_len, _, cache_last_time, _ = encoder(
            audio_signal=torch.randn(1, 20, 16),
            length=torch.tensor([20]),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_len,
            bypass_pre_encode=True,
        )
        assert out.shape == (1, 16, 20) and out_len.tolist() == [20] and cache_last_time.shape == (2, 1, 16, 3)

    @pytest.mark.unit
    def test_integer_conv_context_size_is_symmetric_padding(self):
        # the form the module accepted before the kernel check: one integer for both sides
        conv = ConformerConvolution(d_model=16, kernel_size=5, conv_context_size=2)
        assert conv.conv_context_size == [2, 2]
        assert (conv.depthwise_conv._left_padding, conv.depthwise_conv._right_padding) == (2, 2)
        assert conv(torch.randn(2, 10, 16)).shape == (2, 10, 16)
        layer = ConformerLayer(d_model=16, d_ff=32, conv_kernel_size=5, conv_context_size=2)
        assert layer.conv.conv_context_size == [2, 2]
        with pytest.raises(ValueError, match="conv_context_size"):
            ConformerConvolution(d_model=16, kernel_size=4, conv_context_size=2)

    @pytest.mark.unit
    def test_change_attention_model_keeps_the_attention_block(self):
        encoder = _composed_encoder(self_attn=functools.partial(RelPositionMultiHeadAttention, use_bias=False))
        encoder._cfg = OmegaConf.create(
            {
                "d_model": 16,
                "n_heads": 2,
                "dropout": 0.0,
                "dropout_att": 0.0,
                "dropout_emb": 0.0,
                "pos_emb_max_len": 5000,
            }
        )
        encoder.change_attention_model(att_context_size=[2, 2])
        attention = encoder.layers[0].self_attn
        assert type(attention) is RelPositionMultiHeadAttention and attention.linear_q.bias is None
        assert attention._max_cache_len == 2
        # the stored block fixes the class, so a new attention model needs a block of its own
        with pytest.raises(ValueError, match="self_attn block"):
            encoder.change_attention_model(self_attention_model="abs_pos")
        # a partial has no config form, so it replaces the block only without a config update
        replacement = functools.partial(MultiHeadAttention, use_bias=False)
        with pytest.raises(ValueError, match="config form"):
            encoder.change_attention_model(self_attention_model="abs_pos", self_attn=replacement)
        encoder.change_attention_model(self_attention_model="abs_pos", self_attn=replacement, update_config=False)
        assert type(encoder.layers[0].self_attn) is MultiHeadAttention
        assert encoder.layers[0].self_attn.linear_q.bias is None
        # the replacement is the active block now: a context-only change rebuilds from it
        encoder.change_attention_model(att_context_size=[2, 2])
        for layer in encoder.layers:
            assert type(layer.self_attn) is MultiHeadAttention and layer.self_attn.linear_q.bias is None
            assert layer.self_attn._max_cache_len == 2

    @pytest.mark.unit
    def test_change_attention_model_persists_a_section_replacement(self):
        attention = "nemo.collections.asr.parts.submodules.multi_head_attention."
        config = DictConfig(
            {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 8,
                "n_layers": 2,
                "d_model": 16,
                "n_heads": 2,
                "subsampling_conv_channels": 4,
                "dropout": 0.0,
                "dropout_att": 0.0,
                "dropout_emb": 0.0,
                "pos_emb_max_len": 64,
                "self_attn": {
                    "_target_": attention + "RelPositionMultiHeadAttention",
                    "_partial_": True,
                    "use_bias": False,
                },
            }
        )
        encoder = ConformerEncoder.from_config_dict(config).eval()
        encoder.change_attention_model(
            self_attention_model="abs_pos",
            self_attn={"_target_": attention + "MultiHeadAttention", "_partial_": True, "use_bias": False},
        )
        encoder.change_attention_model(att_context_size=[2, 2])
        for layer in encoder.layers:
            assert type(layer.self_attn) is MultiHeadAttention and layer.self_attn.linear_q.bias is None
        saved = encoder.to_config_dict()
        assert saved["self_attention_model"] == "abs_pos" and saved["self_attn"]["_target_"].endswith(
            ".MultiHeadAttention"
        )
        # the saved config rebuilds the replaced attention, and the weights fit it
        rebuilt = ConformerEncoder.from_config_dict(saved).eval()
        assert (
            type(rebuilt.layers[0].self_attn) is MultiHeadAttention
            and rebuilt.layers[0].self_attn.linear_q.bias is None
        )
        rebuilt.load_state_dict(encoder.state_dict(), strict=True)
        signal, length = torch.randn(1, 8, 40), torch.tensor([40])
        torch.testing.assert_close(
            rebuilt(audio_signal=signal, length=length)[0], encoder(audio_signal=signal, length=length)[0]
        )

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        encoder = ConformerEncoder(feat_in=8, n_layers=1, d_model=16, n_heads=2, subsampling_conv_channels=4)
        layer = encoder.layers[0]
        assert type(layer.feed_forward1).__name__ == "ConformerFeedForward"
        assert type(layer.feed_forward1.activation).__name__ == "Swish"
        # an activation name resolves through the registry, where 'swish' is nn.SiLU like everywhere else in NeMo
        assert type(ConformerFeedForward(16, 32, 0.0, activation="swish").activation) is torch.nn.SiLU
        assert layer.feed_forward1.linear1.bias is not None
        assert type(layer.norm_out) is torch.nn.LayerNorm and layer.norm_out.eps == 1e-5
        assert layer.conv.batch_norm.eps == 1e-5
        assert (layer.conv.depthwise_conv._left_padding, layer.conv.depthwise_conv._right_padding) == (15, 15)
        assert list(encoder.conv_context_size) == [15, 15]
        assert encoder.pre_encode.out.bias is not None and encoder.subsampling_factor == 4
        # a standalone convolution module with an odd kernel keeps symmetric padding
        conv = ConformerConvolution(d_model=16, kernel_size=5)
        assert (conv.depthwise_conv._left_padding, conv.depthwise_conv._right_padding) == (2, 2)
