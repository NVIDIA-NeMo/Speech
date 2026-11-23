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
import copy
import functools
import inspect
import os

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules import transformer_encoder_utils
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    pack_encoder_output,
    packed_slot_layout,
    unpack_encoder_output,
)
from nemo.collections.asr.parts.submodules import conformer_modules
from nemo.collections.asr.parts.submodules import packed_attention as packed_attention_module
from nemo.collections.asr.parts.submodules.broadcast_dropout import BroadcastDropout
from nemo.collections.asr.parts.submodules.conformer_modules import (
    ConformerConvolution,
    ConformerFeedForward,
    ConformerLayer,
    FusedBroadcastDropout,
)
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.common.parts.utils import SquaredReLU
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


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


class TestConformerFeedForwardActivationDropout:
    """Activation, dropout and the second linear layer of the feed-forward as one fused path in training on CUDA."""

    class SeparateSquaredReLU(SquaredReLU):
        """The same activation as a subclass, which the fused path does not take: the separate modules run."""

    @staticmethod
    def _saved(module, x):
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            out = module(x)
        return out, saved

    @classmethod
    def _pair(cls, d_ff, dropout, activation="squared_relu"):
        """The same feed-forward twice: with the fusable activation and with its unfused subclass."""
        torch.manual_seed(0)
        fused = ConformerFeedForward(16, d_ff, dropout, activation=activation, use_bias=False)
        separate = ConformerFeedForward(16, d_ff, dropout, activation=cls.SeparateSquaredReLU(), use_bias=False)
        separate.load_state_dict(fused.state_dict())
        return fused, separate

    @pytest.mark.unit
    def test_eval_and_cpu_run_the_modules(self):
        torch.manual_seed(0)
        module = ConformerFeedForward(16, 32, 0.5, activation="squared_relu")
        x = torch.randn(3, 7, 16)
        module.eval()
        with torch.no_grad():
            torch.testing.assert_close(module(x), module.linear2(module.activation(module.linear1(x))))
        module.train()
        torch.manual_seed(1)
        out = module(x)
        torch.manual_seed(1)
        expected = module.linear2(module.dropout(module.activation(module.linear1(x))))
        torch.testing.assert_close(out, expected)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_training_on_cuda_saves_the_hidden_once(self):
        fused, separate = self._pair(4096, 0.1)
        fused.cuda().train()
        separate.cuda().train()
        x = torch.randn(3, 7, 16, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, saved = self._saved(fused, x)
            _, saved_separate = self._saved(separate, x)
        hidden = [t for t in saved if t.numel() == 3 * 7 * 4096]
        assert len(hidden) == 1 and hidden[0].dtype == torch.bfloat16
        assert out.dtype == torch.bfloat16
        # the separate modules keep at least the activation output, the mask and the second linear layer's input
        assert len({t.untyped_storage().data_ptr() for t in saved_separate if t.numel() == 3 * 7 * 4096}) >= 3
        out.float().sum().backward()
        assert x.grad is not None and fused.linear1.weight.grad is not None and fused.linear2.weight.grad is not None
        assert torch.isfinite(fused.linear1.weight.grad).all() and torch.isfinite(fused.linear2.weight.grad).all()

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_the_fused_path_stands_in_for_the_plain_modules_only(self):
        """Dropout zero, a float64 module, a replaced dropout or linear layer run the modules as they are."""
        x = torch.randn(3, 7, 16, device="cuda", requires_grad=True)
        hidden_count = lambda saved: len({t.untyped_storage().data_ptr() for t in saved if t.numel() == 3 * 7 * 64})
        no_dropout = ConformerFeedForward(16, 64, 0.0, activation="squared_relu").cuda().train()
        _, saved = self._saved(no_dropout, x)
        assert no_dropout._fused_activation(no_dropout.linear1(x)) is None and hidden_count(saved) >= 2
        double = ConformerFeedForward(16, 64, 0.1, activation="squared_relu").cuda().double().train()
        torch.manual_seed(2)
        out = double(x.double())
        torch.manual_seed(2)
        expected = double.linear2(double.dropout(double.activation(double.linear1(x.double()))))
        torch.testing.assert_close(out, expected)

        class ScaledLinear(torch.nn.Linear):
            def forward(self, x):
                return 2 * super().forward(x)

        replaced = ConformerFeedForward(16, 64, 0.1, activation="squared_relu").cuda().train()
        replaced.linear2 = ScaledLinear(64, 16).cuda()
        with torch.no_grad():
            replaced.linear2.weight.zero_()
            replaced.linear2.bias.fill_(1.0)
        assert torch.equal(replaced(x), torch.full((3, 7, 16), 2.0, device="cuda"))
        replaced.linear2 = torch.nn.Linear(64, 16).cuda()
        replaced.dropout = torch.nn.Identity()
        assert replaced._fused_activation(replaced.linear1(x)) is None
        assert replaced(x).shape == (3, 7, 16)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    @pytest.mark.parametrize("activation", ["squared_relu", "swish", "relu", "gelu"])
    def test_fused_path_matches_the_modules_in_expectation(self, activation):
        torch.manual_seed(0)
        module = ConformerFeedForward(16, 64, 0.5, activation=activation).cuda()
        x = torch.randn(4, 50, 16, device="cuda")
        module.eval()
        with torch.no_grad():
            expected = module(x)
            module.train()
            assert module._fused_activation(module.linear1(x)) is not None
            fused = torch.stack([module(x) for _ in range(400)])
            separate = torch.stack(
                [module.linear2(module.dropout(module.activation(module.linear1(x)))) for _ in range(400)]
            )
        # dropout is unbiased: the mean over draws is the evaluation output, with the same spread for both paths
        torch.testing.assert_close(fused.mean(0), expected, atol=0.08, rtol=0.0)
        torch.testing.assert_close(fused.std(0).mean(), separate.std(0).mean(), rtol=0.05, atol=0.0)
        assert not torch.equal(fused[0], fused[1])

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_encoder_trains_through_the_fused_path(self):
        torch.manual_seed(0)
        encoder = _composed_encoder(dropout=0.1).cuda().train()
        features = torch.randn(2, 8, 40, device="cuda")
        lengths = torch.tensor([40, 25], device="cuda")
        states, _ = encoder(audio_signal=features, length=lengths)
        states.sum().backward()
        assert all(p.grad is not None for p in encoder.parameters())


class TestDropoutBroadcast:
    """`dropout_broadcast`: one mask over the feature axis per sequence, shared over the sequence's positions."""

    @staticmethod
    def _dropouts(encoder):
        yield encoder.pos_enc.dropout
        for layer in encoder.layers:
            yield layer.dropout
            yield layer.feed_forward1.dropout
            yield layer.feed_forward2.dropout

    @pytest.mark.unit
    def test_default_drops_per_element(self):
        encoder = _composed_encoder(dropout=0.1, dropout_pre_encoder=0.1)
        assert all(type(module) is torch.nn.Dropout and module.p == 0.1 for module in self._dropouts(encoder))
        assert type(ConformerFeedForward(16, 32, 0.1).dropout) is torch.nn.Dropout
        assert type(ConformerLayer(16, 32).dropout) is torch.nn.Dropout

    @pytest.mark.unit
    def test_option_reaches_every_layer_and_leaves_evaluation_unchanged(self):
        torch.manual_seed(0)
        encoder = _composed_encoder(dropout=0.1, dropout_pre_encoder=0.1, dropout_broadcast=True)
        assert all(isinstance(module, BroadcastDropout) and module.p == 0.1 for module in self._dropouts(encoder))
        # the layers take the kernel form, the positional encoding the plain module
        assert all(type(module) is FusedBroadcastDropout for module in list(self._dropouts(encoder))[1:])
        assert type(encoder.pos_enc.dropout) is BroadcastDropout
        assert type(encoder.layers[0].self_attn.dropout) is torch.nn.Dropout
        default = _composed_encoder(dropout=0.1, dropout_pre_encoder=0.1)
        default.load_state_dict(encoder.state_dict())
        features = torch.randn(2, 8, 40)
        lengths = torch.tensor([40, 25])
        with torch.no_grad():
            states, _ = encoder(audio_signal=features, length=lengths)
            expected, _ = default(audio_signal=features, length=lengths)
        assert torch.equal(states, expected)

        # a feed-forward block without the argument keeps working with the default
        class PlainFeedForward(ConformerFeedForward):
            def __init__(self, d_model, d_ff, dropout, use_bias=True):
                super().__init__(d_model, d_ff, dropout, use_bias=use_bias)

        plain = _composed_encoder(dropout=0.1, feed_forward=PlainFeedForward)
        assert type(plain.layers[0].feed_forward1.dropout) is torch.nn.Dropout
        with pytest.raises(TypeError):
            _composed_encoder(dropout=0.1, dropout_broadcast=True, feed_forward=PlainFeedForward)

    @pytest.mark.unit
    def test_option_reaches_the_input_dropout_of_every_attention_model(self):
        """The pre-encoder dropout follows the option in the positional encodings and in the rope path."""
        pre = dict(dropout=0.1, dropout_pre_encoder=0.2, dropout_emb=0.1, dropout_broadcast=True)
        rel_pos = _composed_encoder(**pre)
        assert type(rel_pos.pos_enc.dropout) is BroadcastDropout and rel_pos.pos_enc.dropout.p == 0.2
        # the positional-embedding dropout stays per element, as the attention dropout
        assert type(rel_pos.pos_enc.dropout_emb) is torch.nn.Dropout
        abs_pos = _composed_encoder(self_attention_model="abs_pos", **pre)
        assert type(abs_pos.pos_enc.dropout) is BroadcastDropout and abs_pos.pos_enc.dropout.p == 0.2
        # the local-attention encoding receives the keyword through its own constructor (its rate is `dropout`)
        local = _composed_encoder(self_attention_model="rel_pos_local_attn", att_context_size=[2, 2], **pre)
        assert type(local.pos_enc.dropout) is BroadcastDropout and local.pos_enc.dropout.p == 0.1
        rope = _composed_encoder(self_attention_model="rope", **pre)
        assert type(rope.dropout_pre_encoder) is BroadcastDropout and rope.dropout_pre_encoder.p == 0.2
        assert type(_composed_encoder(self_attention_model="rope", dropout_pre_encoder=0.2).dropout_pre_encoder) is (
            torch.nn.Dropout
        )
        # a positional encoding with the option drops whole features of every sequence of its input
        encoding = RelPositionalEncoding(d_model=16, dropout_rate=0.5, max_len=64, dropout_broadcast=True).train()
        encoding.extend_pe(5, torch.device("cpu"), torch.float32)
        torch.manual_seed(0)
        out, _ = encoding(torch.ones(2, 5, 16))
        kept = out != 0
        assert torch.equal(kept, kept[:, :1].expand_as(kept)) and 0 < kept[0, 0].sum() < 16
        assert not torch.equal(kept[0, 0], kept[1, 0])
        assert type(RelPositionalEncoding(d_model=16, dropout_rate=0.5).dropout) is torch.nn.Dropout
        # a change of the attention model keeps the option from the config
        rel_pos._cfg = OmegaConf.create(
            {
                "d_model": 16,
                "n_heads": 2,
                "dropout": 0.1,
                "dropout_att": 0.0,
                "dropout_emb": 0.1,
                "dropout_pre_encoder": 0.2,
                "dropout_broadcast": True,
                "pos_emb_max_len": 5000,
            }
        )
        rel_pos.change_attention_model(att_context_size=[2, 2])
        assert type(rel_pos.pos_enc.dropout) is BroadcastDropout
        rel_pos.change_attention_model(self_attention_model="rope")
        assert type(rel_pos.dropout_pre_encoder) is BroadcastDropout and rel_pos.dropout_pre_encoder.p == 0.2

    @pytest.mark.unit
    @pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=pytest.mark.run_only_on("GPU"))])
    def test_fused_module_keeps_the_semantics_of_the_plain_module(self, device_type):
        """Padded states get one mask per sequence over time, packed rows the mask of their sequence."""
        torch.manual_seed(0)
        module = FusedBroadcastDropout(0.5).to(device_type).train()
        assert repr(module) == "FusedBroadcastDropout(p=0.5)"
        x = torch.ones(4, 7, 1000, device=device_type, requires_grad=True)
        out = module(x)
        kept = out != 0
        assert torch.equal(kept, kept[:, :1].expand_as(kept))
        assert not torch.equal(kept[0, 0], kept[3, 0])
        assert 0.4 < kept[:, 0].float().mean().item() < 0.6
        torch.testing.assert_close(out[kept], torch.full_like(out[kept], 2.0))
        out.sum().backward()
        assert torch.equal(x.grad != 0, kept) and torch.equal(x.grad[kept], torch.full_like(out[kept], 2.0))
        assert not torch.equal(module(x) != 0, kept)
        # sequence-packed states with their packing
        lengths = torch.tensor([4, 1, 0, 6], device=device_type)
        context = PackedAttentionContext.from_lengths(lengths)
        packed = torch.ones(11, 1000, device=device_type, dtype=torch.bfloat16, requires_grad=True)
        out = module(packed, context)
        assert out.dtype == torch.bfloat16 and set(out.unique().tolist()) == {0.0, 2.0}
        kept = (out != 0).split(lengths.tolist())
        assert all(torch.equal(rows, rows[:1].expand_as(rows)) for rows in kept)
        assert not torch.equal(kept[0][0], kept[3][0])
        out.float().sum().backward()
        assert torch.equal(packed.grad != 0, out != 0)
        # what the plain module refuses, the fused one refuses with the same words
        with pytest.raises(ValueError, match="packing context"):
            module(torch.ones(11, 1000, device=device_type))
        with pytest.raises(ValueError, match="packing context of 11 tokens"):
            module(torch.ones(5, 1000, device=device_type), context)
        module.eval()
        assert module(x) is x and module(packed, context) is packed
        assert FusedBroadcastDropout(0.0).train()(x) is x
        # a rate of one is torch's range, served by the plain path
        assert (FusedBroadcastDropout(1.0).to(device_type).train()(x) == 0).all()

    @pytest.mark.unit
    def test_feed_forward_on_cpu_runs_the_modules_with_the_broadcast_mask(self):
        torch.manual_seed(0)
        module = ConformerFeedForward(16, 32, 0.5, activation="squared_relu", dropout_broadcast=True).train()
        x = torch.randn(3, 7, 16)
        torch.manual_seed(1)
        out = module(x)
        torch.manual_seed(1)
        expected = module.linear2(module.dropout(module.activation(module.linear1(x))))
        torch.testing.assert_close(out, expected)
        context = PackedAttentionContext.from_lengths(torch.tensor([5, 6]))
        packed = torch.randn(11, 16)
        torch.manual_seed(2)
        out = module(packed, context=context)
        torch.manual_seed(2)
        expected = module.linear2(module.dropout(module.activation(module.linear1(packed)), context))
        torch.testing.assert_close(out, expected)
        with pytest.raises(ValueError, match="packing context"):
            module(packed)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_feed_forward_fuses_with_the_broadcast_mask(self, monkeypatch):
        from nemo.collections.asr.parts.triton.activation_dropout import activation_dropout_mask

        torch.manual_seed(0)
        module = ConformerFeedForward(16, 64, 0.5, activation="squared_relu", dropout_broadcast=True).cuda().train()
        x = torch.randn(4, 50, 16, device="cuda")
        assert module._fused_activation(module.linear1(x)) == "squared_relu"
        calls = []
        fused = conformer_modules.activation_dropout_linear

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return fused(*args, **kwargs)

        monkeypatch.setattr(conformer_modules, "activation_dropout_linear", spy)
        out = module(x)
        assert calls == [{"broadcast": True, "sequence_ids": None, "num_sequences": None}] and out.shape == (4, 50, 16)
        context = PackedAttentionContext.from_lengths(torch.tensor([5, 6], device="cuda"))
        packed_out = module(torch.randn(11, 16, device="cuda"), context=context)
        assert packed_out.shape == (11, 16) and calls[-1]["num_sequences"] == 2
        assert calls[-1]["sequence_ids"] is context.sequence_ids
        with pytest.raises(ValueError, match="padded states"):
            module(torch.randn(11, 16, device="cuda"))
        # the training form: packed states under bfloat16 autocast, forward and backward through the fused path
        packed = torch.randn(11, 16, device="cuda", requires_grad=True)
        seed = torch.tensor([21], dtype=torch.int32, device="cuda")
        monkeypatch.setattr(conformer_modules, "activation_dropout_linear", functools.partial(fused, seed=seed))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed_out = module(packed, context=context)
            hidden = module.linear1(packed)
        assert packed_out.dtype == torch.bfloat16
        # the value: the modules under the kernel's mask of the rows' sequences
        keep = activation_dropout_mask((2, 64), 0.5, seed, packed.device)[context.sequence_ids]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = module.linear2(
                torch.where(keep, module.activation(hidden.float()) / 0.5, 0.0).to(torch.bfloat16)
            )
        torch.testing.assert_close(packed_out.float(), expected.float(), rtol=2e-2, atol=2e-2)
        packed_out.float().sum().backward()
        assert packed.grad.dtype == torch.float32 and torch.isfinite(packed.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in module.parameters())
        monkeypatch.setattr(conformer_modules, "activation_dropout_linear", fused)
        # the expectation over draws is the evaluation output, as with the per-element mask
        module.eval()
        with torch.no_grad():
            expected = module(x)
            module.train()
            draws = torch.stack([module(x) for _ in range(1000)])
        torch.testing.assert_close(draws.mean(0), expected, atol=0.1, rtol=0.0)
        assert not torch.equal(draws[0], draws[1])

    @pytest.mark.unit
    @pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=pytest.mark.run_only_on("GPU"))])
    def test_packed_forward_passes_the_packing_to_the_dropouts(self, device_type):
        """The packed encoder forward with the option gives every token the mask of its sequence."""
        torch.manual_seed(0)
        encoder = _packed_encoder(dropout=0.3, dropout_pre_encoder=0.2, dropout_broadcast=True).to(device_type).train()
        lengths = torch.tensor([40, 25, 61], device=device_type)
        features = _packed_test_features(lengths, dtype=torch.float32, device=device_type)
        captured = []
        encoder.layers[0].dropout.register_forward_hook(lambda module, args, out: captured.append((args[0], out)))
        packed = encoder.forward_sequence_packed(features, lengths)
        assert len(captured) == 4 and torch.isfinite(packed.data).all()
        for branch, out in captured:
            # the zero pattern of the rows with nonzero inputs (a zero frame stays zero under any mask)
            kept = [
                (rows != 0)[inputs.ne(0).all(1)]
                for inputs, rows in zip(branch.split(packed.lengths.tolist()), out.split(packed.lengths.tolist()))
            ]
            assert all(torch.equal(rows, rows[:1].expand_as(rows)) for rows in kept)
            assert not torch.equal(kept[0][0], kept[2][0])
        packed.data.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
        # the default stays per token
        plain = _packed_encoder(dropout=0.3, dropout_pre_encoder=0.2).to(device_type).train()
        captured.clear()
        plain.layers[0].dropout.register_forward_hook(lambda module, args, out: captured.append((args[0], out)))
        plain.forward_sequence_packed(features, lengths)
        kept = captured[0][1] != 0
        assert not torch.equal(kept[:1].expand_as(kept), kept)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_encoder_trains_with_the_option(self):
        torch.manual_seed(0)
        encoder = _composed_encoder(dropout=0.1, dropout_broadcast=True).cuda().train()
        features = torch.randn(2, 8, 40, device="cuda")
        lengths = torch.tensor([40, 25], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states, _ = encoder(audio_signal=features, length=lengths)
        states.float().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


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


class TestConformerConvolutionPointwise:
    """The pointwise convolutions run as linear() on a contiguous time-major input."""

    @pytest.mark.unit
    @pytest.mark.parametrize("norm_type", ["batch_norm", "layer_norm"])
    def test_pointwise_inputs_are_contiguous(self, norm_type, monkeypatch):
        linear = torch.nn.functional.linear
        inputs = []

        def recording_linear(x, weight, bias=None):
            inputs.append(x)
            return linear(x, weight, bias)

        monkeypatch.setattr(torch.nn.functional, "linear", recording_linear)
        conv = ConformerConvolution(d_model=16, kernel_size=5, norm_type=norm_type).eval()
        x = torch.randn(2, 7, 16)
        with torch.no_grad():
            out = conv(x)
        assert out.shape == x.shape
        assert len(inputs) == 2 and all(i.is_contiguous() for i in inputs)


class TestDefaultDeviceConstruction:
    """Parameters must follow the ambient default device, e.g. when a serving engine builds the model on GPU."""

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA default device")
    @pytest.mark.parametrize("untie_biases", [False, True])
    def test_conformer_encoder_parameters_follow_default_device(self, untie_biases):
        with torch.device("cuda"):
            encoder = ConformerEncoder(
                feat_in=8, n_layers=2, d_model=8, n_heads=2, self_attention_model="rel_pos", untie_biases=untie_biases
            )
        misplaced = [name for name, p in encoder.named_parameters() if p.device.type != "cuda"]
        assert misplaced == []
        x = torch.randn(1, 8, 32, device="cuda")
        encoder(audio_signal=x, length=torch.tensor([32], device="cuda"))

    @pytest.mark.unit
    def test_rel_position_attention_parameters_follow_default_device(self):
        with torch.device("meta"):
            attn = RelPositionMultiHeadAttention(
                n_head=2, n_feat=8, dropout_rate=0.0, pos_bias_u=None, pos_bias_v=None
            )
        assert attn.pos_bias_u.device.type == "meta"
        assert attn.pos_bias_v.device.type == "meta"


def _packed_encoder(**overrides):
    """A small Conformer with the strided subsampling block, an even conv kernel and no dropout."""
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
            out_bias=False,
        ),
        feed_forward=functools.partial(ConformerFeedForward, activation="squared_relu", use_bias=False),
        conv=functools.partial(ConformerConvolution, kernel_size=4, norm_eps=1e-3),
        norm=functools.partial(torch.nn.LayerNorm, eps=1e-6),
    )
    options.update(overrides)
    return ConformerEncoder(**options)


def _packed_test_features(lengths, feat_in=8, dtype=torch.float64, device="cpu"):
    torch.manual_seed(0)
    return torch.randn(len(lengths), feat_in, int(max(lengths)), dtype=dtype, device=device)


def _valid_frames(padded_bdt, lengths):
    """Rows of a padded (B, D, T) encoder output inside the lengths, in packing order."""
    padded = padded_bdt.transpose(1, 2)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    return padded[mask]


class TestPackedConformerConvolution:
    @pytest.mark.unit
    @pytest.mark.parametrize("kernel_size", [4, 31])
    def test_matches_padded_in_eval(self, kernel_size):
        torch.manual_seed(1)
        module = ConformerConvolution(d_model=6, kernel_size=kernel_size).double().eval()
        lengths = torch.tensor([9, 2, 40])
        padded_in = torch.randn(3, 40, 6, dtype=torch.float64, requires_grad=True)
        pad_mask = torch.arange(40)[None, :] >= lengths[:, None]
        padded_out = module(padded_in, pad_mask=pad_mask)
        packed_in = padded_in.detach().clone().requires_grad_()
        packed = pack_encoder_output(packed_in, lengths)
        packed_out = module.forward_packed(packed.data, packed_slot_layout(packed, gap=kernel_size - 1))
        valid = torch.arange(40)[None, :] < lengths[:, None]
        torch.testing.assert_close(packed_out, padded_out[valid])
        padded_out[valid].square().sum().backward()
        packed_out.square().sum().backward()
        torch.testing.assert_close(packed_in.grad[valid], padded_in.grad[valid])


class TestPackedConformerEncoder:
    """forward_sequence_packed against forward at the valid frames, for the three attention types."""

    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ["rel_pos", "abs_pos", "rope"])
    def test_matches_padded_in_eval(self, self_attention_model):
        torch.manual_seed(3)
        encoder = _packed_encoder(self_attention_model=self_attention_model).double().eval()
        lengths = torch.tensor([50, 7, 31, 1])
        features = _packed_test_features(lengths)
        padded_out, padded_lengths = encoder(audio_signal=features, length=lengths)
        packed = encoder.forward_sequence_packed(features, lengths)
        assert isinstance(packed, PackedEncoderActivations)
        assert packed.lengths.tolist() == padded_lengths.tolist()
        torch.testing.assert_close(packed.data, _valid_frames(padded_out, padded_lengths))
        torch.testing.assert_close(
            unpack_encoder_output(packed, total_length=padded_out.shape[-1]).transpose(1, 2),
            padded_out * (torch.arange(padded_out.shape[-1])[None, None, :] < padded_lengths[:, None, None]),
        )

    @pytest.mark.unit
    def test_parameter_gradients_match_padded(self):
        torch.manual_seed(4)
        padded_encoder = _packed_encoder().double().eval()
        packed_encoder = copy.deepcopy(padded_encoder)
        lengths = torch.tensor([50, 7, 31])
        features = _packed_test_features(lengths)
        padded_out, padded_lengths = padded_encoder(audio_signal=features, length=lengths)
        _valid_frames(padded_out, padded_lengths).square().mean().backward()
        packed_encoder.forward_sequence_packed(features, lengths).data.square().mean().backward()
        padded_grads = dict(padded_encoder.named_parameters())
        for name, parameter in packed_encoder.named_parameters():
            torch.testing.assert_close(parameter.grad, padded_grads[name].grad, msg=name)

    @pytest.mark.unit
    def test_accepts_packed_features_and_pre_encoded_states(self):
        torch.manual_seed(5)
        encoder = _packed_encoder().double().eval()
        lengths = torch.tensor([20, 13])
        features = _packed_test_features(lengths)
        from_padded = encoder.forward_sequence_packed(features, lengths)
        from_packed = encoder.forward_sequence_packed(pack_encoder_output(features.transpose(1, 2), lengths))
        torch.testing.assert_close(from_packed.data, from_padded.data)
        states = torch.randn(2, 5, 16, dtype=torch.float64)
        state_lengths = torch.tensor([5, 2])
        padded_out, _ = encoder(audio_signal=states, length=state_lengths, bypass_pre_encode=True)
        packed_out = encoder.forward_sequence_packed(states, state_lengths, bypass_pre_encode=True)
        torch.testing.assert_close(packed_out.data, _valid_frames(padded_out, state_lengths))
        with pytest.raises(ValueError, match="length must match"):
            encoder.forward_sequence_packed(pack_encoder_output(features.transpose(1, 2), lengths), lengths - 1)

    @pytest.mark.unit
    def test_output_padding_is_zero_whatever_the_input_padding(self):
        """Normalized packed features carry a (B, feat_in) padding tensor; the encoder output pads with zeros."""
        torch.manual_seed(5)
        lengths = torch.tensor([4, 2])
        for encoder, width in ((_packed_encoder(pre_encode=None, subsampling_factor=1), 8), (_packed_encoder(), 8)):
            encoder = encoder.double().eval()
            features = pack_encoder_output(torch.randn(2, 4, width, dtype=torch.float64), lengths)
            features = PackedEncoderActivations(
                features.data,
                features.lengths,
                features.cu_seqlens,
                features.max_seqlen,
                padding_value=torch.full((2, width), 7.0, dtype=torch.float64),
            )
            out = encoder.forward_sequence_packed(features)
            assert out.padding_value == 0.0
            unpacked = unpack_encoder_output(out)
            assert unpacked.shape == (2, int(out.lengths.max()), 16)
            assert torch.equal(unpacked[1, out.lengths[1] :], torch.zeros_like(unpacked[1, out.lengths[1] :]))
        states = torch.randn(2, 5, 16, dtype=torch.float64)
        packed_states = pack_encoder_output(states, torch.tensor([5, 2]))
        packed_states = PackedEncoderActivations(
            packed_states.data,
            packed_states.lengths,
            packed_states.cu_seqlens,
            packed_states.max_seqlen,
            padding_value=torch.full((2, 16), 7.0, dtype=torch.float64),
        )
        out = encoder.forward_sequence_packed(packed_states, bypass_pre_encode=True)
        assert out.padding_value == 0.0 and torch.equal(unpack_encoder_output(out)[1, 2:], torch.zeros(3, 16).double())

    @pytest.mark.unit
    def test_activation_checkpoint_wrappers_run_packed(self):
        """SpeechLM's activation checkpointing wraps the frontend and the layers; the packed path keeps both.

        The wrapped layer's packed forward runs once in the forward and once more in the backward (the recompute),
        and outputs, lengths and gradients equal the unwrapped encoder's.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        torch.manual_seed(5)
        lengths = torch.tensor([20, 13])
        features = _packed_test_features(lengths).requires_grad_()
        plain = _packed_encoder().double().eval()
        wrapped = copy.deepcopy(plain)
        wrapped.pre_encode = checkpoint_wrapper(wrapped.pre_encode)
        for index in range(len(wrapped.layers)):
            wrapped.layers[index] = checkpoint_wrapper(wrapped.layers[index])
        calls = []
        layer = wrapped.layers[0]._checkpoint_wrapped_module
        layer_forward_packed = layer.forward_packed

        def counting_forward_packed(*args, **kwargs):
            calls.append(1)
            return layer_forward_packed(*args, **kwargs)

        layer.forward_packed = counting_forward_packed

        reference = plain.forward_sequence_packed(features, lengths)
        wrapped_features = features.detach().clone().requires_grad_()
        out = wrapped.forward_sequence_packed(wrapped_features, lengths)
        torch.testing.assert_close(out.data, reference.data)
        assert out.lengths.tolist() == reference.lengths.tolist()
        reference.data.square().sum().backward()
        out.data.square().sum().backward()
        torch.testing.assert_close(wrapped_features.grad, features.grad)
        for (name, parameter), (_, wrapped_parameter) in zip(plain.named_parameters(), wrapped.named_parameters()):
            torch.testing.assert_close(wrapped_parameter.grad, parameter.grad, msg=name)
        assert len(calls) == 2

    @pytest.mark.unit
    def test_batch_norm_training_semantics(self):
        """Training-mode BatchNorm differs by the padded frames; it agrees without padding and with LayerNorm."""
        torch.manual_seed(6)
        encoder = _packed_encoder().double().train()
        uneven = torch.tensor([50, 7, 31])
        features = _packed_test_features(uneven)
        padded_out, padded_lengths = encoder(audio_signal=features, length=uneven)
        packed = encoder.forward_sequence_packed(features, uneven)
        difference = (packed.data - _valid_frames(padded_out, padded_lengths)).abs().max()
        assert difference > 1e-6

        even = torch.tensor([50, 50, 50])
        padded_out, padded_lengths = encoder(audio_signal=features, length=even)
        packed = encoder.forward_sequence_packed(features, even)
        torch.testing.assert_close(packed.data, _valid_frames(padded_out, padded_lengths))

        layer_norm_encoder = (
            _packed_encoder(conv=functools.partial(ConformerConvolution, kernel_size=4, norm_type="layer_norm"))
            .double()
            .train()
        )
        padded_out, padded_lengths = layer_norm_encoder(audio_signal=features, length=uneven)
        packed = layer_norm_encoder.forward_sequence_packed(features, uneven)
        torch.testing.assert_close(packed.data, _valid_frames(padded_out, padded_lengths))

    @pytest.mark.unit
    def test_empty_batch_runs_through_the_encoder(self):
        """A rank whose share of a batch is empty gets an empty output and a zero gradient for every parameter.

        The empty paths of the frontend and the attention alone did not carry the batch through the layers:
        the depthwise convolution cannot run over the empty slot layout and now skips it.
        The running statistics of the batch normalization stay untouched in training mode
        (the batch counter advances as in every training step).
        """
        default_encoder = ConformerEncoder(
            feat_in=8, n_layers=1, d_model=16, n_heads=2, subsampling_factor=4, subsampling_conv_channels=4
        )
        for encoder in (_packed_encoder().double().train(), default_encoder.double().eval()):
            features = torch.empty(0, 8, 0, dtype=torch.float64, requires_grad=True)
            out = encoder.forward_sequence_packed(features, torch.empty(0, dtype=torch.long))
            assert out.data.shape == (0, 16)
            assert out.lengths.tolist() == [] and out.cu_seqlens.tolist() == [0] and out.max_seqlen == 0
            out.data.sum().backward()
            assert features.grad is not None and features.grad.shape == features.shape
            for name, parameter in encoder.named_parameters():
                assert parameter.grad is not None and not parameter.grad.any(), name
            for name, buffer in encoder.named_buffers():
                if name.endswith("running_mean"):
                    assert not buffer.any(), name
                elif name.endswith("running_var"):
                    assert torch.equal(buffer, torch.ones_like(buffer)), name

    @pytest.mark.unit
    def test_single_valid_frame_in_training_normalizes_to_the_bias(self):
        """``BatchNorm1d`` refuses one value per channel in training; the packed convolution module defines it.

        Two input frames at the sixfold subsampling leave one valid frame (and an empty utterance beside it):
        the frame is normalized to the bias, the running mean moves towards the frame,
        the running variance keeps its value, and the step completes forward and backward with finite gradients.
        The padded forward of the same batch normalizes over the valid and the padded frame together.
        """
        torch.manual_seed(7)
        encoder = _packed_encoder().double().train()
        lengths = torch.tensor([2, 0])
        features = _packed_test_features(lengths).requires_grad_()
        out = encoder.forward_sequence_packed(features, lengths)
        assert out.lengths.tolist() == [1, 0] and torch.isfinite(out.data).all()
        out.data.square().sum().backward()
        assert features.grad is not None and torch.isfinite(features.grad).all()
        for name, parameter in encoder.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        for name, module in encoder.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert int(module.num_batches_tracked) == 1, name
                assert torch.equal(module.running_var, torch.ones_like(module.running_var)), name
                assert module.running_mean.abs().sum() > 0, name
        padded_out, padded_lengths = encoder(audio_signal=features.detach(), length=lengths)
        assert padded_lengths.tolist() == [1, 0] and torch.isfinite(padded_out).all()

        conv = ConformerConvolution(d_model=6, kernel_size=4).double().train()
        torch.nn.init.normal_(conv.batch_norm.bias)
        frame = torch.randn(1, 6, dtype=torch.float64, requires_grad=True)
        normalized = conv._batch_norm_one_frame(frame)
        torch.testing.assert_close(normalized.detach(), conv.batch_norm.bias.detach()[None])
        torch.testing.assert_close(conv.batch_norm.running_mean, 0.1 * frame.detach()[0])
        assert torch.equal(conv.batch_norm.running_var, torch.ones(6, dtype=torch.float64))
        normalized.sum().backward()
        assert frame.grad is not None and not frame.grad.any() and conv.batch_norm.weight.grad is not None

    @pytest.mark.unit
    def test_interctc_capture_is_rejected(self):
        """The InterCTC consumer expects padded (B, D, T) tensors, so the packed forward refuses the capture."""
        encoder = _packed_encoder().double().eval()
        lengths = torch.tensor([20, 13])
        features = _packed_test_features(lengths)
        guid = getattr(encoder, "model_guid", None)
        encoder.update_access_cfg({"interctc": {"capture_layers": [0]}, "detach": True}, guid=guid)
        encoder.set_access_enabled(access_enabled=True, guid=guid)
        try:
            with pytest.raises(NotImplementedError, match="InterCTC"):
                encoder.forward_sequence_packed(features, lengths)
            # the padded forward still captures
            encoder(audio_signal=features, length=lengths)
            registry = {}
            for module_registry in encoder.get_module_registry(encoder).values():
                registry.update(module_registry)
            assert registry["interctc/layer_output_0"][0].shape[1] == 16  # (B, D, T)
        finally:
            encoder.reset_registry()
            encoder.set_access_enabled(access_enabled=False, guid=guid)
            encoder.interctc_capture_at_layers = None

    @pytest.mark.unit
    def test_rejects_unsupported_configurations(self):
        features = _packed_test_features(torch.tensor([20]))
        with pytest.raises(NotImplementedError, match="full attention context"):
            _packed_encoder(att_context_size=[4, 4]).double().forward_sequence_packed(features, torch.tensor([20]))
        with pytest.raises(NotImplementedError, match="rel_pos_local_attn"):
            local = _packed_encoder(self_attention_model="rel_pos_local_attn", att_context_size=[4, 4]).double()
            local.forward_sequence_packed(features, torch.tensor([20]))

    @pytest.mark.unit
    def test_attention_dropout_runs_on_cpu(self):
        """Off CUDA the per-utterance reference attention applies the dropout, with the position bias too."""
        torch.manual_seed(8)
        encoder = _packed_encoder(dropout_att=0.1).double().train()
        assert encoder.supports_sequence_packed_output
        lengths = torch.tensor([20, 13])
        packed = encoder.forward_sequence_packed(_packed_test_features(lengths), lengths)
        assert packed.data.shape[1] == 16 and torch.isfinite(packed.data).all()

    @pytest.mark.unit
    def test_state_dict_unchanged(self):
        """The packed forward adds no parameter or buffer: the keys of a small encoder as recorded on `main`."""
        encoder = ConformerEncoder(
            feat_in=8, n_layers=1, d_model=16, n_heads=2, subsampling_conv_channels=4, subsampling_factor=4
        )
        recorded = {
            "pre_encode": ["conv.0.bias", "conv.0.weight", "conv.2.bias", "conv.2.weight", "out.bias", "out.weight"],
            "layers.0.feed_forward1": ["linear1.bias", "linear1.weight", "linear2.bias", "linear2.weight"],
            "layers.0.feed_forward2": ["linear1.bias", "linear1.weight", "linear2.bias", "linear2.weight"],
            "layers.0.self_attn": [
                "linear_k.bias",
                "linear_k.weight",
                "linear_out.bias",
                "linear_out.weight",
                "linear_pos.weight",
                "linear_q.bias",
                "linear_q.weight",
                "linear_v.bias",
                "linear_v.weight",
                "pos_bias_u",
                "pos_bias_v",
            ],
            "layers.0.conv": [
                "batch_norm.bias",
                "batch_norm.num_batches_tracked",
                "batch_norm.running_mean",
                "batch_norm.running_var",
                "batch_norm.weight",
                "depthwise_conv.bias",
                "depthwise_conv.weight",
                "pointwise_conv1.bias",
                "pointwise_conv1.weight",
                "pointwise_conv2.bias",
                "pointwise_conv2.weight",
            ],
        }
        norms = ["norm_conv", "norm_feed_forward1", "norm_feed_forward2", "norm_out", "norm_self_att"]
        keys = [f"{prefix}.{name}" for prefix, names in recorded.items() for name in names]
        keys += [f"layers.0.{norm}.{parameter}" for norm in norms for parameter in ("bias", "weight")]
        assert sorted(encoder.state_dict()) == sorted(keys)
        assert "layers.0.self_attn.linear_pos.weight" in _packed_encoder().state_dict()

    @pytest.mark.unit
    def test_block_arguments_keep_their_positions(self):
        """``packed_rel_pos_attention`` is keyword-only, after the block arguments: a positional ``None`` after
        ``use_triton`` still binds to ``pre_encode``, as it did before the option existed."""
        parameters = list(inspect.signature(ConformerEncoder.__init__).parameters.values())[1:]
        names = [parameter.name for parameter in parameters]
        assert names.index("pre_encode") == names.index("use_triton") + 1
        option = inspect.signature(ConformerEncoder.__init__).parameters["packed_rel_pos_attention"]
        assert option.kind is inspect.Parameter.KEYWORD_ONLY and option.default == "triton"
        sizes = {"feat_in": 8, "n_layers": 1, "d_model": 16}
        positional = [
            sizes.get(parameter.name, parameter.default) for parameter in parameters[: names.index("pre_encode")]
        ]
        encoder = ConformerEncoder(*positional, None)
        assert isinstance(encoder.pre_encode, ConvSubsampling) and encoder.packed_rel_pos_attention == "triton"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPackedConformerEncoderCuda:
    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ["rel_pos", "abs_pos", "rope"])
    def test_matches_padded_outputs_and_gradients(self, self_attention_model):
        torch.manual_seed(8)
        padded_encoder = _packed_encoder(self_attention_model=self_attention_model, d_model=32, n_heads=2)
        padded_encoder = padded_encoder.cuda().eval()
        packed_encoder = copy.deepcopy(padded_encoder)
        lengths = torch.tensor([300, 7, 150, 1, 260], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        padded_out, padded_lengths = padded_encoder(audio_signal=features, length=lengths)
        packed = packed_encoder.forward_sequence_packed(features, lengths)
        padded_valid = _valid_frames(padded_out, padded_lengths)
        torch.testing.assert_close(packed.data, padded_valid, atol=1e-4, rtol=1e-4)
        padded_valid.square().mean().backward()
        packed.data.square().mean().backward()
        padded_grads = dict(padded_encoder.named_parameters())
        for name, parameter in packed_encoder.named_parameters():
            torch.testing.assert_close(parameter.grad, padded_grads[name].grad, atol=1e-4, rtol=1e-3, msg=name)

    @staticmethod
    def _half_precision_flash_attention_available():
        return transformer_encoder_utils._get_flash_attention_varlen_for_device(torch.device("cuda")) is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ["rel_pos", "abs_pos", "rope"])
    def test_bf16_autocast_matches_padded_outputs_and_gradients(self, self_attention_model):
        """rel_pos runs the Triton kernel with the position term; abs_pos and rope run variable-length FlashAttention."""
        if self_attention_model != "rel_pos" and not self._half_precision_flash_attention_available():
            pytest.skip("requires a variable-length FlashAttention provider")
        torch.manual_seed(9)
        padded_encoder = _packed_encoder(self_attention_model=self_attention_model, d_model=32, n_heads=2)
        padded_encoder = padded_encoder.cuda().eval()
        packed_encoder = copy.deepcopy(padded_encoder)
        lengths = torch.tensor([300, 7, 150], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            padded_out, padded_lengths = padded_encoder(audio_signal=features, length=lengths)
            packed = packed_encoder.forward_sequence_packed(features, lengths)
        padded_valid = _valid_frames(padded_out, padded_lengths).float()
        torch.testing.assert_close(packed.data.float(), padded_valid, atol=5e-2, rtol=5e-2)
        padded_valid.square().mean().backward()
        packed.data.float().square().mean().backward()
        padded_grads = dict(padded_encoder.named_parameters())
        for name, parameter in packed_encoder.named_parameters():
            # bf16 rounds differently in the padded and the packed kernels, so the gradients are compared in norm
            reference = padded_grads[name].grad
            assert (parameter.grad - reference).norm() <= 0.1 * reference.norm() + 1e-5, name

    @pytest.mark.unit
    @pytest.mark.parametrize("self_attention_model", ["abs_pos", "rope"])
    def test_attention_dropout_runs_through_flash_attention_in_bf16(self, self_attention_model):
        if not self._half_precision_flash_attention_available():
            pytest.skip("requires a variable-length FlashAttention provider")
        torch.manual_seed(10)
        encoder = _packed_encoder(self_attention_model=self_attention_model, dropout_att=0.1, d_model=32, n_heads=2)
        encoder = encoder.cuda().train()
        lengths = torch.tensor([300, 7, 150], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed = encoder.forward_sequence_packed(features, lengths)
        assert packed.data.shape == (int(packed.lengths.sum()), 32) and torch.isfinite(packed.data).all()
        packed.data.float().square().mean().backward()
        assert encoder.layers[0].self_attn.linear_q.weight.grad is not None

    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_rel_pos_attention_dropout_runs_through_the_kernel(self, monkeypatch):
        """Attention dropout with rel_pos trains packed, in float32 and under bf16 autocast, without FlexAttention."""

        def no_flex_attention(*args, **kwargs):
            raise AssertionError("FlexAttention must not run")

        monkeypatch.setattr(packed_attention_module, "_flex_packed_attention", no_flex_attention)
        torch.manual_seed(11)
        encoder = _packed_encoder(self_attention_model="rel_pos", dropout_att=0.1, d_model=32, n_heads=2)
        encoder = encoder.cuda().train()
        lengths = torch.tensor([300, 7, 150], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        for autocast in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                packed = encoder.forward_sequence_packed(features, lengths)
                again = encoder.forward_sequence_packed(features, lengths)
            assert packed.data.shape == (int(packed.lengths.sum()), 32) and torch.isfinite(packed.data).all()
            # the dropout draws a fresh mask per call
            assert not torch.equal(packed.data, again.data)
            encoder.zero_grad()
            packed.data.float().square().mean().backward()
            assert encoder.layers[0].self_attn.linear_pos.weight.grad is not None
            assert encoder.layers[0].self_attn.pos_bias_v.grad.abs().sum() > 0

    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="requires Triton")
    def test_flex_attention_stays_selectable_for_rel_pos(self, monkeypatch):
        """``packed_rel_pos_attention: flex`` keeps the backend before the kernel: FlexAttention over the rounded
        table, which has no dropout, so attention dropout in training mode is refused as before."""
        with pytest.raises(ValueError, match="packed_rel_pos_attention"):
            _packed_encoder(packed_rel_pos_attention="cuda")
        torch.manual_seed(8)
        encoder = _packed_encoder(self_attention_model="rel_pos", dropout_att=0.1, packed_rel_pos_attention="flex")
        encoder = encoder.cuda().train()
        reason = encoder.sequence_packed_unsupported_reason()
        assert reason is not None and "FlexAttention" in reason
        encoder.eval()
        assert encoder.sequence_packed_unsupported_reason() is None

        def no_kernel(*args, **kwargs):
            raise AssertionError("the Triton kernel must not run")

        monkeypatch.setattr(packed_attention_module, "packed_rel_pos_attention", no_kernel)
        lengths = torch.tensor([300, 7, 150], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        padded_out, padded_lengths = encoder(audio_signal=features, length=lengths)
        packed = encoder.forward_sequence_packed(features, lengths)
        torch.testing.assert_close(packed.data, _valid_frames(padded_out, padded_lengths), atol=1e-4, rtol=1e-4)

    @pytest.mark.unit
    @pytest.mark.parametrize("backend", ["flex", "triton"])
    def test_float64_attention_dropout_runs_the_reference_on_either_backend(self, monkeypatch, backend):
        """In float64 ``packed_attention`` runs the dense reference attention, with dropout, whichever backend the
        encoder selects, and the capability check at the encoder entry point follows that dispatch rule."""

        def no_flex_attention(*args, **kwargs):
            raise AssertionError("FlexAttention must not run")

        def no_kernel(*args, **kwargs):
            raise AssertionError("the Triton kernel must not run")

        monkeypatch.setattr(packed_attention_module, "_flex_packed_attention", no_flex_attention)
        monkeypatch.setattr(packed_attention_module, "packed_rel_pos_attention", no_kernel, raising=False)
        torch.manual_seed(5)
        encoder = _packed_encoder(
            self_attention_model="rel_pos", dropout_att=0.1, d_model=32, n_heads=2, packed_rel_pos_attention=backend
        )
        encoder = encoder.cuda().double().train()
        assert encoder.sequence_packed_unsupported_reason() is None
        lengths = torch.tensor([40, 7, 25], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float64, device="cuda")
        packed = encoder.forward_sequence_packed(features, lengths)
        again = encoder.forward_sequence_packed(features, lengths)
        assert packed.data.dtype == torch.float64 and torch.isfinite(packed.data).all()
        # the dropout draws a fresh mask per call
        assert not torch.equal(packed.data, again.data)
        packed.data.square().mean().backward()
        assert encoder.layers[0].self_attn.linear_pos.weight.grad is not None

    @pytest.mark.parametrize("self_attention_model", ["abs_pos", "rope"])
    def test_flash_and_flex_backends_match_padded_in_bf16(self, self_attention_model):
        """The same weights and inputs through both CUDA attention backends, each against the padded forward.

        Outputs, lengths, the input gradient at the valid frames and the parameter gradients are compared;
        the two packed backends are closer to each other than either is to the padded forward,
        whose kernels round bf16 differently.
        """
        if not self._half_precision_flash_attention_available():
            pytest.skip("requires a variable-length FlashAttention provider")
        torch.manual_seed(11)
        reference_encoder = _packed_encoder(self_attention_model=self_attention_model, d_model=32, n_heads=2)
        reference_encoder = reference_encoder.cuda().eval()
        lengths = torch.tensor([300, 7, 150, 1, 129], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        valid_input = torch.arange(features.shape[-1], device="cuda")[None, None, :] < lengths[:, None, None]

        def run(backend):
            encoder = copy.deepcopy(reference_encoder)
            x = features.detach().clone().requires_grad_()
            with pytest.MonkeyPatch.context() as patch:
                if backend == "flex":
                    patch.setattr(packed_attention_module, "_flash_attention_varlen", lambda queries: None)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if backend == "padded":
                        padded_out, out_lengths = encoder(audio_signal=x, length=lengths)
                        out = _valid_frames(padded_out, out_lengths)
                    else:
                        packed = encoder.forward_sequence_packed(x, lengths)
                        out, out_lengths = packed.data, packed.lengths
            out.float().square().mean().backward()
            grads = {name: parameter.grad for name, parameter in encoder.named_parameters()}
            return out.float(), out_lengths.tolist(), (x.grad * valid_input).float(), grads

        padded = run("padded")
        for backend in ("flash", "flex"):
            out, out_lengths, input_grad, grads = run(backend)
            assert out_lengths == padded[1]
            torch.testing.assert_close(out, padded[0], atol=5e-2, rtol=5e-2)
            assert (input_grad - padded[2]).norm() <= 0.1 * padded[2].norm() + 1e-5
            for name, grad in grads.items():
                assert (grad - padded[3][name]).norm() <= 0.1 * padded[3][name].norm() + 1e-5, name
        flash, flex = run("flash"), run("flex")
        torch.testing.assert_close(flash[0], flex[0], atol=2e-2, rtol=2e-2)
        assert (flash[2] - flex[2]).norm() <= 0.05 * flex[2].norm() + 1e-5
        for name, grad in flash[3].items():
            assert (grad - flex[3][name]).norm() <= 0.05 * flex[3][name].norm() + 1e-5, name

    @pytest.mark.unit
    def test_attention_dropout_without_flash_attention_is_rejected(self, monkeypatch):
        """FlexAttention has no dropout: rel_pos attention without its Triton kernel, and abs_pos attention
        in float32, refuse dropout_att.

        The refusal comes from the support check that ``supports_sequence_packed_output`` shares,
        so packed perception modules keep their padded path for these configurations instead;
        the kernel is switched off here, since with it rel_pos attention keeps its dropout.
        """
        monkeypatch.setattr(packed_attention_module, "TRITON_AVAILABLE", False)
        lengths = torch.tensor([30, 7], device="cuda")
        features = _packed_test_features(lengths, dtype=torch.float32, device="cuda")
        for self_attention_model in ("rel_pos", "abs_pos"):
            encoder = _packed_encoder(self_attention_model=self_attention_model, dropout_att=0.1).cuda().train()
            assert not encoder.supports_sequence_packed_output
            with pytest.raises(NotImplementedError, match="dropout_att"):
                encoder.forward_sequence_packed(features, lengths)
            encoder.eval()
            assert encoder.supports_sequence_packed_output
            encoder.train()
            if self_attention_model == "abs_pos":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    assert encoder.supports_sequence_packed_output == self._half_precision_flash_attention_available()
        # a head width that FlashAttention does not take (44 is the FastConformer Small head) has no dropout either
        narrow = _packed_encoder(self_attention_model="abs_pos", dropout_att=0.1, d_model=88, n_heads=2).cuda().train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not narrow.supports_sequence_packed_output
            with pytest.raises(NotImplementedError, match="dropout_att"):
                narrow.forward_sequence_packed(features, lengths)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPackedFastConformerSmoke:
    """A released FastConformer on real speech: the packed forward against the padded one.

    Parakeet 110M (hybrid TDT and CTC, ``dw_striding`` subsampling, rel_pos attention of width 64) encodes two AN4
    recordings padded and packed in evaluation mode.
    The padded forward runs the fused Triton kernel of the subsampling stack where Triton is available,
    while the packed form runs the stack layer by layer, so the two differ by that kernel's precision
    (1.7e-2 at most and 1.8e-4 on average over the two recordings, the same as the fused padded forward's
    deviation from the plain one); with the fusion off they agree to float32 rounding (1.5e-5 at most).
    """

    @pytest.mark.unit
    @pytest.mark.with_downloads
    def test_parakeet_110m_matches_padded_on_real_audio(self, test_data_dir):
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

        # loaded here rather than through the session fixture, so that a run without --with_downloads downloads nothing
        model = ASRModel.from_pretrained("parakeet-tdt_ctc-110m", map_location="cuda").eval()
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        # a five-second and a one-second recording, so the padded batch holds real padding
        clips = [("ffmm", "cen2-ffmm-b.wav"), ("fash", "an253-fash-b.wav")]
        samples = [
            torch.as_tensor(
                AudioSegment.from_file(
                    os.path.join(test_data_dir, "an4_speaker", "an4", "wav", "an4_clstk", speaker, clip),
                    target_sr=16000,
                ).samples
            )
            for speaker, clip in clips
        ]
        audio = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True).cuda()
        audio_lengths = torch.tensor([len(clip) for clip in samples], device="cuda")
        encoder = model.encoder
        stack = encoder.pre_encode.conv
        fused = stack.fuse_triton
        with torch.no_grad():
            features, feature_lengths = model.preprocessor(input_signal=audio, length=audio_lengths)
            packed = encoder.forward_sequence_packed(features, feature_lengths)
            errors = {}
            try:
                for fuse_triton in sorted({False, fused}):
                    stack.fuse_triton = fuse_triton
                    padded_out, padded_lengths = encoder(audio_signal=features, length=feature_lengths)
                    assert packed.lengths.tolist() == padded_lengths.tolist()
                    errors[fuse_triton] = (packed.data - _valid_frames(padded_out, padded_lengths)).abs().max().item()
            finally:
                stack.fuse_triton = fused
        assert errors[False] < 1e-4, errors
        assert errors.get(True, 0.0) < 5e-2, errors


class TestCacheAwareTraining:
    """Testing that training matches cache-aware streaming."""

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["vggnet", "dw_striding_conv1d", "striding_conv1d"])
    def test_training_prefix_requires_causal_convolution(self, cache_aware_model_cfg, subsampling):
        encoder_cfg = cache_aware_model_cfg.encoder
        encoder_cfg.subsampling = subsampling
        encoder = ConformerEncoder.from_config_dict(encoder_cfg).cpu().train()
        features = torch.randn(1, encoder_cfg.feat_in, 16)
        expected_frames = features.size(-1)
        if subsampling == "striding_conv1d":
            expected_frames += encoder.streaming_cfg.pre_encode_cache_size[1]

        class ReachedFrontend(Exception):
            pass

        def check_input(module, args, kwargs):
            assert kwargs["x"].size(1) == kwargs["lengths"].item() == expected_frames
            raise ReachedFrontend

        with (
            encoder.pre_encode.register_forward_pre_hook(check_input, with_kwargs=True),
            pytest.raises(ReachedFrontend),
        ):
            encoder(audio_signal=features, length=torch.tensor([features.size(-1)]))

    @pytest.mark.unit
    def test_activation_checkpointing_preserves_pad_and_drop(self, cache_aware_model_cfg):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        torch.manual_seed(0)
        encoder = ConformerEncoder.from_config_dict(cache_aware_model_cfg.encoder).cpu().train()
        wrapped = copy.deepcopy(encoder)
        wrapped.pre_encode = checkpoint_wrapper(wrapped.pre_encode)
        features = torch.randn(1, cache_aware_model_cfg.encoder.feat_in, 128)
        lengths = torch.tensor([features.size(-1)])

        reference, reference_len = encoder(audio_signal=features, length=lengths)
        output, output_len = wrapped(audio_signal=features, length=lengths)
        assert output_len.tolist() == reference_len.tolist() == [16]
        torch.testing.assert_close(output, reference)

        grad = torch.randn_like(reference)
        reference.backward(grad)
        output.backward(grad)
        for parameter, wrapped_parameter in zip(encoder.parameters(), wrapped.parameters()):
            torch.testing.assert_close(wrapped_parameter.grad, parameter.grad)

    @pytest.mark.parametrize("n_frames", [7, 32])
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    def test_noncausal_train_matches_eval(self, cache_aware_model_cfg, n_frames, subsampling):
        """Noncausal training preserves output lengths and frame alignment."""
        torch.manual_seed(0)
        encoder_cfg = cache_aware_model_cfg.encoder
        encoder_cfg.update(
            n_layers=1, d_model=16, subsampling=subsampling, subsampling_conv_channels=8, causal_downsampling=False
        )
        model = ConformerEncoder.from_config_dict(encoder_cfg).cpu()
        feats = torch.randn(1, encoder_cfg.feat_in, n_frames)
        lengths = torch.tensor([n_frames])
        with torch.no_grad():
            model.eval()
            eval_out, eval_len = model(audio_signal=feats, length=lengths)
            model.train()
            train_out, train_len = model(audio_signal=feats, length=lengths)

        factor = encoder_cfg.subsampling_factor
        assert train_len.tolist() == eval_len.tolist() == [(n_frames + factor - 1) // factor]
        torch.testing.assert_close(train_out, eval_out)

    @pytest.mark.parametrize(
        "lengths,stream_in_train_mode",
        [((128,), False), ((127,), False), ((127, 73), False), pytest.param((128,), True, id="cached-training")],
    )
    @pytest.mark.parametrize(
        "att_context_size,context_index",
        [([16, 3], 0), ([[16, 3], [16, 1]], 0), ([[16, 3], [16, 1]], 1)],
    )
    def test_train_matches_streaming(
        self, cache_aware_model_cfg, lengths, stream_in_train_mode, att_context_size, context_index
    ):
        """Training matches real buffered streaming for partial chunks and ragged batches."""
        torch.manual_seed(0)
        encoder_cfg = cache_aware_model_cfg.encoder
        num_contexts = len(att_context_size) if isinstance(att_context_size[0], list) else 1
        encoder_cfg.att_context_size = att_context_size
        # Select each supported training lookahead deterministically for comparison with eval.
        encoder_cfg.att_context_probs = [float(i == context_index) for i in range(num_contexts)]
        model = EncDecCTCModel(cfg=cache_aware_model_cfg).cpu()
        encoder = model.encoder
        encoder.set_default_att_context_size(encoder.att_context_size_all[context_index])
        buffer = CacheAwareStreamingAudioBuffer(model, pad_and_drop_preencoded=True)
        for length in lengths:
            buffer.append_processed_signal(torch.randn(1, encoder_cfg.feat_in, length))
        feats, input_lengths = buffer.get_all_audios()

        model.train()
        with torch.no_grad():
            train_out, train_len = encoder(audio_signal=feats, length=input_lengths)

        model.train(stream_in_train_mode)
        cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=len(lengths))
        stream_out = [[] for _ in lengths]
        with torch.no_grad():
            for chunk, chunk_len in buffer:
                out, out_len, cache_channel, cache_time, cache_len = encoder.cache_aware_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=chunk_len,
                    cache_last_channel=cache_channel,
                    cache_last_time=cache_time,
                    cache_last_channel_len=cache_len,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    drop_extra_pre_encoded=encoder.streaming_cfg.drop_extra_pre_encoded,
                )
                for i, length in enumerate(out_len):
                    stream_out[i].append(out[i, :, :length])

        for i, outputs in enumerate(stream_out):
            streamed = torch.cat(outputs, dim=-1)
            assert streamed.shape[-1] == train_len[i].item()
            torch.testing.assert_close(train_out[i, :, : train_len[i]], streamed, atol=1e-5, rtol=1e-5)
