# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.transformer_encoder import (
    FeatureStacking,
    TransformerEncoder,
    TransformerEncoderConfig,
)


class TestTransformerEncoderConfig:
    @pytest.mark.unit
    def test_default_config(self):
        cfg = TransformerEncoderConfig()
        assert cfg.feat_in == 128
        assert cfg.d_model == 512
        assert cfg.n_heads == 8
        assert cfg.n_layers == 17
        assert cfg.drop_rate == 0.1
        assert cfg.qkv_bias is False
        assert cfg.qk_norm is False
        assert cfg.ff_expansion == 4.0
        assert cfg.pre_block_norm is True
        assert cfg.subsampling_factor == 4
        assert cfg.attn_mode == "full"
        assert cfg.self_attention_model == "rel_pos"

    @pytest.mark.unit
    def test_custom_config(self):
        cfg = TransformerEncoderConfig(
            feat_in=128, d_model=1280, n_heads=16, n_layers=32, qk_norm=True, self_attention_model="abs_pos"
        )
        assert cfg.feat_in == 128
        assert cfg.d_model == 1280
        assert cfg.n_heads == 16
        assert cfg.n_layers == 32
        assert cfg.qk_norm is True
        assert cfg.self_attention_model == "abs_pos"


class TestFeatureStacking:
    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling_factor", [2, 4, 8])
    def test_output_shape(self, subsampling_factor):
        B, C, T = 2, 80, 400
        stacking = FeatureStacking(subsampling_factor=subsampling_factor, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400, 300])

        out, out_lengths = stacking(x, lengths)
        expected_t = stacking.compute_num_out_frames(T)
        assert out.shape == (B, expected_t, 256)
        assert out_lengths[0].item() == expected_t

    @pytest.mark.unit
    def test_padding_when_not_divisible(self):
        B, C, T = 1, 80, 401
        subsampling_factor = 4
        stacking = FeatureStacking(subsampling_factor=subsampling_factor, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([401])

        out, out_lengths = stacking(x, lengths)
        expected_t = stacking.compute_num_out_frames(T)
        assert out.shape == (B, expected_t, 256)
        assert out_lengths[0].item() == expected_t

    @pytest.mark.unit
    def test_length_shorter_than_batch(self):
        """Output length must be ceil(sample_length / factor), not dependent on batch T."""
        B, C, T = 2, 80, 403
        subsampling_factor = 4
        stacking = FeatureStacking(subsampling_factor=subsampling_factor, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([401, 397])

        _, out_lengths = stacking(x, lengths)
        assert out_lengths[0].item() == stacking.compute_num_out_frames(401)
        assert out_lengths[1].item() == stacking.compute_num_out_frames(397)

    @pytest.mark.unit
    def test_no_padding_when_divisible(self):
        B, C, T = 1, 80, 400
        stacking = FeatureStacking(subsampling_factor=4, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400])

        out, out_lengths = stacking(x, lengths)
        assert out.shape == (B, stacking.compute_num_out_frames(T), 256)
        assert out_lengths[0].item() == stacking.compute_num_out_frames(T)


class TestBypassPreEncode:
    """Testing bypass pre-encode functionality."""

    def test_bypass_pre_encode_forward(self):
        """Testing that forward works with "bypass pre-encode" mode.

        Forwards are wrapped in ``torch.no_grad()`` so the test runs on CPU as well as GPU:
        FlexAttention's CPU path refuses to run when any input requires gradients (parameters
        of an ``nn.Module`` do by default), and we are only checking output shapes here, never
        calling ``.backward()``.
        """
        # For pre-encoded embeddings, the shape is (batch_size, n_frames, emb_dim)
        batch_size = 2
        n_frames, emb_dim, feat_out = 17, 64, 8  # emb_dim=64 with n_heads=4 -> head_dim=16 (>= 16)
        random_input = torch.rand((batch_size, n_frames, emb_dim))
        random_length = torch.tensor([n_frames] * batch_size, dtype=torch.int64)

        model = TransformerEncoder(
            feat_in=10,
            n_layers=3,
            d_model=emb_dim,
            n_heads=4,
            feat_out=feat_out,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
        )
        model.train()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        with torch.no_grad():
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
        n_frames, emb_dim, feat_in, feat_out = 17, 64, 10, 8  # emb_dim=64 with n_heads=4 -> head_dim=16 (>= 16)

        pre_encode_input = torch.rand((batch_size, n_frames, emb_dim))
        feat_input = torch.rand((batch_size, feat_in, n_frames))
        input_length = torch.tensor([n_frames] * batch_size, dtype=torch.int64)

        model = TransformerEncoder(
            feat_in=feat_in,
            n_layers=3,
            d_model=emb_dim,
            n_heads=4,
            feat_out=feat_out,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
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
        # NB: forwards that actually reach FlexAttention are wrapped in ``torch.no_grad()`` so
        # the test passes on CPU (FlexAttention's CPU path refuses inputs that require grad).
        # The ``pytest.raises(ValueError)`` blocks above/below intentionally do *not* need this
        # wrapper because the shape check in ``TransformerEncoder.forward()`` raises before any
        # attention computation.
        model.train()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        with torch.no_grad():
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
        with torch.no_grad():
            fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)

        model.eval()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)

    @pytest.mark.unit
    def test_bypass_pre_encode_matches_manual_pre_encode(self):
        """``bypass_pre_encode=True`` must skip *only* the pre-encoder.

        Running the pre-encoder by hand and feeding its output back in with
        ``bypass_pre_encode=True`` should reproduce the full forward
        (``bypass_pre_encode=False``) exactly, because the positional-encoding, norm and
        Transformer-block stack downstream of the pre-encoder is identical on both paths.
        """
        B, feat_in, T, d_model, feat_out = 2, 32, 64, 64, 8  # d_model=64 with n_heads=4 -> head_dim=16 (>= 16)
        model = TransformerEncoder(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            feat_out=feat_out,
            subsampling_factor=4,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
        )
        model.eval()

        mel = torch.randn(B, feat_in, T)
        lengths = torch.tensor([T, T - 8], dtype=torch.int64)

        with torch.no_grad():
            out_full, len_full = model(audio_signal=mel, length=lengths, bypass_pre_encode=False)

            # Reproduce just the pre-encoder, then bypass it on the next call.
            pre_x, pre_len = model.pre_encode(mel, lengths)
            out_bypass, len_bypass = model(audio_signal=pre_x, length=pre_len, bypass_pre_encode=True)

        assert out_full.shape == out_bypass.shape == (B, feat_out, pre_x.shape[1])
        assert torch.equal(len_full, len_bypass)
        assert torch.allclose(out_full, out_bypass, atol=1e-5)


class TestTransformerEncoder:
    @pytest.mark.unit
    def test_model_creation(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert len(model.layers) == 2

    @pytest.mark.unit
    def test_model_creation_with_qk_norm(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, qk_norm=True)
        attn = model.layers[0].attn
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    @pytest.mark.unit
    def test_model_creation_without_qk_norm(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, qk_norm=False)
        attn = model.layers[0].attn
        assert not hasattr(attn, 'q_norm')
        assert not hasattr(attn, 'k_norm')

    @pytest.mark.unit
    def test_invalid_attn_mode(self):
        with pytest.raises(ValueError, match="not yet supported"):
            TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, attn_mode="sliding_window")

    @pytest.mark.unit
    def test_head_dim_below_16_raises(self):
        """head_dim = d_model // n_heads must be >= 16 (PyTorch FlexAttention CUDA requirement).

        The check happens at construction time, so an unsupported (d_model, n_heads) pair raises
        before any forward pass.
        """
        # d_model=32, n_heads=4 -> head_dim=8 (< 16).
        with pytest.raises(ValueError, match="per-head embedding dimension >= 16"):
            TransformerEncoder(feat_in=128, d_model=32, n_heads=4, n_layers=2)

    @pytest.mark.unit
    def test_causal_forward_cpu(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, attn_mode="causal")
        model.eval()

        x = torch.randn(2, 80, 400)
        lengths = torch.tensor([400, 300])

        with torch.no_grad():
            out, out_lengths = model(x, lengths)

        assert out.shape == (2, 64, 100)
        assert out_lengths.tolist() == [100, 75]
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_causal_future_does_not_affect_past(self):
        """Output at position t must be invariant to changes at positions > t."""
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, attn_mode="causal")
        model.eval()

        B, C, T = 1, 80, 400
        x_a = torch.randn(B, C, T)
        x_b = x_a.clone()
        # Perturb only the second half of frames.
        x_b[:, :, T // 2 :] = torch.randn(B, C, T - T // 2)
        lengths = torch.tensor([T])

        with torch.no_grad():
            out_a, _ = model(x_a, lengths)
            out_b, _ = model(x_b, lengths)

        # Output frames covering only past + present should be identical.
        # First half of *output* frames corresponds to first half of input frames after subsampling.
        safe_t = (T // 2) // model.pre_encode.subsampling_factor
        assert torch.allclose(out_a[:, :, :safe_t], out_b[:, :, :safe_t], atol=1e-5)

    @pytest.mark.unit
    def test_freeze_unfreeze_partial_restores_prior_state(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        for p in model.final_norm.parameters():
            p.requires_grad = False
        prior = {n: p.requires_grad for n, p in model.named_parameters()}

        model.freeze()
        assert all(not p.requires_grad for p in model.parameters())
        assert not model.training

        model.unfreeze(partial=True)
        assert {n: p.requires_grad for n, p in model.named_parameters()} == prior
        assert model.training

    @pytest.mark.unit
    def test_forward_cpu(self):
        """Forward pass on CPU uses unfused FlexAttention fallback."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, subsampling_factor=4)
        model.eval()

        B, C, T = 2, 128, 400
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400, 300])

        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 300 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_forward_cpu_with_qk_norm(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, qk_norm=True)
        model.eval()

        x = torch.randn(1, 128, 200)
        lengths = torch.tensor([200])

        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape == (1, 64, 50)
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_basic(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, subsampling_factor=4)
        model = model.cuda().to(torch.bfloat16)

        B, C, T = 2, 128, 400
        x = torch.randn(B, C, T, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([400, 300], device='cuda')

        model.eval()
        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 300 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_with_qk_norm(self):
        model = TransformerEncoder(
            feat_in=128, d_model=128, n_heads=8, n_layers=2, drop_rate=0.0, qk_norm=True, subsampling_factor=8
        )
        model = model.cuda().to(torch.bfloat16)

        B, C, T = 2, 128, 800
        x = torch.randn(B, C, T, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([800, 640], device='cuda')

        model.eval()
        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 128, T // 8)
        assert out_lengths[1].item() == 640 // 8
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_output_channels_first(self):
        """Verify output is (B, D, T) channels-first as expected by downstream decoders."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=1, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16)

        x = torch.randn(1, 128, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200], device='cuda')

        model.eval()
        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape[1] == 64  # D dimension
        assert out.shape[2] == 200 // 4  # T dimension

    @pytest.mark.run_only_on('GPU')
    def test_eval_deterministic(self):
        """In eval mode with no dropout, repeated forward passes should produce identical output."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).eval()

        x = torch.randn(1, 128, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200], device='cuda')

        with torch.no_grad():
            out1, _ = model(audio_signal=x, length=lengths)
            out2, _ = model(audio_signal=x, length=lengths)

        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.run_only_on('GPU')
    def test_padding_does_not_affect_valid_output(self):
        """Padding frames should not change the encoded output at valid positions."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).eval()

        T_valid = 200
        x_short = torch.randn(1, 128, T_valid, device='cuda', dtype=torch.bfloat16)
        lengths_short = torch.tensor([T_valid], device='cuda')

        T_padded = 400
        x_long = torch.zeros(1, 128, T_padded, device='cuda', dtype=torch.bfloat16)
        x_long[:, :, :T_valid] = x_short
        lengths_long = torch.tensor([T_valid], device='cuda')

        with torch.no_grad():
            out_short, len_short = model(audio_signal=x_short, length=lengths_short)
            out_long, len_long = model(audio_signal=x_long, length=lengths_long)

        assert len_short[0].item() == len_long[0].item()
        valid_t = len_short[0].item()
        # bf16 + different block mask shapes cause small numerical differences in Triton kernels
        assert torch.allclose(out_short[:, :, :valid_t], out_long[:, :, :valid_t], atol=5e-2)

    @pytest.mark.run_only_on('GPU')
    def test_backward_pass(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).train()

        x = torch.randn(2, 128, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200, 160], device='cuda')

        out, _ = model(audio_signal=x, length=lengths)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestSelfAttentionModel:
    """Tests for the ``self_attention_model`` positional encoding option."""

    @pytest.mark.unit
    def test_default_is_rel_pos(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2)
        assert model.self_attention_model == "rel_pos"

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["abs_pos", "rel_pos", "no_pos"])
    def test_valid_modes_are_accepted(self, mode):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model=mode)
        assert model.self_attention_model == mode

    @pytest.mark.unit
    def test_none_aliases_no_pos(self):
        """Passing ``self_attention_model=None`` must be equivalent to ``"no_pos"``."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model=None)
        assert model.self_attention_model == "no_pos"
        assert model.pos_enc is None

    @pytest.mark.unit
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            TransformerEncoder(
                feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model="rel_pos_local_attn"
            )

    @pytest.mark.unit
    def test_rel_pos_attention_params_allocated(self):
        """rel_pos mode allocates the Transformer-XL bias parameters per attention layer."""
        d_model, n_heads, n_layers = 64, 4, 2
        model = TransformerEncoder(
            feat_in=128, d_model=d_model, n_heads=n_heads, n_layers=n_layers, self_attention_model="rel_pos"
        )
        head_dim = d_model // n_heads
        assert model.pos_enc is not None
        for layer in model.layers:
            attn = layer.attn
            assert attn.linear_pos is not None
            assert attn.pos_bias_u is not None
            assert attn.pos_bias_v is not None
            assert attn.pos_bias_u.shape == (n_heads, head_dim)
            assert attn.pos_bias_v.shape == (n_heads, head_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["abs_pos", "no_pos"])
    def test_non_rel_pos_modes_have_no_rel_params(self, mode):
        """abs_pos and no_pos modes must not allocate the rel-pos parameters."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model=mode)
        for layer in model.layers:
            attn = layer.attn
            assert attn.linear_pos is None
            assert attn.pos_bias_u is None
            assert attn.pos_bias_v is None

    @pytest.mark.unit
    def test_no_pos_has_no_positional_encoding_module(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model="no_pos")
        assert model.pos_enc is None
        # set_max_audio_length is invoked in __init__; it must not crash for no_pos and must
        # still record the requested max length so update_max_seq_length works normally.
        assert model.max_audio_length == model.pos_emb_max_len

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["abs_pos", "rel_pos", "no_pos", None])
    def test_forward_each_mode_cpu(self, mode):
        """Each ``self_attention_model`` choice (including ``None``) must produce a valid forward."""
        model = TransformerEncoder(
            feat_in=128,
            d_model=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            subsampling_factor=4,
            self_attention_model=mode,
        )
        model.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 160 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_rel_pos_broadcasts_when_T_differs_from_n_heads(self):
        """Regression test for the Transformer-XL bias broadcasting.

        ``pos_bias_{u,v}`` has shape ``(H, D)`` and must broadcast against the head axis of
        ``q`` which has shape ``(B, H, T, D)``. A naive add would right-align ``H`` against
        ``T`` and either crash (``T != H``) or silently apply the bias on the wrong axis
        (``T == H``). This test exercises a configuration where ``T_attn != n_heads`` so the
        broken broadcast would surface as an error.
        """
        # 200 input frames / subsampling_factor=4 -> 50 attention frames; n_heads=4 -> T != H.
        model = TransformerEncoder(
            feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, self_attention_model="rel_pos"
        )
        model.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert not torch.isnan(out).any()
