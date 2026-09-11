# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import dataclasses

import numpy as np
import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.modules.transformer_encoder import (
    FeatureStacking,
    StreamingTransformerEncoder,
    TransformerEncoder,
    TransformerEncoderConfig,
    _make_chunked_limited_mod,
    _make_sliding_window_mod,
)
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import RotaryPositionalEncoding


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
        assert cfg.rope_base == 10000.0
        assert cfg.rotary_fraction == 1.0

    @pytest.mark.unit
    def test_custom_config(self):
        cfg = TransformerEncoderConfig(
            feat_in=128,
            d_model=1280,
            n_heads=16,
            n_layers=32,
            qk_norm=True,
            self_attention_model="rope",
            rope_base=500000.0,
            rotary_fraction=0.5,
        )
        assert cfg.feat_in == 128
        assert cfg.d_model == 1280
        assert cfg.n_heads == 16
        assert cfg.n_layers == 32
        assert cfg.qk_norm is True
        assert cfg.self_attention_model == "rope"
        assert cfg.rope_base == 500000.0
        assert cfg.rotary_fraction == 0.5


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


class TestRotaryPositionalEncoding:
    @pytest.mark.unit
    @pytest.mark.parametrize("rotary_fraction", [0.0, -0.1, 1.1])
    def test_invalid_rotary_fraction_raises(self, rotary_fraction):
        with pytest.raises(ValueError, match="rotary_fraction must be in"):
            RotaryPositionalEncoding(d_k=16, rotary_fraction=rotary_fraction)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("d_k", "rotary_fraction"),
        [
            (16, 0.05),  # int(16 * 0.05) == 0
            (18, 0.5),  # int(18 * 0.5) == 9
        ],
    )
    def test_invalid_effective_rotary_dim_raises(self, d_k, rotary_fraction):
        with pytest.raises(ValueError, match="Effective rotary dim"):
            RotaryPositionalEncoding(d_k=d_k, rotary_fraction=rotary_fraction)

    @pytest.mark.unit
    def test_partial_rotation_preserves_tail_and_norm(self):
        rope = RotaryPositionalEncoding(d_k=8, rotary_fraction=0.5, max_len=4)
        rope.extend_pe(length=4, device=torch.device("cpu"), dtype=torch.float32)
        q = torch.randn(2, 3, 4, 8)
        k = torch.randn(2, 3, 4, 8)

        q_rot, k_rot = rope(q, k)

        torch.testing.assert_close(q_rot[..., 4:], q[..., 4:])
        torch.testing.assert_close(k_rot[..., 4:], k[..., 4:])
        torch.testing.assert_close(q_rot[..., :4].norm(dim=-1), q[..., :4].norm(dim=-1))
        torch.testing.assert_close(k_rot[..., :4].norm(dim=-1), k[..., :4].norm(dim=-1))
        # Position zero has angle zero and therefore remains unchanged.
        torch.testing.assert_close(q_rot[:, :, 0], q[:, :, 0])
        torch.testing.assert_close(k_rot[:, :, 0], k[:, :, 0])

    @pytest.mark.unit
    def test_query_cache_offset_matches_full_sequence_positions(self):
        rope = RotaryPositionalEncoding(d_k=8, max_len=4)
        rope.extend_pe(length=4, device=torch.device("cpu"), dtype=torch.float32)
        full_q = torch.randn(1, 2, 4, 8)

        full_q_rot, full_k_rot = rope(full_q, full_q)
        cached_q_rot, cached_k_rot = rope(full_q[:, :, -2:], full_q)

        torch.testing.assert_close(cached_q_rot, full_q_rot[:, :, -2:])
        torch.testing.assert_close(cached_k_rot, full_k_rot)


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


class TestDropout:
    """Dropout wiring.

    Every other test in this file builds encoders with ``drop_rate=0.0``, so nothing here
    exercised the dropout path -- which is how the FFN branch ended up being dropped twice
    (once inside ``FeedForward.net``, once by ``TransformerBlock``'s residual dropout),
    giving an effective ``1 - 0.9**2 = 0.19`` instead of the configured 0.1.
    """

    @pytest.mark.unit
    def test_ffn_drops_branch_output_exactly_once(self):
        """The FFN residual branch must be dropped at the configured rate, not twice."""
        torch.manual_seed(0)
        p = 0.2
        cfg = TransformerEncoderConfig(d_model=64, n_heads=4, n_layers=1, drop_rate=p, ff_expansion=4)
        from nemo.collections.asr.modules.transformer_encoder import FeedForward, TransformerBlock

        # FeedForward's own output must be dropout-free; TransformerBlock owns the branch dropout.
        ffn = FeedForward(cfg).train()
        assert not isinstance(ffn.net[-1], torch.nn.Dropout), (
            "FeedForward.net must not end in Dropout -- TransformerBlock already drops this "
            "module's output on the residual branch, and stacking the two doubles the rate."
        )

        # Measure the branch's realized drop rate end to end through the block.
        block = TransformerBlock(cfg).train()
        x = torch.ones(1, 8192, 64)
        with torch.no_grad():
            branch = block.drop(block.ffn(block.norm2(x)))
        zero_frac = (branch == 0).float().mean().item()
        assert zero_frac == pytest.approx(p, abs=0.02), (
            f"FFN branch zero fraction {zero_frac:.4f} should be ~{p} (dropped once); "
            f"~{1 - (1 - p) ** 2:.2f} means it is being dropped twice."
        )

    @pytest.mark.unit
    def test_ffn_state_dict_keys_are_checkpoint_compatible(self):
        """FFN Linear indices are state_dict keys -- they must stay net.0 / net.3.

        ``nn.Dropout`` has no parameters, so only its *position* matters. Removing the
        trailing Dropout keeps the Linears at indices 0 and 3; removing the inner one
        would renumber the second Linear to ``net.2.*`` and break every existing
        checkpoint. This test pins that down.
        """
        cfg = TransformerEncoderConfig(d_model=64, n_heads=4, n_layers=1, drop_rate=0.1, ff_expansion=4)
        from nemo.collections.asr.modules.transformer_encoder import FeedForward

        keys = sorted(k for k, _ in FeedForward(cfg).named_parameters())
        assert keys == ['net.0.bias', 'net.0.weight', 'net.3.bias', 'net.3.weight'], (
            f"FFN parameter keys changed to {keys}; this breaks loading existing "
            "StreamingTransformerEncoder checkpoints."
        )

    @pytest.mark.unit
    def test_eval_mode_disables_dropout(self):
        """Inference must be deterministic and dropout-free at any drop_rate."""
        torch.manual_seed(0)
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.5).eval()
        x = torch.randn(2, 128, 32)
        lengths = torch.tensor([32, 32])
        with torch.no_grad():
            a, _ = model(x, lengths)
            b, _ = model(x, lengths)
        torch.testing.assert_close(a, b)


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
    @pytest.mark.parametrize("mode", ["abs_pos", "rel_pos", "rope", "no_pos"])
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
    @pytest.mark.parametrize("mode", ["abs_pos", "rope", "no_pos"])
    def test_non_rel_pos_modes_have_no_rel_params(self, mode):
        """Non-relative modes must not allocate the rel-pos parameters."""
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
    @pytest.mark.parametrize("mode", ["abs_pos", "rel_pos", "rope", "no_pos", None])
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

    @pytest.mark.unit
    def test_rope_uses_shared_rotary_pos_enc(self):
        """rope mode builds a single ``RotaryPositionalEncoding`` reused by every attention layer.

        The cos/sin buffers are computed once on the shared module (see ``TransformerEncoder``),
        so each layer's ``attn.rope`` must be the *same* object as ``model.pos_enc``.
        """
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=3, self_attention_model="rope")
        assert isinstance(model.pos_enc, RotaryPositionalEncoding)
        for layer in model.layers:
            attn = layer.attn
            assert attn._uses_rope is True
            assert attn.rope is model.pos_enc

    @pytest.mark.unit
    def test_rope_partial_rotation_forward_cpu(self):
        """``rotary_fraction`` < 1.0 rotates only part of each head dim (exercises the pass-through split)."""
        model = TransformerEncoder(
            feat_in=128,
            d_model=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            subsampling_factor=4,
            self_attention_model="rope",
            rotary_fraction=0.5,
        )
        model.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_rope_padding_does_not_affect_valid_output(self):
        """Masked padding keys must not change valid RoPE encoder outputs."""
        model = TransformerEncoder(
            feat_in=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            self_attention_model="rope",
        ).eval()
        valid = torch.randn(1, 5, 64)
        padded = torch.cat((valid, torch.randn(1, 3, 64)), dim=1)
        lengths = torch.tensor([5])

        with torch.no_grad():
            out_valid, valid_lengths = model(valid, lengths, bypass_pre_encode=True)
            out_padded, padded_lengths = model(padded, lengths, bypass_pre_encode=True)

        assert valid_lengths.tolist() == padded_lengths.tolist() == [5]
        torch.testing.assert_close(out_padded[:, :, :5], out_valid, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_rope_causal_mask_blocks_future_frames(self):
        """Changing future embeddings must not affect past outputs in causal RoPE mode."""
        model = TransformerEncoder(
            feat_in=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            self_attention_model="rope",
            attn_mode="causal",
        ).eval()
        x_a = torch.randn(1, 8, 64)
        x_b = x_a.clone()
        x_b[:, 4:] = torch.randn_like(x_b[:, 4:])
        lengths = torch.tensor([8])

        with torch.no_grad():
            out_a, _ = model(x_a, lengths, bypass_pre_encode=True)
            out_b, _ = model(x_b, lengths, bypass_pre_encode=True)

        torch.testing.assert_close(out_a[:, :, :4], out_b[:, :, :4], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_rope_forward_with_checkpoint_wrapped_feature_stacking(self):
        """The pre-encoder type dispatch must work through PyTorch's checkpoint wrapper."""
        model = TransformerEncoder(
            feat_in=16,
            d_model=64,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            self_attention_model="rope",
        ).eval()
        model.pre_encode = checkpoint_wrapper(model.pre_encode)
        audio = torch.randn(2, 16, 32)
        lengths = torch.tensor([32, 24])

        with torch.no_grad():
            out, out_lengths = model(audio, lengths)

        assert out.shape == (2, 64, 8)
        assert out_lengths.tolist() == [8, 6]

    def test_base_build_mask_mod_unchanged(self):
        """The refactored ``_build_mask_mod`` hook must preserve the base full/causal masks."""
        full = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=1, attn_mode="full")
        causal = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=1, attn_mode="causal")
        length = torch.tensor([4, 4], dtype=torch.int64)

        def allowed(mod, q, kv):
            return bool(mod(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(kv)))

        full_mod = full._build_mask_mod(length)
        causal_mod = causal._build_mask_mod(length)
        # Full: any valid (non-pad) key is attendable, including future keys.
        assert allowed(full_mod, 1, 3)
        assert allowed(full_mod, 3, 1)
        # Padding (kv >= length) is masked in both.
        assert not allowed(full_mod, 1, 4)
        # Causal: future keys are masked, past/present allowed.
        assert allowed(causal_mod, 3, 1)
        assert not allowed(causal_mod, 1, 3)


class TestSlidingWindowMod:
    """Unit tests for the ``_make_sliding_window_mod`` FlexAttention mask factory."""

    @staticmethod
    def _allowed(mod, q, kv):
        return bool(mod(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(kv)))

    @pytest.mark.unit
    def test_bounded_window(self):
        """left=2, right=1 -> query q attends to kv in [q - 2, q + 1]."""
        mod = _make_sliding_window_mod(2, 1)
        assert self._allowed(mod, 5, 3)  # lower edge
        assert self._allowed(mod, 5, 5)  # self
        assert self._allowed(mod, 5, 6)  # upper edge (look-ahead)
        assert not self._allowed(mod, 5, 2)  # beyond left context
        assert not self._allowed(mod, 5, 7)  # beyond right context

    @pytest.mark.unit
    def test_unlimited_left_is_causal(self):
        """left<0, right=0 -> unlimited past, no look-ahead (causal)."""
        mod = _make_sliding_window_mod(-1, 0)
        assert self._allowed(mod, 5, 0)  # far past allowed
        assert self._allowed(mod, 5, 5)  # self allowed
        assert not self._allowed(mod, 5, 6)  # future masked

    @pytest.mark.unit
    def test_unlimited_right_only_left_bound(self):
        """right<0, left=1 -> unlimited future, only one frame of past."""
        mod = _make_sliding_window_mod(1, -1)
        assert self._allowed(mod, 5, 4)  # one frame back allowed
        assert self._allowed(mod, 5, 99)  # far future allowed
        assert not self._allowed(mod, 5, 3)


class TestChunkedLimitedMod:
    """Unit tests for the ``_make_chunked_limited_mod`` FlexAttention mask factory."""

    @staticmethod
    def _allowed(mod, q, kv):
        return bool(mod(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(kv)))

    @pytest.mark.unit
    def test_own_chunk_plus_left_chunks(self):
        """left=4, right=1 -> chunk_size 2, left_chunks 2: query in chunk 3 sees chunks 1..3."""
        mod = _make_chunked_limited_mod(4, 1)
        assert self._allowed(mod, 6, 7)  # own chunk, in-chunk look-ahead
        assert self._allowed(mod, 7, 6)  # own chunk, backwards
        assert self._allowed(mod, 6, 2)  # two chunks back (left_chunks = 4 // 2 = 2)
        assert not self._allowed(mod, 6, 1)  # three chunks back -> masked
        assert not self._allowed(mod, 6, 8)  # next chunk -> masked

    @pytest.mark.unit
    def test_left_context_quantized_to_whole_chunks(self):
        """left is floored to whole chunks: left=5, chunk_size=4 -> 1 left chunk (4 frames)."""
        mod = _make_chunked_limited_mod(5, 3)
        assert self._allowed(mod, 8, 4)  # start of the single visible left chunk
        assert not self._allowed(mod, 8, 3)  # 5 frames back, but a chunk earlier -> masked

    @pytest.mark.unit
    def test_unlimited_left(self):
        """left<0 -> every earlier chunk is visible, still no cross-chunk look-ahead."""
        mod = _make_chunked_limited_mod(-1, 3)
        assert self._allowed(mod, 100, 0)
        assert self._allowed(mod, 100, 103)  # own chunk (100..103)
        assert not self._allowed(mod, 100, 104)

    @pytest.mark.unit
    def test_matches_conformer_reference_mask(self):
        """Bit-for-bit agreement with ``ConformerEncoder._create_masks``'s chunked_limited branch."""
        T, left, right = 24, 5, 2
        chunk_size = right + 1
        left_chunks_num = left // chunk_size
        chunk_idx = torch.div(torch.arange(T), chunk_size, rounding_mode="trunc")
        diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
        reference = torch.logical_and(diff_chunks <= left_chunks_num, diff_chunks >= 0)

        mod = _make_chunked_limited_mod(left, right)
        ours = torch.tensor([[self._allowed(mod, q, kv) for kv in range(T)] for q in range(T)])
        assert torch.equal(ours, reference)


class TestStreamingTransformerEncoder:
    """Tests for the sliding-window streaming encoder and its cache-aware interface."""

    @pytest.mark.unit
    def test_satisfies_streaming_encoder_interface(self):
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        # Must pass the isinstance(encoder, StreamingEncoder) gate in AudioPerceptionModule.
        assert isinstance(enc, StreamingEncoder)
        assert isinstance(enc, TransformerEncoder)
        for method in ("cache_aware_stream_step", "get_initial_cache_state", "setup_streaming_params"):
            assert callable(getattr(enc, method))

    @pytest.mark.unit
    def test_streaming_cfg_tracks_att_context(self):
        """``streaming_cfg`` is (re)built from ``att_context_size`` — the left context sizes the
        rolling cache; FeatureStacking needs no pre-encode overlap."""
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_size=[7, 0])
        assert enc.streaming_cfg.pre_encode_cache_size == 0
        assert enc.streaming_cfg.last_channel_cache_size == 7
        # Retuning the window rebuilds the cfg.
        enc.set_default_att_context_size([12, 0])
        assert enc.streaming_cfg.last_channel_cache_size == 12

    @pytest.mark.unit
    def test_initial_cache_state_shapes(self):
        """A rolling cache pre-allocates ``left`` frames per layer (padded); ``cache_last_time`` is
        a zero-width placeholder (no conv) and valid length starts at 0.

        ``cache_last_time`` must keep ``ConformerEncoder``'s 4-D rank
        ``(n_layers, B, d_model, conv_cache)`` even though its width is 0: the shared
        cache-aware inference tooling slices it as ``cache_last_time[:, slot_ids, :, :]``,
        so a 3-D placeholder breaks ``asr_streaming_infer.py`` at runtime.
        """
        d_model, n_layers, left, B = 64, 3, 7, 2
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=d_model, n_heads=4, n_layers=n_layers, att_context_size=[left, 0]
        )
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=B)
        assert clc.shape == (n_layers, B, left, d_model)
        assert clt.shape == (n_layers, B, d_model, 0)
        assert clt.dim() == 4, "cache_last_time must be 4-D for parity with ConformerEncoder"
        assert clcl.shape == (B,)
        assert clcl.sum().item() == 0
        # A full cache (left < 0) starts empty and grows.
        enc.set_default_att_context_size([-1, 0])
        clc_full, _, _ = enc.get_initial_cache_state(batch_size=B)
        assert clc_full.shape == (n_layers, B, 0, d_model)

    @pytest.mark.unit
    def test_default_att_context_size_is_full(self):
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        assert enc.att_context_size == [-1, -1]

    @pytest.mark.unit
    def test_set_default_att_context_size(self):
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_size=[70, 1])
        assert enc.att_context_size == [70, 1]
        # StreamingSTTModel retunes the look-ahead per chunk by reassigning this attribute.
        enc.set_default_att_context_size([70, 5])
        assert enc.att_context_size == [70, 5]

    @pytest.mark.unit
    def test_invalid_att_context_size_raises(self):
        with pytest.raises(ValueError, match=r"\[left, right\] pair"):
            StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_size=[1, 2, 3])

    @pytest.mark.unit
    def test_attn_mode_kwarg_is_ignored(self):
        """Unlike the base (which rejects it), the streaming encoder swallows ``attn_mode``."""
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=64, n_heads=4, n_layers=2, attn_mode="sliding_window", att_context_size=[3, 0]
        )
        assert enc.attn_mode == "sliding_window"
        assert enc.att_context_size == [3, 0]

    @pytest.mark.unit
    def test_forward_cpu_shape(self):
        enc = StreamingTransformerEncoder(
            feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, att_context_size=[10, 1]
        )
        enc.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, out_lengths = enc(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths.tolist() == [T // 4, 160 // 4]
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_sliding_window_limits_receptive_field(self):
        """With a single layer and window [1, 1], output at frame i must be invariant to
        input changes outside [i - 1, i + 1]. Uses ``bypass_pre_encode`` + ``no_pos`` so
        frame indices map 1:1 and no positional mixing obscures the receptive field."""
        B, T, d_model = 1, 8, 64
        enc = StreamingTransformerEncoder(
            feat_in=d_model,
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="no_pos",
            att_context_size=[1, 1],
        )
        enc.eval()

        x = torch.randn(B, T, d_model)
        length = torch.tensor([T], dtype=torch.int64)
        x_perturbed = x.clone()
        x_perturbed[:, 5, :] = torch.randn(d_model)  # change only frame 5

        with torch.no_grad():
            out, _ = enc(audio_signal=x, length=length, bypass_pre_encode=True)
            out_perturbed, _ = enc(audio_signal=x_perturbed, length=length, bypass_pre_encode=True)

        # out is (B, d_model, T). Frame 2's window {1, 2, 3} excludes frame 5 -> unchanged.
        assert torch.allclose(out[:, :, 2], out_perturbed[:, :, 2], atol=1e-6)
        # Frame 4's window {3, 4, 5} includes frame 5 -> must change.
        assert not torch.allclose(out[:, :, 4], out_perturbed[:, :, 4], atol=1e-6)

    @pytest.mark.unit
    def test_full_window_attends_everywhere(self):
        """Contrast to the windowed case: att_context_size [-1, -1] is full attention, so a
        distant perturbation *does* reach every output frame."""
        B, T, d_model = 1, 8, 64
        enc = StreamingTransformerEncoder(
            feat_in=d_model,
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="no_pos",
            att_context_size=[-1, -1],
        )
        enc.eval()

        x = torch.randn(B, T, d_model)
        length = torch.tensor([T], dtype=torch.int64)
        x_perturbed = x.clone()
        x_perturbed[:, 5, :] = torch.randn(d_model)

        with torch.no_grad():
            out, _ = enc(audio_signal=x, length=length, bypass_pre_encode=True)
            out_perturbed, _ = enc(audio_signal=x_perturbed, length=length, bypass_pre_encode=True)

        assert not torch.allclose(out[:, :, 2], out_perturbed[:, :, 2], atol=1e-6)

    @staticmethod
    def _stream_sequence(enc, x, chunk_len, n_chunks, batch_size, bypass_pre_encode):
        """Feed ``x`` through the cache-aware streaming path chunk by chunk and return the
        concatenated encoder output ``(B, D, T')``. ``x`` is ``(B, T, d_model)`` when
        ``bypass_pre_encode`` else ``(B, feat_in, T_in)``; ``chunk_len`` is in input frames."""
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=batch_size, dtype=x.dtype, device=x.device)
        outs = []
        for c in range(n_chunks):
            if bypass_pre_encode:
                chunk = x[:, c * chunk_len : (c + 1) * chunk_len, :]
            else:
                chunk = x[:, :, c * chunk_len : (c + 1) * chunk_len]
            chunk_frames = chunk.shape[1] if bypass_pre_encode else chunk.shape[2]
            clen = torch.tensor([chunk_frames] * batch_size, dtype=torch.int64)
            enc_out, _, clc, clt, clcl = enc.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=clen,
                cache_last_channel=clc,
                cache_last_time=clt,
                cache_last_channel_len=clcl,
                bypass_pre_encode=bypass_pre_encode,
            )
            outs.append(enc_out)
        return torch.cat(outs, dim=2), clcl

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sam, left, chunk, n_chunks, batch, bypass",
        [
            ("rel_pos", 3, 2, 4, 1, True),  # warm-up: cache fills over several chunks (chunk < left)
            ("rel_pos", 2, 4, 4, 1, True),  # rolling: chunk > left
            ("rel_pos", 3, 2, 4, 2, True),  # batched
            ("no_pos", 3, 2, 4, 1, True),  # no positional encoding
            ("rope", 3, 2, 4, 1, True),  # rotary position embedding (warm-up)
            ("rope", 2, 4, 4, 1, True),  # rotary position embedding (rolling)
            ("rope", 3, 2, 4, 1, False),  # rope with FeatureStacking subsampling
            ("rel_pos", -1, 2, 4, 1, True),  # full (unbounded) cache
            ("rel_pos", 3, 2, 4, 1, False),  # with FeatureStacking subsampling (aligned chunks)
        ],
    )
    def test_streaming_matches_full_forward(self, sam, left, chunk, n_chunks, batch, bypass):
        """The defining guarantee: causal chunk-by-chunk streaming with the KV cache must equal the
        full-sequence causal forward. Requires ``right == 0`` (a frame never needs a future frame)."""
        torch.manual_seed(0)
        d_model, sub, feat_in = 64, 4, 32
        enc = StreamingTransformerEncoder(
            feat_in=feat_in if not bypass else d_model,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model=sam,
            subsampling_factor=sub,
            att_context_size=[left, 0],
        )
        enc.eval()

        if bypass:
            x = torch.randn(batch, chunk * n_chunks, d_model)
            chunk_len = chunk
            in_len = x.shape[1]
        else:
            # ``chunk`` encoder frames == ``chunk * sub`` input frames; keep chunks aligned to the
            # subsampling factor so each chunk subsamples independently (matches the full forward).
            x = torch.randn(batch, feat_in, chunk * sub * n_chunks)
            chunk_len = chunk * sub
            in_len = x.shape[2]
        lengths = torch.tensor([in_len] * batch, dtype=torch.int64)

        with torch.no_grad():
            full_out, _ = enc(audio_signal=x, length=lengths, bypass_pre_encode=bypass)
            stream_out, final_valid_len = self._stream_sequence(enc, x, chunk_len, n_chunks, batch, bypass)

        assert stream_out.shape == full_out.shape
        assert torch.allclose(full_out, stream_out, atol=1e-5)
        # Valid cache length saturates at ``left`` for a rolling cache (whole utterance for full).
        expected = chunk * n_chunks if left < 0 else min(chunk * n_chunks, left)
        assert final_valid_len.tolist() == [expected] * batch

    # ------------------------------------------------------------------ #
    # chunked_limited (chunk-aligned look-ahead)
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_invalid_att_context_style_raises(self):
        with pytest.raises(ValueError, match="att_context_style"):
            StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_style="bogus")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (70, 13, 70),  # chunk 14 divides 70 exactly -> full left context
            (70, 27, 56),  # chunk 28 -> 2 whole chunks, left quantized down from 70
            (70, 3, 68),  # chunk 4 -> 17 whole chunks
            (2, 13, 0),  # left smaller than one chunk -> chunk attends to itself only
            (-1, 13, -1),  # unbounded left -> full (growing) cache
        ],
    )
    def test_chunked_limited_cache_size_quantizes_to_chunks(self, left, right, expected):
        """The rolling cache holds whole chunks only, so it matches what the mask can actually see."""
        enc = StreamingTransformerEncoder(
            feat_in=80,
            d_model=64,
            n_heads=4,
            n_layers=2,
            att_context_style="chunked_limited",
            att_context_size=[left, right],
        )
        assert enc.cache_size == expected
        clc, _, _ = enc.get_initial_cache_state(batch_size=2)
        assert clc.shape[2] == max(expected, 0)
        # ``streaming_cfg`` reports a concrete size, substituting ``max_context`` for an unbounded
        # left context (Conformer parity); ``cache_size < 0`` stays the encoder's "grow" marker.
        if expected >= 0:
            assert enc.streaming_cfg.last_channel_cache_size == expected
        else:
            assert enc.streaming_cfg.last_channel_cache_size == 10000

    @pytest.mark.unit
    def test_chunked_limited_lookahead_does_not_compound_across_layers(self):
        """The property that makes chunk-aligned look-ahead streamable.

        A sliding ``right`` window lets each layer peek one window further ahead, so an N-layer
        stack sees ``N * right`` frames of future — unreproducible chunk-by-chunk. Chunk-aligned
        attention stops every layer at the same boundary, so the look-ahead stays inside the
        current chunk no matter how deep the stack.
        """
        n_layers, chunk, sub, feat_in = 4, 8, 4, 32
        right = chunk - 1

        def first_affected_frame(style):
            torch.manual_seed(0)
            enc = StreamingTransformerEncoder(
                feat_in=feat_in,
                d_model=64,
                n_heads=4,
                n_layers=n_layers,
                drop_rate=0.0,
                dropout_pre_encoder=0.0,
                dropout_emb=0.0,
                self_attention_model="rope",
                subsampling_factor=sub,
                att_context_style=style,
                att_context_size=[24, right],
            ).eval()
            T = 16 * chunk * sub
            x = torch.randn(1, feat_in, T)
            lengths = torch.tensor([T], dtype=torch.int64)
            perturb_from = 8 * chunk  # encoder frame index — the start of chunk 8
            x2 = x.clone()
            x2[:, :, perturb_from * sub :] = torch.randn_like(x2[:, :, perturb_from * sub :])
            with torch.no_grad():
                a, _ = enc(audio_signal=x, length=lengths)
                b, _ = enc(audio_signal=x2, length=lengths)
            diff = (a - b).abs().mean(dim=1)[0]
            return perturb_from - int((diff > 1e-6).nonzero()[0])

        # Chunk-aligned: only the frames of the perturbed frame's own chunk can see it, and the
        # perturbation starts on a chunk boundary — so nothing before it moves at all.
        assert first_affected_frame("chunked_limited") == 0
        # Sliding window with the same ``right``: the reach compounds well past one chunk.
        assert first_affected_frame("sliding_window") > chunk

    @pytest.mark.unit
    @pytest.mark.parametrize("sam", ["rope", "rel_pos", "no_pos"])
    @pytest.mark.parametrize("left, chunk, n_chunks", [(24, 8, 5), (6, 4, 6), (3, 8, 4)])
    def test_chunked_limited_streaming_matches_full_forward(self, sam, left, chunk, n_chunks):
        """The payoff: with chunk-aligned look-ahead, chunk-by-chunk streaming is still exact.

        Contrast with :meth:`test_streaming_matches_full_forward`, which requires ``right == 0``.
        Here ``right = chunk - 1`` gives a full chunk of look-ahead and streaming still reproduces
        the full forward, because the current chunk is entirely present in the KV span.
        """
        torch.manual_seed(0)
        d_model, sub, feat_in = 64, 4, 32
        enc = StreamingTransformerEncoder(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=4,
            n_layers=3,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model=sam,
            subsampling_factor=sub,
            att_context_style="chunked_limited",
            att_context_size=[left, chunk - 1],
        )
        enc.eval()

        x = torch.randn(1, feat_in, chunk * sub * n_chunks)
        lengths = torch.tensor([x.shape[2]], dtype=torch.int64)

        with torch.no_grad():
            full_out, _ = enc(audio_signal=x, length=lengths)
            stream_out, final_valid_len = self._stream_sequence(enc, x, chunk * sub, n_chunks, 1, False)

        assert stream_out.shape == full_out.shape
        assert torch.allclose(full_out, stream_out, atol=1e-5)
        assert final_valid_len.tolist() == [min(chunk * n_chunks, enc.cache_size)]

    @pytest.mark.unit
    def test_chunked_limited_streaming_ragged_batch_matches_full_forward(self):
        """Batched streaming stays exact over every stream's valid frames when lengths differ."""
        torch.manual_seed(0)
        sub, feat_in, chunk, n_chunks = 4, 32, 8, 5
        enc = StreamingTransformerEncoder(
            feat_in=feat_in,
            d_model=64,
            n_heads=4,
            n_layers=3,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            subsampling_factor=sub,
            att_context_style="chunked_limited",
            att_context_size=[16, chunk - 1],
        )
        enc.eval()

        valid_enc = [chunk * n_chunks, 3 * chunk + 5]  # stream 1 ends mid-chunk
        x = torch.randn(2, feat_in, chunk * sub * n_chunks)
        x[1, :, valid_enc[1] * sub :] = 0.0  # zero-pad the tail, as streaming inference does
        lengths = torch.tensor([v * sub for v in valid_enc], dtype=torch.int64)

        with torch.no_grad():
            full_out, _ = enc(audio_signal=x, length=lengths)
            clc, clt, clcl = enc.get_initial_cache_state(batch_size=2)
            outs = []
            for c in range(n_chunks):
                step = chunk * sub
                chunk_lens = torch.tensor([max(0, min(step, int(l) - c * step)) for l in lengths], dtype=torch.int64)
                out, _, clc, clt, clcl = enc.cache_aware_stream_step(
                    processed_signal=x[:, :, c * step : (c + 1) * step],
                    processed_signal_length=chunk_lens,
                    cache_last_channel=clc,
                    cache_last_time=clt,
                    cache_last_channel_len=clcl,
                )
                outs.append(out)
            stream_out = torch.cat(outs, dim=2)

        # Frames past a stream's end are discarded by the caller, so only valid frames must match.
        for b, v in enumerate(valid_enc):
            assert torch.allclose(full_out[b, :, :v], stream_out[b, :, :v], atol=1e-5)

    @pytest.mark.unit
    def test_chunked_limited_rejects_oversized_streaming_chunk(self):
        """A streaming step wider than one attention chunk would silently break exactness."""
        enc = StreamingTransformerEncoder(
            feat_in=64,
            d_model=64,
            n_heads=4,
            n_layers=1,
            att_context_style="chunked_limited",
            att_context_size=[8, 3],  # chunk_size = 4
        )
        enc.eval()
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=1)
        with torch.no_grad(), pytest.raises(ValueError, match="chunked_limited streaming"):
            enc.cache_aware_stream_step(
                processed_signal=torch.randn(1, 5, 64),  # 5 > chunk_size 4
                processed_signal_length=torch.tensor([5]),
                cache_last_channel=clc,
                cache_last_time=clt,
                cache_last_channel_len=clcl,
                bypass_pre_encode=True,
            )

    # ------------------------------------------------------------------ #
    # multi chunk-size training (att_context_size as a list of pairs)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _multi_ctx_encoder(att_context_size, att_context_probs=None, n_layers=2, **kw):
        return StreamingTransformerEncoder(
            feat_in=128,
            feat_out=-1,
            d_model=256,
            n_heads=8,
            n_layers=n_layers,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            qk_norm=True,
            subsampling="feature_stacking",
            subsampling_factor=8,
            self_attention_model="rope",
            att_context_style="chunked_limited",
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            **kw,
        )

    @pytest.mark.unit
    def test_multi_att_context_size_parsed(self):
        """A list of [left, right] pairs configures multi chunk-size training."""
        enc = self._multi_ctx_encoder([[70, 1], [70, 6], [70, 13]])
        assert enc.att_context_size_all == [[70, 1], [70, 6], [70, 13]]
        assert enc.att_context_probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])
        # Eval/streaming default is the first entry, as in ConformerEncoder.
        assert enc.att_context_size == [70, 1]

    @pytest.mark.unit
    def test_single_pair_still_accepted(self):
        """Backward compatible: a bare pair becomes a one-entry list and disables sampling."""
        enc = self._multi_ctx_encoder([70, 13])
        assert enc.att_context_size_all == [[70, 13]]
        assert enc.att_context_size == [70, 13]
        enc.train()
        assert {tuple(enc._sample_att_context_size()) for _ in range(20)} == {(70, 13)}

    @pytest.mark.unit
    def test_sampling_only_in_training(self):
        """Training samples per batch; eval is deterministic so val loss and streaming are stable."""
        enc = self._multi_ctx_encoder([[70, 1], [70, 6], [70, 13]])
        enc.train()
        seen = {tuple(enc._sample_att_context_size()) for _ in range(200)}
        assert seen == {(70, 1), (70, 6), (70, 13)}, f"not all contexts sampled: {seen}"
        enc.eval()
        assert {tuple(enc._sample_att_context_size()) for _ in range(50)} == {(70, 1)}

    @pytest.mark.unit
    def test_att_context_probs_respected(self):
        """A degenerate distribution must pin the corresponding context."""
        enc = self._multi_ctx_encoder([[70, 1], [70, 13]], att_context_probs=[0.0, 1.0])
        enc.train()
        assert {tuple(enc._sample_att_context_size()) for _ in range(100)} == {(70, 13)}

    @pytest.mark.unit
    def test_invalid_att_context_probs_raise(self):
        with pytest.raises(ValueError, match="att_context_probs has"):
            self._multi_ctx_encoder([[70, 1], [70, 13]], att_context_probs=[1.0])
        with pytest.raises(ValueError, match="must sum to 1"):
            self._multi_ctx_encoder([[70, 1], [70, 13]], att_context_probs=[0.3, 0.3])

    @pytest.mark.unit
    def test_malformed_att_context_size_raises(self):
        with pytest.raises(ValueError, match=r"\[left, right\] pairs"):
            self._multi_ctx_encoder([[70, 1, 2], [70, 13]])
        with pytest.raises(ValueError, match="must not be empty"):
            self._multi_ctx_encoder([])

    @pytest.mark.unit
    def test_set_default_att_context_size_pins_eval_context_only(self):
        """How inference selects a chunk size — and what pinning does NOT do.

        Pinning fixes the eval/streaming context. Training keeps sampling over the configured set
        (matching ConformerEncoder), so pinning never silently narrows a training run.
        """
        enc = self._multi_ctx_encoder([[70, 1], [70, 6], [70, 13]])
        enc.set_default_att_context_size([70, 13])
        assert enc.att_context_size == [70, 13]
        assert enc.streaming_cfg.chunk_size == 8 * 14  # input frames for a 14-frame chunk
        enc.eval()
        assert {tuple(enc._sample_att_context_size()) for _ in range(20)} == {(70, 13)}
        enc.train()
        assert enc.att_context_size_all == [[70, 1], [70, 6], [70, 13]]
        assert len({tuple(enc._sample_att_context_size()) for _ in range(200)}) == 3

    @pytest.mark.unit
    @pytest.mark.parametrize("context", [[70, 1], [70, 6], [70, 13]])
    def test_every_configured_context_streams_exactly(self, context):
        """The guarantee that makes multi chunk-size training usable: each context a batch may be
        trained at must also stream exactly, so one checkpoint serves every chunk size."""
        torch.manual_seed(0)
        sub, n_chunks = 8, 5
        enc = self._multi_ctx_encoder([[70, 1], [70, 6], [70, 13]], n_layers=3)
        enc.eval()
        enc.set_default_att_context_size(context)
        chunk = context[1] + 1
        x = torch.randn(1, 128, chunk * sub * n_chunks)
        lengths = torch.tensor([x.shape[2]], dtype=torch.int64)
        with torch.no_grad():
            full_out, _ = enc(audio_signal=x, length=lengths)
            stream_out, _ = self._stream_sequence(enc, x, chunk * sub, n_chunks, 1, False)
        assert stream_out.shape == full_out.shape
        assert torch.allclose(full_out, stream_out, atol=1e-5)

    # ------------------------------------------------------------------ #
    # streaming_cfg / pre-encode cache (shared cache-aware tooling contract)
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_streaming_cfg_exposes_full_cache_aware_config(self):
        """``speech_to_text_cache_aware_streaming_infer.py`` and ``CacheAwareStreamingAudioBuffer``
        read fields off ``streaming_cfg`` directly; a missing one is an AttributeError mid-stream."""
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=64, n_heads=4, n_layers=2, subsampling_factor=8, att_context_size=[70, 0]
        )
        for field in dataclasses.fields(CacheAwareStreamingConfig):
            assert hasattr(enc.streaming_cfg, field.name), f"streaming_cfg is missing {field.name}"
        # Units follow the Conformer: chunk/shift in input frames, valid_out_len in encoder frames.
        assert enc.streaming_cfg.chunk_size == 8 * enc.streaming_cfg.valid_out_len
        assert enc.streaming_cfg.shift_size == enc.streaming_cfg.chunk_size
        assert enc.streaming_cfg.cache_drop_size == 0
        assert enc.streaming_cfg.last_channel_num == 2
        assert enc.streaming_cfg.last_time_num == 0

    @pytest.mark.unit
    def test_setup_streaming_params_honors_chunk_and_left_chunks(self):
        """The overrides the infer script passes for offline/full-context models must take effect."""
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=64, n_heads=4, n_layers=2, subsampling_factor=8, att_context_size=[70, 0]
        )
        # Causal sliding window implies a 1-frame step; the script can widen it for throughput.
        assert enc.streaming_cfg.chunk_size == 8
        enc.setup_streaming_params(chunk_size=14, shift_size=14)
        assert enc.streaming_cfg.chunk_size == 8 * 14
        assert enc.streaming_cfg.valid_out_len == 14
        # left_chunks retunes the window itself, so mask and cache cannot disagree.
        configured_contexts = [list(context) for context in enc.att_context_size_all]
        enc.setup_streaming_params(chunk_size=14, shift_size=14, left_chunks=3)
        assert enc.att_context_size[0] == 42
        assert enc.att_context_size_all == configured_contexts
        assert enc.streaming_cfg.last_channel_cache_size == 42 == enc.cache_size

    @pytest.mark.unit
    def test_setup_streaming_params_rejects_unsupported_overrides(self):
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=64, n_heads=4, n_layers=2, subsampling_factor=8, att_context_size=[70, 0]
        )
        with pytest.raises(ValueError, match="shift_size"):
            enc.setup_streaming_params(chunk_size=14, shift_size=7)  # overlapping chunks
        enc2 = StreamingTransformerEncoder(
            feat_in=80,
            d_model=64,
            n_heads=4,
            n_layers=2,
            subsampling_factor=8,
            att_context_style="chunked_limited",
            att_context_size=[8, 3],
        )
        for chunk_size in (2, 5):
            with pytest.raises(ValueError, match="must equal the chunked_limited attention chunk"):
                enc2.setup_streaming_params(chunk_size=chunk_size)

    @pytest.mark.unit
    def test_pre_encode_cache_size_must_align_to_subsampling(self):
        with pytest.raises(ValueError, match="pre_encode_cache_size"):
            StreamingTransformerEncoder(
                feat_in=80, d_model=64, n_heads=4, n_layers=2, subsampling_factor=8, pre_encode_cache_size=5
            )

    @pytest.mark.unit
    def test_pre_encode_cache_size_derives_drop_extra_and_step0_list(self):
        """``pre_encode_cache_size`` is emitted as ``[0, P]`` so the streaming buffer prepends
        nothing on the first step, matching ``calc_drop_extra_pre_encoded`` returning 0 there."""
        enc = StreamingTransformerEncoder(
            feat_in=80,
            d_model=64,
            n_heads=4,
            n_layers=2,
            subsampling_factor=8,
            pre_encode_cache_size=16,
            att_context_size=[70, 0],
        )
        assert enc.streaming_cfg.pre_encode_cache_size == [0, 16]
        assert enc.streaming_cfg.drop_extra_pre_encoded == 2  # 16 input frames / 8 = 2 encoder frames
        # Default (no look-back) stays a plain scalar 0, unchanged from before.
        plain = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, subsampling_factor=8)
        assert plain.streaming_cfg.pre_encode_cache_size == 0
        assert plain.streaming_cfg.drop_extra_pre_encoded == 0

    @pytest.mark.unit
    def test_no_cache_means_no_implicit_drop(self):
        """A caller with no cache is passing a whole utterance, not a chunk with look-back.

        ``conformer_stream_step(processed_signal=<whole audio>)`` — which the cache-aware infer
        script uses for ``compare_vs_offline`` — reaches the encoder with ``cache_last_channel=None``
        and ``drop_extra_pre_encoded=None``. Defaulting to the configured drop there would silently
        truncate the offline reference it is meant to compare against.
        """
        sub, feat_in, frames = 8, 32, 6
        enc = StreamingTransformerEncoder(
            feat_in=feat_in,
            d_model=64,
            n_heads=4,
            n_layers=2,
            self_attention_model="rope",
            subsampling_factor=sub,
            att_context_size=[16, 0],
            pre_encode_cache_size=sub,  # -> drop_extra_pre_encoded == 1
        )
        enc.eval()
        assert enc.streaming_cfg.drop_extra_pre_encoded == 1
        x = torch.randn(1, feat_in, frames * sub)
        with torch.no_grad():
            out, out_len, *_ = enc.cache_aware_stream_step(
                processed_signal=x,
                processed_signal_length=torch.tensor([x.shape[2]], dtype=torch.int64),
                cache_last_channel=None,  # no cache -> whole-utterance call
            )
        assert out.shape[2] == frames, "whole-utterance call must not lose leading frames"
        assert out_len.tolist() == [frames]
        # An explicit value is still honoured regardless of the cache state.
        with torch.no_grad():
            out2, _, *_ = enc.cache_aware_stream_step(
                processed_signal=x,
                processed_signal_length=torch.tensor([x.shape[2]], dtype=torch.int64),
                cache_last_channel=None,
                drop_extra_pre_encoded=1,
            )
        assert out2.shape[2] == frames - 1

    @pytest.mark.unit
    @pytest.mark.parametrize("pre_encode_cache_size", [8, 16])
    def test_drop_extra_pre_encoded_makes_lookback_chunks_exact(self, pre_encode_cache_size):
        """Feeding ``[look-back | chunk]`` and dropping the extra pre-encoded frames must return the
        same frames — and the same count — as feeding the bare chunk.

        This mirrors ``CacheAwareStreamingAudioBuffer``: step 0 gets no look-back and drops nothing;
        later steps prepend the real preceding input frames and drop ``drop_extra_pre_encoded``.
        """
        torch.manual_seed(0)
        sub, feat_in, chunk, n_chunks = 8, 32, 4, 5
        enc = StreamingTransformerEncoder(
            feat_in=feat_in,
            d_model=64,
            n_heads=4,
            n_layers=3,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            subsampling_factor=sub,
            att_context_size=[16, 0],
            pre_encode_cache_size=pre_encode_cache_size,
        )
        enc.eval()
        step = chunk * sub
        x = torch.randn(1, feat_in, step * n_chunks)
        lengths = torch.tensor([x.shape[2]], dtype=torch.int64)
        drop = enc.streaming_cfg.drop_extra_pre_encoded

        with torch.no_grad():
            full_out, _ = enc(audio_signal=x, length=lengths)
            clc, clt, clcl = enc.get_initial_cache_state(batch_size=1)
            outs = []
            for c in range(n_chunks):
                start = c * step
                # Step 0 has no history: no look-back, nothing to drop (calc_drop_extra_pre_encoded).
                back = 0 if c == 0 else pre_encode_cache_size
                piece = x[:, :, start - back : start + step]
                out, out_len, clc, clt, clcl = enc.cache_aware_stream_step(
                    processed_signal=piece,
                    processed_signal_length=torch.tensor([piece.shape[2]], dtype=torch.int64),
                    cache_last_channel=clc,
                    cache_last_time=clt,
                    cache_last_channel_len=clcl,
                    drop_extra_pre_encoded=0 if c == 0 else drop,
                )
                assert out.shape[2] == chunk, f"chunk {c}: got {out.shape[2]} frames, expected {chunk}"
                assert out_len.tolist() == [chunk]
                outs.append(out)
            stream_out = torch.cat(outs, dim=2)

        assert stream_out.shape == full_out.shape
        assert torch.allclose(full_out, stream_out, atol=1e-5)

    @pytest.mark.unit
    def test_streaming_abs_pos_not_implemented(self):
        """``abs_pos`` streaming is intentionally unsupported (no position offset in the cache)."""
        enc = StreamingTransformerEncoder(
            feat_in=64, d_model=64, n_heads=4, n_layers=1, self_attention_model="abs_pos", att_context_size=[3, 0]
        )
        enc.eval()
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=1)
        with torch.no_grad(), pytest.raises(NotImplementedError, match="Cache-aware streaming"):
            enc.cache_aware_stream_step(
                processed_signal=torch.randn(1, 2, 64),
                processed_signal_length=torch.tensor([2]),
                cache_last_channel=clc,
                cache_last_time=clt,
                cache_last_channel_len=clcl,
                bypass_pre_encode=True,
            )
