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
        assert cfg.feat_in == 80
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

    @pytest.mark.unit
    def test_custom_config(self):
        cfg = TransformerEncoderConfig(feat_in=128, d_model=1280, n_heads=16, n_layers=32, qk_norm=True)
        assert cfg.feat_in == 128
        assert cfg.d_model == 1280
        assert cfg.n_heads == 16
        assert cfg.n_layers == 32
        assert cfg.qk_norm is True


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


class TestTransformerEncoder:
    @pytest.mark.unit
    def test_model_creation(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert len(model.layers) == 2

    @pytest.mark.unit
    def test_model_creation_with_qk_norm(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, qk_norm=True)
        attn = model.layers[0].attn
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    @pytest.mark.unit
    def test_model_creation_without_qk_norm(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, qk_norm=False)
        attn = model.layers[0].attn
        assert not hasattr(attn, 'q_norm')
        assert not hasattr(attn, 'k_norm')

    @pytest.mark.unit
    def test_invalid_attn_mode(self):
        with pytest.raises(ValueError, match="not yet supported"):
            TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, attn_mode="sliding_window")

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
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, subsampling_factor=4)
        model.eval()

        B, C, T = 2, 80, 400
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400, 300])

        with torch.no_grad():
            out, out_lengths = model(x, lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 300 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_forward_cpu_with_qk_norm(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, qk_norm=True)
        model.eval()

        x = torch.randn(1, 80, 200)
        lengths = torch.tensor([200])

        with torch.no_grad():
            out, _ = model(x, lengths)

        assert out.shape == (1, 64, 50)
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_basic(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, subsampling_factor=4)
        model = model.cuda().to(torch.bfloat16)

        B, C, T = 2, 80, 400
        x = torch.randn(B, C, T, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([400, 300], device='cuda')

        model.eval()
        with torch.no_grad():
            out, out_lengths = model(x, lengths)

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
            out, out_lengths = model(x, lengths)

        assert out.shape == (B, 128, T // 8)
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_output_channels_first(self):
        """Verify output is (B, D, T) channels-first as expected by downstream decoders."""
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=1, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16)

        x = torch.randn(1, 80, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200], device='cuda')

        model.eval()
        with torch.no_grad():
            out, _ = model(x, lengths)

        assert out.shape[1] == 64  # D dimension
        assert out.shape[2] == 200 // 4  # T dimension

    @pytest.mark.run_only_on('GPU')
    def test_eval_deterministic(self):
        """In eval mode with no dropout, repeated forward passes should produce identical output."""
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).eval()

        x = torch.randn(1, 80, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200], device='cuda')

        with torch.no_grad():
            out1, _ = model(x, lengths)
            out2, _ = model(x, lengths)

        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.run_only_on('GPU')
    def test_padding_does_not_affect_valid_output(self):
        """Padding frames should not change the encoded output at valid positions."""
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).eval()

        T_valid = 200
        x_short = torch.randn(1, 80, T_valid, device='cuda', dtype=torch.bfloat16)
        lengths_short = torch.tensor([T_valid], device='cuda')

        T_padded = 400
        x_long = torch.zeros(1, 80, T_padded, device='cuda', dtype=torch.bfloat16)
        x_long[:, :, :T_valid] = x_short
        lengths_long = torch.tensor([T_valid], device='cuda')

        with torch.no_grad():
            out_short, len_short = model(x_short, lengths_short)
            out_long, len_long = model(x_long, lengths_long)

        assert len_short[0].item() == len_long[0].item()
        valid_t = len_short[0].item()
        # bf16 + different block mask shapes cause small numerical differences in Triton kernels
        assert torch.allclose(out_short[:, :, :valid_t], out_long[:, :, :valid_t], atol=5e-2)

    @pytest.mark.run_only_on('GPU')
    def test_backward_pass(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).train()

        x = torch.randn(2, 80, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200, 160], device='cuda')

        out, out_lengths = model(x, lengths)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
