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

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import RoPEMultiHeadAttention, RotaryPositionalEncoding


def _build_encoder(
    self_attention_model='rope',
    n_layers=2,
    d_model=64,
    n_heads=4,
    use_pytorch_sdpa=False,
    use_pytorch_sdpa_backends=None,
    rotary_fraction=1.0,
    rope_base=10000.0,
    pos_emb_max_len=256,
):
    return ConformerEncoder(
        feat_in=80,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        self_attention_model=self_attention_model,
        subsampling_factor=4,
        subsampling_conv_channels=32,
        pos_emb_max_len=pos_emb_max_len,
        rotary_fraction=rotary_fraction,
        rope_base=rope_base,
        use_pytorch_sdpa=use_pytorch_sdpa,
        use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        dropout=0.0,
        dropout_att=0.0,
        dropout_emb=0.0,
        dropout_pre_encoder=0.0,
    ).eval()


class TestRotaryPositionalEncoding:
    @pytest.mark.unit
    def test_rejects_invalid_rotary_fraction(self):
        with pytest.raises(ValueError):
            RotaryPositionalEncoding(d_k=16, rotary_fraction=0.0)
        with pytest.raises(ValueError):
            RotaryPositionalEncoding(d_k=16, rotary_fraction=1.5)

    @pytest.mark.unit
    def test_rejects_odd_effective_dim(self):
        # d_k * rotary_fraction = 16 * 0.1875 = 3, which is odd
        with pytest.raises(ValueError):
            RotaryPositionalEncoding(d_k=16, rotary_fraction=0.1875)

    @pytest.mark.unit
    def test_extend_pe_grows_buffers(self):
        pe = RotaryPositionalEncoding(d_k=16, max_len=128)
        pe.extend_pe(64, device=torch.device('cpu'), dtype=torch.float32)
        assert pe.cos.shape == (64, 16)
        pe.extend_pe(128, device=torch.device('cpu'), dtype=torch.float32)
        assert pe.cos.shape == (128, 16)
        # No-op when buffer is already large enough.
        prev = pe.cos.data_ptr()
        pe.extend_pe(64, device=torch.device('cpu'), dtype=torch.float32)
        assert pe.cos.data_ptr() == prev

    @pytest.mark.unit
    def test_forward_first_token_is_identity(self):
        # Position 0 has zero phase, so cos=1, sin=0 -> rotation is identity.
        pe = RotaryPositionalEncoding(d_k=16, rotary_fraction=1.0)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        q_rot, k_rot = pe(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert torch.allclose(q_rot[:, :, 0, :], q[:, :, 0, :], atol=1e-6)
        assert torch.allclose(k_rot[:, :, 0, :], k[:, :, 0, :], atol=1e-6)

    @pytest.mark.unit
    def test_partial_rotation_leaves_tail_unchanged(self):
        pe = RotaryPositionalEncoding(d_k=16, rotary_fraction=0.5)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        q_rot, k_rot = pe(q, k)
        # The last (d_k - d_k_rot) = 8 dims of each head must pass through untouched.
        assert torch.allclose(q_rot[..., pe.d_k_rot :], q[..., pe.d_k_rot :])
        assert torch.allclose(k_rot[..., pe.d_k_rot :], k[..., pe.d_k_rot :])

    @pytest.mark.unit
    def test_dot_product_translation_invariance(self):
        # The defining property of RoPE: for the same q and k content, <q_m, k_n>
        # depends only on the position difference (m - n). Pick two (m, n) pairs
        # that share the same difference and assert the dot products agree.
        pe = RotaryPositionalEncoding(d_k=16, rotary_fraction=1.0)
        pe.extend_pe(64, device=torch.device('cpu'), dtype=torch.float32)

        torch.manual_seed(0)
        q_content = torch.randn(1, 1, 1, 16)
        k_content = torch.randn(1, 1, 1, 16)

        def dot_at(m, n):
            cos_q = pe.cos[m : m + 1].view(1, 1, 1, 16)
            sin_q = pe.sin[m : m + 1].view(1, 1, 1, 16)
            cos_k = pe.cos[n : n + 1].view(1, 1, 1, 16)
            sin_k = pe.sin[n : n + 1].view(1, 1, 1, 16)
            q_r = pe._apply_rotary(q_content, cos_q, sin_q)
            k_r = pe._apply_rotary(k_content, cos_k, sin_k)
            return (q_r * k_r).sum()

        # Three (m, n) pairs with the same difference n - m = 3.
        d_a = dot_at(2, 5)
        d_b = dot_at(10, 13)
        d_c = dot_at(40, 43)
        assert torch.allclose(d_a, d_b, atol=1e-5)
        assert torch.allclose(d_a, d_c, atol=1e-5)

        # Sanity: a different position difference must yield a different dot product
        # (otherwise the rotation is a no-op or degenerate).
        d_diff = dot_at(2, 7)  # difference 5
        assert not torch.allclose(d_a, d_diff, atol=1e-3)

    @pytest.mark.unit
    def test_rotation_is_not_identity(self):
        # Confirm RoPE actually mutates Q/K at non-zero positions.
        pe = RotaryPositionalEncoding(d_k=16, rotary_fraction=1.0)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)
        q = torch.randn(1, 1, 8, 16)
        k = torch.randn(1, 1, 8, 16)
        q_rot, k_rot = pe(q, k)
        # Tokens after position 0 must change.
        assert not torch.allclose(q_rot[:, :, 1:, :], q[:, :, 1:, :], atol=1e-3)
        assert not torch.allclose(k_rot[:, :, 1:, :], k[:, :, 1:, :], atol=1e-3)

    @pytest.mark.unit
    def test_norm_preservation(self):
        # Rotation is unitary: ||q_rot[..., t, :]||_2 == ||q[..., t, :]||_2 per (batch, head, t).
        # Catches scaling bugs in _apply_rotary.
        pe = RotaryPositionalEncoding(d_k=16, rotary_fraction=1.0)
        pe.extend_pe(64, device=torch.device('cpu'), dtype=torch.float32)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        q_rot, k_rot = pe(q, k)
        q_norm_in = torch.linalg.norm(q, dim=-1)
        q_norm_out = torch.linalg.norm(q_rot, dim=-1)
        k_norm_in = torch.linalg.norm(k, dim=-1)
        k_norm_out = torch.linalg.norm(k_rot, dim=-1)
        assert torch.allclose(q_norm_in, q_norm_out, atol=1e-5)
        assert torch.allclose(k_norm_in, k_norm_out, atol=1e-5)

    @pytest.mark.unit
    def test_reference_equivalence(self):
        # Slow split-half RoPE reference written in explicit-2D-rotation form
        # (no _rotate_half trick, no cat-duplicated cos/sin). Same math as the
        # production code expressed via a disjoint code path, so a bug in either
        # _rotate_half or the cos/sin layout would surface here.
        d_k = 16
        pe = RotaryPositionalEncoding(d_k=d_k, rotary_fraction=1.0)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)

        torch.manual_seed(0)
        q = torch.randn(1, 1, 8, d_k)
        k = torch.randn(1, 1, 8, d_k)
        q_rot, k_rot = pe(q, k)

        d_half = d_k // 2
        positions = torch.arange(8, dtype=torch.float32)
        theta = positions[:, None] * pe.inv_freq[None, :]  # (T, d_half)
        c = theta.cos()
        s = theta.sin()

        def rope_ref(x):
            # Rotate each (x[..., i], x[..., i + d_half]) pair by angle theta[t, i].
            x_a = x[..., :d_half]
            x_b = x[..., d_half:]
            y_a = x_a * c - x_b * s
            y_b = x_a * s + x_b * c
            return torch.cat((y_a, y_b), dim=-1)

        assert torch.allclose(q_rot, rope_ref(q), atol=1e-6)
        assert torch.allclose(k_rot, rope_ref(k), atol=1e-6)

    @pytest.mark.unit
    def test_extend_preserves_existing_positions(self):
        # Extending the cos/sin buffers must not change the values at previously
        # covered positions, otherwise streaming forward calls would silently
        # produce different rotations across the extension boundary.
        pe = RotaryPositionalEncoding(d_k=16, max_len=64)
        pe.extend_pe(64, device=torch.device('cpu'), dtype=torch.float32)
        cos_before = pe.cos[:64].clone()
        sin_before = pe.sin[:64].clone()
        pe.extend_pe(256, device=torch.device('cpu'), dtype=torch.float32)
        assert torch.equal(pe.cos[:64], cos_before)
        assert torch.equal(pe.sin[:64], sin_before)

    @pytest.mark.unit
    def test_non_contiguous_inputs(self):
        # Real-world callers may pass non-contiguous Q/K (e.g. from .transpose()).
        # The rotation must produce the same result as on the contiguous version.
        pe = RotaryPositionalEncoding(d_k=16, rotary_fraction=1.0)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)

        # Build (B, T, H, D) and transpose to (B, H, T, D) -> non-contiguous.
        q_btnd = torch.randn(2, 8, 4, 16)
        k_btnd = torch.randn(2, 8, 4, 16)
        q_nc = q_btnd.transpose(1, 2)
        k_nc = k_btnd.transpose(1, 2)
        assert not q_nc.is_contiguous() and not k_nc.is_contiguous()

        q_rot_nc, k_rot_nc = pe(q_nc, k_nc)
        q_rot_c, k_rot_c = pe(q_nc.contiguous(), k_nc.contiguous())
        assert torch.allclose(q_rot_nc, q_rot_c, atol=1e-6)
        assert torch.allclose(k_rot_nc, k_rot_c, atol=1e-6)


class TestRoPEMultiHeadAttention:
    @pytest.mark.unit
    def test_rejects_pos_enc_with_wrong_d_k(self):
        # n_feat / n_head = 64 / 4 = 16, but pos_enc was built with d_k=32.
        bad_pe = RotaryPositionalEncoding(d_k=32, max_len=64)
        with pytest.raises(ValueError):
            RoPEMultiHeadAttention(n_head=4, n_feat=64, dropout_rate=0.0, pos_enc=bad_pe)

    @pytest.mark.unit
    def test_v_unchanged_by_rotation(self):
        # Confirm the rotation hook is called only with (q, k); V must never reach
        # the positional encoder. Catches a future regression where someone adds
        # V to the rotation hook signature.
        pe = RotaryPositionalEncoding(d_k=16, max_len=32)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)
        attn = RoPEMultiHeadAttention(n_head=4, n_feat=64, dropout_rate=0.0, pos_enc=pe).eval()

        call_args = []
        original_forward = pe.forward

        def spy(q, k):
            call_args.append((q.shape, k.shape))
            return original_forward(q, k)

        attn.pos_enc.forward = spy
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            _ = attn(query=x, key=x, value=x, mask=None)
        assert len(call_args) == 1
        q_shape, k_shape = call_args[0]
        # Both tensors have the same length (16); the layout is (B, H, T, d_k).
        assert q_shape == (2, 4, 16, 16)
        assert k_shape == (2, 4, 16, 16)

    @pytest.mark.unit
    def test_backward_smoke(self):
        # Forward → loss → backward → every learnable param has a non-NaN, non-zero
        # gradient. Mirrors test_transformer_encoder.py::test_backward_pass.
        pe = RotaryPositionalEncoding(d_k=16, max_len=32)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)
        attn = RoPEMultiHeadAttention(n_head=4, n_feat=64, dropout_rate=0.0, pos_enc=pe).train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(query=x, key=x, value=x, mask=None)
        loss = out.sum()
        loss.backward()
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert (param.grad != 0).any(), f"All-zero gradient for {name}"

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("backend", ['MATH', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION'])
    def test_sdpa_backend_smoke_gpu(self, backend):
        # Each SDPA backend must run with RoPE pre-rotation under bf16 autocast
        # (the production training path) without falling back or crashing on
        # shape/dtype constraints. FLASH/EFFICIENT/CUDNN require fp16/bf16;
        # bf16 satisfies all four.
        pe = RotaryPositionalEncoding(d_k=16, max_len=32).to("cuda")
        pe.extend_pe(32, device=torch.device('cuda'), dtype=torch.float32)
        attn = (
            RoPEMultiHeadAttention(
                n_head=4,
                n_feat=64,
                dropout_rate=0.0,
                pos_enc=pe,
                use_pytorch_sdpa=True,
                use_pytorch_sdpa_backends=[backend],
            )
            .to("cuda")
            .eval()
        )
        x = torch.randn(2, 16, 64, device='cuda')
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = attn(query=x, key=x, value=x, mask=None)
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_autocast_gpu(self):
        # Mixed-precision forward (CUDA autocast in bf16) must produce finite output.
        # Exercises the interaction between RoPE's .to(q.dtype) cast and the
        # avoid_float16_autocast_context wrapper in the base MHA.
        pe = RotaryPositionalEncoding(d_k=16, max_len=32).to("cuda")
        pe.extend_pe(32, device=torch.device('cuda'), dtype=torch.float32)
        attn = RoPEMultiHeadAttention(n_head=4, n_feat=64, dropout_rate=0.0, pos_enc=pe).to("cuda").eval()
        x = torch.randn(2, 16, 64, device='cuda')
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = attn(query=x, key=x, value=x, mask=None)
        assert torch.isfinite(out).all()

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "dtype,atol",
        [(torch.float32, 1e-5), (torch.bfloat16, 5e-2), (torch.float16, 1e-2)],
    )
    def test_dtype_stability_gpu(self, dtype, atol):
        # Forward in low precision must stay close to the fp32 reference.
        pe = RotaryPositionalEncoding(d_k=16, max_len=32).to("cuda")
        pe.extend_pe(32, device=torch.device('cuda'), dtype=torch.float32)
        attn = RoPEMultiHeadAttention(n_head=4, n_feat=64, dropout_rate=0.0, pos_enc=pe).to("cuda").eval()
        torch.manual_seed(0)
        x = torch.randn(2, 16, 64, device='cuda')
        with torch.no_grad():
            out_ref = attn(query=x, key=x, value=x, mask=None)

        # `attn.to(dtype=...)` converts every buffer including pos_enc.cos/sin to `dtype`.
        attn_dt = attn.to(dtype=dtype)
        x_dt = x.to(dtype=dtype)
        with torch.no_grad():
            out_dt = attn_dt(query=x_dt, key=x_dt, value=x_dt, mask=None)
        assert torch.isfinite(out_dt).all()
        assert torch.allclose(out_dt.float(), out_ref, atol=atol, rtol=atol)

    @pytest.mark.unit
    def test_streaming_matches_offline(self):
        # The load-bearing test for the cache_len offset logic. Feeding the last
        # `new_len` tokens with the first `cache_len` tokens as KV cache must
        # reproduce the corresponding slice of the offline forward, because RoPE
        # depends only on the (m - n) position difference and the cache layout
        # preserves that.
        pe = RotaryPositionalEncoding(d_k=16, max_len=64)
        pe.extend_pe(32, device=torch.device('cpu'), dtype=torch.float32)
        attn = RoPEMultiHeadAttention(n_head=4, n_feat=64, dropout_rate=0.0, pos_enc=pe).eval()
        attn.cache_drop_size = 0  # required by update_cache

        torch.manual_seed(7)
        full_seq = torch.randn(1, 12, 64)
        cache_len = 8

        with torch.no_grad():
            offline_out = attn(query=full_seq, key=full_seq, value=full_seq, mask=None)
            new_query = full_seq[:, cache_len:]
            cache = full_seq[:, :cache_len]
            streaming_out, _ = attn(query=new_query, key=new_query, value=new_query, mask=None, cache=cache)

        assert torch.allclose(streaming_out, offline_out[:, cache_len:], atol=1e-5)


class TestConformerEncoderRoPE:
    @pytest.mark.unit
    def test_pos_enc_shared_across_layers(self):
        # Critical: every layer must hold the same pos_enc instance so that the
        # encoder's set_max_audio_length / extend_pe grows the buffers used by
        # every layer (not just the first).
        enc = _build_encoder()
        assert all(layer.self_attn.pos_enc is enc.pos_enc for layer in enc.layers)
        # And exercising the shared-extend path: growing the buffer once must be
        # visible from every layer.
        enc.pos_enc.extend_pe(512, device=torch.device('cpu'), dtype=torch.float32)
        assert all(layer.self_attn.pos_enc.cos.size(0) >= 512 for layer in enc.layers)

    @pytest.mark.unit
    def test_sdpa_matches_manual(self):
        # CPU fp32: SDPA falls back to MATH; verify it matches the manual matmul
        # path so RoPE pre-rotation is applied consistently across both code paths.
        enc_manual = _build_encoder(use_pytorch_sdpa=False)
        enc_sdpa = _build_encoder(use_pytorch_sdpa=True)
        enc_sdpa.load_state_dict(enc_manual.state_dict(), strict=False)
        x = torch.randn(2, 80, 200)
        lens = torch.tensor([200, 150])
        with torch.no_grad():
            o_manual, _ = enc_manual(audio_signal=x, length=lens)
            o_sdpa, _ = enc_sdpa(audio_signal=x, length=lens)
        assert torch.allclose(o_manual, o_sdpa, atol=1e-4, rtol=1e-4)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("backend", ['MATH', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION'])
    def test_sdpa_backend_matches_manual_gpu(self, backend):
        # Forward + backward parity vs the manual path under bf16 autocast.
        # RoPE applies to Q/K before the SDPA call, so each backend sees the
        # same rotated tensors and must agree on outputs and gradients within
        # bf16 tolerance. FLASH_ATTENTION is excluded because PyTorch rejects
        # any non-null `attn_mask` on the Flash kernel and the encoder always
        # emits a padding mask; CUDNN and EFFICIENT both accept bool masks.
        # The MHA-level smoke test covers FLASH with mask=None.
        enc_manual = _build_encoder(use_pytorch_sdpa=False).to("cuda")
        enc_sdpa = _build_encoder(use_pytorch_sdpa=True, use_pytorch_sdpa_backends=[backend]).to("cuda")
        enc_sdpa.load_state_dict(enc_manual.state_dict(), strict=False)

        torch.manual_seed(0)
        x_base = torch.randn(2, 80, 200, device='cuda')
        x_manual = x_base.clone().requires_grad_(True)
        x_sdpa = x_base.clone().requires_grad_(True)
        lens = torch.tensor([200, 150], device='cuda')

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            o_manual, _ = enc_manual(audio_signal=x_manual, length=lens)
            o_sdpa, _ = enc_sdpa(audio_signal=x_sdpa, length=lens)

        # Forward parity.
        assert torch.allclose(o_manual.float(), o_sdpa.float(), atol=5e-2, rtol=5e-2)

        # Backward parity: same loss, compare input grads and weight grads.
        o_manual.sum().backward()
        o_sdpa.sum().backward()
        assert torch.allclose(x_manual.grad.float(), x_sdpa.grad.float(), atol=5e-2, rtol=5e-2)
        for (n1, p1), (n2, p2) in zip(enc_manual.named_parameters(), enc_sdpa.named_parameters()):
            assert n1 == n2
            assert p1.grad is not None and p2.grad is not None, f"missing grad for {n1}"
            assert torch.allclose(p1.grad.float(), p2.grad.float(), atol=5e-2, rtol=5e-2), f"grad mismatch for {n1}"

    @pytest.mark.unit
    def test_padding_does_not_leak(self):
        # Output for the valid prefix must be invariant to the values in the
        # padded suffix.
        enc = _build_encoder()
        x = torch.randn(1, 80, 200)
        valid_len = 120
        x1 = x.clone()
        x1[0, :, valid_len:] = torch.randn(80, 200 - valid_len)
        x2 = x.clone()
        x2[0, :, valid_len:] = torch.randn(80, 200 - valid_len)
        lens = torch.tensor([valid_len])
        with torch.no_grad():
            o1, _ = enc(audio_signal=x1, length=lens)
            o2, _ = enc(audio_signal=x2, length=lens)
        valid_out_len = valid_len // 4
        assert torch.allclose(o1[..., :valid_out_len], o2[..., :valid_out_len], atol=1e-5)

    @pytest.mark.unit
    def test_change_attention_model_to_rope(self):
        # Build a rel_pos encoder, swap to rope, run forward.
        enc = _build_encoder(self_attention_model='rel_pos')
        enc._cfg = OmegaConf.create(
            {
                'd_model': 64,
                'n_heads': 4,
                'dropout': 0.0,
                'dropout_att': 0.0,
                'dropout_emb': 0.0,
                'pos_emb_max_len': 256,
                'rope_base': 10000.0,
                'rotary_fraction': 1.0,
            }
        )
        enc.change_attention_model('rope')
        assert isinstance(enc.pos_enc, RotaryPositionalEncoding)
        assert all(layer.self_attn.pos_enc is enc.pos_enc for layer in enc.layers)
        x = torch.randn(2, 80, 200)
        lens = torch.tensor([200, 150])
        out, _ = enc(audio_signal=x, length=lens)
        assert torch.isfinite(out).all()

    @pytest.mark.unit
    def test_change_attention_model_preserves_use_bias_false(self):
        # Regression: the swap loop in change_attention_model was building the new
        # attention without forwarding use_bias, so a use_bias=False model silently
        # gained randomly-initialised bias parameters after a swap.
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

        enc = ConformerEncoder(
            feat_in=80,
            n_layers=2,
            d_model=64,
            n_heads=4,
            self_attention_model='rel_pos',
            subsampling_factor=4,
            subsampling_conv_channels=32,
            pos_emb_max_len=256,
            use_bias=False,
            dropout=0.0,
            dropout_att=0.0,
            dropout_emb=0.0,
            dropout_pre_encoder=0.0,
        ).eval()
        enc._cfg = OmegaConf.create(
            {
                'd_model': 64,
                'n_heads': 4,
                'dropout': 0.0,
                'dropout_att': 0.0,
                'dropout_emb': 0.0,
                'pos_emb_max_len': 256,
                'rope_base': 10000.0,
                'rotary_fraction': 1.0,
                'use_bias': False,
            }
        )
        # Pre-condition: rel_pos attention has no biases.
        for layer in enc.layers:
            assert layer.self_attn.linear_q.bias is None

        enc.change_attention_model('rope')

        # Post-condition: still no biases — use_bias preserved through the swap.
        for layer in enc.layers:
            assert layer.self_attn.linear_q.bias is None
            assert layer.self_attn.linear_k.bias is None
            assert layer.self_attn.linear_v.bias is None
            assert layer.self_attn.linear_out.bias is None

    @pytest.mark.unit
    def test_change_attention_model_preserves_cfg_on_partial_update(self):
        # Regression: ASRModuleMixin.change_attention_model used to write the *raw*
        # kwargs into self.cfg.encoder, so a partial update like
        # `change_attention_model(rotary_fraction=0.5)` left
        # cfg.encoder.self_attention_model = None (corrupting the saved config) and
        # skipped writing the rope fields entirely.
        from omegaconf import DictConfig

        from nemo.collections.asr.models import EncDecCTCModel

        encoder_cfg = {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': 64,
            'n_layers': 2,
            'd_model': 64,
            'n_heads': 4,
            'self_attention_model': 'rope',
            'subsampling_factor': 4,
            'subsampling_conv_channels': 32,
            'pos_emb_max_len': 256,
            'rope_base': 10000.0,
            'rotary_fraction': 1.0,
            'dropout': 0.0,
            'dropout_att': 0.0,
            'dropout_emb': 0.0,
            'dropout_pre_encoder': 0.0,
        }
        decoder_cfg = {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': None,
            'num_classes': 28,
            'vocabulary': list("abcdefghijklmnopqrstuvwxyz '"),
        }
        preproc_cfg = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
        model = EncDecCTCModel(
            cfg=DictConfig(
                {
                    'preprocessor': preproc_cfg,
                    'encoder': encoder_cfg,
                    'decoder': decoder_cfg,
                    'optim': {'name': 'adamw'},
                }
            )
        )

        # Partial update: only rotary_fraction is being changed.
        model.change_attention_model(rotary_fraction=0.5)

        # cfg.encoder must reflect the resolved values, not the None kwargs.
        assert model.cfg.encoder.self_attention_model == 'rope'
        assert model.cfg.encoder.att_context_size is not None
        assert model.cfg.encoder.rotary_fraction == 0.5
        assert model.cfg.encoder.rope_base == 10000.0
        # Live encoder agrees.
        assert model.encoder.self_attention_model == 'rope'
        assert model.encoder.pos_enc.d_k_rot == 8  # d_k=16, fraction=0.5
