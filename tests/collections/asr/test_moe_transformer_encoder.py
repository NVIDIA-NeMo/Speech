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

"""
Tests for MoE Transformer Encoder.

Verifies:
1. Basic forward pass & output shapes
2. Expert initialization from FFN weights
3. Auxiliary load-balancing loss
4. Omni vs switch router sharing
5. Selective layer MoE application
6. State dict remapping from base TransformerEncoder
7. Parameter count comparison (MoE vs dense)
"""

import pytest
import torch

from nemo.collections.asr.modules.moe_transformer_encoder import MoEFeedForward, MoETransformerEncoder, SwitchGate
from nemo.collections.asr.modules.transformer_encoder import FeedForward, TransformerEncoder, TransformerEncoderConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
N_MELS = 80
BATCH = 2
SEQ_LEN = 200  # raw audio frames (pre-subsampling)


def _make_cfg(d_model=D_MODEL, ff_expansion=4.0, n_heads=N_HEADS, **kwargs):
    """Build a TransformerEncoderConfig for unit-testing FeedForward / MoEFeedForward."""
    return TransformerEncoderConfig(
        d_model=d_model, n_heads=n_heads, ff_expansion=ff_expansion, drop_rate=0.0, **kwargs
    )


def _make_input(batch=BATCH, seq_len=SEQ_LEN, n_mels=N_MELS, device=DEVICE):
    """Create a dummy audio input and corresponding lengths."""
    audio = torch.randn(batch, n_mels, seq_len, device=device)
    lengths = torch.full((batch,), seq_len, dtype=torch.long, device=device)
    return audio, lengths


def _make_base_encoder(**kwargs):
    defaults = dict(
        feat_in=N_MELS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        drop_rate=0.0,
        qkv_bias=False,
        attn_mode="full",
        subsampling="feature_stacking",
        qk_norm=False,
        self_attention_model="rel_pos",
    )
    defaults.update(kwargs)
    return TransformerEncoder(**defaults).to(DEVICE)


def _make_moe_encoder(**kwargs):
    defaults = dict(
        feat_in=N_MELS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        drop_rate=0.0,
        qkv_bias=False,
        attn_mode="full",
        subsampling="feature_stacking",
        qk_norm=False,
        moe_num_experts=4,
        moe_top_k=1,
        moe_router_type='omni',
        moe_layer_indices=None,
        moe_load_balance_loss_weight=0.01,
        moe_jitter_eps=0.0,
        moe_init_from_ffn=True,
    )
    defaults.update(kwargs)
    return MoETransformerEncoder(**defaults).to(DEVICE)


# ---------------------------------------------------------------------------
# Test 1: Basic forward pass — correct output shapes
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_output_shape_matches_base(self):
        """MoE encoder should produce the same output shape as the base encoder."""
        base = _make_base_encoder()
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            base_out, base_len = base(audio, lengths)
            moe_out, moe_len = moe(audio, lengths)

        assert base_out.shape == moe_out.shape, f"Shape mismatch: base {base_out.shape} vs moe {moe_out.shape}"
        assert torch.equal(base_len, moe_len)

    def test_output_shape_values(self):
        """Output should be (B, d_model, T') where T' is subsampled time."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            out, out_len = moe(audio, lengths)

        assert out.shape[0] == BATCH
        assert out.shape[1] == D_MODEL
        assert out.shape[2] > 0

    def test_no_nan_in_output(self):
        """Output should not contain NaN or Inf."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            out, _ = moe(audio, lengths)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        moe = _make_moe_encoder()
        for bs in [1, 3, 5]:
            audio, lengths = _make_input(batch=bs)
            with torch.no_grad():
                out, out_len = moe(audio, lengths)
            assert out.shape[0] == bs

    def test_gradients_flow(self):
        """Gradients should flow through MoE layers."""
        moe = _make_moe_encoder()
        moe_ffn = moe.layers[moe.moe_layer_indices[0]].ffn
        assert isinstance(moe_ffn, MoEFeedForward)

        hidden = torch.randn(BATCH, 8, D_MODEL, device=DEVICE)
        loss = moe_ffn(hidden).sum()
        loss.backward()

        assert any(
            param.grad is not None for param in moe_ffn.experts.parameters()
        ), "No gradients reached expert parameters"


# ---------------------------------------------------------------------------
# Test 2: Expert initialization from FFN weights
# ---------------------------------------------------------------------------


class TestExpertInitialization:
    def test_experts_initialized_from_ffn(self):
        """When moe_init_from_ffn=True, all experts should start with identical weights."""
        moe = _make_moe_encoder(moe_init_from_ffn=True, moe_num_experts=3)

        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            assert isinstance(moe_ffn, MoEFeedForward)

            ref_state = moe_ffn.experts[0].state_dict()
            for expert_idx in range(1, moe_ffn.num_experts):
                expert_state = moe_ffn.experts[expert_idx].state_dict()
                for key in ref_state:
                    assert torch.equal(
                        ref_state[key], expert_state[key]
                    ), f"Layer {layer_idx}, expert {expert_idx}, key '{key}' differs from expert 0"

    def test_experts_random_when_init_disabled(self):
        """When moe_init_from_ffn=False, experts should have different random weights."""
        moe = _make_moe_encoder(moe_init_from_ffn=False, moe_num_experts=3)

        any_different = False
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            state0 = moe_ffn.experts[0].state_dict()
            state1 = moe_ffn.experts[1].state_dict()
            for key in state0:
                if not torch.equal(state0[key], state1[key]):
                    any_different = True
                    break
            if any_different:
                break
        assert any_different, "Experts should have different random weights when init_from_ffn=False"


# ---------------------------------------------------------------------------
# Test 3: Auxiliary load-balancing loss
# ---------------------------------------------------------------------------


class TestAuxiliaryLoss:
    def test_aux_loss_is_scalar(self):
        """get_moe_auxiliary_loss() should return a scalar tensor after forward."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            moe(audio, lengths)

        loss = moe.get_moe_auxiliary_loss()
        assert loss is not None, "Auxiliary loss should not be None after forward pass"
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    def test_aux_loss_is_positive(self):
        """Auxiliary loss should be non-negative."""
        moe = _make_moe_encoder()
        audio, lengths = _make_input()

        with torch.no_grad():
            moe(audio, lengths)

        loss = moe.get_moe_auxiliary_loss()
        assert loss >= 0, f"Auxiliary loss should be non-negative, got {loss.item()}"

    def test_aux_loss_none_before_forward(self):
        """Before any forward pass, auxiliary loss should be None."""
        moe = _make_moe_encoder()
        loss = moe.get_moe_auxiliary_loss()
        assert loss is None

    def test_aux_loss_weighted(self):
        """Auxiliary loss should scale with moe_load_balance_loss_weight."""
        moe_low = _make_moe_encoder(moe_load_balance_loss_weight=0.001)
        moe_high = _make_moe_encoder(moe_load_balance_loss_weight=1.0)

        # Share the same weights so the raw aux losses are identical
        moe_high.load_state_dict(moe_low.state_dict(), strict=False)

        audio, lengths = _make_input()
        with torch.no_grad():
            moe_low(audio, lengths)
            moe_high(audio, lengths)

        loss_low = moe_low.get_moe_auxiliary_loss()
        loss_high = moe_high.get_moe_auxiliary_loss()
        assert loss_high > loss_low, "Higher weight should produce larger auxiliary loss"

    def test_aux_loss_backprop(self):
        """Auxiliary loss should be differentiable."""
        moe = _make_moe_encoder()
        hidden = torch.randn(BATCH, 8, D_MODEL, device=DEVICE)

        # Populate every MoE layer's auxiliary loss without involving the
        # unrelated attention path, whose FlexAttention CPU backward is not
        # implemented in the PyTorch version used by CPU CI.
        for layer_idx in moe.moe_layer_indices:
            moe.layers[layer_idx].ffn(hidden)

        aux_loss = moe.get_moe_auxiliary_loss()
        assert aux_loss is not None
        aux_loss.backward()

        assert (
            moe.omni_router.w_gate.weight.grad is not None
        ), "Auxiliary loss gradients should reach router parameters"


# ---------------------------------------------------------------------------
# Test 4: Omni vs switch router sharing
# ---------------------------------------------------------------------------


class TestRouterStrategies:
    def test_omni_router_shared(self):
        """In omni mode, all MoE layers should share the same router object."""
        moe = _make_moe_encoder(moe_router_type='omni')

        routers = set()
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            routers.add(id(moe_ffn.router))

        assert len(routers) == 1, f"Omni router should be shared, found {len(routers)} distinct routers"

    def test_switch_router_independent(self):
        """In switch mode, each MoE layer should have its own router."""
        moe = _make_moe_encoder(moe_router_type='switch')

        routers = set()
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            routers.add(id(moe_ffn.router))

        assert len(routers) == len(moe.moe_layer_indices), (
            f"Switch router should be independent per layer, found {len(routers)} "
            f"distinct routers for {len(moe.moe_layer_indices)} layers"
        )

    def test_omni_router_is_module_attribute(self):
        """The shared omni router should be a registered module attribute."""
        moe = _make_moe_encoder(moe_router_type='omni')
        assert moe.omni_router is not None
        assert isinstance(moe.omni_router, SwitchGate)

    def test_switch_has_no_omni_router(self):
        """In switch mode, omni_router should be None."""
        moe = _make_moe_encoder(moe_router_type='switch')
        assert moe.omni_router is None

    def test_invalid_router_type_raises(self):
        """Invalid router type should raise ValueError."""
        with pytest.raises(ValueError, match="moe_router_type"):
            _make_moe_encoder(moe_router_type='invalid')


# ---------------------------------------------------------------------------
# Test 5: Selective layer MoE application
# ---------------------------------------------------------------------------


class TestSelectiveLayers:
    def test_only_specified_layers_have_moe(self):
        """Only layers in moe_layer_indices should have MoEFeedForward."""
        moe_indices = [0, 2]
        moe = _make_moe_encoder(moe_layer_indices=moe_indices)

        for i in range(N_LAYERS):
            ffn = moe.layers[i].ffn
            if i in moe_indices:
                assert isinstance(ffn, MoEFeedForward), f"Layer {i} should be MoE"
            else:
                assert isinstance(ffn, FeedForward), f"Layer {i} should be dense FFN"

    def test_all_layers_when_none(self):
        """When moe_layer_indices=None, all layers should be MoE."""
        moe = _make_moe_encoder(moe_layer_indices=None)

        for i in range(N_LAYERS):
            assert isinstance(
                moe.layers[i].ffn, MoEFeedForward
            ), f"Layer {i} should be MoE when moe_layer_indices=None"

    def test_single_layer_moe(self):
        """MoE on a single layer should work."""
        moe = _make_moe_encoder(moe_layer_indices=[1])
        audio, lengths = _make_input()

        with torch.no_grad():
            out, _ = moe(audio, lengths)

        assert not torch.isnan(out).any()
        assert isinstance(moe.layers[1].ffn, MoEFeedForward)
        assert isinstance(moe.layers[0].ffn, FeedForward)

    def test_invalid_layer_index_raises(self):
        """Layer index out of range should raise ValueError."""
        with pytest.raises(ValueError, match="invalid index"):
            _make_moe_encoder(moe_layer_indices=[N_LAYERS + 5])


# ---------------------------------------------------------------------------
# Test 6: State dict remapping from base TransformerEncoder
# ---------------------------------------------------------------------------


class TestStateDictRemapping:
    def test_load_base_into_moe(self):
        """Loading a base TransformerEncoder state dict into MoE encoder should work."""
        base = _make_base_encoder()
        moe = _make_moe_encoder(moe_num_experts=2)

        base_state = base.state_dict()
        moe.load_state_dict(base_state, strict=False)

        # Verify expert weights match the original FFN weights
        for layer_idx in moe.moe_layer_indices:
            base_ffn_prefix = f"layers.{layer_idx}.ffn."
            moe_ffn = moe.layers[layer_idx].ffn

            for param_name, base_param in base_state.items():
                if param_name.startswith(base_ffn_prefix):
                    suffix = param_name[len(base_ffn_prefix) :]
                    # Skip if this is already an expert/router key
                    if suffix.startswith("experts.") or suffix.startswith("router."):
                        continue
                    for expert in moe_ffn.experts:
                        # The FeedForward has params like net.0.weight, net.0.bias, net.3.weight, net.3.bias
                        flat_params = dict(expert.named_parameters())
                        if suffix in flat_params:
                            assert torch.allclose(
                                flat_params[suffix], base_param.to(DEVICE)
                            ), f"Layer {layer_idx}, suffix '{suffix}' mismatch after loading"

    def test_moe_to_moe_load(self):
        """Loading a MoE state dict into another MoE encoder should work perfectly."""
        moe1 = _make_moe_encoder(moe_num_experts=2)
        moe2 = _make_moe_encoder(moe_num_experts=2)

        moe2.load_state_dict(moe1.state_dict())

        audio, lengths = _make_input()
        with torch.no_grad():
            out1, _ = moe1(audio, lengths)
            out2, _ = moe2(audio, lengths)

        assert torch.allclose(out1, out2, atol=1e-6), "Same weights should produce same output"

    def test_router_not_in_base_state(self):
        """Router keys should not exist in base encoder state dict."""
        base = _make_base_encoder()
        for key in base.state_dict():
            assert 'router' not in key
            assert 'w_gate' not in key


# ---------------------------------------------------------------------------
# Test 7: Parameter count comparison
# ---------------------------------------------------------------------------


class TestParameterCount:
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def test_moe_has_more_params(self):
        """MoE encoder should have more total parameters than base."""
        base = _make_base_encoder()
        moe = _make_moe_encoder(moe_num_experts=4)
        assert self._count_params(moe) > self._count_params(base)

    def test_single_expert_similar_to_base(self):
        """With 1 expert, MoE should have only router params extra."""
        base = _make_base_encoder()
        moe = _make_moe_encoder(moe_num_experts=1)

        base_params = self._count_params(base)
        moe_params = self._count_params(moe)

        # The only extra params should be the router: d_model * 1 (for 1 expert)
        router_params = D_MODEL * 1  # omni router: Linear(d_model, 1) has d_model params (no bias)
        assert moe_params == base_params + router_params, (
            f"Expected {base_params + router_params} params, got {moe_params}. " f"Diff = {moe_params - base_params}"
        )


# ---------------------------------------------------------------------------
# Test 8: SwitchGate unit tests
# ---------------------------------------------------------------------------


class TestSwitchGate:
    def test_output_shape(self):
        gate = SwitchGate(d_model=32, num_experts=4).to(DEVICE)
        x = torch.randn(10, 32, device=DEVICE)
        probs = gate(x)
        assert probs.shape == (10, 4)

    def test_probabilities_sum_to_one(self):
        gate = SwitchGate(d_model=32, num_experts=4).to(DEVICE)
        x = torch.randn(10, 32, device=DEVICE)
        probs = gate(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10, device=DEVICE), atol=1e-5)

    def test_jitter_only_during_training(self):
        gate = SwitchGate(d_model=32, num_experts=4, jitter_eps=1.0).to(DEVICE)
        x = torch.randn(5, 32, device=DEVICE)

        gate.eval()
        torch.manual_seed(42)
        out1 = gate(x)
        torch.manual_seed(42)
        out2 = gate(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# ---------------------------------------------------------------------------
# Test 9: MoEFeedForward unit tests
# ---------------------------------------------------------------------------


class TestMoEFeedForward:
    def test_output_shape(self):
        moe_ff = MoEFeedForward(cfg=_make_cfg(32), num_experts=4, top_k=1).to(DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        out = moe_ff(x)
        assert out.shape == (2, 10, 32)

    def test_top_k_2(self):
        """Top-k=2 should also work correctly."""
        moe_ff = MoEFeedForward(cfg=_make_cfg(32), num_experts=4, top_k=2).to(DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        out = moe_ff(x)
        assert out.shape == (2, 10, 32)
        assert not torch.isnan(out).any()

    def test_external_router(self):
        """MoEFeedForward should accept an external router."""
        router = SwitchGate(d_model=32, num_experts=4).to(DEVICE)
        moe_ff = MoEFeedForward(cfg=_make_cfg(32), num_experts=4, router=router).to(DEVICE)
        assert moe_ff.router is router

    def test_expert_counts_recorded(self):
        """After forward, _expert_counts should be a length-num_experts long tensor
        whose total equals num_tokens * top_k.
        """
        E, K = 5, 2
        moe_ff = MoEFeedForward(cfg=_make_cfg(16), num_experts=E, top_k=K).to(DEVICE)
        x = torch.randn(3, 7, 16, device=DEVICE)
        with torch.no_grad():
            moe_ff(x)
        counts = moe_ff._expert_counts
        assert counts is not None
        assert counts.shape == (E,)
        assert counts.dtype == torch.long
        assert int(counts.sum().item()) == 3 * 7 * K

    def test_gate_prob_sum_recorded(self):
        """After forward, _gate_prob_sum / _num_tokens should sum to ~1."""
        E = 4
        moe_ff = MoEFeedForward(cfg=_make_cfg(16), num_experts=E, top_k=1).to(DEVICE)
        x = torch.randn(2, 5, 16, device=DEVICE)
        with torch.no_grad():
            moe_ff(x)
        prob_sum = moe_ff._gate_prob_sum
        n = moe_ff._num_tokens
        assert prob_sum is not None
        assert prob_sum.shape == (E,)
        assert prob_sum.dtype == torch.float32
        assert n == 2 * 5
        rho = prob_sum / n
        assert torch.allclose(rho.sum(), torch.tensor(1.0, device=DEVICE), atol=1e-4)


# ---------------------------------------------------------------------------
# Test 10: Configurable FFN inner dim via ff_expansion
# ---------------------------------------------------------------------------


class TestConfigurableFFNExpansion:
    def test_feedforward_default_is_4x(self):
        """ff_expansion=4.0 -> inner dim = 4 * d_model; net[0]/net[3] are the Linears."""
        ff = FeedForward(_make_cfg(64, ff_expansion=4.0))
        # net = [Linear(d, h), GELU, Dropout, Linear(h, d), Dropout]
        assert ff.net[0].weight.shape == (256, 64)
        assert ff.net[3].weight.shape == (64, 256)

    def test_feedforward_thin_ff_expansion(self):
        """Sub-1x ff_expansion produces a thin FFN (e.g. 0.5 -> 32)."""
        ff = FeedForward(_make_cfg(64, ff_expansion=0.5))
        assert ff.net[0].weight.shape == (32, 64)
        assert ff.net[3].weight.shape == (64, 32)

    def test_encoder_ff_expansion(self):
        """ff_expansion=0.5 -> every block FFN inner dim = int(d_model * 0.5)."""
        enc = _make_base_encoder(ff_expansion=0.5)
        h = int(D_MODEL * 0.5)
        for layer in enc.layers:
            assert layer.ffn.net[0].out_features == h
            assert layer.ffn.net[0].weight.shape == (h, D_MODEL)

    def test_moe_encoder_experts_ff_expansion(self):
        """MoETransformerEncoder sizes each expert FFN via ff_expansion."""
        moe = _make_moe_encoder(ff_expansion=0.5, moe_init_from_ffn=False, moe_num_experts=3)
        h = int(D_MODEL * 0.5)
        assert moe.ff_hidden_size == h
        for layer_idx in moe.moe_layer_indices:
            moe_ffn = moe.layers[layer_idx].ffn
            assert isinstance(moe_ffn, MoEFeedForward)
            for expert in moe_ffn.experts:
                assert expert.net[0].weight.shape == (h, D_MODEL)

    def test_moe_thin_param_count(self):
        """Sanity-check parameter count for a thin MoE configuration."""
        d, E, n = D_MODEL, 3, N_LAYERS
        ff_expansion = 24.0 / d  # int(ff_expansion * d) = 24
        h = int(ff_expansion * d)
        moe = _make_moe_encoder(
            ff_expansion=ff_expansion,
            moe_init_from_ffn=False,
            moe_num_experts=E,
            moe_router_type='omni',
        )
        assert moe.ff_hidden_size == h

        # Per-expert FFN: Linear(d, h) + Linear(h, d), both with bias
        per_expert = (d * h + h) + (h * d + d)
        encoder_total = n * E * per_expert
        # Lower bound only -- attention, layernorms, router, pre-encoder and
        # positional params all add on top of the expert FFN params.
        actual = sum(p.numel() for p in moe.parameters())
        assert actual >= encoder_total, f"Expected at least {encoder_total} expert-FFN params, got {actual}"


# ---------------------------------------------------------------------------
# Test 11: Optimized dispatch loop equivalence & correctness
# ---------------------------------------------------------------------------


def _reference_dispatch(moe_ff: MoEFeedForward, x: torch.Tensor) -> torch.Tensor:
    """Naive O(top_k * num_experts) reference implementation -- used to validate
    the optimized in-tree dispatch in MoEFeedForward.forward.
    """
    batch, seq, d = x.shape
    x_flat = x.reshape(-1, d)

    gate_probs = moe_ff.router(x_flat)
    top_k_probs, top_k_indices = torch.topk(gate_probs, moe_ff.top_k, dim=-1)
    if moe_ff.top_k > 1:
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

    out = torch.zeros_like(x_flat)
    for k in range(moe_ff.top_k):
        expert_indices = top_k_indices[:, k]
        expert_weights = top_k_probs[:, k]
        for i in range(moe_ff.num_experts):
            mask = expert_indices == i
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = moe_ff.experts[i](expert_input)
                out[mask] += expert_output * expert_weights[mask].unsqueeze(-1)
    return out.reshape(batch, seq, d)


class TestOptimizedDispatchEquivalence:
    def test_top_k_1_matches_reference(self):
        """Optimized dispatch (top_k=1) matches the naive reference."""
        torch.manual_seed(0)
        moe_ff = MoEFeedForward(cfg=_make_cfg(16), num_experts=4, top_k=1).to(DEVICE)
        moe_ff.eval()
        x = torch.randn(2, 9, 16, device=DEVICE)
        with torch.no_grad():
            out_optimized = moe_ff(x)
            out_reference = _reference_dispatch(moe_ff, x)
        assert torch.allclose(out_optimized, out_reference, atol=1e-5)

    def test_top_k_2_matches_reference(self):
        """Optimized dispatch (top_k=2) matches the naive reference."""
        torch.manual_seed(1)
        moe_ff = MoEFeedForward(cfg=_make_cfg(16), num_experts=4, top_k=2).to(DEVICE)
        moe_ff.eval()
        x = torch.randn(2, 9, 16, device=DEVICE)
        with torch.no_grad():
            out_optimized = moe_ff(x)
            out_reference = _reference_dispatch(moe_ff, x)
        assert torch.allclose(out_optimized, out_reference, atol=1e-5)

    def test_top_k_4_of_8_matches_reference(self):
        """Higher top_k of more experts also matches the reference."""
        torch.manual_seed(2)
        moe_ff = MoEFeedForward(cfg=_make_cfg(16), num_experts=8, top_k=4).to(DEVICE)
        moe_ff.eval()
        x = torch.randn(2, 9, 16, device=DEVICE)
        with torch.no_grad():
            out_optimized = moe_ff(x)
            out_reference = _reference_dispatch(moe_ff, x)
        assert torch.allclose(out_optimized, out_reference, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 12: MoE diagnostic metrics (cumulative buffers + get_moe_metrics)
# ---------------------------------------------------------------------------


class TestMoEMetrics:
    def _train_step_then_metrics(self, moe, audio, lengths, n_steps=1):
        """Run `n_steps` training-mode forwards and return get_moe_metrics()."""
        moe.train()
        for _ in range(n_steps):
            with torch.no_grad():
                moe(audio, lengths)
        return moe.get_moe_metrics(distributed=False, reset=False)

    def test_metrics_none_before_any_forward(self):
        moe = _make_moe_encoder()
        out = moe.get_moe_metrics(distributed=False, reset=False)
        assert out is None, "Should return None when no MoE forward has run"

    def test_metrics_present_after_training_forward(self):
        moe = _make_moe_encoder(moe_num_experts=4, moe_top_k=2)
        audio, lengths = _make_input()
        out = self._train_step_then_metrics(moe, audio, lengths, n_steps=2)
        assert out is not None
        assert "scalars" in out and "per_layer" in out
        for key in (
            "moe/load_cv",
            "moe/load_cv_max",
            "moe/load_max_over_ideal",
            "moe/load_min_over_ideal",
            "moe/router_entropy_norm",
            "moe/router_entropy_norm_min",
            "moe/dead_experts",
            "moe/dead_experts_pct",
        ):
            assert key in out["scalars"], f"missing scalar {key}"
            assert torch.is_tensor(out["scalars"][key])
            assert out["scalars"][key].dim() == 0

    def test_metrics_per_layer_shapes(self):
        E, K, L = 4, 2, N_LAYERS
        moe = _make_moe_encoder(moe_num_experts=E, moe_top_k=K)
        audio, lengths = _make_input()
        out = self._train_step_then_metrics(moe, audio, lengths)
        pl = out["per_layer"]
        assert pl["load"].shape == (L, E)
        assert pl["rho"].shape == (L, E)
        assert pl["cv"].shape == (L,)
        assert pl["entropy_norm"].shape == (L,)
        assert pl["dead"].shape == (L,)

    def test_metrics_load_sums_to_topk_per_layer(self):
        """Per-layer load fractions f must sum to top_k."""
        E, K = 4, 2
        moe = _make_moe_encoder(moe_num_experts=E, moe_top_k=K)
        audio, lengths = _make_input()
        out = self._train_step_then_metrics(moe, audio, lengths)
        load = out["per_layer"]["load"]
        assert torch.allclose(load.sum(dim=-1), torch.full((load.shape[0],), float(K)), atol=1e-3)

    def test_metrics_rho_sums_to_one_per_layer(self):
        E = 4
        moe = _make_moe_encoder(moe_num_experts=E, moe_top_k=1)
        audio, lengths = _make_input()
        out = self._train_step_then_metrics(moe, audio, lengths)
        rho = out["per_layer"]["rho"]
        assert torch.allclose(rho.sum(dim=-1), torch.ones(rho.shape[0]), atol=1e-3)

    def test_reset_zeros_buffers(self):
        moe = _make_moe_encoder(moe_num_experts=3)
        audio, lengths = _make_input()
        moe.train()
        with torch.no_grad():
            moe(audio, lengths)
        assert moe._cum_counts is not None
        assert moe._cum_counts.sum().item() > 0
        moe.reset_moe_metrics()
        assert moe._cum_counts.sum().item() == 0
        assert moe._cum_prob_sum.sum().item() == 0.0
        assert moe._cum_tokens.sum().item() == 0

    def test_get_moe_metrics_with_reset(self):
        """get_moe_metrics(reset=True) should clear the cumulative state."""
        moe = _make_moe_encoder(moe_num_experts=3)
        audio, lengths = _make_input()
        moe.train()
        with torch.no_grad():
            moe(audio, lengths)
        moe.get_moe_metrics(distributed=False, reset=True)
        assert moe._cum_counts.sum().item() == 0
        assert moe._cum_prob_sum.sum().item() == 0.0
        assert moe._cum_tokens.sum().item() == 0

    def test_eval_mode_does_not_accumulate(self):
        """Eval / inference forwards must not allocate cumulative tensors."""
        moe = _make_moe_encoder(moe_num_experts=3)
        audio, lengths = _make_input()
        moe.eval()
        with torch.no_grad():
            moe(audio, lengths)
        # Either the cumulative tensors were never allocated, or they remain
        # all zeros. Both are acceptable.
        if moe._cum_counts is not None:
            assert moe._cum_counts.sum().item() == 0
            assert moe._cum_tokens.sum().item() == 0

    def test_dead_expert_detected(self):
        """If we manually zero a hot expert's counts, dead_experts should rise."""
        moe = _make_moe_encoder(moe_num_experts=4, moe_top_k=1)
        audio, lengths = _make_input()
        moe.train()
        with torch.no_grad():
            moe(audio, lengths)
        # Force one expert to look dead in every layer.
        moe._cum_counts[:, 0] = 0
        out = moe.get_moe_metrics(distributed=False, reset=False)
        assert out["scalars"]["moe/dead_experts"].item() >= float(len(moe.moe_layer_indices))

    def test_cumulative_state_not_in_state_dict(self):
        """Cumulative MoE state must not leak into the saved state_dict.

        Implementation detail: cumulative state is held as plain attributes,
        not via register_buffer, both to keep the state out of checkpoints
        and to prevent DDP's default broadcast_buffers behavior from
        overwriting per-rank accumulation each step.
        """
        moe = _make_moe_encoder()
        audio, lengths = _make_input()
        moe.train()
        with torch.no_grad():
            moe(audio, lengths)
        sd = moe.state_dict()
        for key in ("_cum_counts", "_cum_prob_sum", "_cum_tokens"):
            assert not any(k.endswith(key) for k in sd), f"Cumulative MoE state {key!r} leaked into state_dict"

    def test_cumulative_state_is_not_a_named_buffer(self):
        """Hard guard: must NOT be a registered nn.Module buffer.

        register_buffer would expose the tensor to DDP's broadcast_buffers,
        which would overwrite per-rank accumulation. This test ensures we
        keep the plain-attribute design.
        """
        moe = _make_moe_encoder()
        named_buffers = {n for n, _ in moe.named_buffers()}
        for key in ("_cum_counts", "_cum_prob_sum", "_cum_tokens"):
            assert key not in named_buffers, (
                f"{key!r} is a registered buffer; this will desync per-rank "
                f"accumulation under DDP. Keep it as a plain attribute."
            )


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
