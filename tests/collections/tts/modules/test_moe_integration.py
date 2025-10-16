# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Integration tests for MoE implementation.

These tests verify the complete integration of MoE components:
- Modules (moe_modules.py)
- Losses (moe_loss.py)
- Transformer (transformer_2501.py)
- Model (magpietts.py)

These tests should catch API mismatches and integration issues.
"""

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.losses.moe_loss import MoEAuxiliaryLoss, MoELoadBalancingLoss, MoERouterZLoss
from nemo.collections.tts.modules.moe_modules import MoERouter, PositionwiseConvFFMoE
from nemo.collections.tts.modules.transformer_2501 import Transformer, TransformerLayer


@pytest.mark.unit
class TestMoELosses:
    """Test MoE loss modules."""

    def test_load_balancing_loss(self):
        """Test MoELoadBalancingLoss."""
        loss_fn = MoELoadBalancingLoss(num_experts=8, loss_scale=0.01)

        router_probs = torch.softmax(torch.randn(2, 10, 8), dim=-1)
        loss = loss_fn(router_probs=router_probs)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_router_z_loss(self):
        """Test MoERouterZLoss."""
        loss_fn = MoERouterZLoss(loss_scale=0.001)

        router_logits = torch.randn(2, 10, 8)
        loss = loss_fn(router_logits=router_logits)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_auxiliary_loss(self):
        """Test MoEAuxiliaryLoss returns tuple of 3 values."""
        loss_fn = MoEAuxiliaryLoss(
            num_experts=8,
            load_balancing_loss_scale=0.01,
            router_z_loss_scale=0.001,
        )

        router_logits = torch.randn(2, 10, 8)
        router_probs = torch.softmax(router_logits, dim=-1)

        result = loss_fn(router_logits=router_logits, router_probs=router_probs)
        assert len(result) == 3, "MoEAuxiliaryLoss should return tuple of 3 values"

        load_balancing_loss, router_z_loss, total_loss = result

        # Check values are scalars
        assert load_balancing_loss.ndim == 0
        assert router_z_loss.ndim == 0
        assert total_loss.ndim == 0

        # Check total is sum of components
        expected_total = load_balancing_loss + router_z_loss
        assert torch.allclose(total_loss, expected_total)


@pytest.mark.unit
class TestMoEModuleIntegration:
    """Test MoE modules return correct values for loss computation."""

    def test_router_returns_four_values(self):
        """Test that MoERouter returns (weights, indices, logits, probs)."""
        router = MoERouter(d_model=64, num_experts=4, top_k=2)

        x = torch.randn(2, 10, 64)
        x_mask = torch.ones(2, 10)
        output = router(x, x_mask)

        # Should return 4 values
        assert len(output) == 4
        weights, indices, logits, probs = output

        # Check shapes
        assert weights.shape == (2, 10, 2)  # (B, T, top_k)
        assert indices.shape == (2, 10, 2)  # (B, T, top_k)
        assert logits.shape == (2, 10, 4)  # (B, T, num_experts)
        assert probs.shape == (2, 10, 4)  # (B, T, num_experts)

    def test_moe_ffn_returns_four_values(self):
        """Test that PositionwiseConvFFMoE returns (output, logits, probs, indices)."""
        moe_ffn = PositionwiseConvFFMoE(
            d_model=64,
            d_ffn=256,
            p_dropout=0.0,
            num_experts=4,
            top_k_experts=2,
            kernel_size=1,
        )

        x = torch.randn(2, 10, 64)
        x_mask = torch.ones(2, 10)

        with torch.no_grad():
            output_tuple = moe_ffn(x, x_mask)

        # Should return 4 values
        assert len(output_tuple) == 4
        output, logits, probs, indices = output_tuple

        # Check shapes
        assert output.shape == (2, 10, 64)  # Same as input
        assert logits.shape == (2, 10, 4)  # (B, T, num_experts)
        assert probs.shape == (2, 10, 4)  # (B, T, num_experts)
        assert indices.shape == (2, 10, 2)  # (B, T, top_k)

    def test_transformer_layer_with_moe_returns_routing_info(self):
        """Test that TransformerLayer with MoE returns moe_routing_info."""
        layer = TransformerLayer(
            d_model=64,
            d_ffn=256,
            sa_n_heads=4,
            kernel_size=1,
            p_dropout=0.0,
            has_xattn=False,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        x = torch.randn(2, 10, 64)
        x_mask = torch.ones(2, 10).bool()

        with torch.no_grad():
            output_dict = layer(x, x_mask)

        # Check output structure
        assert 'output' in output_dict
        assert 'attn_probabilities' in output_dict
        assert 'moe_routing_info' in output_dict

        # Check MoE routing info
        assert output_dict['moe_routing_info'] is not None
        assert 'router_logits' in output_dict['moe_routing_info']
        assert 'router_probs' in output_dict['moe_routing_info']
        assert 'expert_indices' in output_dict['moe_routing_info']

    def test_transformer_with_moe_collects_routing_info(self):
        """Test that Transformer collects routing info from all layers."""
        model = Transformer(
            n_layers=3,
            d_model=64,
            d_ffn=256,
            sa_n_heads=4,
            kernel_size=1,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        x = torch.randn(2, 10, 64)
        x_mask = torch.ones(2, 10).bool()

        with torch.no_grad():
            output_dict = model(x, x_mask)

        # Check output structure
        assert 'output' in output_dict
        assert 'moe_routing_info' in output_dict

        # Should have routing info from all layers
        assert output_dict['moe_routing_info'] is not None
        assert len(output_dict['moe_routing_info']) == 3  # n_layers

        # Each layer should have logits, probs, and indices
        for layer_routing in output_dict['moe_routing_info']:
            assert 'router_logits' in layer_routing
            assert 'router_probs' in layer_routing
            assert 'expert_indices' in layer_routing


@pytest.mark.unit
class TestMoEEndToEnd:
    """End-to-end integration tests for complete MoE pipeline."""

    def test_complete_moe_pipeline(self):
        """Test complete flow: Transformer → routing_info → Loss computation."""
        # Create transformer with MoE
        transformer = Transformer(
            n_layers=2,
            d_model=64,
            d_ffn=256,
            sa_n_heads=4,
            kernel_size=1,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
            router_jitter_noise=0.0,
            routing_strategy="top_k",
        )

        # Create loss module
        loss_module = MoEAuxiliaryLoss(
            num_experts=4,
            load_balancing_loss_scale=0.01,
            router_z_loss_scale=0.001,
        )

        # Forward pass
        x = torch.randn(2, 10, 64)
        x_mask = torch.ones(2, 10).bool()

        transformer.train()
        output_dict = transformer(x, x_mask)

        # Extract routing info
        moe_routing_info = output_dict['moe_routing_info']
        assert moe_routing_info is not None
        assert len(moe_routing_info) == 2  # n_layers

        # Stack routing info for loss computation (as done in magpietts.py)
        all_logits = torch.stack([info['router_logits'] for info in moe_routing_info], dim=0)
        all_probs = torch.stack([info['router_probs'] for info in moe_routing_info], dim=0)

        # Reshape
        merged_logits = all_logits.view(-1, all_logits.size(2), all_logits.size(3))
        merged_probs = all_probs.view(-1, all_probs.size(2), all_probs.size(3))

        # Repeat mask for each layer (for mask-aware loss computation)
        n_layers = len(moe_routing_info)
        merged_mask = x_mask.unsqueeze(0).repeat(n_layers, 1, 1).view(-1, x_mask.size(1))

        load_balancing_loss, router_z_loss, total_loss = loss_module(
            router_logits=merged_logits, router_probs=merged_probs, x_mask=merged_mask
        )

        assert load_balancing_loss.item() >= 0
        assert router_z_loss.item() >= 0
        assert total_loss.item() >= 0

    def test_transformer_layer_parameter_compatibility(self):
        """Test that TransformerLayer accepts correct MoE parameters."""
        # These parameters should work
        valid_params = {
            'd_model': 64,
            'd_ffn': 256,
            'sa_n_heads': 4,
            'kernel_size': 1,
            'p_dropout': 0.0,
            'has_xattn': False,
            'use_moe': True,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_jitter_noise': 0.01,
            'routing_strategy': 'top_k',
        }

        # Should not raise
        layer = TransformerLayer(**valid_params)
        assert layer.use_moe is True

    def test_transformer_layer_rejects_loss_coefficients(self):
        """Test that TransformerLayer rejects loss coefficient parameters (they belong at model level)."""
        params = {
            'd_model': 64,
            'd_ffn': 256,
            'sa_n_heads': 4,
            'kernel_size': 1,
            'p_dropout': 0.0,
            'has_xattn': False,
            'use_moe': True,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_load_balancing_loss_coeff': 0.01,  # Should cause error - belongs at model level
        }

        # Should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            layer = TransformerLayer(**params)

    def test_transformer_parameter_compatibility(self):
        """Test that Transformer accepts correct MoE parameters."""
        # These parameters should work
        valid_params = {
            'n_layers': 2,
            'd_model': 64,
            'd_ffn': 256,
            'sa_n_heads': 4,
            'kernel_size': 1,
            'use_moe': True,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_jitter_noise': 0.01,
            'routing_strategy': 'sinkhorn',
        }

        # Should not raise
        transformer = Transformer(**valid_params)
        assert transformer.use_moe is True

    def test_transformer_rejects_loss_coefficients(self):
        """Test that Transformer rejects loss coefficient parameters (they belong at model level)."""
        params = {
            'n_layers': 2,
            'd_model': 64,
            'd_ffn': 256,
            'sa_n_heads': 4,
            'kernel_size': 1,
            'use_moe': True,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_z_loss_coeff': 0.001,  # Should cause error - belongs at model level
        }

        # Should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            transformer = Transformer(**params)

    def test_moe_ffn_parameter_compatibility(self):
        """Test that PositionwiseConvFFMoE accepts only valid parameters."""
        # Valid parameters
        valid_params = {
            'd_model': 64,
            'd_ffn': 256,
            'p_dropout': 0.0,
            'num_experts': 4,
            'top_k_experts': 2,
            'kernel_size': 1,
            'router_jitter_noise': 0.01,
            'routing_strategy': 'top_k',
        }

        # Should not raise
        moe_ffn = PositionwiseConvFFMoE(**valid_params)
        assert moe_ffn.num_experts == 4

    def test_moe_ffn_rejects_loss_coefficients(self):
        """Test that PositionwiseConvFFMoE rejects loss coefficient parameters (they belong at model level)."""
        params = {
            'd_model': 64,
            'd_ffn': 256,
            'p_dropout': 0.0,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_load_balancing_loss_coeff': 0.01,  # Should cause error - belongs at model level
        }

        # Should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            moe_ffn = PositionwiseConvFFMoE(**params)


@pytest.mark.unit
class TestMoEConfigYAMLCompatibility:
    """Test that YAML config structure works with refactored code."""

    def test_transformer_from_yaml_config(self):
        """Test creating Transformer from YAML config."""
        # Simulate config from YAML (decoder section)
        config_dict = {
            'n_layers': 2,
            'd_model': 64,
            'd_ffn': 256,
            'sa_n_heads': 4,
            'kernel_size': 1,
            'p_dropout': 0.0,
            'has_xattn': False,
            'is_causal': True,
            'use_moe': True,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_jitter_noise': 0.0,
            'routing_strategy': 'top_k',
            'router_load_balancing_loss_coeff': 0.01,
            'router_z_loss_coeff': 0.001,
        }

        # Filter out loss coefficients before passing to Transformer
        # Loss coefficients should only be in model config, not passed to Transformer module
        config_dict.pop('router_load_balancing_loss_coeff', None)
        config_dict.pop('router_z_loss_coeff', None)

        transformer = Transformer(**config_dict)
        assert transformer.use_moe is True

    def test_config_with_loss_coefficients_must_be_filtered(self):
        """Test that loss coefficients in config dict must be filtered out before passing to Transformer."""
        config_with_loss_coeffs = {
            'n_layers': 2,
            'd_model': 64,
            'd_ffn': 256,
            'sa_n_heads': 4,
            'kernel_size': 1,
            'use_moe': True,
            'num_experts': 4,
            'top_k_experts': 2,
            'router_load_balancing_loss_coeff': 0.01,
            'router_z_loss_coeff': 0.001,
        }

        with pytest.raises(TypeError) as exc_info:
            transformer = Transformer(**config_with_loss_coeffs)
        assert "unexpected keyword argument" in str(exc_info.value)

        config_filtered = config_with_loss_coeffs.copy()
        config_filtered.pop('router_load_balancing_loss_coeff', None)
        config_filtered.pop('router_z_loss_coeff', None)

        transformer = Transformer(**config_filtered)
        assert transformer.use_moe is True


@pytest.mark.unit
class TestMoELossIntegration:
    """Test integration between modules and losses."""

    def test_routing_info_compatible_with_loss(self):
        """Test that routing info from modules can be used with loss."""
        # Create module
        moe_ffn = PositionwiseConvFFMoE(
            d_model=32,
            d_ffn=128,
            p_dropout=0.0,
            num_experts=4,
            top_k_experts=2,
        )

        # Create loss
        loss_fn = MoEAuxiliaryLoss(
            num_experts=4,
            load_balancing_loss_scale=0.01,
            router_z_loss_scale=0.001,
        )

        # Forward through module
        x = torch.randn(2, 10, 32)
        x_mask = torch.ones(2, 10)

        moe_ffn.train()
        output, router_logits, router_probs, expert_indices = moe_ffn(x, x_mask)

        load_balancing_loss, router_z_loss, total_loss = loss_fn(
            router_logits=router_logits, router_probs=router_probs, x_mask=x_mask
        )

        # Should work without errors
        assert total_loss.item() >= 0

    def test_transformer_routing_info_compatible_with_loss(self):
        """Test that Transformer routing info can be used with loss module."""
        # Create transformer
        transformer = Transformer(
            n_layers=2,
            d_model=32,
            d_ffn=128,
            sa_n_heads=2,
            kernel_size=1,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        # Create loss
        loss_fn = MoEAuxiliaryLoss(num_experts=4)

        # Forward
        x = torch.randn(2, 10, 32)
        x_mask = torch.ones(2, 10).bool()

        transformer.train()
        output_dict = transformer(x, x_mask)

        # Extract routing info (as done in magpietts.py)
        moe_routing_info = output_dict['moe_routing_info']
        all_logits = torch.stack([info['router_logits'] for info in moe_routing_info], dim=0)
        all_probs = torch.stack([info['router_probs'] for info in moe_routing_info], dim=0)

        merged_logits = all_logits.view(-1, all_logits.size(2), all_logits.size(3))
        merged_probs = all_probs.view(-1, all_probs.size(2), all_probs.size(3))

        # Repeat mask for each layer
        n_layers = len(moe_routing_info)
        merged_mask = x_mask.unsqueeze(0).repeat(n_layers, 1, 1).view(-1, x_mask.size(1))

        load_balancing_loss, router_z_loss, total_loss = loss_fn(
            router_logits=merged_logits, router_probs=merged_probs, x_mask=merged_mask
        )

        assert total_loss.requires_grad

    def test_loss_backward_compatibility(self):
        """Test that loss computation doesn't break backward pass."""
        transformer = Transformer(
            n_layers=2,
            d_model=32,
            d_ffn=128,
            sa_n_heads=2,
            kernel_size=1,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        loss_fn = MoEAuxiliaryLoss(num_experts=4)

        x = torch.randn(2, 10, 32)
        x_mask = torch.ones(2, 10).bool()

        transformer.train()
        output_dict = transformer(x, x_mask)

        # Compute losses
        moe_routing_info = output_dict['moe_routing_info']
        all_logits = torch.stack([info['router_logits'] for info in moe_routing_info], dim=0)
        all_probs = torch.stack([info['router_probs'] for info in moe_routing_info], dim=0)
        merged_logits = all_logits.view(-1, all_logits.size(2), all_logits.size(3))
        merged_probs = all_probs.view(-1, all_probs.size(2), all_probs.size(3))

        # Repeat mask for each layer
        n_layers = len(moe_routing_info)
        merged_mask = x_mask.unsqueeze(0).repeat(n_layers, 1, 1).view(-1, x_mask.size(1))

        load_balancing_loss, router_z_loss, moe_total_loss = loss_fn(
            router_logits=merged_logits, router_probs=merged_probs, x_mask=merged_mask
        )

        # Add dummy task loss
        task_loss = output_dict['output'].sum()
        total_loss = task_loss + moe_total_loss

        # Should be able to backward
        total_loss.backward()

        # Check gradients exist
        assert transformer.layers[0].pos_ff.router.router.weight.grad is not None
