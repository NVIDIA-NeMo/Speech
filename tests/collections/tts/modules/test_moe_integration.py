# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

These tests verify cross-module contracts between:
- Modules (moe_modules.py)
- Losses (moe_loss.py)
- Transformer (transformer_2501.py)
"""

import pytest
import torch

from nemo.collections.tts.losses.moe_loss import MoEAuxiliaryLoss
from nemo.collections.tts.modules.moe_modules import PositionwiseConvFFMoE
from nemo.collections.tts.modules.transformer_2501 import Transformer, TransformerLayer


@pytest.mark.unit
class TestMoEIntegration:
    """Integration tests for MoE pipeline: modules, losses, and config handling."""

    def test_complete_moe_pipeline(self):
        """Test complete flow: Transformer -> routing_info -> Loss computation."""
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

        loss_module = MoEAuxiliaryLoss(
            num_experts=4,
            load_balancing_loss_scale=0.01,
            router_z_loss_scale=0.001,
        )

        x = torch.randn(2, 10, 64)
        x_mask = torch.ones(2, 10).bool()

        transformer.train()
        output_dict = transformer(x, x_mask)

        # Extract routing info
        moe_routing_info = output_dict['moe_routing_info']
        assert moe_routing_info is not None
        assert len(moe_routing_info) == 2  # n_layers

        all_logits = torch.stack([info['router_logits'] for info in moe_routing_info], dim=0)
        all_probs = torch.stack([info['router_probs'] for info in moe_routing_info], dim=0)

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

    def test_transformer_from_yaml_config(self):
        """Test creating Transformer from YAML-style config dict."""
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
        }

        transformer = Transformer(**config_dict)
        assert transformer.use_moe is True

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (
                TransformerLayer,
                {
                    'd_model': 64,
                    'd_ffn': 256,
                    'sa_n_heads': 4,
                    'kernel_size': 1,
                    'p_dropout': 0.0,
                    'has_xattn': False,
                    'use_moe': True,
                    'num_experts': 4,
                    'top_k_experts': 2,
                    'router_load_balancing_loss_coeff': 0.01,
                },
            ),
            (
                Transformer,
                {
                    'n_layers': 2,
                    'd_model': 64,
                    'd_ffn': 256,
                    'sa_n_heads': 4,
                    'kernel_size': 1,
                    'use_moe': True,
                    'num_experts': 4,
                    'top_k_experts': 2,
                    'router_z_loss_coeff': 0.001,
                },
            ),
            (
                PositionwiseConvFFMoE,
                {
                    'd_model': 64,
                    'd_ffn': 256,
                    'p_dropout': 0.0,
                    'num_experts': 4,
                    'top_k_experts': 2,
                    'router_load_balancing_loss_coeff': 0.01,
                },
            ),
        ],
        ids=["TransformerLayer", "Transformer", "PositionwiseConvFFMoE"],
    )
    def test_loss_coefficients_rejected_by_modules(self, cls, kwargs):
        """Test that MoE modules reject loss coefficient parameters (they belong at model level)."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            cls(**kwargs)
