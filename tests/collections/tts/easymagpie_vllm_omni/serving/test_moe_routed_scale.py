# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the pinned vLLM runner's routed/shared-expert scaling contract."""
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.backbone_patches import patch_moe_routed_scale
from torch import nn
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('shared_experts', [False, True])
def test_routed_scale_matches_native_with_and_without_shared_experts(dtype, shared_experts):
    backbone = _backbone(shared_experts)
    patched = patch_moe_routed_scale(backbone)
    values = torch.tensor([[1.0, -2.0, 3.0]], dtype=dtype)
    actual = backbone.layers[0].mixer(values)
    expected = values * 2.5 + (values * 0.5 if shared_experts else 0)
    torch.testing.assert_close(actual, expected)
    assert patched == int(shared_experts)


def test_routed_scale_patch_is_idempotent():
    backbone = _backbone(True)
    assert patch_moe_routed_scale(backbone) == 1
    assert patch_moe_routed_scale(backbone) == 0
    values = torch.tensor([[2.0]], dtype=torch.float16)
    torch.testing.assert_close(backbone.layers[0].mixer(values), values * 3.0)


def _backbone(has_shared_experts):
    class NemotronHMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_scaling_factor = 2.5
            self.shared_experts = nn.Identity() if has_shared_experts else None

        def forward(self, values):
            shared = values * 0.5 if self.shared_experts is not None else None
            # Exercise the actual pinned vLLM runner's dtype-dependent scaling,
            # without constructing expert weights or allocating a GPU.
            shared, routed = MoERunner._maybe_apply_routed_scale_to_output(
                SimpleNamespace(routed_scaling_factor=self.routed_scaling_factor), shared, values.clone()
            )
            return routed if shared is None else routed + shared

    return SimpleNamespace(layers=[SimpleNamespace(mixer=NemotronHMoE())])
