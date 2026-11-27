# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Nemotron-H expert compatibility for EasyMagpie checkpoints."""
from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import ReLUSquaredActivation, get_act_fn

logger = init_logger(__name__)


def patch_shared_expert_activation(backbone) -> int:
    """Make shared experts honor ``mlp_hidden_act`` from the model config.

    Nemotron-H's shared experts default to ReLU². NeMo uses the configured
    activation for both shared and routed experts.
    """
    activation_name = getattr(getattr(backbone, "config", None), "mlp_hidden_act", None)
    if not isinstance(activation_name, str) or not activation_name:
        raise ValueError("Nemotron-H config must provide a non-empty mlp_hidden_act")

    expected_type = type(get_act_fn(activation_name))
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        se = getattr(mixer, "shared_experts", None)
        if se is None:
            continue
        if isinstance(se.act_fn, expected_type):
            continue
        if not isinstance(se.act_fn, ReLUSquaredActivation):
            raise RuntimeError(
                "vLLM Nemotron-H shared-expert activation implementation changed; "
                "review the compatibility patch before replacing it"
            )
        se.act_fn = get_act_fn(activation_name)
        patched += 1
    logger.info("%s shared-expert activation fix installed on %d layers", activation_name, patched)
    return patched


def patch_moe_routed_scale(backbone) -> int:
    """Restore the FP16 scale deferred by the shared-expert MoE runner."""
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        scale = float(getattr(mixer, "routed_scaling_factor", 1.0))
        if scale == 1.0 or mixer.shared_experts is None:
            continue

        def _scale_output(_mod, _inp, out, _scale=scale):
            # FusedMoE only defers the scale in FP16; leave other dtypes alone.
            if isinstance(out, torch.Tensor) and out.dtype == torch.float16:
                return out * _scale
            return out

        mixer.register_forward_hook(_scale_output)
        patched += 1
    logger.info("FP16 MoE routed-scale fix installed on %d layers", patched)
    return patched
