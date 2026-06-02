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
"""Backbone-side patches applied at model ``__init__``.

Runtime fixes for the constructed ``NemotronHModel`` backbone. They live with
the model because they're inherent to running EasyMagpie SmallMamba
(``mlp_hidden_act=silu``) on vLLM's NemotronH implementation. Mirrors the
EasyMagpie vLLM *sidecar* (``easymagpie_vllm/backbone_patches.py``).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)


class _SiluActivation(nn.Module):
    """``nn.Module`` wrapper around ``F.silu`` (so vLLM's NemotronHMLP can hold it)."""

    def forward(self, x):
        return F.silu(x)


def patch_silu_shared_experts(backbone) -> int:
    """Replace ``shared_experts.act_fn`` with SiLU on every NemotronHMoE layer.

    vLLM's ``NemotronHMLP`` hard-codes ReLU² for ``shared_experts`` (ignoring
    ``config.mlp_hidden_act``). SmallMamba trained with SiLU, so the mismatch
    blows up shared-expert norms ~5× and the per-layer cosine drops to ≈-0.7 by
    layer 30. Patching only ``act_fn`` (not the whole forward) keeps
    ``NemotronHMLP.forward`` in charge so torch.compile / CUDA-graph capture
    continue to wrap it unchanged.

    Args:
        backbone: the ``NemotronHModel`` instance.

    Returns:
        Number of layers patched.
    """
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        se = getattr(mixer, "shared_experts", None)
        if se is None:
            continue
        se.act_fn = _SiluActivation()
        patched += 1
    logger.info("SiLU shared_experts fix installed on %d layers", patched)
    return patched
