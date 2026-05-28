"""Backbone-side patches applied at model ``__init__``.

Runtime fixes for the constructed NemotronH backbone. They live with the
model because they're inherent to running SmallMamba (``mlp_hidden_act=silu``)
on vLLM 0.19.1's NemotronH implementation.
"""
from __future__ import annotations

import logging

import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class _SiluActivation(nn.Module):
    """``nn.Module`` wrapper around ``F.silu`` (so vLLM's NemotronHMLP can hold it)."""

    def forward(self, x):
        return F.silu(x)


def patch_silu_shared_experts(backbone) -> int:
    """Replace ``shared_experts.act_fn`` with SiLU on every NemotronHMoE layer.

    vLLM's ``NemotronHMLP`` hard-codes ReLU² for ``shared_experts`` (ignoring
    ``config.mlp_hidden_act``). SmallMamba trained with SiLU, so the
    mismatch blows up shared-expert norms ~5× and the per-layer cosine
    drops to ≈-0.7 by layer 30. Patching only ``act_fn`` (not the whole
    forward) keeps ``NemotronHMLP.forward`` in charge so torch.compile /
    CUDA-graph capture continue to wrap it unchanged.

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
