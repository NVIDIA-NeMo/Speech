"""Backbone-side patches V2 needs at __init__ time.

Runtime fixes applied to the constructed NemotronH backbone — they belong
with the model class, not the server, because they're an inherent property
of running SmallMamba checkpoints (``mlp_hidden_act=silu``) on the vLLM
0.19.1 NemotronH implementation.
"""
from __future__ import annotations

import logging

import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class _SiluActivation(nn.Module):
    def forward(self, x):
        return F.silu(x)


def patch_silu_shared_experts(backbone) -> int:
    """Replace ``shared_experts.act_fn`` on every NemotronHMoE layer with SiLU.

    vLLM's ``NemotronHMLP.forward`` (used as ``shared_experts`` inside
    ``NemotronHMoE``) hard-codes ``self.act_fn = ReLUSquaredActivation()``
    in nemotron_h.py line 118, ignoring ``config.mlp_hidden_act``. For
    SmallMamba (``mlp_hidden_act="silu"``) this produces shared-expert
    output norms ~5x production's, compounding across the hybrid backbone
    (verified via prefill_diff.py: per-layer cosine drops to -0.7 by
    layer 30 without this fix; restoring SiLU gives cos 1.0 at every
    layer). The routed experts (``FusedMoE``) correctly use SILU_NO_MUL
    already, so we only need to patch ``shared_experts.act_fn``.

    Replacing only ``act_fn`` (not the whole forward) is the minimal,
    decode-path-safe fix: vLLM's NemotronHMLP.forward stays in charge,
    so any CUDA-graph / compile capture continues to wrap it normally.

    ``backbone`` is V2's composed ``self.backbone`` (a ``NemotronHModel``).
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
