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

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.v1.attention.backends.mamba_attn as _mamba_attn
from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_mamba_streaming_decode() -> None:
    """Treat 1-token streaming extends as decodes so FULL decode cudagraphs work.

    EasyMagpie's streaming-input path keeps extending each request's prompt with
    every chunk, so ``num_computed_tokens < num_prompt_tokens`` (the engine's
    ``is_prefilling`` flag) stays True for the whole stream. vLLM's Mamba2
    metadata builder calls
    :func:`vllm.v1.attention.backends.utils.split_decodes_and_prefills` with
    ``treat_short_extends_as_decodes=False``, so every single-token decode step
    is classified as a *prefill* (``num_prefills>0``).

    That collides with the cudagraph dispatcher, which keys only on query length:
    a uniform ``query_len==1`` batch dispatches the **FULL decode** graph
    regardless of ``is_prefilling``. Two failures result:

    * the replayed decode graph runs the decode Mamba kernels while the metadata
      says prefill, and
    * because ``num_prefills>0``, ``_update_metadata_for_cudagraph_capture``
      never refreshes the persistent ``state_indices_tensor_d`` buffer, so the
      captured kernel reads the capture-time dummy slot (0) instead of the
      request's real Mamba-cache slot -> garbage hidden states.

    Forcing ``treat_short_extends_as_decodes=True`` makes single-token extends
    classify as decodes (``num_prefills==0``), which both matches the dispatched
    FULL decode graph and re-enables the per-step ``state_indices_tensor_d``
    refresh. Multi-token context prefills (``query_len>1``) still classify as
    prefills, so this is safe for mixed batches. Advancing Mamba state by one
    token via the decode kernels is semantically identical to a 1-token prefill
    chunk (it reads the slot's state and writes the advanced state back in
    place), so no state update is lost — the only requirement is exactly one new
    token per streamed step (``SamplingParams(max_tokens=1)``).

    Idempotent and process-global; the EasyMagpie plugin only ever serves this
    model so the global patch is acceptable.
    """
    orig = _mamba_attn.split_decodes_and_prefills
    if getattr(orig, "_easymagpie_patched", False):
        return

    def patched(
        common_attn_metadata,
        decode_threshold: int = 1,
        require_uniform: bool = False,
        treat_short_extends_as_decodes: bool = True,
    ):
        return orig(
            common_attn_metadata,
            decode_threshold=decode_threshold,
            require_uniform=require_uniform,
            treat_short_extends_as_decodes=True,
        )

    patched._easymagpie_patched = True
    _mamba_attn.split_decodes_and_prefills = patched
    logger.info("Mamba streaming-decode classification patch installed")


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


def patch_moe_routed_scale(backbone) -> int:
    """Restore ``routed_scaling_factor`` on the NemotronHMoE output in FP16.

    vLLM's ``FusedMoE`` uses an FP16 overflow trick: with
    ``apply_routed_scale_to_output=True`` it does **not** multiply the routed
    output by ``s`` (=routed_scaling_factor); in FP16 it instead divides the
    *shared* output by ``s`` and relies on the decoder layer to keep the whole
    residual stream scaled by ``1/s`` (see ``DeepseekV2DecoderLayer.forward``).
    NemotronH's decoder layer never applies that compensation, so in FP16 the
    MoE block emits ``routed_raw + shared/s == (s*routed + shared)/s`` — the
    correct value divided by ``s``. The MoE contribution to the residual ends up
    ``s``× too small and the error accumulates across the MoE layers.

    We re-multiply each MoE mixer's output by ``s`` in FP16::

        s * (routed_raw + shared/s) = s*routed_raw + shared

    which matches the NeMo reference. FP32/BF16 already take the correct
    ``fused_output *= s`` branch, so the hook is a no-op there.

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
        scale = float(getattr(mixer, "routed_scaling_factor", 1.0))
        if scale == 1.0:
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
