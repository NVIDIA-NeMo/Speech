# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Address-stable, fixed-shape CFG inputs for decode graph replay."""
from __future__ import annotations

import torch
from easymagpie_vllm_omni.cfg_pairing import CFGPairingLayout
from vllm.config import CompilationMode, CUDAGraphMode


def supports_cfg_graphs(compilation) -> bool:
    """Allow decode-only capture or compiled backbone prefill with full decode capture.

    The outer model and V2 predictor stay eager during piecewise prefill; only
    the decorated Nemotron-H backbone is compiled. Both policies need the same
    address-stable pairing buffers for full decode replay.
    """
    return (getattr(compilation, 'mode', None), getattr(compilation, 'cudagraph_mode', None)) in (
        (CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY),
        (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE),
    )


def uses_cfg_decode_graphs(config) -> bool:
    """Select the fixed-shape path only for the explicit paired synthesis scheduler."""
    scheduler = config.scheduler_config.scheduler_cls
    if isinstance(scheduler, type):
        scheduler = f"{scheduler.__module__}.{scheduler.__qualname__}"
    return (
        scheduler == "easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler"
        and not config.model_config.enforce_eager
        and supports_cfg_graphs(config.compilation_config)
    )


class CFGGraphBuffers:
    """Physical-row ownership staged by the host before each model invocation.

    Capture always executes the same operations, including when no pair is live.
    Private rows gather their public owner's sampled frame; unpaired rows own
    themselves. Inactive padding is masked independently from branch ownership.
    These are scratch inputs, never cache allocations or checkpoint weights.
    """

    def __init__(self, capacity: int, *, device=None) -> None:
        self.owners = torch.arange(capacity, device=device)
        self.partners = torch.arange(capacity, device=device)
        self.active = torch.zeros(capacity, dtype=torch.bool, device=device)

    def update(self, layout: CFGPairingLayout | None, num_tokens: int, padded_tokens: int) -> None:
        """Stage the current unpadded mapping outside graph capture/replay.

        Rebuild the entire padded prefix, including identity entries, so slot
        reuse and transitions to conditional-only traffic cannot retain a pair.
        Layout alignment and request identities are validated by the runner.
        """
        if not 0 <= num_tokens <= padded_tokens <= self.owners.numel():
            raise ValueError('CFG graph token count exceeds buffer capacity')
        owners = list(range(padded_tokens))
        partners = list(range(padded_tokens))
        if layout is not None:
            for conditional, unconditional in zip(layout.conditional_rows, layout.unconditional_rows):
                if len(conditional) != len(unconditional):
                    raise ValueError('CFG graph branch lengths must agree')
                for public, private in zip(conditional, unconditional):
                    if not (0 <= public < num_tokens and 0 <= private < num_tokens):
                        raise ValueError('CFG graph row is outside the active batch')
                    owners[private] = public
                    partners[public] = private
        self.owners[:padded_tokens].copy_(torch.tensor(owners, dtype=torch.long, device='cpu'))
        self.partners[:padded_tokens].copy_(torch.tensor(partners, dtype=torch.long, device='cpu'))
        self.active[:padded_tokens].zero_()
        self.active[:num_tokens].fill_(True)
