# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired-request CFG contracts shared by the scheduler, model and serving API."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

CFG_COMPANION_PREFIX = '__easymagpie_cfg_unconditional__:'


@dataclass(frozen=True)
class CFGOptions:
    """Native NeMo convention: ``s * conditional + (1-s) * unconditional``."""

    enabled: bool = False
    scale: float = 2.5
    max_prefill_tokens: int = 256

    @classmethod
    def from_hf_config(cls, config: Any) -> CFGOptions:
        options = cls(
            enabled=bool(getattr(config, 'easymagpie_cfg', False)) or os.environ.get('EASYMAGPIE_CFG_ENABLED') == '1',
            scale=float(os.environ.get('CFG_SCALE', getattr(config, 'easymagpie_cfg_scale', 2.5))),
            max_prefill_tokens=int(getattr(config, 'easymagpie_cfg_max_prefill_tokens', 256)),
        )
        options.resolve_scale(options.scale)
        if options.max_prefill_tokens < 2:
            raise ValueError('easymagpie_cfg_max_prefill_tokens must be at least two')
        return options

    def resolve_scale(self, value: Any = None) -> float:
        value = self.scale if value is None else value
        if isinstance(value, bool):
            raise ValueError('cfg_scale must be a finite non-negative number')
        try:
            scale = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError('cfg_scale must be a finite non-negative number') from exc
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('cfg_scale must be a finite non-negative number')
        return scale


def guided_logits(conditional: torch.Tensor, unconditional: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mix finite, unmasked logits in fp32 before top-k / forbidden-token masking."""
    conditional = conditional.float()
    unconditional = unconditional.float()
    return conditional + (scale - 1.0) * (conditional - unconditional)


def unconditional_prefill(unknown: torch.Tensor, *, offset: int, length: int, conditioning_len: int) -> torch.Tensor:
    """NeMo masks the context with CFG_UNK and removes text/phoneme warm-up rows."""
    result = unknown.new_zeros((length, unknown.numel()))
    context_rows = max(0, min(length, conditioning_len - offset))
    result[:context_rows] = unknown.reshape(1, -1)
    return result


def build_cfg_pairs(records: list[tuple[str, str, int, int, float]]) -> list[tuple[str, int, int, float]]:
    """Find the final query rows of each pair without assuming scheduler row order."""
    grouped: dict[str, dict[str, tuple[int, int, float]]] = {}
    for key, role, start, length, scale in records:
        if role not in {'conditional', 'unconditional'} or length < 1:
            raise ValueError('Invalid CFG pair role or query length')
        roles = grouped.setdefault(key, {})
        if role in roles:
            raise ValueError(f'Duplicate CFG pair role for {key}')
        roles[role] = (start, length, scale)
    pairs = []
    for key, roles in sorted(grouped.items()):
        if set(roles) != {'conditional', 'unconditional'}:
            raise ValueError(f'CFG pair {key} was not scheduled together')
        conditional, unconditional = roles['conditional'], roles['unconditional']
        if conditional[1] != unconditional[1]:
            raise ValueError(f'CFG pair {key} has mismatched query length')
        if conditional[2] != unconditional[2]:
            raise ValueError(f'CFG pair {key} has mismatched scales')
        pairs.append(
            (key, conditional[0] + conditional[1] - 1, unconditional[0] + unconditional[1] - 1, conditional[2])
        )
    return pairs
