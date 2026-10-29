# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Validated request settings and versioned V2 CFG artifact contract."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class AcousticSamplingSettings:
    """Acoustic sampling belongs to a logical request, independent of engine sampling."""

    use_cfg: bool = False
    cfg_scale: float = 2.5
    temperature: float = 0.7
    top_k: int = 80

    @classmethod
    def from_info(cls, info: Mapping) -> AcousticSamplingSettings:
        use_cfg = info.get("use_cfg", False)
        if type(use_cfg) is not bool:
            raise ValueError("use_cfg must be a boolean")
        scale = _finite_number(info.get("cfg_scale", 2.5), "cfg_scale")
        temperature = _finite_number(info.get("temperature", 0.7), "temperature")
        top_k = info.get("top_k", info.get("topk", 80))
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("acoustic top_k must be a positive integer")
        return cls(use_cfg, scale, temperature, top_k)


def validate_cfg_capability(capability, *, predictor_type: str, token_id, text_vocab_size=None) -> None:
    """Older artifacts omit the capability and remain usable without CFG."""
    if capability is None:
        return
    if (
        not isinstance(capability, dict)
        or set(capability) != {"version", "context_cas_disabled"}
        or type(capability.get("version")) is not int
        or capability["version"] != 1
        or type(capability.get("context_cas_disabled")) is not bool
        or predictor_type != "acoustic_codes_predictor"
        or type(token_id) is not int
        or token_id < 0
        or (text_vocab_size is not None and token_id >= text_vocab_size)
    ):
        raise ValueError("Invalid V2 CFG capability metadata; reconvert the NeMo checkpoint")


def validate_cfg_context_embedding(tensor, embedding_dim: int) -> None:
    """Validate before casting/copying so malformed integer weights cannot pass."""
    import torch

    if tensor.shape != (1, embedding_dim) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"cfg_context_embedding must be finite floating point with shape (1, {embedding_dim})")


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)
