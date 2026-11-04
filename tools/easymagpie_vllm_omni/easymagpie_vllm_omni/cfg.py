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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class AcousticSamplingSettings:
    """Acoustic sampling belongs to a logical request, independent of engine sampling.

    Attributes:
        use_cfg: Whether to guide acoustic predictions with an unconditional branch.
        cfg_scale: Guidance scale applied before acoustic sampling.
        temperature: Acoustic sampling temperature.
        top_k: Number of acoustic logits retained by the cutoff.
    """

    use_cfg: bool = False
    cfg_scale: float = 2.5
    temperature: float = 0.7
    top_k: int = 80

    @property
    def uses_fixed_sampling(self) -> bool:
        """Whether shared legacy sampling and guidance constants apply."""
        return self.temperature == 0.7 and self.top_k == 80 and (not self.use_cfg or self.cfg_scale == 2.5)

    @classmethod
    def from_info(cls, info: Mapping[str, Any]) -> AcousticSamplingSettings:
        """Validate request metadata and construct acoustic sampling settings.

        Args:
            info: Additional request metadata.

        Returns:
            Validated per-request acoustic sampling settings.

        Raises:
            ValueError: If CFG is not boolean or acoustic sampling settings are invalid.
        """
        use_cfg = info.get("use_cfg", False)
        if type(use_cfg) is not bool:
            raise ValueError("use_cfg must be a boolean")
        scale = _finite_number(info.get("cfg_scale", 2.5), "cfg_scale")
        temperature = _finite_number(info.get("temperature", 0.7), "temperature")
        top_k = info.get("top_k", info.get("topk", 80))
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("acoustic top_k must be a positive integer")
        return cls(use_cfg, scale, temperature, top_k)


def validate_cfg_capability(
    capability: object,
    *,
    predictor_type: str,
    token_id: object,
    text_vocab_size: int | None = None,
) -> None:
    """Validate versioned CFG capability metadata when present.

    Older artifacts omit the capability and remain usable without CFG.

    Args:
        capability: Capability metadata to validate, including malformed inputs.
        predictor_type: Acoustic predictor identifier stored in the converted configuration.
        token_id: CFG unknown-token identifier to validate.
        text_vocab_size: Optional text vocabulary size used to bound the CFG token ID.

    Raises:
        ValueError: If advertised CFG metadata, predictor type, or token bounds are invalid.
    """
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


def validate_cfg_context_embedding(
    tensor: torch.Tensor,
    embedding_dim: int,
) -> None:
    """Validate before casting/copying so malformed integer weights cannot pass.

    Args:
        tensor: Embedding tensor to validate.
        embedding_dim: Expected width of the single context embedding.

    Raises:
        ValueError: If the tensor has the wrong shape, dtype, or nonfinite values.
    """
    import torch

    if tensor.shape != (1, embedding_dim) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"cfg_context_embedding must be finite floating point with shape (1, {embedding_dim})")


def _finite_number(
    value: object,
    name: str,
) -> float:
    """Validate a finite real-valued acoustic setting.

    Args:
        value: Value to validate or stage.
        name: Request or attribute name.

    Returns:
        The validated value converted to a Python float.

    Raises:
        ValueError: If the value is boolean, nonnumeric, or nonfinite.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)
