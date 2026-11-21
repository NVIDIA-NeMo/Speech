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

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch

from nemo.core.classes import Loss

__all__ = ["CompositeLoss", "LossTerm"]


@dataclass(frozen=True)
class LossTerm:
    """
    One loss as the sum over its elements and the number of elements that normalizes it.

    ``summed / count`` is the mean of the loss over the elements it counts
    (target tokens, target labels, frames or sequences),
    so each term defines its own normalization.

    Args:
        summed: scalar tensor, the loss summed over its elements
        count: scalar tensor, the number of elements the sum runs over
    """

    summed: torch.Tensor
    count: torch.Tensor

    @classmethod
    def from_elements(cls, losses: torch.Tensor, mask: Optional[torch.Tensor] = None) -> "LossTerm":
        """
        Per-element losses, normalized by the number of valid elements.

        Args:
            losses: per-element losses of any shape; masked-out elements may hold any finite value
            mask: valid elements, same shape as ``losses``, boolean or 0/1 values; all elements are valid if None

        Returns:
            the term with the masked sum and the number of valid elements
        """
        if mask is None:
            return cls(losses.sum(), torch.tensor(losses.numel(), device=losses.device, dtype=losses.dtype))
        mask = mask.to(losses.dtype)
        return cls((losses * mask).sum(), mask.sum())

    @classmethod
    def from_sequences(cls, losses: torch.Tensor, lengths: torch.Tensor) -> "LossTerm":
        """
        Per-sequence losses, normalized by the per-sequence element counts,
        for example a CTC loss per sequence normalized by the target lengths.

        Args:
            losses: per-sequence losses of shape ``(B,)``
            lengths: elements per sequence of shape ``(B,)``

        Returns:
            the term with the summed losses and the summed lengths
        """
        return cls(losses.sum(), lengths.to(losses.dtype).sum())


class CompositeLoss(Loss):
    """
    Sums separately normalized loss terms with per-term scales, and reports the normalized value of each term.

    Every term is a :class:`LossTerm` with its own element count and contributes ``scale * summed / count``:
    a cross-entropy over target tokens stays a mean per token and a CTC loss stays a mean per target label,
    whatever the other terms count.
    A term with scale 0 is reported but not trained on.

    ``normalization`` says how the mean is formed across the ranks of data-parallel training,
    whose element counts differ from batch to batch:

    - ``"per_rank"``: every rank normalizes by its own count;
      the gradient averaging of data-parallel training then weights the ranks equally
      (the mean of the per-rank means).
    - ``"global"``: the counts are summed over all ranks,
      so the result is the mean over all elements of the global batch;
      every rank's loss is scaled by the world size to undo the gradient averaging,
      and the reported values are the global means as well.

    Without an initialized process group the two are the same.

    Args:
        scales: scale per term name; terms not listed use ``default_scale``
        default_scale: scale of the terms that are not listed in ``scales``
        normalization: ``"per_rank"`` or ``"global"``, see above
    """

    NORMALIZATIONS = ("per_rank", "global")

    def __init__(
        self,
        scales: Optional[Mapping[str, float]] = None,
        default_scale: float = 1.0,
        normalization: str = "per_rank",
    ):
        super().__init__()
        if normalization not in self.NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {self.NORMALIZATIONS}, got {normalization!r}")
        self.scales = {str(name): float(scale) for name, scale in (scales or {}).items()}
        self.default_scale = float(default_scale)
        self.normalization = normalization

    def scale(self, name: str) -> float:
        """The scale of the term ``name``."""
        return self.scales.get(name, self.default_scale)

    def forward(self, terms: Mapping[str, LossTerm]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            terms: the loss terms by name; every name in ``scales`` must be present

        Returns:
            the loss to train on, and the detached normalized value of every term by name
        """
        if not terms:
            raise ValueError("CompositeLoss needs at least one term")
        unknown = sorted(set(self.scales) - set(terms))
        if unknown:
            raise ValueError(f"Scales configured for terms that were not provided: {unknown}")
        names = list(terms)
        summed = torch.stack([terms[name].summed for name in names])
        counts = torch.stack([terms[name].count.to(summed.dtype) for name in names])
        world_size = _world_size()
        if self.normalization == "global" and world_size > 1:
            reduced = torch.cat([summed.detach(), counts]).clone()
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            global_summed, counts = reduced[: len(names)], reduced[len(names) :]
            denominators = counts.clamp(min=1.0)
            # The data-parallel average over the ranks divides by the world size; the global mean must not.
            normalized = summed * world_size / denominators
            reported = global_summed / denominators
        else:
            denominators = counts.clamp(min=1.0)
            normalized = summed / denominators
            reported = normalized.detach()
        scales = torch.tensor([self.scale(name) for name in names], dtype=summed.dtype, device=summed.device)
        total = (normalized * scales).sum()
        return total, {name: value for name, value in zip(names, reported.unbind())}


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1
