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

"""Dropout with one mask over the feature axis per sequence, shared by every position of the sequence."""

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext

__all__ = ['BroadcastDropout']


class BroadcastDropout(nn.Module):
    """Dropout with one mask over the last axis per sequence, shared by every position of the sequence.

    ``torch.nn.Dropout`` draws a mask per element.
    This module draws one mask of the feature width for every sequence of the batch
    and broadcasts it over the time axis, the second to last one,
    so a sequence loses the same features at every position, and every sequence its own;
    the kept features are scaled by ``1 / (1 - p)``.
    Sequence-packed states, token-flat rows without a batch axis, get the same masks:
    with their packing context every row takes the mask of its sequence.

    Args:
        p: dropout probability in ``[0, 1]``, as for ``torch.nn.Dropout``
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def extra_repr(self) -> str:
        return f"p={self.p}"

    def forward(self, x: torch.Tensor, context: PackedAttentionContext | None = None) -> torch.Tensor:
        """
        Args:
            x: padded states with time on the second to last axis and features on the last one,
                shape ``(..., T, D)`` with at least one batch axis;
                or, with ``context``, sequence-packed states of shape ``(T_total, D)``
            context: the packing of token-flat states, which gives the sequence of every row

        Returns:
            ``x`` itself in evaluation mode or with ``p = 0``;
            otherwise ``x`` with the dropped features zeroed and the kept ones scaled, same shape and dtype.
        """
        if not self.training or self.p == 0.0:
            return x
        if context is not None:
            if x.dim() != 2 or x.shape[0] != context.total_tokens:
                raise ValueError(
                    f"BroadcastDropout with a packing context of {context.total_tokens} tokens needs states"
                    f" of shape (T_total, D), got {tuple(x.shape)}"
                )
            masks = F.dropout(x.new_ones(context.lengths.numel(), x.shape[-1]), self.p)
            return x * masks.index_select(0, context.sequence_ids)
        if x.dim() < 3:
            raise ValueError(
                f"BroadcastDropout needs padded states of shape (..., T, D) with a batch axis, got {tuple(x.shape)};"
                " sequence-packed states (T_total, D) need their packing context"
            )
        return x * F.dropout(x.new_ones(*x.shape[:-2], 1, x.shape[-1]), self.p)
