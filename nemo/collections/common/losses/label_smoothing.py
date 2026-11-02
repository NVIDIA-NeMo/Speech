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

from typing import Optional, Sequence, Tuple, Union

import torch

__all__ = ["label_smoothed_log_prob_gradient"]


def label_smoothed_log_prob_gradient(
    log_probs: torch.Tensor,
    smoothing: Union[float, torch.Tensor],
    *,
    dim: int = -1,
    exclude_labels: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Returns ``log_probs`` unchanged, with a gradient that corresponds to label smoothing.

    For a loss of the form ``-sum_i target[i] * log_probs[i]`` (cross-entropy, negative log-likelihood),
    the gradient with respect to ``log_probs`` is ``-target``.
    This function keeps the loss value
    but replaces that gradient by the gradient of the same loss with the smoothed targets:
    ``1 - smoothing`` on the target label and ``smoothing / (C - 1)`` on each other label,
    ``C`` being the number of labels.
    The reported loss therefore stays the unsmoothed one,
    while the parameters receive the gradient of the label-smoothed loss.

    The gradient of the loss with respect to ``log_probs``
    must be minus a (scaled) probability distribution over ``dim``.
    This holds for a negative log-likelihood on log-probabilities,
    and for a CTC or RNNT loss whose backward is taken with respect to the log-probabilities.
    ``torch.nn.functional.ctc_loss`` returns the gradient with respect to the logits behind the log-probabilities
    (``softmax - posterior``), so it is not compatible with this function.

    The idea and the form of the gradient transformation follow ``label_smoothed_log_prob_gradient`` of the RETURNN
    frontend (https://github.com/rwth-i6/returnn, ``returnn.frontend.label_smoothing``),
    which trains attention encoder-decoder models with label smoothing this way.

    Args:
        log_probs: log-probabilities of any shape with the label dimension ``dim``
        smoothing: label smoothing weight in ``[0, 1]``, a float or a scalar tensor (for a scheduled value);
            a tensor value is not range-checked, which would synchronize with the device
        dim: the label dimension
        exclude_labels: labels that receive no smoothing mass and keep their gradient, such as a blank label;
            the mass of the other labels is then smoothed among those ``C - len(exclude_labels)`` labels,
            with the floor ``smoothing / (C - len(exclude_labels) - 1)``

    Returns:
        ``log_probs``, with the smoothed gradient attached
    """
    if isinstance(smoothing, torch.Tensor):
        if smoothing.dim() != 0:
            raise ValueError(f"smoothing must be a scalar tensor, got shape {tuple(smoothing.shape)}")
    elif not 0.0 <= smoothing <= 1.0:
        raise ValueError(f"smoothing must be in [0, 1], got {smoothing!r}")
    elif smoothing == 0:
        return log_probs
    num_labels = log_probs.shape[dim]
    keep = None
    if not exclude_labels:
        if num_labels < 2:
            raise ValueError(f"Label smoothing needs at least two labels, got {num_labels}")
        floor = smoothing / (num_labels - 1)
        factor = 1.0 - num_labels * floor
    else:
        excluded = sorted(set(int(label) for label in exclude_labels))
        if excluded[0] < 0 or excluded[-1] >= num_labels:
            raise ValueError(f"exclude_labels must be in [0, {num_labels}), got {excluded}")
        if len(excluded) >= num_labels - 1:
            raise ValueError(
                f"Label smoothing needs at least two participating labels, "
                f"got {num_labels} labels with {len(excluded)} excluded"
            )
        floor = smoothing / (num_labels - len(excluded) - 1)
        factor = 1.0 - (num_labels - len(excluded)) * floor
        keep = torch.ones(num_labels, dtype=log_probs.dtype, device=log_probs.device)
        keep[excluded] = 0.0
        shape = [1] * log_probs.dim()
        shape[dim] = num_labels
        keep = keep.view(shape)
        factor = keep * factor + (1.0 - keep)
        floor = keep * floor
    return _LabelSmoothedLogProbGradient.apply(log_probs, factor, floor, keep, dim)


class _LabelSmoothedLogProbGradient(torch.autograd.Function):
    """Identity in the forward pass, label-smoothed gradient in the backward pass."""

    @staticmethod
    def forward(
        ctx,
        log_probs: torch.Tensor,
        factor: Union[float, torch.Tensor],
        floor: Union[float, torch.Tensor],
        keep: Optional[torch.Tensor],
        dim: int,
    ) -> torch.Tensor:
        """Returns ``log_probs`` and keeps the smoothing factors for the backward pass."""
        ctx.factor = factor
        ctx.floor = floor
        ctx.keep = keep
        ctx.dim = dim
        return log_probs

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """Scales the gradient and shifts it by the floor times the redistributed mass."""
        # The incoming gradient is minus a (scaled) distribution,
        # so its absolute sum over the included labels is the mass to redistribute.
        mass = grad.abs() if ctx.keep is None else grad.abs() * ctx.keep
        total = mass.sum(dim=ctx.dim, keepdim=True)
        return grad * ctx.factor - ctx.floor * total, None, None, None, None
