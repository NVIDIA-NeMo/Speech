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

"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz) with an AdamW fallback.

Muon (Keller Jordan et al., 2024, https://kellerjordan.github.io/posts/muon/) updates a weight matrix
with an approximately semi-orthogonal matrix derived from the momentum-averaged gradient,
computed with a quintic Newton-Schulz iteration in bfloat16.
Parameters that are not matrices (biases, normalization scales, convolution filters)
and matrices with a dimension of at least ``adam_dim_threshold`` (large embeddings and output projections)
are updated with AdamW at a fixed fraction of the learning rate,
as the public reference trainings (modded-nanogpt, nanochat) do with their embeddings and heads.
An embedding below the threshold is a matrix like any other and takes the Muon update
unless its parameter group sets ``use_muon=False``, see :class:`Muon`.
One optimizer instance handles both kinds,
so one learning-rate schedule drives both with a fixed ratio.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

__all__ = ['Muon', 'update_scale', 'zeropower_via_newtonschulz5']

SCALE_MODES = ('fan_in', 'fan_out', 'none')


class Muon(Optimizer):
    """Muon for weight matrices with an AdamW fallback for the other parameters, see the module docstring.

    Which parameters take the Muon update is decided per parameter:
    with the default ``use_muon=None`` a parameter takes it
    when it is a matrix whose larger dimension is below ``adam_dim_threshold``,
    so biases, normalization scales, convolution filters and wide embeddings or output projections fall back to AdamW.
    The shape alone does not tell an embedding from a hidden matrix:
    a small-vocabulary embedding (``1026 x 512``) is below the default threshold and takes the Muon update,
    so put embeddings and tied output projections that are to stay on AdamW
    into a parameter group with ``use_muon=False`` (and usually ``weight_decay=0.0``)::

        Muon([{'params': hidden}, {'params': [embedding.weight], 'use_muon': False, 'weight_decay': 0.0}], lr=0.02)

    A parameter group can force the choice either way with ``use_muon=True`` or ``use_muon=False``;
    a forced parameter with more than two dimensions (a convolution filter)
    is orthogonalized as the matrix ``(out_channels, -1)``.

    The orthogonalized update of a ``(rows, columns)`` matrix is multiplied by a shape-dependent factor.
    ``scale_mode='fan_in'`` (the default) uses ``sqrt(max(1, rows / columns))``,
    which for the ``(out, in)`` layout of ``torch.nn.Linear``
    gives every update the root mean square ``1 / sqrt(fan_in)`` (the modded-nanogpt convention).
    ``'fan_out'`` uses ``sqrt(max(1, columns / rows))``, the root mean square ``1 / sqrt(fan_out)``,
    which is the same rule applied to the transposed ``(in, out)`` layout of other frameworks.
    ``'none'`` applies no factor.
    The same matrix in the two layouts receives the same update, transposed, when the two modes are paired,
    up to the bfloat16 rounding of the Newton-Schulz iteration.

    Weight decay is decoupled:
    it multiplies the parameter by ``1 - lr * weight_decay`` for Muon parameters
    and by ``1 - lr * adam_lr_ratio * weight_decay`` for AdamW parameters,
    so the relative decay per step is the same fraction of the respective learning rate;
    set ``weight_decay: 0.0`` on a parameter group to exclude it.

    Args:
        params: iterable of parameters or of parameter-group dicts
        lr: learning rate of the Muon update
        momentum: momentum factor of the gradient buffer
        nesterov: apply the momentum buffer in Nesterov form
        ns_steps: number of Newton-Schulz iterations
        weight_decay: decoupled weight decay
        scale_mode: ``'fan_in'``, ``'fan_out'`` or ``'none'``, see above
        adam_betas: coefficients of the AdamW moving averages
        adam_eps: term added to the AdamW denominator
        adam_lr_ratio: the AdamW learning rate is ``lr * adam_lr_ratio``
        adam_dim_threshold: matrices with a dimension of at least this size fall back to AdamW
        use_muon: force (``True``) or exclude (``False``) the Muon update; ``None`` selects by shape
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        scale_mode: str = 'fan_in',
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_lr_ratio: float = 0.025,
        adam_dim_threshold: int = 8192,
        use_muon: Optional[bool] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid number of Newton-Schulz steps: {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not (0.0 <= adam_betas[0] < 1.0 and 0.0 <= adam_betas[1] < 1.0):
            raise ValueError(f"Invalid AdamW betas: {adam_betas}")
        if adam_eps < 0.0:
            raise ValueError(f"Invalid AdamW epsilon: {adam_eps}")
        if adam_lr_ratio < 0.0:
            raise ValueError(f"Invalid AdamW learning rate ratio: {adam_lr_ratio}")
        if adam_dim_threshold < 1:
            raise ValueError(f"Invalid AdamW dimension threshold: {adam_dim_threshold}")
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"Unknown scale_mode {scale_mode!r}, expected one of {SCALE_MODES}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            scale_mode=scale_mode,
            adam_betas=tuple(adam_betas),
            adam_eps=adam_eps,
            adam_lr_ratio=adam_lr_ratio,
            adam_dim_threshold=adam_dim_threshold,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group['scale_mode'] not in SCALE_MODES:
                raise ValueError(f"Unknown scale_mode {group['scale_mode']!r}, expected one of {SCALE_MODES}")
            if group['use_muon']:
                for param in group['params']:
                    if param.ndim < 2:
                        raise ValueError(
                            f"use_muon=True needs at least two dimensions, got shape {tuple(param.shape)}"
                        )

    @staticmethod
    def selects_muon(param: torch.Tensor, *, use_muon: Optional[bool], adam_dim_threshold: int) -> bool:
        """Whether ``param`` takes the Muon update under the given group options.

        Args:
            param: the parameter
            use_muon: the group's ``use_muon`` option, see the class docstring
            adam_dim_threshold: the group's ``adam_dim_threshold`` option

        Returns:
            True for the Muon update, False for the AdamW fallback
        """
        if use_muon is not None:
            return bool(use_muon)
        return param.ndim == 2 and max(param.shape) < adam_dim_threshold

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure: re-evaluates the model and returns the loss

        Returns:
            the loss returned by ``closure``, or None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                if self.selects_muon(
                    param, use_muon=group['use_muon'], adam_dim_threshold=group['adam_dim_threshold']
                ):
                    _muon_update(
                        param,
                        param.grad,
                        state,
                        lr=group['lr'],
                        momentum=group['momentum'],
                        nesterov=group['nesterov'],
                        ns_steps=group['ns_steps'],
                        weight_decay=group['weight_decay'],
                        scale_mode=group['scale_mode'],
                    )
                else:
                    _adamw_update(
                        param,
                        param.grad,
                        state,
                        lr=group['lr'] * group['adam_lr_ratio'],
                        betas=group['adam_betas'],
                        eps=group['adam_eps'],
                        weight_decay=group['weight_decay'],
                    )
        return loss


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix with a quintic Newton-Schulz iteration.

    For the singular value decomposition ``grad = U S V^T`` the result approximates ``U V^T``:
    the iteration pushes the singular values towards one, not exactly onto it,
    with the modded-nanogpt coefficients,
    which are tuned for a fast start of the iteration rather than for convergence to exactly orthogonal factors.
    The iteration runs in bfloat16 on the wide orientation of the matrix (rows <= columns),
    so transposing the input transposes the result;
    the result has the dtype of the input.

    Args:
        grad: a matrix of shape (rows, columns), any float dtype
        steps: number of Newton-Schulz iterations

    Returns:
        the approximately semi-orthogonal matrix, same shape and dtype as ``grad``
    """
    if grad.ndim != 2:
        raise ValueError(f"Expected a matrix, got a tensor of shape {tuple(grad.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transpose = grad.size(0) > grad.size(1)
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transpose:
        x = x.T
    return x.to(grad.dtype)


def update_scale(shape: Sequence[int], scale_mode: str) -> float:
    """Factor on the orthogonalized update of a ``(rows, columns)`` matrix.

    Args:
        shape: the two dimensions of the matrix
        scale_mode: ``'fan_in'``, ``'fan_out'`` or ``'none'``, see :class:`Muon`

    Returns:
        the factor
    """
    rows, columns = shape
    if scale_mode == 'fan_in':
        return max(1.0, rows / columns) ** 0.5
    if scale_mode == 'fan_out':
        return max(1.0, columns / rows) ** 0.5
    if scale_mode == 'none':
        return 1.0
    raise ValueError(f"Unknown scale_mode {scale_mode!r}, expected one of {SCALE_MODES}")


def _muon_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: Dict[str, Any],
    *,
    lr: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    weight_decay: float,
    scale_mode: str,
) -> None:
    if 'momentum_buffer' not in state:
        state['momentum_buffer'] = torch.zeros_like(param)
    buf = state['momentum_buffer']
    buf.mul_(momentum).add_(grad)
    update = grad.add(buf, alpha=momentum) if nesterov else buf
    matrix = update if update.ndim == 2 else update.reshape(update.size(0), -1)
    orthogonal = zeropower_via_newtonschulz5(matrix, steps=ns_steps)
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)
    param.add_(orthogonal.reshape_as(param), alpha=-lr * update_scale(matrix.shape, scale_mode))


def _adamw_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: Dict[str, Any],
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    if 'exp_avg' not in state:
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param)
        state['exp_avg_sq'] = torch.zeros_like(param)
    state['step'] += 1
    step = state['step']
    beta1, beta2 = betas
    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)
    param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
