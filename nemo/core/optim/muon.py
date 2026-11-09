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

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

    ``foreach`` selects the form of the step.
    The multi-tensor form (``True``, the default) runs one batched Newton-Schulz iteration
    for all matrices of one shape
    and the momentum, weight decay and AdamW arithmetic as ``torch._foreach`` operations,
    a few dozen kernel launches per step instead of a sequence per parameter;
    it allocates the stacked copies of the largest shape group and their iteration intermediates
    while it runs.
    The per-parameter loop (``False``) updates one parameter after the other.
    Both forms keep the same per-parameter state, so a state dict moves between them,
    and compute the same arithmetic for float32 parameters;
    their updates differ only where the batched matrix products of the Newton-Schulz iteration
    round differently in bfloat16.
    With bfloat16 or float16 parameters on CPU the multi-tensor form applies the scalar coefficients
    (``momentum``, the AdamW coefficients, the decay factor) rounded to the parameter dtype,
    as the CPU ``torch._foreach`` scalar operations do; the CUDA operations apply them exactly.

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
        foreach: the multi-tensor form of the step (``True``, faster, more transient memory)
            or the per-parameter loop (``False``), see above
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
        foreach: bool = True,
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
            foreach=foreach,
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

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        # an optimizer state saved before the option existed has no `foreach` in its parameter groups
        for group in self.param_groups:
            group.setdefault('foreach', True)

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
            muon_params: List[torch.Tensor] = []
            adamw_params: List[torch.Tensor] = []
            for param in group['params']:
                if param.grad is None:
                    continue
                if self.selects_muon(
                    param, use_muon=group['use_muon'], adam_dim_threshold=group['adam_dim_threshold']
                ):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
            muon_options = dict(
                lr=group['lr'],
                momentum=group['momentum'],
                nesterov=group['nesterov'],
                ns_steps=group['ns_steps'],
                weight_decay=group['weight_decay'],
                scale_mode=group['scale_mode'],
            )
            adamw_options = dict(
                lr=group['lr'] * group['adam_lr_ratio'],
                betas=group['adam_betas'],
                eps=group['adam_eps'],
                weight_decay=group['weight_decay'],
            )
            if group['foreach']:
                if muon_params:
                    _muon_update_foreach(
                        muon_params,
                        [param.grad for param in muon_params],
                        [self.state[param] for param in muon_params],
                        **muon_options,
                    )
                if adamw_params:
                    _adamw_update_foreach(
                        adamw_params,
                        [param.grad for param in adamw_params],
                        [self.state[param] for param in adamw_params],
                        **adamw_options,
                    )
            else:
                for param in muon_params:
                    _muon_update(param, param.grad, self.state[param], **muon_options)
                for param in adamw_params:
                    _adamw_update(param, param.grad, self.state[param], **adamw_options)
        return loss


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix or of a batch of matrices with a quintic Newton-Schulz iteration.

    For the singular value decomposition ``grad = U S V^T`` the result approximates ``U V^T``:
    the iteration pushes the singular values towards one, not exactly onto it,
    with the modded-nanogpt coefficients,
    which are tuned for a fast start of the iteration rather than for convergence to exactly orthogonal factors.
    The iteration runs in bfloat16 on the wide orientation of the matrix (rows <= columns),
    so transposing the input transposes the result;
    the result has the dtype of the input.
    A batch of matrices is iterated as one, with batched matrix products.

    Args:
        grad: a matrix of shape (rows, columns) or a batch of matrices of shape (n, rows, columns), any float dtype
        steps: number of Newton-Schulz iterations

    Returns:
        the approximately semi-orthogonal matrix or matrices, same shape and dtype as ``grad``
    """
    if grad.ndim not in (2, 3):
        raise ValueError(f"Expected a matrix or a batch of matrices, got a tensor of shape {tuple(grad.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transpose = grad.size(-2) > grad.size(-1)
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transpose:
        x = x.mT
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


def _muon_update_foreach(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    states: List[Dict[str, Any]],
    *,
    lr: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    weight_decay: float,
    scale_mode: str,
) -> None:
    bufs: List[torch.Tensor] = []
    for param, state in zip(params, states):
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(param)
        bufs.append(state['momentum_buffer'])
    torch._foreach_mul_(bufs, momentum)
    torch._foreach_add_(bufs, grads)
    if weight_decay != 0:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
    # one batched Newton-Schulz iteration per matrix shape; torch.stack needs one device and dtype per batch
    batches: Dict[Tuple[torch.device, torch.dtype, int, int], List[int]] = {}
    for index, param in enumerate(params):
        rows, columns = param.size(0), math.prod(param.shape[1:])
        batches.setdefault((param.device, param.dtype, rows, columns), []).append(index)
    for (_, _, rows, columns), indices in batches.items():
        source = grads if nesterov else bufs
        update = torch.stack([source[index].reshape(rows, columns) for index in indices])
        if nesterov:
            update.add_(torch.stack([bufs[index].reshape(rows, columns) for index in indices]), alpha=momentum)
        orthogonal = zeropower_via_newtonschulz5(update, steps=ns_steps)
        torch._foreach_add_(
            [params[index] for index in indices],
            [matrix.reshape_as(params[index]) for index, matrix in zip(indices, orthogonal.unbind(0))],
            alpha=-lr * update_scale((rows, columns), scale_mode),
        )


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


def _adamw_update_foreach(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    states: List[Dict[str, Any]],
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    # the bias corrections depend on the step count, which differs between parameters
    # when one of them had no gradient in an earlier step
    by_step: Dict[int, List[int]] = {}
    for index, (param, state) in enumerate(zip(params, states)):
        if 'exp_avg' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)
        state['step'] += 1
        by_step.setdefault(state['step'], []).append(index)
    beta1, beta2 = betas
    for step, indices in by_step.items():
        step_params = [params[index] for index in indices]
        step_grads = [grads[index] for index in indices]
        exp_avgs = [states[index]['exp_avg'] for index in indices]
        exp_avg_sqs = [states[index]['exp_avg_sq'] for index in indices]
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, step_grads, alpha=1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, step_grads, step_grads, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denoms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denoms, bias_correction2**0.5)
        torch._foreach_add_(denoms, eps)
        if weight_decay != 0:
            torch._foreach_mul_(step_params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(step_params, exp_avgs, denoms, value=-lr / bias_correction1)
