# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CTC loss over sequence-packed log probabilities.

The log probabilities of a packed batch are one token-flat ``(T_total, D)`` buffer,
utterance ``b`` at the rows ``[cu_seqlens[b], cu_seqlens[b + 1])``.
The kernels read that buffer and write the gradient in the same layout,
so the loss needs neither the padded ``(B, T, D)`` copy that ``torch.nn.functional.ctc_loss`` takes
nor its padded gradient.

One program per utterance walks the utterance's frames.
The forward program runs the alpha recursion over the extended label sequence
``blank, l_1, blank, l_2, ..., l_U, blank`` (``S = 2 U + 1`` states, one lane per state)
and stores every alpha row into ``(T_total, S_max)``, as the padded kernel stores its ``(B, T, S)`` alphas.
The backward program runs the beta recursion from the last frame,
forms the state posteriors from the stored alphas,
and writes each frame's gradient row into the token-flat gradient:
``exp(log_probs) - posterior`` summed per class,
which is the gradient of the padded kernel
(the gradient with respect to the logits when ``log_probs`` is a ``log_softmax`` output).
The posteriors of the blank states are reduced in registers;
the label states are scattered with atomic adds,
as the padded CUDA kernel does for its non-blank labels.

The design follows RETURNN's packed full-sum loss
(``ctc_loss_packed`` and ``FastBaumWelchPackedOp`` in ``returnn/torch/util/native_op.py`` and ``returnn/native_op.py``
of https://github.com/rwth-i6/returnn):
the scores are addressed in the token-flat buffer through per-sequence start offsets,
the posteriors are scattered back into a buffer of the same layout,
the gradient is ``exp(log_probs) - posterior`` masked per frame,
the shapes are static (no host synchronization in the kernels, so a CUDA graph can capture them),
and the state count follows the longest target of the batch rather than the width of the targets buffer,
so an over-allocated buffer costs nothing (RETURNN's content-sized FSA edge layout).
It differs in three points:
the CTC topology is built into the recursion instead of a generic FSA edge list,
each program walks the frames of its own utterance instead of frame-wise kernels over all edges of the batch,
and the alphas are stored while the betas are recomputed in the backward,
instead of the full ``(T_total, D)`` posterior tensor.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["PACKED_CTC_LOSS_DTYPES", "packed_ctc_loss"]

# the dtypes the kernels read; the recursion runs in float32 either way
PACKED_CTC_LOSS_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def packed_ctc_loss(
    log_probs: Tensor,
    cu_seqlens: Tensor,
    lengths: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    *,
    blank: int,
    zero_infinity: bool = False,
) -> Tensor:
    """CTC loss per utterance over token-flat log probabilities.

    Same semantics as ``torch.nn.functional.ctc_loss`` with ``reduction="none"``:
    the loss of an utterance that cannot be aligned is ``inf``,
    or zero with ``zero_infinity``, which also zeroes its gradient,
    and the gradient is the one with respect to the logits of a ``log_softmax``.
    The label scatter of the backward uses atomic adds,
    so the summation order of repeated labels varies between runs,
    as it does in the padded CUDA kernel.

    Args:
        log_probs: log probabilities of all utterances concatenated along time, shape ``(T_total, D)``
            with the blank among the ``D`` classes; float32, bfloat16 or float16, computed in float32.
        cu_seqlens: cumulative frame counts, shape ``(B + 1,)``, integer.
        lengths: frames per utterance, shape ``(B,)``, integer.
        targets: padded target ids, shape ``(B, U)``, integer.
        target_lengths: target ids per utterance, shape ``(B,)``, integer, at most ``U``.
        blank: index of the blank class.
        zero_infinity: replace an infinite loss and its gradient by zero.

    Returns:
        The loss per utterance, shape ``(B,)``, float32.
    """
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must have shape (T_total, D), got {tuple(log_probs.shape)}.")
    if not log_probs.is_cuda:
        raise RuntimeError("The packed CTC loss kernel requires CUDA tensors.")
    if log_probs.dtype not in PACKED_CTC_LOSS_DTYPES:
        raise TypeError(f"log_probs must be float32, bfloat16 or float16, got {log_probs.dtype}.")
    batch_size = lengths.numel()
    if cu_seqlens.shape != (batch_size + 1,):
        raise ValueError(f"cu_seqlens must have shape ({batch_size + 1},), got {tuple(cu_seqlens.shape)}.")
    if targets.ndim != 2 or targets.shape[0] != batch_size:
        raise ValueError(f"targets must have shape ({batch_size}, U), got {tuple(targets.shape)}.")
    if target_lengths.shape != (batch_size,):
        raise ValueError(f"target_lengths must have shape ({batch_size},), got {tuple(target_lengths.shape)}.")
    if not 0 <= blank < log_probs.shape[1]:
        raise ValueError(f"blank must be a class index below {log_probs.shape[1]}, got {blank}.")
    # The same host check as the padded kernel, and the longest target sizes the states,
    # so that an over-allocated targets buffer costs nothing.
    # A graph capture cannot sync: there the width stands in, and the sampler guarantees the bound.
    longest = targets.shape[1]
    if batch_size and not torch.cuda.is_current_stream_capturing():
        longest = int(target_lengths.max())
        if longest > targets.shape[1]:
            raise ValueError(f"target_lengths must not exceed the targets width {targets.shape[1]}, got {longest}.")
    return _PackedCTCLoss.apply(
        log_probs.contiguous(),
        cu_seqlens.contiguous(),
        lengths.contiguous(),
        targets.contiguous(),
        target_lengths.contiguous(),
        blank,
        zero_infinity,
        2 * longest + 1,
    )


class _PackedCTCLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, cu_seqlens, lengths, targets, target_lengths, blank, zero_infinity, num_states_max):
        batch_size, target_width = targets.shape
        alpha = torch.empty((log_probs.shape[0], num_states_max), device=log_probs.device, dtype=torch.float32)
        loss = torch.empty((batch_size,), device=log_probs.device, dtype=torch.float32)
        if batch_size:
            _alpha_kernel[(batch_size,)](
                log_probs,
                cu_seqlens,
                lengths,
                targets,
                target_lengths,
                alpha,
                loss,
                log_probs.shape[1],
                target_width,
                num_states_max,
                blank,
                BLOCK_S=_state_block(num_states_max),
                num_warps=_num_warps(num_states_max),
            )
        ctx.save_for_backward(log_probs, cu_seqlens, lengths, targets, target_lengths, alpha, loss)
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity
        ctx.num_states_max = num_states_max
        if zero_infinity:
            return torch.where(torch.isposinf(loss), 0.0, loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, cu_seqlens, lengths, targets, target_lengths, alpha, loss = ctx.saved_tensors
        batch_size, target_width = targets.shape
        num_states_max = ctx.num_states_max
        # zeros, not empty: a frame outside every utterance (a gapped layout) has no gradient
        grad = torch.zeros(log_probs.shape, device=log_probs.device, dtype=torch.float32)
        if batch_size:
            _beta_grad_kernel[(batch_size,)](
                log_probs,
                cu_seqlens,
                lengths,
                targets,
                target_lengths,
                alpha,
                loss,
                grad_output.contiguous().to(torch.float32),
                grad,
                log_probs.shape[1],
                target_width,
                num_states_max,
                ctx.blank,
                ZERO_INFINITY=ctx.zero_infinity,
                BLOCK_S=_state_block(num_states_max),
                BLOCK_D=1024,
                num_warps=_num_warps(num_states_max),
            )
        return grad.to(log_probs.dtype), None, None, None, None, None, None, None


def _state_block(num_states_max: int) -> int:
    return max(16, triton.next_power_of_2(num_states_max))


def _num_warps(num_states_max: int) -> int:
    return 4 if num_states_max <= 512 else 8


@triton.jit
def _logsumexp3(a, b, c):
    peak = tl.maximum(tl.maximum(a, b), c)
    finite_peak = tl.where(peak == float("-inf"), 0.0, peak)
    total = tl.exp(a - finite_peak) + tl.exp(b - finite_peak) + tl.exp(c - finite_peak)
    return tl.where(peak == float("-inf"), float("-inf"), finite_peak + tl.log(total))


@triton.jit
def _extended_labels(targets_ptr, b, target_width, state, valid, blank):
    """``l'_s``: the blank at even states, ``targets[b, (s - 1) // 2]`` at odd states; blank beyond the utterance."""
    is_label = valid & (state % 2 == 1)
    return tl.load(targets_ptr + b * target_width + (state - 1) // 2, mask=is_label, other=blank).to(tl.int32)


@triton.jit
def _frame_log_probs(log_probs_ptr, row, num_classes, label, valid):
    return tl.load(log_probs_ptr + row * num_classes + label, mask=valid, other=float("-inf")).to(tl.float32)


@triton.jit
def _alpha_kernel(
    log_probs_ptr,
    cu_seqlens_ptr,
    lengths_ptr,
    targets_ptr,
    target_lengths_ptr,
    alpha_ptr,
    loss_ptr,
    num_classes,
    target_width,
    num_states_max,
    blank,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(cu_seqlens_ptr + b).to(tl.int64)
    num_frames = tl.load(lengths_ptr + b).to(tl.int32)
    num_labels = tl.load(target_lengths_ptr + b).to(tl.int32)
    num_states = 2 * num_labels + 1
    state = tl.arange(0, BLOCK_S)
    valid = state < num_states
    label = _extended_labels(targets_ptr, b, target_width, state, valid, blank)
    # the skip over the blank between two different labels; a target id equal to the blank is a label like any other,
    # as in the padded kernel
    label_before = tl.gather(label, tl.maximum(state - 2, 0), 0)
    can_skip = valid & (state >= 2) & (state % 2 == 1) & (label != label_before)

    loss = tl.where(num_labels == 0, 0.0, float("inf"))
    if num_frames > 0:
        # the paths start in the first blank or the first label
        alpha = tl.where(
            valid & (state < 2), _frame_log_probs(log_probs_ptr, start, num_classes, label, valid), float("-inf")
        )
        tl.store(alpha_ptr + start * num_states_max + state, alpha, mask=valid)
        for t in range(1, num_frames):
            row = start + t
            stay = alpha
            advance = tl.where(state >= 1, tl.gather(alpha, tl.maximum(state - 1, 0), 0), float("-inf"))
            skip = tl.where(can_skip, tl.gather(alpha, tl.maximum(state - 2, 0), 0), float("-inf"))
            alpha = _logsumexp3(stay, advance, skip) + _frame_log_probs(log_probs_ptr, row, num_classes, label, valid)
            alpha = tl.where(valid, alpha, float("-inf"))
            tl.store(alpha_ptr + row * num_states_max + state, alpha, mask=valid)
        # the paths end in the last blank or the last label
        last = tl.max(tl.where(state == num_states - 1, alpha, float("-inf")), 0)
        before_last = tl.max(tl.where(state == num_states - 2, alpha, float("-inf")), 0)
        loss = -_logsumexp3(last, before_last, float("-inf"))
    tl.store(loss_ptr + b, loss)


@triton.jit
def _beta_grad_kernel(
    log_probs_ptr,
    cu_seqlens_ptr,
    lengths_ptr,
    targets_ptr,
    target_lengths_ptr,
    alpha_ptr,
    loss_ptr,
    grad_output_ptr,
    grad_ptr,
    num_classes,
    target_width,
    num_states_max,
    blank,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(cu_seqlens_ptr + b).to(tl.int64)
    num_frames = tl.load(lengths_ptr + b).to(tl.int32)
    num_labels = tl.load(target_lengths_ptr + b).to(tl.int32)
    num_states = 2 * num_labels + 1
    state = tl.arange(0, BLOCK_S)
    valid = state < num_states
    label = _extended_labels(targets_ptr, b, target_width, state, valid, blank)
    label_after = tl.gather(label, tl.minimum(state + 2, BLOCK_S - 1), 0)
    can_skip = valid & (state + 2 < num_states) & (state % 2 == 1) & (label != label_after)
    is_label = valid & (state % 2 == 1)
    is_blank = valid & (state % 2 == 0)

    nll = tl.load(loss_ptr + b)
    scale = tl.load(grad_output_ptr + b)
    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)
    classes = tl.arange(0, BLOCK_D)

    beta = tl.full([BLOCK_S], float("-inf"), tl.float32)
    for i in range(0, num_frames):
        t = num_frames - 1 - i
        row = start + t
        lp = _frame_log_probs(log_probs_ptr, row, num_classes, label, valid)
        if i == 0:
            beta = tl.where(valid & (state >= num_states - 2), lp, float("-inf"))
        else:
            stay = beta
            advance = tl.where(
                valid & (state + 1 < num_states), tl.gather(beta, tl.minimum(state + 1, BLOCK_S - 1), 0), float("-inf")
            )
            skip = tl.where(can_skip, tl.gather(beta, tl.minimum(state + 2, BLOCK_S - 1), 0), float("-inf"))
            beta = tl.where(valid, _logsumexp3(stay, advance, skip) + lp, float("-inf"))
        alpha = tl.load(alpha_ptr + row * num_states_max + state, mask=valid, other=float("-inf"))
        # the posterior of state s at frame t; nothing when the gradient is zero, so that inf - inf never enters
        posterior = tl.where(valid & (scale != 0.0), tl.exp(alpha + beta + nll - lp) * scale, 0.0)
        blank_posterior = tl.sum(tl.where(is_blank, posterior, 0.0), 0)
        for class_start in range(0, num_classes, BLOCK_D):
            c = class_start + classes
            class_valid = c < num_classes
            probs = tl.exp(tl.load(log_probs_ptr + row * num_classes + c, mask=class_valid, other=0.0).to(tl.float32))
            grad_row = probs * scale - tl.where(c == blank, blank_posterior, 0.0)
            tl.store(grad_ptr + row * num_classes + c, grad_row, mask=class_valid)
        # the label posteriors go into the stored row, so the stores must be complete first
        tl.debug_barrier()
        tl.atomic_add(grad_ptr + row * num_classes + label, -posterior, mask=is_label)
