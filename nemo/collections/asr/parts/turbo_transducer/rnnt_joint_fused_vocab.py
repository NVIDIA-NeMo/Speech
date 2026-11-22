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


import torch

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.types import TransducerLossInputs, check_turbo_joint_backend
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_consistency_triton import (
        rnnt_joint_fused_vocab_consistency_triton,
    )
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import rnnt_joint_fused_vocab_triton


def rnnt_joint_fused_vocab(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = False,
    weight_grad_full_precision: bool = False,
    backend: str | TurboTransducerBackend | None = TurboTransducerBackend.TRITON,
    mm_full_precision: bool = False,
    backward_pruning_range_size: int | None = None,
    backward_mode: str | None = None,
    pruning_range_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target/blank log-probabilities without materializing the vocabulary logits.

    Only the Triton backend is implemented (``backend=None`` selects it).

    BF16 activations use BF16 matmul operands even with FP32 master parameters. ``mm_full_precision=True``
    promotes operands to FP32; ``use_high_precision_mm=True`` also promotes operands and requests IEEE input
    precision in Triton. Gradient accumulation precision is controlled separately by the gradient flags.

    ``backward_pruning_range_size`` enables approximate backward on that many contiguous target positions per frame.
    Windows maximize incoming target/blank gradient magnitude. Forward remains unchanged.
    None or non-positive widths disable pruning. ``backward_mode`` optionally selects the ``'atomic'``
    or ``'chunked'`` algorithm independently of pruning.

    ``pruning_range_starts`` evaluates the Joint for the pruned lattice: ``joint_hidden`` is given only
    for the pruning ranges, ``[B, T, range_size, hidden_dim]``, and the integer tensor ``[B, T]`` defines the first
    target position of the range for each frame (see ``rnnt_pruning``). Forward and backward are computed only
    inside the ranges. The outputs have the shape of the full lattice ``[B, T, U+1]`` with U = ``targets.shape[1]``
    and are ``-inf`` outside the ranges, so the loss computed from them is the loss of the pruned lattice with
    the exact gradient. Incompatible with ``backward_pruning_range_size``.

    Returns:
        target and blank log-probabilities, ``[B, T, U+1]``
    """
    _check_backend(backend)
    if backward_pruning_range_size is not None and backward_pruning_range_size <= 0:
        backward_pruning_range_size = None
    return rnnt_joint_fused_vocab_triton(
        joint_hidden=joint_hidden,
        targets=targets,
        tgt_lengths=tgt_lengths,
        src_lengths=src_lengths,
        weight=weight,
        bias=bias,
        blank_id=blank_id,
        use_high_precision_mm=use_high_precision_mm,
        input_grad_full_precision=input_grad_full_precision,
        weight_grad_full_precision=weight_grad_full_precision,
        mm_full_precision=mm_full_precision,
        backward_pruning_range_size=backward_pruning_range_size,
        backward_mode=backward_mode,
        pruning_range_starts=pruning_range_starts,
    )


def rnnt_joint_fused_vocab_with_consistency(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    consistency_symmetrical: bool = True,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = False,
    weight_grad_full_precision: bool = False,
    backend: str | TurboTransducerBackend | None = TurboTransducerBackend.TRITON,
    mm_full_precision: bool = False,
    backward_pruning_range_size: int | None = None,
    backward_mode: str | None = None,
    pruning_range_starts: torch.Tensor | None = None,
) -> TransducerLossInputs:
    """
    Same as `rnnt_joint_fused_vocab`, but for a double batch (teacher rows [0, B), student rows [B, 2B) with identical
    lengths and targets) also computes the per-position consistency (KL) loss [B, T, U+1] inside the kernels.
    See `rnnt_joint_fused_vocab_consistency_triton` for the semantics.

    With the pruned lattice, `pruning_range_starts` [B, T] are shared by the teacher and the student rows:
    the joint hidden is [2B, T, range_size, hidden_dim], and the consistency loss is computed only for
    the positions of the ranges, [B, T, range_size] (the mean over the valid positions is the mean over the ranges).
    """
    _check_backend(backend)
    if backward_pruning_range_size is not None and backward_pruning_range_size <= 0:
        backward_pruning_range_size = None
    return rnnt_joint_fused_vocab_consistency_triton(
        joint_hidden=joint_hidden,
        targets=targets,
        tgt_lengths=tgt_lengths,
        src_lengths=src_lengths,
        weight=weight,
        bias=bias,
        blank_id=blank_id,
        consistency_symmetrical=consistency_symmetrical,
        use_high_precision_mm=use_high_precision_mm,
        input_grad_full_precision=input_grad_full_precision,
        weight_grad_full_precision=weight_grad_full_precision,
        mm_full_precision=mm_full_precision,
        backward_pruning_range_size=backward_pruning_range_size,
        backward_mode=backward_mode,
        pruning_range_starts=pruning_range_starts,
    )


def _check_backend(backend: str | TurboTransducerBackend | None):
    """Only the Triton backend is implemented: reject other backends, require Triton"""
    check_turbo_joint_backend(backend)
    if not TRITON_AVAILABLE:
        raise ModuleNotFoundError("Triton is required for the `fused_vocab` Joint, but it is not installed")
