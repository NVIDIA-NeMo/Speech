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

from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import rnnt_logprobs


def rnnt_best_path_align(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RNN-T best path alignment using Viterbi decoding.

    Finds the maximum-likelihood path through the RNN-T lattice and returns
    frame indices where each target token was emitted.

    Lattice structure:
    - States: (t, u) where t is frames consumed (0..T), u is tokens emitted (0..U)
    - Blank arc: (t, u) → (t+1, u) with score blank_logprobs[b, t, u] - advance frame, no emit
    - Emit arc: (t, u) → (t, u+1) with score target_logprobs[b, t, u] - emit symbol, stay on frame

    Args:
        target_logprobs: Log probabilities for target labels [B, T, U+1].
            target_logprobs[b, t, u] is the log probability of emitting target u at frame t.
        blank_logprobs: Log probabilities for blank label [B, T, U+1].
            blank_logprobs[b, t, u] is the log probability of emitting blank at state (t, u).
        src_lengths: Lengths of source sequences [B]. Each value T_b indicates valid frames 0..T_b-1.
        tgt_lengths: Lengths of target sequences [B]. Each value U_b indicates valid targets 0..U_b-1.

    Returns:
        alignments: Tensor [B, U_max] where alignments[b, u] = frame index at which target u was emitted.
            For positions u >= tgt_lengths[b], the value is 0 (padding).

    Raises:
        ValueError: If any sequence has T=0 and U>0 (impossible to align).
    """
    device = target_logprobs.device
    dtype = target_logprobs.dtype
    batch_size, src_length_max, tgt_length_max_plus_1 = target_logprobs.shape
    tgt_length_max = tgt_length_max_plus_1 - 1

    # Handle edge case: tgt_length_max == 0 (nothing to align)
    if tgt_length_max == 0:
        return torch.zeros([batch_size, 0], dtype=torch.long, device=device)

    # Validate: T=0 with U>0 is impossible
    # We check this by looking at whether any sequence has src_length=0 and tgt_length>0
    # To avoid CUDA-CPU sync, we compute the mask and check only on CPU tensors
    # or skip validation if inputs are on GPU (will produce invalid results silently)
    if device.type == "cpu":
        impossible_mask = (src_lengths == 0) & (tgt_lengths > 0)
        if impossible_mask.any():
            raise ValueError("Cannot align sequences with T=0 and U>0: impossible alignment")

    NEG_INF = float("-inf")

    # Pre-allocate tensors for entire batch
    # alpha[b, t, u] = max log-probability to reach state (t, u) for batch element b
    # State (t, u) means: consumed t frames, emitted u symbols
    alpha = torch.full([batch_size, src_length_max + 1, tgt_length_max + 1], NEG_INF, dtype=dtype, device=device)
    alpha[:, 0, 0] = 0.0  # Start state for all batch elements

    # Backpointers: 0 = came from blank (vertical move), 1 = came from emit (horizontal move)
    backptr = torch.zeros([batch_size, src_length_max + 1, tgt_length_max + 1], dtype=torch.int8, device=device)

    # Vectorized forward pass using diagonal wavefront
    # States on the same anti-diagonal (where src_index + tgt_index = diagonal) are independent
    for diagonal in range(1, src_length_max + tgt_length_max + 1):
        # Compute valid (src_index, tgt_index) pairs on this diagonal
        src_index_min = max(0, diagonal - tgt_length_max)
        src_index_max = min(diagonal, src_length_max)

        src_on_diag = torch.arange(src_index_min, src_index_max + 1, device=device)
        tgt_on_diag = diagonal - src_on_diag

        # Blank transition: (src_index-1, tgt_index) -> (src_index, tgt_index)
        # Can only happen if src_index > 0
        # Use clamped indices to avoid out-of-bounds, then mask invalid with NEG_INF
        blank_src_prev = (src_on_diag - 1).clamp(min=0)
        blank_valid_mask = src_on_diag >= 1  # [num_positions]

        # Gather alpha and logprobs using clamped indices (all accesses are valid)
        alpha_from_blank = alpha[:, blank_src_prev, tgt_on_diag]
        blank_lp = blank_logprobs[:, blank_src_prev, tgt_on_diag]
        blank_score = alpha_from_blank + blank_lp

        # Emit transition: (src_index, tgt_index-1) -> (src_index, tgt_index)
        # Can only happen if tgt_index > 0 and src_index < src_length_max
        emit_tgt_prev = (tgt_on_diag - 1).clamp(min=0)
        emit_src_clamped = src_on_diag.clamp(max=src_length_max - 1)
        emit_valid_mask = (tgt_on_diag >= 1) & (src_on_diag < src_length_max)

        # Gather alpha and logprobs using clamped indices
        alpha_from_emit = alpha[:, emit_src_clamped, emit_tgt_prev]
        target_lp = target_logprobs[:, emit_src_clamped, emit_tgt_prev]
        emit_score = alpha_from_emit + target_lp

        # Create validity masks based on sequence lengths
        # A state is valid if src_index <= src_lengths[b] and tgt_index <= tgt_lengths[b]
        valid_src = src_on_diag[None, :] <= src_lengths[:, None]  # [batch_size, num_positions]
        valid_tgt = tgt_on_diag[None, :] <= tgt_lengths[:, None]  # [batch_size, num_positions]
        valid_state = valid_src & valid_tgt

        # For blank transitions: source state (src-1, tgt) must be valid and src >= 1
        blank_src_prev_unclamped = src_on_diag - 1
        blank_src_valid = (blank_src_prev_unclamped[None, :] <= src_lengths[:, None]) & blank_valid_mask[None, :]
        blank_score = torch.where(blank_src_valid & valid_state, blank_score, NEG_INF)

        # For emit transitions: source state (src, tgt-1) must be valid, src < src_lengths[b], and tgt >= 1
        emit_tgt_prev_unclamped = tgt_on_diag - 1
        emit_tgt_valid = (emit_tgt_prev_unclamped[None, :] <= tgt_lengths[:, None]) & emit_valid_mask[None, :]
        emit_src_valid = src_on_diag[None, :] < src_lengths[:, None]
        emit_score = torch.where(emit_tgt_valid & emit_src_valid & valid_state, emit_score, NEG_INF)

        # Select max and update alpha and backptr
        new_alpha = torch.maximum(blank_score, emit_score)
        new_backptr = (emit_score > blank_score).to(torch.int8)

        # Scatter to alpha and backptr tensors
        alpha[:, src_on_diag, tgt_on_diag] = new_alpha
        backptr[:, src_on_diag, tgt_on_diag] = new_backptr

    # Vectorized backtracking: trace all paths in parallel
    # We'll build alignments by iterating through target positions from U-1 down to 0
    # and recording the frame at which each symbol was emitted
    alignments = torch.zeros([batch_size, tgt_length_max], dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, device=device)

    # Start at end state for each batch: (src_lengths[b], tgt_lengths[b])
    cur_src = src_lengths.clone()
    cur_tgt = tgt_lengths.clone()

    # Maximum number of steps is src_length_max + tgt_length_max
    for _ in range(src_length_max + tgt_length_max):
        # Get backpointer at current position for all batch elements
        bp = backptr[batch_indices, cur_src, cur_tgt]

        # Determine active elements (still tracing)
        active = (cur_src > 0) | (cur_tgt > 0)

        # If emit (bp == 1), record frame index
        # When backpointer is 1, we came from (cur_src, cur_tgt-1) via emit
        # This means symbol at position cur_tgt-1 was emitted at frame cur_src
        is_emit = (bp == 1) & active
        emit_tgt_idx = cur_tgt - 1

        # Create mask for valid emit positions
        valid_emit = is_emit & (emit_tgt_idx >= 0) & (emit_tgt_idx < tgt_length_max)

        # Update alignments for valid emits
        # We need to set alignments[b, emit_tgt_idx[b]] = cur_src[b] for valid_emit[b]
        # Use conditional scatter: only update where valid_emit is True
        # To avoid overwriting with 0, we add cur_src only where valid_emit is True
        emit_idx_clamped = emit_tgt_idx.clamp(min=0, max=tgt_length_max - 1)

        # Get current values at the positions we might update
        current_values = alignments.gather(dim=1, index=emit_idx_clamped.unsqueeze(1)).squeeze(1)

        # Compute new values: keep current if not valid_emit, else use cur_src
        new_values = torch.where(valid_emit, cur_src, current_values)

        # Scatter the new values
        alignments.scatter_(dim=1, index=emit_idx_clamped.unsqueeze(1), src=new_values.unsqueeze(1))

        # Update position: blank (bp=0) -> src-=1, emit (bp=1) -> tgt-=1
        cur_src = torch.where(active & (bp == 0), cur_src - 1, cur_src)
        cur_tgt = torch.where(active & (bp == 1), cur_tgt - 1, cur_tgt)

    return alignments


def align_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given logits, calculate RNN-T alignment between targets and logits using Viterbi decoding.

    Finds the maximum-likelihood path through the RNN-T lattice and returns
    frame indices where each target token was emitted.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        src_lengths: optional tensor with lengths for source utterances
        tgt_lengths: optional tensor with lengths for targets

    Returns:
        Tensor of size [B, U] with frame indices for each target token.
        alignments[b, u] = frame index at which target u was emitted.
    """
    device = logits.device
    batch_size, src_length_max, tgt_length_max_plus_1, _ = logits.shape
    tgt_length_max = tgt_length_max_plus_1 - 1

    if src_lengths is None:
        src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
    if tgt_lengths is None:
        tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max, dtype=torch.long, device=device)

    target_logprobs, blank_logprobs = rnnt_logprobs(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )

    alignments = rnnt_best_path_align(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )

    return alignments


def align_with_joint_simple(
    encoder_output: torch.Tensor,
    predictor_output: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    joint: AbstractRNNTJoint,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ...
    """
    encoder_output = joint.project_encoder(encoder_output)
    predictor_output = joint.project_prednet(predictor_output)

    joint_output = joint.joint_after_projection(
        f=encoder_output,
        g=predictor_output,
    )

    return align_from_logits(
        logits=joint_output,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )


def align_with_joint(
    encoder_output: torch.Tensor,
    predictor_output: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    joint: AbstractRNNTJoint,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ...
    """
    encoder_output = joint.project_encoder(encoder_output)
    predictor_output = joint.project_prednet(predictor_output)

    device = encoder_output.device
    batch_size, src_length_max, _ = encoder_output.shape
    tgt_length_max_plus_1 = predictor_output.shape[1]
    tgt_length_max = tgt_length_max_plus_1 - 1

    if src_lengths is None:
        src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
    if tgt_lengths is None:
        tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max, dtype=torch.long, device=device)

    raise NotImplementedError
