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

"""
Pruning ranges for the transducer lattice.

A pruning range is a window of `range_size` contiguous target positions for each frame. The ranges of a batch are
defined by the starts of the windows, an integer tensor `[B, T]`: frame `src_index` of utterance `batch_index`
keeps the target positions `[start, start + range_size)`. If the Joint is evaluated only inside the ranges
(the log-probabilities are `-inf` outside), the ranges must be connected, see `make_pruning_ranges_connected`.
"""

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import _select_pruning_starts
    from nemo.collections.asr.parts.turbo_transducer.rnnt_loss_triton import _rnnt_loss_bwd, _rnnt_loss_fwd

# A range of a single position can't emit a target: the lattice is not connected
MIN_CONNECTED_RANGE_SIZE = 2


def rnnt_pruning_range_starts(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    range_size: int,
    paired: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate connected pruning ranges from the log-probabilities of the full lattice.

    For each frame, the window with the largest occupation probability is selected: the probability that
    an alignment passes through the positions of the window, computed with the forward-backward algorithm
    (plain RNN-T, FastEmit is not applied). The windows are then adjusted to form a connected lattice,
    see `make_pruning_ranges_connected`. No gradient is computed.

    Args:
        target_logprobs: log-probabilities for target labels, [B, T, U+1]
        blank_logprobs: log-probabilities for the blank label, [B, T, U+1]
        src_lengths: source lengths, [B]
        tgt_lengths: target lengths, [B]
        range_size: number of target positions in a range, at least 2
        paired: the batch is a double batch [teacher; student] with identical lengths in both halves:
            one range is selected for each teacher/student pair of frames by the summed occupation probabilities

    Returns:
        starts of the ranges, int32 [B, T] ([B // 2, T] if `paired`), and the RNN-T loss of the full lattice
        for each utterance, [B] (infinite if there is no alignment with non-zero probability)
    """
    if not TRITON_AVAILABLE:
        raise ModuleNotFoundError("Triton is required to estimate the pruning ranges")
    _check_range_size(range_size)
    with torch.no_grad():
        target_logprobs = target_logprobs.detach().contiguous()
        blank_logprobs = blank_logprobs.detach().contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        full_lattice_loss, alpha = _rnnt_loss_fwd(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            fastemit_lambda=0.0,
        )
        # gradient of the loss for the log-probabilities: negated posterior probabilities of the transitions
        grad_target, grad_blank = _rnnt_loss_bwd(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            alpha=alpha,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            grad_rnnt_loss=torch.ones_like(full_lattice_loss),
            fastemit_lambda=0.0,
        )
        starts = _select_pruning_starts(grad_target, grad_blank, src_lengths, tgt_lengths, range_size, paired=paired)
        num_utterances = starts.shape[0]
        starts = make_pruning_ranges_connected(
            starts, src_lengths[:num_utterances], tgt_lengths[:num_utterances], range_size
        )
    return starts, full_lattice_loss


def make_pruning_ranges_connected(
    starts: torch.Tensor, src_lengths: torch.Tensor, tgt_lengths: torch.Tensor, range_size: int
) -> torch.Tensor:
    """
    Adjust the starts of the pruning ranges, so that the pruned lattice contains alignments
    (same constraints as for the pruned RNN-T loss in k2). For each utterance:

    - the range of the first frame starts at 0, the range of the last frame ends at the last target position
      (starts at `max(tgt_length + 1 - range_size, 0)`);
    - the starts do not decrease with the frame index;
    - the start increases by at most `range_size - 1` per frame: consecutive ranges overlap,
      the blank transition from the last position of a range leads to the next range.

    A frame can emit at most `range_size - 1` targets inside its range. If `tgt_length > (range_size - 1) *
    src_length`, the constraints can't be satisfied: the last range does not reach the last target position,
    and the loss of the pruned lattice is infinite for the utterance.

    Args:
        starts: starts of the ranges, [B, T]; values for frames beyond `src_lengths` are ignored
        src_lengths: source lengths, [B]
        tgt_lengths: target lengths, [B]
        range_size: number of target positions in a range, at least 2

    Returns:
        adjusted starts, int32 [B, T]. Padded frames have the start of the last frame
    """
    _check_range_size(range_size)
    num_frames = starts.shape[1]
    frame_indices = torch.arange(num_frames, device=starts.device)
    last_start = (tgt_lengths.long() + 1 - range_size).clamp(min=0).unsqueeze(1)
    is_before_last_frame = frame_indices.unsqueeze(0) < src_lengths.unsqueeze(1) - 1
    starts = torch.where(is_before_last_frame, starts.long().clamp(min=0).minimum(last_start), last_start)
    starts = _suffix_min(starts)  # non-decreasing
    # `slack` is the distance to the largest start reachable from 0: the step constraint is equivalent to
    # non-decreasing slack, and the start of the first frame is 0 if the slack is non-negative
    max_reachable_start = (range_size - 1) * frame_indices
    slack = _suffix_min(max_reachable_start.unsqueeze(0) - starts).clamp(min=0)
    return (max_reachable_start.unsqueeze(0) - slack).to(torch.int32)


def _suffix_min(values: torch.Tensor) -> torch.Tensor:
    """Minimum over the suffix `[index, end)` along the last dimension: the largest non-decreasing lower bound"""
    return values.flip(-1).cummin(dim=-1).values.flip(-1)


def _check_range_size(range_size: int):
    if type(range_size) is not int or range_size < MIN_CONNECTED_RANGE_SIZE:
        raise ValueError(
            f"Pruning range size must be an integer >= {MIN_CONNECTED_RANGE_SIZE} for a connected lattice, "
            f"got {range_size}"
        )
