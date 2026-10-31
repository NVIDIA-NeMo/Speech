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
"""Utilities for SOT-style speaker tokens and speaker-activity alignment."""
# pylint: disable=import-error

import re
from itertools import permutations
from typing import Optional, Sequence

import numpy as np
import torch

SPEAKER_TOKEN_PATTERN = re.compile(r"<spk:(\d+)>")
_SPEAKER_TOKEN_SPLIT_PATTERN = re.compile(r"(<spk:\d+>)")

# Alignment only resolves the RTTM column permutation; cap its DTW input at
# 1,200 frames to bound long-session cost without upsampling short inputs.
_DEFAULT_ALIGNMENT_FRAME_SECONDS = 0.08
_DEFAULT_MAX_ALIGNMENT_FRAMES = int(round(96.0 / _DEFAULT_ALIGNMENT_FRAME_SECONDS))
# Full permutation search is exact and affordable through six active speakers
# (6! = 720), but becomes a dataloader-scale denial of service for the
# eight-speaker targets used by SpeechLM (8! = 40,320). Seven- and eight-speaker
# examples use a bounded shortlist; callers can pass ``None`` for strict
# exhaustive parity when offline preprocessing time is acceptable.
_DEFAULT_MAX_ALIGNMENT_PERMUTATIONS = 720
_ALIGNMENT_TIMELINE_QUANTILES = np.linspace(0.1, 0.9, 9, dtype=np.float32)

__all__ = [
    "SPEAKER_TOKEN_PATTERN",
    "collate_speaker_activity_targets",
    "dtw_cost",
    "dtw_cost_batch",
    "ensure_single_speaker_sot",
    "fix_speaker_activity",
    "get_speaker_token_index_map",
    "get_text_speaker_char_counts",
    "has_speaker_tokens",
    "parse_speaker_tokens",
    "sl_to_wl_sot",
    "speaker_activity_from_cut",
    "speaker_freq_cost_batch",
]


def has_speaker_tokens(text: Optional[str]) -> bool:
    """Return True if text contains SOT speaker tags such as ``<spk:0>``.

    Args:
        text (Optional[str]): Input text that may contain speaker tags.

    Returns:
        bool: True if at least one ``<spk:N>`` speaker tag is present.
    """
    return bool(text and SPEAKER_TOKEN_PATTERN.search(text))


def get_speaker_token_index_map(text: Optional[str], num_speakers: int) -> Optional[dict[str, int]]:
    """Return canonical SOT speaker labels mapped to their target columns.

    Args:
        text (Optional[str]): SOT text containing ``<spk:N>`` speaker tags.
        num_speakers (int): Number of available target speaker columns.

    Returns:
        Optional[dict[str, int]]: Unique exact token strings mapped to their numeric indices, or None when the text
            has no speaker tokens or contains a non-canonical/out-of-range token.
    """
    if not text:
        return None

    speaker_to_idx = {}
    for match in SPEAKER_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        speaker_idx = int(match.group(1))
        if token != f"<spk:{speaker_idx}>" or speaker_idx >= num_speakers:
            return None
        speaker_to_idx[token] = speaker_idx
    return speaker_to_idx or None


def sl_to_wl_sot(text: str) -> str:
    """Convert segment-level SOT text to word-level SOT text.

    Args:
        text (str): Segment-level SOT text where a speaker tag precedes each segment.

    Returns:
        str: Word-level SOT text where a speaker tag precedes every word.
    """
    parts = _SPEAKER_TOKEN_SPLIT_PATTERN.split(text)
    result = []
    current_token = None
    for part in parts:
        if _SPEAKER_TOKEN_SPLIT_PATTERN.fullmatch(part):
            current_token = part
            continue
        words = part.split()
        if current_token is None:
            result.extend(words)
            continue
        for word in words:
            result.append(current_token)
            result.append(word)
    return " ".join(result)


def parse_speaker_tokens(text: str) -> list[int]:
    """Extract one forward-filled speaker index per word from SOT text.

    Args:
        text (str): SOT text containing ``<spk:N>`` speaker tags.

    Returns:
        list[int]: Speaker index for each word; words before the first tag are dropped.
    """
    parts = _SPEAKER_TOKEN_SPLIT_PATTERN.split(text)
    spk_seq: list[int] = []
    current_spk = -1
    for part in parts:
        match = SPEAKER_TOKEN_PATTERN.fullmatch(part)
        if match:
            current_spk = int(match.group(1))
            continue
        if current_spk < 0:
            continue
        for _ in part.split():
            spk_seq.append(current_spk)
    return spk_seq


def get_text_speaker_char_counts(text: str, num_speakers: int) -> np.ndarray:
    """Estimate per-speaker text mass from word character counts.

    Args:
        text (str): SOT text containing ``<spk:N>`` speaker tags.
        num_speakers (int): Number of speaker slots in the output vector.

    Returns:
        np.ndarray: Shape ``(num_speakers,)`` normalized character-count distribution.
    """
    parts = _SPEAKER_TOKEN_SPLIT_PATTERN.split(text)
    char_counts = np.zeros(num_speakers, dtype=np.float32)
    current_spk = -1
    for part in parts:
        match = SPEAKER_TOKEN_PATTERN.fullmatch(part)
        if match:
            current_spk = int(match.group(1))
            continue
        if current_spk < 0 or current_spk >= num_speakers:
            continue
        for word in part.split():
            char_counts[current_spk] += len(word)
    total = char_counts.sum()
    if total > 0:
        char_counts /= total
    return char_counts


def ensure_single_speaker_sot(text: Optional[str]) -> tuple[str, int, bool]:
    """Prefix no-speaker text with the ``<spk:0>`` SOT speaker tag.

    Existing SOT text is returned unchanged with ``speaker_index=-1`` and ``changed=False``.

    Args:
        text (Optional[str]): Input text, possibly without speaker tags.

    Returns:
        tuple[str, int, bool]: ``(text, speaker_index, changed)`` where ``changed``
            indicates whether a tag was inserted.
    """
    text = text or ""
    if has_speaker_tokens(text):
        return text, -1, False
    return f"<spk:0> {text}", 0, True


def dtw_cost_batch(
    activity: np.ndarray,
    spk_seq_arr: np.ndarray,
    perm_batch: np.ndarray,
    num_speakers: int,
    token_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute DTW costs for a batch of speaker-column permutations.

    Args:
        activity (np.ndarray): Shape ``(T, N)`` frame-level speaker activity.
        spk_seq_arr (np.ndarray): Shape ``(num_tokens,)`` per-word speaker indices.
        perm_batch (np.ndarray): Shape ``(P, N)`` speaker-column permutations to score.
        num_speakers (int): Number of valid speakers (tokens at/above this are ignored).
        token_weights (Optional[np.ndarray]): Shape ``(num_tokens,)`` per-token cost weights.

    Returns:
        np.ndarray: Shape ``(P,)`` normalized DTW cost for each permutation.
    """
    num_tokens = spk_seq_arr.shape[0]
    num_frames = activity.shape[0]
    num_perms = perm_batch.shape[0]
    if num_tokens == 0 or num_frames == 0:
        return np.full(num_perms, np.float32(np.inf))

    # Speaker activity is binary. Keep only this compact (P, T, N) view and
    # construct one float32 local-cost row at a time; materializing the former
    # (P, words, T) cube took multiple gigabytes on hour-long transcripts.
    activity = np.asarray(activity, dtype=np.bool_)
    valid = spk_seq_arr < num_speakers
    activity_permuted = activity[:, perm_batch].transpose(1, 0, 2)  # (P, T, N), bool
    activity_sum = np.maximum(np.count_nonzero(activity, axis=1), 1).astype(np.float32)
    cols = np.where(valid, spk_seq_arr, 0)

    def local_cost_row(token_idx: int) -> np.ndarray:
        if not valid[token_idx]:
            local = np.ones((num_perms, num_frames), dtype=np.float32)
        else:
            selected = activity_permuted[:, :, cols[token_idx]]
            local = 1.0 - selected.astype(np.float32) / activity_sum[np.newaxis, :]
        if token_weights is not None:
            local *= np.float32(token_weights[token_idx])
        return local

    prev_row = np.cumsum(local_cost_row(0), axis=1, dtype=np.float32)

    for token_idx in range(1, num_tokens):
        local = local_cost_row(token_idx)

        # Vectorized equivalent of:
        #   cur[j] = local[j] + min(prev[j], prev[j - 1], cur[j - 1])
        # Unrolling the horizontal recurrence yields a prefix sum plus a prefix
        # minimum, removing the Python loop over every activity frame.
        local_prefix = np.cumsum(local, axis=1, dtype=np.float32)
        candidates = np.empty_like(prev_row)
        candidates[:, 0] = prev_row[:, 0]
        np.minimum(prev_row[:, 1:], prev_row[:, :-1], out=candidates[:, 1:])
        candidates[:, 1:] -= local_prefix[:, :-1]
        np.minimum.accumulate(candidates, axis=1, out=candidates)
        prev_row = local_prefix + candidates

    return prev_row[:, num_frames - 1] / (num_tokens + num_frames)


def speaker_freq_cost_batch(text_freq: np.ndarray, rttm_freq: np.ndarray, perm_batch: np.ndarray) -> np.ndarray:
    """L1 mismatch between text and RTTM speaker frequency under each permutation.

    Args:
        text_freq (np.ndarray): Shape ``(N,)`` per-speaker text frequency distribution.
        rttm_freq (np.ndarray): Shape ``(N,)`` per-speaker RTTM activity distribution.
        perm_batch (np.ndarray): Shape ``(P, N)`` speaker-column permutations to score.

    Returns:
        np.ndarray: Shape ``(P,)`` L1 distance for each permutation.
    """
    rttm_freq_perm = rttm_freq[perm_batch]
    return np.abs(text_freq - rttm_freq_perm).sum(axis=1).astype(np.float32)


def speaker_timeline_cost_batch(
    activity: np.ndarray,
    spk_seq_arr: np.ndarray,
    perm_batch: np.ndarray,
    num_speakers: int,
) -> np.ndarray:
    """Cheap temporal-distribution mismatch used to shortlist DTW permutations.

    For each text speaker and RTTM column, compare nine normalized occurrence
    quantiles.  This preserves coarse speaker order even when speakers have
    equal aggregate duration, while avoiding the factorial DTW cost.  It is
    only a candidate-ranking heuristic: shortlisted permutations are still
    scored by the original word/frame DTW objective.
    """
    activity = np.asarray(activity, dtype=np.bool_)
    text_quantiles = np.zeros((num_speakers, _ALIGNMENT_TIMELINE_QUANTILES.size), dtype=np.float32)
    activity_quantiles = np.zeros_like(text_quantiles)

    token_denominator = max(spk_seq_arr.size - 1, 1)
    frame_denominator = max(activity.shape[0] - 1, 1)
    text_present = np.zeros(num_speakers, dtype=np.bool_)
    activity_present = np.zeros(num_speakers, dtype=np.bool_)

    for speaker_idx in range(num_speakers):
        token_positions = np.flatnonzero(spk_seq_arr == speaker_idx)
        if token_positions.size:
            text_present[speaker_idx] = True
            text_quantiles[speaker_idx] = np.quantile(token_positions, _ALIGNMENT_TIMELINE_QUANTILES).astype(
                np.float32
            ) / np.float32(token_denominator)

        frame_positions = np.flatnonzero(activity[:, speaker_idx])
        if frame_positions.size:
            activity_present[speaker_idx] = True
            activity_quantiles[speaker_idx] = np.quantile(frame_positions, _ALIGNMENT_TIMELINE_QUANTILES).astype(
                np.float32
            ) / np.float32(frame_denominator)

    pair_cost = np.abs(text_quantiles[:, np.newaxis, :] - activity_quantiles[np.newaxis, :, :]).mean(axis=2)
    pair_cost[~text_present, :] = 0.0
    pair_cost[:, ~activity_present] += 1.0

    output_speakers = np.arange(num_speakers, dtype=np.intp)
    return pair_cost[output_speakers[np.newaxis, :], perm_batch].sum(axis=1).astype(np.float32)


def _shortlist_alignment_permutations(
    activity: np.ndarray,
    spk_seq_arr: np.ndarray,
    perm_batch: np.ndarray,
    text_freq: np.ndarray,
    rttm_freq: np.ndarray,
    num_speakers: int,
    max_permutations: Optional[int],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return DTW candidates, full frequency costs, and optional exact-class expansion.

    Permutations that differ only on text-unused output slots have identical
    DTW cost.  Score one representative per such equivalence class and expand
    those scores back across the original full permutation order so the final
    float32 frequency-cost and tie behavior remains exhaustive.  Only when the
    number of distinct classes itself exceeds ``max_permutations`` do we apply
    the bounded frequency/timeline shortlist.
    """
    freq_costs = speaker_freq_cost_batch(text_freq, rttm_freq, perm_batch)
    if max_permutations is not None and max_permutations <= 0:
        raise ValueError(f"max_alignment_permutations must be positive or None, got {max_permutations}.")

    remapped_slots = sorted(
        set(spk_seq_arr[(spk_seq_arr >= 0) & (spk_seq_arr < num_speakers)].tolist())
        | set(np.flatnonzero(text_freq[:num_speakers]).tolist())
    )
    if remapped_slots:
        class_keys = perm_batch[:, remapped_slots]
        _, representative_indices, class_inverse = np.unique(
            class_keys, axis=0, return_index=True, return_inverse=True
        )
    else:
        representative_indices = np.array([0], dtype=np.intp)
        class_inverse = np.zeros(perm_batch.shape[0], dtype=np.intp)

    if max_permutations is None or representative_indices.size <= max_permutations:
        return representative_indices.astype(np.intp, copy=False), freq_costs, class_inverse

    timeline_costs = speaker_timeline_cost_batch(activity, spk_seq_arr, perm_batch, num_speakers)
    ranking_costs = freq_costs + timeline_costs

    # Keep the best-ranked representative from each distinct DTW class.  This
    # avoids spending the bounded budget on permutations that differ only in
    # output columns that are zeroed after alignment.
    candidate_indices = []
    selected_classes = set()
    for permutation_idx in np.argsort(ranking_costs, kind="stable"):
        class_idx = int(class_inverse[permutation_idx])
        if class_idx in selected_classes:
            continue
        candidate_indices.append(int(permutation_idx))
        selected_classes.add(class_idx)
        if len(candidate_indices) == max_permutations:
            break

    # Identity is a deterministic baseline.  If its class was not shortlisted,
    # replace the last heuristic candidate with it.
    identity_class = int(class_inverse[0])
    if identity_class not in selected_classes:
        candidate_indices[-1] = 0
    candidate_indices = np.asarray(candidate_indices, dtype=np.intp)
    candidate_indices.sort()
    return candidate_indices, freq_costs, None


def dtw_cost(
    activity: np.ndarray,
    spk_seq_arr: np.ndarray,
    perm: Sequence[int],
    num_speakers: int,
    token_weights: Optional[np.ndarray] = None,
) -> float:
    """Compute DTW cost for a single speaker-column permutation.

    Args:
        activity (np.ndarray): Shape ``(T, N)`` frame-level speaker activity.
        spk_seq_arr (np.ndarray): Shape ``(num_tokens,)`` per-word speaker indices.
        perm (Sequence[int]): Speaker-column permutation to score.
        num_speakers (int): Number of valid speakers (tokens at/above this are ignored).
        token_weights (Optional[np.ndarray]): Shape ``(num_tokens,)`` per-token cost weights.

    Returns:
        float: Normalized DTW cost for the permutation.
    """
    perm_batch = np.array([perm], dtype=np.intp)
    costs = dtw_cost_batch(activity, spk_seq_arr, perm_batch, num_speakers, token_weights)
    return float(costs[0])


def _coarsen_activity_for_alignment(activity: np.ndarray, max_frames: Optional[int]) -> np.ndarray:
    """Majority-pool binary activity into at most ``max_frames`` proportional bins.

    Every source frame contributes to exactly one bin. A speaker is active in a
    coarse bin only when active for more than half of its source frames, so very
    short turns do not dominate alignment of hour-long sessions.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_alignment_frames must be positive or None, got {max_frames}.")

    activity = np.asarray(activity, dtype=np.bool_)
    if max_frames is None or activity.shape[0] <= max_frames:
        # Never upsample: this preserves the 80 ms floor and cannot create empty
        # or duplicated alignment frames for short utterances.
        return activity

    num_frames, num_speakers = activity.shape
    # Integer proportional boundaries produce exactly ``max_frames`` non-empty
    # bins when num_frames > max_frames. For non-integral ratios, bin widths differ
    # by at most one source frame.
    edges = np.arange(max_frames + 1, dtype=np.int64) * num_frames // max_frames
    cumulative = np.empty((num_frames + 1, num_speakers), dtype=np.uint32)
    cumulative[0] = 0
    np.cumsum(activity, axis=0, dtype=np.uint32, out=cumulative[1:])
    bin_counts = cumulative[edges[1:]] - cumulative[edges[:-1]]
    bin_widths = np.diff(edges).astype(np.uint32)
    return bin_counts * 2 > bin_widths[:, np.newaxis]


def fix_speaker_activity(
    cut_or_text,
    speaker_activity: torch.Tensor,
    num_speakers: int,
    max_permutable: Optional[int] = None,
    max_alignment_frames: Optional[int] = _DEFAULT_MAX_ALIGNMENT_FRAMES,
    max_alignment_permutations: Optional[int] = _DEFAULT_MAX_ALIGNMENT_PERMUTATIONS,
) -> torch.Tensor:
    """Align RTTM speaker-activity columns with SOT speaker-token order.

    Args:
        cut_or_text (Union[Cut, str]): A Lhotse cut with a ``text`` attribute, or raw SOT text.
        speaker_activity (torch.Tensor): Shape ``(T, N)`` frame-level activity to reorder.
        num_speakers (int): Number of speakers used to bound the permutation search.
        max_permutable (Optional[int]): Max active speakers to brute-force permute over;
            defaults to ``num_speakers + 1``.
        max_alignment_frames (Optional[int]): Maximum number of activity frames passed
            to DTW. Longer binary sequences are majority-pooled into this many proportional bins;
            the default 1,200 frames corresponds to 96 seconds at the standard 80 ms
            target rate, and therefore to 3.0-second bins for a one-hour session. Set
            to ``None`` to disable coarsening. The returned tensor stays full-resolution.
        max_alignment_permutations (Optional[int]): Maximum number of speaker-column
            permutations passed to word/frame DTW. Full search is retained below
            this bound. Larger searches are shortlisted using speaker-frequency
            and temporal-distribution costs, always including identity. Set to
            ``None`` to force exhaustive DTW.

    Returns:
        torch.Tensor: Shape ``(T, N)`` activity with columns reordered to match text speaker order.
    """
    text = getattr(cut_or_text, "text", cut_or_text) or ""
    if not text:
        return speaker_activity

    _, num_activity_speakers = speaker_activity.shape
    active_frames = speaker_activity.sum(dim=0)
    num_active = min(int((active_frames > 0).sum().item()), num_activity_speakers)

    spk_seq = parse_speaker_tokens(text)
    if not spk_seq:
        return speaker_activity

    speakers_in_text = sorted(set(spk_seq))
    spk_seq_arr = np.array(spk_seq, dtype=np.intp)
    num_tokens = len(spk_seq_arr)
    activity_np = speaker_activity.detach().cpu().numpy().astype(np.bool_, copy=False)

    token_counts = np.bincount(spk_seq_arr, minlength=num_activity_speakers).astype(np.float32)
    token_counts = np.maximum(token_counts, 1.0)
    token_weights = (num_tokens / token_counts)[spk_seq_arr]

    text_freq = get_text_speaker_char_counts(text, num_activity_speakers)
    rttm_freq = activity_np.sum(axis=0).astype(np.float32)
    rttm_total = rttm_freq.sum()
    if rttm_total > 0:
        rttm_freq /= rttm_total

    identity_perm = list(range(num_activity_speakers))
    max_permutable = max_permutable if max_permutable is not None else num_speakers + 1
    if num_active > 0 and num_active <= max_permutable:
        perm_active = np.array(list(permutations(range(num_active))), dtype=np.intp)
        perm_batch = np.zeros((perm_active.shape[0], num_activity_speakers), dtype=np.intp)
        perm_batch[:, :num_active] = perm_active
        perm_batch[:, num_active:] = np.arange(num_active, num_activity_speakers)

        alignment_activity = _coarsen_activity_for_alignment(activity_np, max_alignment_frames)
        candidate_indices, freq_costs, class_inverse = _shortlist_alignment_permutations(
            alignment_activity,
            spk_seq_arr,
            perm_batch,
            text_freq,
            rttm_freq,
            num_activity_speakers,
            max_alignment_permutations,
        )
        candidate_permutations = perm_batch[candidate_indices]
        dtw_costs = dtw_cost_batch(
            alignment_activity,
            spk_seq_arr,
            candidate_permutations,
            num_activity_speakers,
            token_weights,
        )
        if class_inverse is not None:
            # Preserve the original full-permutation frequency summation and
            # np.argmin tie order while reusing one expensive DTW score per
            # mathematically identical class.
            expanded_dtw_costs = dtw_costs[class_inverse]
            best_perm = perm_batch[int(np.argmin(expanded_dtw_costs + freq_costs))].tolist()
        else:
            best_candidate = int(np.argmin(dtw_costs + freq_costs[candidate_indices]))
            best_perm = candidate_permutations[best_candidate].tolist()
    else:
        best_perm = identity_perm

    fixed = speaker_activity[:, best_perm].clone()
    cols_to_zero = [idx for idx in range(num_activity_speakers) if idx not in speakers_in_text]
    if cols_to_zero:
        fixed[:, cols_to_zero] = 0.0

    return fixed


def speaker_activity_from_cut(
    cut,
    num_speakers: int,
    num_sample_per_mel_frame: int,
    num_mel_frame_per_target_frame: int,
    no_rttm_to_ones: bool = True,
    boundary_segments: bool = True,
    text: Optional[str] = None,
    return_permutation_resolved: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, bool]:
    """Build frame-level speaker activity targets from a Lhotse cut.

    Args:
        cut (Cut): Lhotse cut carrying RTTM/supervision speaker information.
        num_speakers (int): Number of speaker slots in the target tensor.
        num_sample_per_mel_frame (int): Audio samples per mel frame.
        num_mel_frame_per_target_frame (int): Mel frames per output target frame.
        no_rttm_to_ones (bool): If True, emit all-ones targets when no RTTM is present.
        boundary_segments (bool): If True, include boundary segments when building targets.
        text (Optional[str]): SOT text used to derive an exact ``<spk:N>`` label-to-column mapping.
        return_permutation_resolved (bool): If True, also return whether the RTTM labels exactly matched the text
            speaker tokens and were placed directly in their numeric target columns.

    Returns:
        Union[torch.Tensor, tuple[torch.Tensor, bool]]: Shape ``(T, num_speakers)`` frame-level speaker activity
            targets, optionally paired with the permutation-resolved detection status.
    """
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import speaker_to_target

    return speaker_to_target(
        a_cut=cut,
        num_speakers=num_speakers,
        num_sample_per_mel_frame=num_sample_per_mel_frame,
        num_mel_frame_per_asr_frame=num_mel_frame_per_target_frame,
        boundary_segments=boundary_segments,
        no_rttm_to_ones=no_rttm_to_ones,
        preferred_speaker_to_idx_map=get_speaker_token_index_map(text, num_speakers),
        return_speaker_mapping_status=return_permutation_resolved,
    )


def collate_speaker_activity_targets(
    speaker_activities: list[torch.Tensor],
    audio_lens: torch.Tensor,
    num_speakers: int,
    num_sample_per_mel_frame: int,
    num_mel_frame_per_target_frame: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate and length-compute speaker activity targets.

    Args:
        speaker_activities (list[torch.Tensor]): Per-example ``(T, N)`` activity tensors.
        audio_lens (torch.Tensor): Shape ``(B,)`` per-example audio sample lengths.
            Retained for API compatibility; target lengths are taken from the
            generated activity tensors themselves.
        num_speakers (int): Number of speaker columns to pad the targets to. Inputs with
            more columns are rejected to avoid silently discarding speaker supervision.
        num_sample_per_mel_frame (int): Audio samples per mel frame.
        num_mel_frame_per_target_frame (int): Mel frames per output target frame.
        dtype (torch.dtype): Output dtype for the collated targets.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(targets, target_length)`` where ``targets`` is
            ``(B, T, num_speakers)`` and ``target_length`` is ``(B,)``.
    """
    from lhotse.dataset.collation import collate_matrices

    # `collate_matrices` pads the time axis (dim 0) to the batch max but requires a
    # uniform speaker axis (dim 1). `speaker_to_target` emits one column per speaker
    # found in each cut's RTTM, so a batch mixing different speaker counts crashes
    # inside `collate_matrices`. Normalize every per-example target to exactly
    # `num_speakers` columns before collating. Never truncate here: that would silently
    # turn an 8-speaker PR-RTTM example into 4-speaker supervision when a recipe forgot
    # to override the historical default.
    normalized = []
    for activity in speaker_activities:
        n_spk = activity.shape[1]
        if n_spk > num_speakers:
            raise ValueError(
                f"Speaker activity has {n_spk} columns, but multispeaker_cfg.num_speakers={num_speakers}. "
                "Increase num_speakers instead of dropping speaker supervision."
            )
        elif n_spk < num_speakers:
            activity = torch.nn.functional.pad(activity, (0, num_speakers - n_spk), mode="constant", value=0.0)
        normalized.append(activity)

    targets = collate_matrices(normalized).to(dtype)
    # These tensors have already been generated on the target-frame grid. Their
    # actual time dimensions are therefore the authoritative valid lengths.
    # Recomputing them from loaded audio lengths can differ by a few frames after
    # resampling/augmentation or duration rounding and can exceed the collated
    # tensor's time dimension.
    target_length = torch.tensor([activity.shape[0] for activity in normalized], dtype=torch.long)
    return targets, target_length
