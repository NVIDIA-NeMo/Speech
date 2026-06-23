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
"""Utilities for SOT-style speaker tokens and speaker-activity alignment."""
# pylint: disable=import-error

import re
from itertools import permutations
from typing import Optional, Sequence

import numpy as np
import torch

SPEAKER_TOKEN_PATTERN = re.compile(r"<spk:(\d+)>")
_SPEAKER_TOKEN_SPLIT_PATTERN = re.compile(r"(<spk:\d+>)")

__all__ = [
    "SPEAKER_TOKEN_PATTERN",
    "collate_speaker_activity_targets",
    "dtw_cost",
    "dtw_cost_batch",
    "ensure_single_speaker_sot",
    "fix_speaker_activity",
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

    valid = spk_seq_arr < num_speakers
    activity_permuted = activity[:, perm_batch].transpose(1, 0, 2)  # (P, T, N)
    activity_sum = np.maximum(activity.sum(axis=1), 1.0).astype(np.float32)
    cols = np.where(valid, spk_seq_arr, 0)
    local = 1.0 - activity_permuted[:, :, cols].transpose(0, 2, 1) / activity_sum
    local[:, ~valid, :] = 1.0

    if token_weights is not None:
        local = local * token_weights[np.newaxis, :, np.newaxis]

    inf = np.float32(np.inf)
    prev_row = np.cumsum(local[:, 0, :], axis=1).astype(np.float32)

    for token_idx in range(1, num_tokens):
        cur_row = np.full((num_perms, num_frames), inf, dtype=np.float32)
        cur_row[:, 0] = prev_row[:, 0] + local[:, token_idx, 0]
        for frame_idx in range(1, num_frames):
            cur_row[:, frame_idx] = (
                np.minimum(
                    np.minimum(prev_row[:, frame_idx], prev_row[:, frame_idx - 1]),
                    cur_row[:, frame_idx - 1],
                )
                + local[:, token_idx, frame_idx]
            )
        prev_row = cur_row

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


def fix_speaker_activity(
    cut_or_text,
    speaker_activity: torch.Tensor,
    num_speakers: int,
    max_permutable: Optional[int] = None,
) -> torch.Tensor:
    """Align RTTM speaker-activity columns with SOT speaker-token order.

    Args:
        cut_or_text (Union[Cut, str]): A Lhotse cut with a ``text`` attribute, or raw SOT text.
        speaker_activity (torch.Tensor): Shape ``(T, N)`` frame-level activity to reorder.
        num_speakers (int): Number of speakers used to bound the permutation search.
        max_permutable (Optional[int]): Max active speakers to brute-force permute over;
            defaults to ``num_speakers + 1``.

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
    activity_np = speaker_activity.detach().cpu().numpy().astype(np.float32)

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

        dtw_costs = dtw_cost_batch(activity_np, spk_seq_arr, perm_batch, num_activity_speakers, token_weights)
        freq_costs = speaker_freq_cost_batch(text_freq, rttm_freq, perm_batch)
        best_perm = perm_batch[int(np.argmin(dtw_costs + freq_costs))].tolist()
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
) -> torch.Tensor:
    """Build frame-level speaker activity targets from a Lhotse cut.

    Args:
        cut (Cut): Lhotse cut carrying RTTM/supervision speaker information.
        num_speakers (int): Number of speaker slots in the target tensor.
        num_sample_per_mel_frame (int): Audio samples per mel frame.
        num_mel_frame_per_target_frame (int): Mel frames per output target frame.
        no_rttm_to_ones (bool): If True, emit all-ones targets when no RTTM is present.
        boundary_segments (bool): If True, include boundary segments when building targets.

    Returns:
        torch.Tensor: Shape ``(T, num_speakers)`` frame-level speaker activity targets.
    """
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import speaker_to_target

    return speaker_to_target(
        a_cut=cut,
        num_speakers=num_speakers,
        num_sample_per_mel_frame=num_sample_per_mel_frame,
        num_mel_frame_per_asr_frame=num_mel_frame_per_target_frame,
        boundary_segments=boundary_segments,
        no_rttm_to_ones=no_rttm_to_ones,
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
        num_speakers (int): Number of speaker columns to pad/truncate the targets to.
        num_sample_per_mel_frame (int): Audio samples per mel frame.
        num_mel_frame_per_target_frame (int): Mel frames per output target frame.
        dtype (torch.dtype): Output dtype for the collated targets.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(targets, target_length)`` where ``targets`` is
            ``(B, T, num_speakers)`` and ``target_length`` is ``(B,)``.
    """
    from lhotse.dataset.collation import collate_matrices
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length

    # `collate_matrices` pads the time axis (dim 0) to the batch max but requires a
    # uniform speaker axis (dim 1). `speaker_to_target` emits one column per speaker
    # found in each cut's RTTM -- e.g. a 5-speaker cut yields (T, 5) even when
    # `num_speakers=4` -- so a batch mixing different speaker counts crashes inside
    # `collate_matrices`. Normalize every per-example target to exactly `num_speakers`
    # columns (truncate extras / zero-pad missing) BEFORE collating; this is what the
    # original post-collate clamp intended, just moved ahead of the collate.
    normalized = []
    for activity in speaker_activities:
        n_spk = activity.shape[1]
        if n_spk > num_speakers:
            activity = activity[:, :num_speakers]
        elif n_spk < num_speakers:
            activity = torch.nn.functional.pad(activity, (0, num_speakers - n_spk), mode="constant", value=0.0)
        normalized.append(activity)

    targets = collate_matrices(normalized).to(dtype)
    target_length = torch.tensor(
        [
            get_hidden_length_from_sample_length(al, num_sample_per_mel_frame, num_mel_frame_per_target_frame)
            for al in audio_lens
        ]
    )
    return targets, target_length
