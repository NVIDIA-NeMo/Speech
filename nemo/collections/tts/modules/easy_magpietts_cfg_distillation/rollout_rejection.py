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

"""Rollout rejection rules for EasyMagpieTTS CFG distillation."""

from typing import Optional

import torch
from torch import Tensor

from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.distillation_types import TeacherOutput
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.teacher_streaming_state import TeacherStreamingState
from nemo.utils import logging


def apply_rollout_upper_bound_rejection(
    state: TeacherStreamingState,
    batch: dict[str, Tensor],
    upper_rejection_threshold: Optional[float],
    rejection_weight: Optional[float],
) -> TeacherStreamingState:
    """Reject unfinished rollouts that reach their per-item upper length bound.

    Args:
        state: Current streaming rollout state.
        batch: Batch containing ground-truth audio lengths and optional metadata.
        upper_rejection_threshold: Maximum allowed generated-length fraction of the
            ground-truth audio length, or `None` to disable this rejection rule.
        rejection_weight: Optional loss weight assigned to rejected items.

    Returns:
        Streaming rollout state updated with rejected items, final audio end
        indices and optional sample weights.
    """
    gt_audio_lens = batch["audio_codes_lens"]

    if upper_rejection_threshold is None or gt_audio_lens is None:
        return state

    if len(state.pred_codes) == 0:
        return state

    # Indices in pred_codes use the global rollout-frame coordinate system.
    current_frame_count = state.current_frame_idx
    upper_thresholds = torch.round(upper_rejection_threshold * gt_audio_lens).long().to(state.device)

    # Compare the threshold with the number of frames generated for each item,
    # not with the global rollout position. Items may start audio generation at
    # different global positions because their context lengths differ.
    has_started = state.audio_prediction_start_idx >= 0
    start_indices = state.audio_prediction_start_idx.clamp_min(0)
    generated_lens = current_frame_count - start_indices

    should_reject = has_started & (~state.finished) & (generated_lens >= upper_thresholds)

    if not should_reject.any():
        return state

    state.finished = state.finished | should_reject

    if state.rejected is None:
        state.rejected = torch.zeros_like(state.finished)

    state.rejected = state.rejected | should_reject
    newly_rejected = should_reject & (state.audio_prediction_end_idx == -1)

    if newly_rejected.any():
        state.audio_prediction_end_idx = torch.where(
            condition=newly_rejected,
            input=torch.full_like(state.audio_prediction_end_idx, current_frame_count),
            other=state.audio_prediction_end_idx,
        )

    if rejection_weight is not None:
        if state.sample_weights is None:
            state.sample_weights = torch.ones(
                state.batch_size,
                dtype=torch.float,
                device=state.device,
            )
        state.sample_weights = torch.where(
            condition=should_reject,
            input=torch.full_like(state.sample_weights, rejection_weight),
            other=state.sample_weights,
        )

    dataset_names = batch.get("dataset_names")
    languages = batch.get("languages")

    for item_idx in torch.nonzero(should_reject, as_tuple=False).flatten().tolist():
        dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
        language = languages[item_idx] if languages is not None else "Unknown"
        logging.info(
            f"Item {item_idx} rejected by upper length bound at generation step "
            f"{len(state.pred_codes)} (current length: {current_frame_count}, "
            f"dataset: {dataset_name}, language: {language})."
        )
    return state


def apply_rollout_max_steps_rejection(
    state: TeacherStreamingState,
    batch: dict[str, Tensor],
    rejection_weight: Optional[float],
) -> TeacherStreamingState:
    """Reject rollouts that remain unfinished at the decoding-step limit.

    Args:
        state: Current streaming rollout state.
        batch: Batch containing optional dataset and language metadata.
        rejection_weight: Optional loss weight assigned to rejected items.

    Returns:
        Streaming rollout state updated with rejected unfinished items, final audio
        boundaries and optional sample weights.
    """
    should_reject = ~state.finished

    if not should_reject.any():
        return state

    current_frame_count = state.current_frame_idx
    has_started = state.audio_prediction_start_idx >= 0
    never_started = should_reject & (~has_started)

    # A never-started item must have zero finalized length. Using the current
    # position as both start and end prevents it from inheriting predictions
    # generated for other batch items.
    state.audio_prediction_start_idx = torch.where(
        condition=never_started,
        input=torch.full_like(state.audio_prediction_start_idx, current_frame_count),
        other=state.audio_prediction_start_idx,
    )
    state.audio_prediction_end_idx = torch.where(
        condition=should_reject & (state.audio_prediction_end_idx == -1),
        input=torch.full_like(state.audio_prediction_end_idx, current_frame_count),
        other=state.audio_prediction_end_idx,
    )
    state.finished = state.finished | should_reject

    if state.rejected is None:
        state.rejected = torch.zeros_like(state.finished)

    state.rejected = state.rejected | should_reject

    if rejection_weight is not None:
        if state.sample_weights is None:
            state.sample_weights = torch.ones(
                state.batch_size,
                dtype=torch.float,
                device=state.device,
            )
        state.sample_weights = torch.where(
            condition=should_reject,
            input=torch.full_like(state.sample_weights, rejection_weight),
            other=state.sample_weights,
        )

    dataset_names = batch.get("dataset_names")
    languages = batch.get("languages")

    for item_idx in torch.nonzero(should_reject, as_tuple=False).flatten().tolist():
        dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
        language = languages[item_idx] if languages is not None else "Unknown"
        logging.info(
            f"Item {item_idx} rejected after reaching max_decoder_steps, "
            f"(global frame position: {current_frame_count}, "
            f"audio generation started: {bool(has_started[item_idx].item())}, "
            f"dataset: {dataset_name}, language: {language})."
        )

    return state


def apply_rollout_lower_bound_rejection(
    teacher_output: TeacherOutput,
    batch: dict[str, Tensor],
    lower_rejection_threshold: Optional[float],
    rejection_weight: Optional[float],
) -> TeacherOutput:
    """Apply lower-bound rejection to finalized teacher rollouts.

    Marks previously accepted rollouts as rejected when their finalized generated
    audio-code length is at or below the inclusive per-item lower bound. Optionally
    assigns the configured loss weight to newly rejected items.

    Args:
        teacher_output: Finalized teacher-rollout predictions and metadata.
        batch: Batch containing `"audio_codes_lens"` and optional sample metadata.
        lower_rejection_threshold: Fraction of each ground-truth audio-code length
            used to compute the inclusive lower rejection bound. A rollout at or
            below the rounded bound is rejected, or `None` to disable this rule.
        rejection_weight: Optional loss weight assigned to newly rejected items.

    Returns:
        Teacher output updated with lower-bound rejection flags and sample weights.

    Raises:
        KeyError: If `batch` does not contain `"audio_codes_lens"`.
    """
    dataset_names = batch.get("dataset_names")
    languages = batch.get("languages")
    gt_audio_lens = batch["audio_codes_lens"]

    if lower_rejection_threshold is None or gt_audio_lens is None:
        return teacher_output

    device = teacher_output.lens.device
    lower_thresholds = torch.round(lower_rejection_threshold * gt_audio_lens).long().to(device)
    should_reject = teacher_output.lens <= lower_thresholds

    # Do not apply lower-bound rejection to samples already rejected earlier.
    if teacher_output.rejected is not None:
        should_reject = should_reject & (~teacher_output.rejected)

    if not should_reject.any():
        return teacher_output

    if teacher_output.rejected is None:
        rejected = should_reject.clone()
    else:
        rejected = teacher_output.rejected | should_reject

    teacher_output.rejected = rejected

    if teacher_output.sample_weights is None:
        sample_weights = torch.ones_like(teacher_output.lens, dtype=torch.float, device=device)
    else:
        sample_weights = teacher_output.sample_weights.clone()

    if rejection_weight is not None:
        sample_weights = torch.where(
            condition=should_reject,
            input=torch.full_like(sample_weights, rejection_weight),
            other=sample_weights,
        )

    teacher_output.sample_weights = sample_weights

    for item_idx in torch.nonzero(should_reject, as_tuple=False).flatten().tolist():
        dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
        language = languages[item_idx] if languages is not None else "Unknown"
        logging.info(
            f"Item {item_idx} rejected by lower length bound after finalization "
            f"(threshold: {lower_thresholds[item_idx].item()}, "
            f"predicted length: {teacher_output.lens[item_idx].item()}, "
            f"dataset: {dataset_name}, language: {language})."
        )
    return teacher_output
