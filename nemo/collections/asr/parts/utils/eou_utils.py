# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import LabelLoopingStateItem
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# Special tokens that close an utterance in ASR-EOU models (end-of-utterance and end-of-backchannel).
# These models are trained on single-utterance samples, so decoding must restart after emitting one of them.
UTTERANCE_END_TOKENS = ('<EOU>', '<EOB>')


@dataclass
class EOUResult:
    """
    A dataclass to store the EOU results.
    Args:
        latency: List of latencies in seconds.
        early_cutoff: List of early cutoffs in seconds.
        true_positives: Number of true positives.
        false_negatives: Number of false negatives.
        false_positives: Number of false positives.
        num_utterances: Number of utterances.
        num_predictions: Number of predictions.
        missing: Number of missing predictions.
    """

    latency: list
    early_cutoff: list
    true_positives: int
    false_negatives: int
    false_positives: int
    num_utterances: int
    num_predictions: int
    missing: int

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the EOUResult dataclass to a dictionary.
        Returns:
            Dict: A dictionary representation of the EOUResult.
        """
        return {
            'latency': self.latency,
            'early_cutoff': self.early_cutoff,
            'true_positives': self.true_positives,
            'false_negatives': self.false_negatives,
            'false_positives': self.false_positives,
            'num_utterances': self.num_utterances,
            'num_predictions': self.num_predictions,
            'missing': self.missing,
        }


def flatten_nested_list(nested_list: List[List[float]]) -> List[float]:
    """
    Flatten a nested list into a single list.
    Args:
        nested_list (List[List]): A nested list to be flattened.
    Returns:
        List: A flattened list.
    """
    return [item for sublist in nested_list for item in sublist]


def evaluate_eou(
    *, prediction: List[dict], reference: List[dict], threshold: float, collar: float, do_sorting: bool = True
) -> EOUResult:
    """
    Evaluate end of utterance predictions against reference labels.
    Each item in predicition/reference is a dictionary in SegLST containing:
    {
        "session_id": str,
        "start_time": float,  # start time in seconds
        "end_time": float,  # end time in seconds
        "words": str,  # transcription of the utterance
        "audio_filepath": str,  # only in prediction
        "eou_prob": float, # only in prediction, probability of EOU in range [0.1]
        "eou_pred": bool, # only in prediction
        "full_text": str, # only in prediction, which is the full transcription up to the end_time
    }

    Args:
        predictions (List[dict]): List of dictionaries containing predictions.
        references (List[dict]): List of dictionaries containing reference labels.
        threshold (float): Threshold on "eou_prob" for considering a prediction as EOU. If it is None or
            non-positive, the boolean "eou_pred" field is used instead when every prediction provides it.
        collar (float): Collar time in seconds for matching predictions to references.
        do_sorting (bool): Whether to sort the predictions and references by start time.
    Returns:
        EOUResult: A dataclass containing the evaluation results.
    """

    latency = []
    early_cutoff = []
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    num_utterances = len(reference)
    num_predictions = len(prediction)
    missing = 0
    earlycut_ids = set()
    predicted_eou = prediction
    if threshold is not None and threshold > 0:
        predicted_eou = [p for p in prediction if p["eou_prob"] > threshold]
    elif all(["eou_pred" in p for p in prediction]):
        # If eou_pred is available, use it
        predicted_eou = [p for p in prediction if p["eou_pred"]]

    if do_sorting:
        predicted_eou = sorted(predicted_eou, key=lambda x: x["start_time"])
        reference = sorted(reference, key=lambda x: x["start_time"])

    p_idx = 0
    r_idx = 0
    for p_idx in range(len(predicted_eou)):
        p = predicted_eou[p_idx]
        p_start = p["start_time"]
        p_end = p["end_time"]

        while r_idx < len(reference) and reference[r_idx]["end_time"] < p_start:
            # Current reference ends before the current predicted utterance starts, find the next reference
            r_idx += 1

        if r_idx >= len(reference):
            # No more references to compare against
            false_positives += 1
            continue

        r = reference[r_idx]
        r_start = r["start_time"]
        r_end = r["end_time"]

        if np.abs(p_end - r_end) <= collar:
            # Correctly predicted EOU
            true_positives += 1
            latency.append(p_end - r_end)
            if r_idx in earlycut_ids:
                # If this reference was already missed due to early cutoff, we do not count it again
                earlycut_ids.remove(r_idx)
            r_idx += 1
        elif r_start <= p_end < r_end - collar:
            # Early cutoff
            # current predicted EOU is within the current reference utterance
            false_positives += 1
            early_cutoff.append(r_end - p_end)
            earlycut_ids.add(r_idx)
        elif r_end + collar < p_end:
            # Late EOU
            # Current predicted EOU is after the current reference ends
            false_negatives += 1
            latency.append(p_end - r_end)
            if r_idx in earlycut_ids:
                # If this reference was already missed due to early cutoff, we do not count it again
                earlycut_ids.remove(r_idx)
            r_idx += 1
        else:
            # p_end <= r_start
            # Current predicted EOU is before the current reference starts
            false_positives += 1

    if r_idx < len(reference):
        # There are remaining references that were not matched
        false_negatives += len(reference) - r_idx
        missing += len(reference) - r_idx

    # Only references counted by the block above can be discounted here. An early cutoff
    # that r_idx already advanced past was never added, so subtracting it drives both
    # counts negative.
    counted_earlycut = sum(1 for idx in earlycut_ids if idx >= r_idx)
    missing -= counted_earlycut  # Remove the references that were missed due to early cutoff
    false_negatives -= counted_earlycut  # Remove the references that were missed due to early cutoff
    return EOUResult(
        latency=latency,
        early_cutoff=early_cutoff,
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        num_utterances=num_utterances,
        num_predictions=num_predictions,
        missing=missing,
    )


def get_SegLST_from_frame_labels(frame_labels: List[int], frame_len_in_secs: float = 0.08) -> List[dict]:
    """
    Convert frame labels to SegLST format.
    Each positively labeled frame marks the end of an utterance, so a segment spans from the end of the
    previous utterance to the absolute time of the current EOU frame.
    Args:
        frame_labels (List[int]): List of frame labels.
        frame_len_in_secs (float): Length of each frame in seconds.
    Returns:
        List[dict]: List of dictionaries in SegLST format.
    """
    seg_lst = []
    start_time = 0.0
    for i, label in enumerate(frame_labels):
        if label > 0:
            end_time = frame_len_in_secs * i
            seg_lst.append({"start_time": start_time, "end_time": end_time, "eou_prob": label})
            start_time = end_time
    return seg_lst


def cal_eou_metrics_from_frame_labels(
    *,
    prediction: List[float],
    reference: List[float],
    threshold: float = 0.5,
    collar: float = 0,
    frame_len_in_secs: float = 0.08,
) -> EOUResult:
    """
    Calculate EOU metrics from lists of predictions and references.
    Args:
        prediction (List): List of floats containing predicted EOU probabilities.
        reference (List): List of binary floats containing reference EOU probabilities.
        threshold (float): Threshold for considering a prediction as EOU.
        collar (float): Collar time in seconds for matching predictions to references.
        frame_len_in_secs (float): Length of each frame in seconds.
    """

    if len(prediction) != len(reference):
        raise ValueError(
            f"Prediction ({len(prediction)}) and reference ({len(reference)}) lists must have the same length."
        )

    pred_seg_lst = get_SegLST_from_frame_labels(prediction, frame_len_in_secs)
    ref_seg_lst = get_SegLST_from_frame_labels(reference, frame_len_in_secs)
    eou_metrics = evaluate_eou(
        prediction=pred_seg_lst, reference=ref_seg_lst, threshold=threshold, collar=collar, do_sorting=False
    )
    return eou_metrics


def get_percentiles(values: List[float], percentiles: List[float], tag: str = "") -> Dict[str, float]:
    """
    Get the percentiles of a list of values.
    Args:
        values: list of values
        percentiles: list of percentiles
    Returns:
        metrics: Dict of percentiles
    """
    if len(values) == 0:
        return [0.0] * len(percentiles)
    results = np.percentile(values, percentiles).tolist()
    metrics = {}
    if tag:
        tag += "_"
    for i, p in enumerate(percentiles):
        metrics[f'{tag}p{int(p)}'] = float(results[i])
    return metrics


def aggregate_eou_metrics(eou_metrics: List[EOUResult], target_percentiles: List = [50, 90, 95]) -> Dict[str, float]:
    """
    Aggregate EOU metrics to produce metrics for logging.
    Args:
        eou_metrics: List of EOUResult objects.
        target_percentiles: List of target percentiles.
    Returns:
        Dict: A dictionary containing the aggregated EOU metrics.
    """
    num_eou_utterances = sum([x.num_utterances for x in eou_metrics])
    eou_latency = flatten_nested_list([x.latency for x in eou_metrics])
    eou_early_cutoff = flatten_nested_list([x.early_cutoff for x in eou_metrics])

    eou_avg_num_early_cutoff = len(eou_early_cutoff) / num_eou_utterances if num_eou_utterances > 0 else 0.0
    if len(eou_latency) == 0:
        eou_latency = [0.0]
    if len(eou_early_cutoff) == 0:
        eou_early_cutoff = [0.0]

    eou_missing = [x.missing for x in eou_metrics]

    metrics = {}
    eou_latency_metrics = get_percentiles(eou_latency, target_percentiles, tag='latency')
    eou_early_cutoff_metrics = get_percentiles(eou_early_cutoff, target_percentiles, tag='early_cutoff')

    metrics.update(eou_latency_metrics)
    metrics.update(eou_early_cutoff_metrics)

    metrics['early_cutoff_rate'] = eou_avg_num_early_cutoff
    metrics['miss_rate'] = sum(eou_missing) / num_eou_utterances if num_eou_utterances > 0 else 0.0

    return metrics


def get_utterance_end_token_ids(tokenizer) -> List[int]:
    """
    Get the ids of utterance-end special tokens (``<EOU>``, ``<EOB>``) present in the tokenizer vocabulary.

    Args:
        tokenizer: NeMo tokenizer (or None).
    Returns:
        List of token ids; empty if the tokenizer has no utterance-end tokens.
    """
    if tokenizer is None:
        return []
    token_ids = []
    for token in UTTERANCE_END_TOKENS:
        try:
            token_id = tokenizer.token_to_id(token)
            # unknown tokens are mapped to <unk> by most tokenizers, so verify the round trip
            if token_id is not None and token_id >= 0 and tokenizer.ids_to_tokens([token_id]) == [token]:
                token_ids.append(int(token_id))
        except Exception:  # tokenizers without single-token lookup (e.g. aggregate) have no utterance-end tokens
            continue
    return token_ids


def get_hypotheses_lengths(hypotheses: Optional[Sequence[Optional[Hypothesis]]], batch_size: int) -> List[int]:
    """
    Number of tokens in each (possibly missing) partial hypothesis carried between streaming steps.

    Args:
        hypotheses: partial hypotheses (or None before the first streaming step).
        batch_size: number of streams.
    Returns:
        List of token counts.
    """
    if hypotheses is None:
        return [0] * batch_size
    return [0 if hyp is None or hyp.y_sequence is None else len(hyp.y_sequence) for hyp in hypotheses]


def find_utterance_end_emissions(
    hypotheses: Sequence[Optional[Hypothesis]], prev_lengths: Sequence[int], token_ids: Sequence[int]
) -> List[bool]:
    """
    Check which hypotheses emitted an utterance-end token after their first ``prev_lengths[i]`` tokens.

    Args:
        hypotheses: hypotheses after the current (streaming) decoding step.
        prev_lengths: number of tokens in each hypothesis before the current step.
        token_ids: utterance-end token ids.
    Returns:
        List of flags, True for the hypotheses that emitted an utterance-end token in the current step.
    """
    flags = []
    for hyp, prev_length in zip(hypotheses, prev_lengths):
        if hyp is None or hyp.y_sequence is None or len(token_ids) == 0:
            flags.append(False)
            continue
        new_tokens = _to_list(hyp.y_sequence)[prev_length:]
        flags.append(any(token in token_ids for token in new_tokens))
    return flags


def reset_hypothesis_decoder_state_(hypothesis: Hypothesis, decoding_computer=None) -> Hypothesis:
    """
    Reset (in place) the decoder (prediction network) state of a partial hypothesis to the start-of-sequence state,
    keeping the decoded tokens and the time offset, so that the next streaming step starts a new utterance.

    Args:
        hypothesis: partial hypothesis carried between streaming steps.
        decoding_computer: label-looping decoding computer used to produce ``hypothesis.dec_state`` (if any).
    Returns:
        The same hypothesis.
    """
    state = hypothesis.dec_state
    if isinstance(state, LabelLoopingStateItem) and decoding_computer is not None:
        # label-looping state: reset predictor state and last label, but keep the number of decoded frames,
        # which offsets the timestamps of the next streaming step
        batched_state = decoding_computer.merge_to_batched_state([state])
        decoded_lengths = batched_state.decoded_lengths.clone()
        batched_state = decoding_computer.reset_state_by_mask(
            batched_state, torch.ones_like(decoded_lengths, dtype=torch.bool)
        )
        batched_state.decoded_lengths = decoded_lengths
        new_state = decoding_computer.split_batched_state(batched_state)[0]
        new_state.time_jump = state.time_jump
        hypothesis.dec_state = new_state
    else:
        hypothesis.dec_state = None
    hypothesis.last_token = None
    return hypothesis


def truncate_hypothesis_after_first_utterance_end_(
    hypothesis: Hypothesis, token_ids: Sequence[int], num_frames: int
) -> Optional[int]:
    """
    If the hypothesis emitted an utterance-end token before its last frame, truncate it (in place)
    right after the first such token.

    Args:
        hypothesis: raw (not post-processed) hypothesis with per-token frame indices in ``timestamp``.
        token_ids: utterance-end token ids.
        num_frames: number of encoder frames decoded for this hypothesis.
    Returns:
        Index of the frame where the utterance-end token was emitted if the hypothesis was truncated, else None.
    """
    tokens = _to_list(hypothesis.y_sequence)
    pos = next((i for i, token in enumerate(tokens) if token in token_ids), None)
    if pos is None:
        return None
    frame = int(hypothesis.timestamp[pos])
    if frame >= num_frames - 1:
        return None
    hypothesis.y_sequence = hypothesis.y_sequence[: pos + 1]
    hypothesis.timestamp = hypothesis.timestamp[: pos + 1]
    if hypothesis.token_duration is not None and len(hypothesis.token_duration) > pos + 1:
        hypothesis.token_duration = hypothesis.token_duration[: pos + 1]
    if hypothesis.non_blank_step_confidence_precomputed is not None:
        hypothesis.non_blank_step_confidence_precomputed = hypothesis.non_blank_step_confidence_precomputed[: pos + 1]
    if hypothesis.alignments is not None:
        hypothesis.alignments = hypothesis.alignments[: frame + 1]
    if hypothesis.frame_confidence is not None:
        hypothesis.frame_confidence = hypothesis.frame_confidence[: frame + 1]
    return frame


def append_hypothesis_segment_(hypothesis: Hypothesis, segment: Hypothesis, frame_offset: int) -> Hypothesis:
    """
    Append (in place) a raw hypothesis decoded from a later segment of the same audio.

    Args:
        hypothesis: raw hypothesis of the preceding audio.
        segment: raw hypothesis of the segment starting at encoder frame ``frame_offset``.
        frame_offset: encoder frame index where ``segment`` starts.
    Returns:
        The merged hypothesis.
    """
    if isinstance(segment.timestamp, torch.Tensor):
        segment.timestamp = segment.timestamp + frame_offset
    elif segment.timestamp is not None:
        segment.timestamp = [t + frame_offset for t in segment.timestamp]
    if isinstance(hypothesis.token_duration, torch.Tensor) and isinstance(segment.token_duration, torch.Tensor):
        token_duration = torch.cat((hypothesis.token_duration, segment.token_duration), dim=0)
    else:
        token_duration = hypothesis.token_duration
    hypothesis.merge_(segment)
    hypothesis.token_duration = token_duration
    return hypothesis


def _to_list(tokens) -> list:
    return tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
