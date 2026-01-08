# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List

import numpy as np


@dataclass
class EOUResult:
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
        threshold (float): Threshold for considering a prediction as EOU.
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
    elif all([hasattr(p, "eou_pred") for p in prediction]):
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

    missing -= len(earlycut_ids)  # Remove the references that were missed due to early cutoff
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
            end_time = start_time + frame_len_in_secs * i
            seg_lst.append({"start_time": start_time, "end_time": end_time, "eou_prob": label})
            start_time = end_time
    return seg_lst


def cal_eou_metrics_from_frame_labels(
    *, prediction: List, reference: List, threshold: float = 0.5, collar: float = 0, frame_len_in_secs: float = 0.08
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
    # Aggregate EOU metrics
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
