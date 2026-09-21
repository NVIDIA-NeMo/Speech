# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import List, Union

import numpy as np
import pytest

from nemo.collections.asr.parts.utils.eou_utils import (
    EOUResult,
    cal_eou_metrics_from_frame_labels,
    evaluate_eou,
    get_SegLST_from_frame_labels,
)


def make_eou_frame_labels(
    duration: float, eou_time: Union[float, List[float]], frame_len_in_secs: float = 0.08
) -> List[float]:
    """
    Make EOU frame labels.
    Args:
        duration (float): Duration of the audio in seconds.
        eou_time (float or List[float]): Time(s) of the EOU in seconds.
        frame_len_in_secs (float): Length of each frame in seconds.
    Returns:
        List[float]: List of EOU frame labels.
    """
    eou_times = [eou_time] if isinstance(eou_time, (int, float)) else eou_time

    labels = [0] * int(np.ceil(duration / frame_len_in_secs) + 1)
    for t in eou_times:
        if t < 0 or t > duration:
            raise ValueError(f"EOU time ({t}) is out of range for duration ({duration}).")
        labels[int(np.ceil(t / frame_len_in_secs))] = 1
    return labels


class TestEOUMetrics:
    @pytest.mark.unit
    def test_cal_eou_metrics_from_frame_labels(self):
        duration = 1.6
        eou_time = 0.64
        frame_len_in_secs = 0.08
        ref_labels = make_eou_frame_labels(duration, eou_time, frame_len_in_secs)

        # Test case 1: Early cutoff
        pred_eou_time = 0.32
        preds = make_eou_frame_labels(duration, pred_eou_time, frame_len_in_secs)
        eou_metrics: EOUResult = cal_eou_metrics_from_frame_labels(
            prediction=preds, reference=ref_labels, frame_len_in_secs=frame_len_in_secs
        )
        assert eou_metrics.true_positives == 0
        assert eou_metrics.false_positives == 1
        assert eou_metrics.false_negatives == 0
        assert eou_metrics.num_utterances == 1
        assert eou_metrics.num_predictions == 1
        assert eou_metrics.missing == 0
        assert eou_metrics.latency == []
        assert np.isclose(eou_metrics.early_cutoff, [0.32])

        # Test case 2: Latency
        pred_eou_time = 0.96
        preds = make_eou_frame_labels(duration, pred_eou_time, frame_len_in_secs)
        eou_metrics: EOUResult = cal_eou_metrics_from_frame_labels(
            prediction=preds, reference=ref_labels, frame_len_in_secs=frame_len_in_secs
        )
        assert eou_metrics.true_positives == 0
        assert eou_metrics.false_positives == 0
        assert eou_metrics.false_negatives == 1
        assert eou_metrics.num_utterances == 1
        assert eou_metrics.num_predictions == 1
        assert eou_metrics.missing == 0
        assert np.isclose(eou_metrics.latency, [0.32])
        assert eou_metrics.early_cutoff == []

        # Test case 3: miss detection
        preds = [0] * len(ref_labels)
        eou_metrics: EOUResult = cal_eou_metrics_from_frame_labels(
            prediction=preds, reference=ref_labels, frame_len_in_secs=frame_len_in_secs
        )
        assert eou_metrics.true_positives == 0
        assert eou_metrics.false_positives == 0
        assert eou_metrics.false_negatives == 1
        assert eou_metrics.num_utterances == 1
        assert eou_metrics.num_predictions == 0
        assert eou_metrics.missing == 1
        assert eou_metrics.latency == []
        assert eou_metrics.early_cutoff == []

    @pytest.mark.unit
    def test_evaluate_eou_filters_on_eou_pred_when_no_threshold(self):
        # scripts/asr_eou/eval_eou_metrics.py calls evaluate_eou() with threshold=None, so the per-segment
        # `eou_pred` flag is the only signal available to reject a hypothesised end of utterance.
        reference = [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 1.0, "end_time": 2.0},
        ]
        prediction = [
            {"start_time": 0.0, "end_time": 0.4, "eou_prob": 0.10, "eou_pred": False},
            {"start_time": 0.0, "end_time": 0.7, "eou_prob": 0.20, "eou_pred": False},
            {"start_time": 0.0, "end_time": 1.0, "eou_prob": 0.95, "eou_pred": True},
            {"start_time": 1.0, "end_time": 1.5, "eou_prob": 0.15, "eou_pred": False},
            {"start_time": 1.0, "end_time": 2.0, "eou_prob": 0.98, "eou_pred": True},
        ]

        eou_metrics: EOUResult = evaluate_eou(
            prediction=prediction, reference=reference, threshold=None, collar=0.0, do_sorting=True
        )

        # Only the two `eou_pred=True` segments count as predicted EOUs, and both match a reference exactly.
        assert eou_metrics.true_positives == 2
        assert eou_metrics.false_positives == 0
        assert eou_metrics.false_negatives == 0
        assert eou_metrics.missing == 0
        assert eou_metrics.early_cutoff == []
        assert np.allclose(eou_metrics.latency, [0.0, 0.0])
        # `num_predictions` reports how many segments were submitted, not how many survived filtering.
        assert eou_metrics.num_predictions == 5

    @pytest.mark.unit
    def test_evaluate_eou_keeps_segments_without_eou_pred(self):
        # SegLST segments built from frame labels carry no `eou_pred` key, so no filtering may happen.
        reference = [{"start_time": 0.0, "end_time": 1.0}]
        prediction = [{"start_time": 0.0, "end_time": 1.0, "eou_prob": 1.0}]

        eou_metrics: EOUResult = evaluate_eou(
            prediction=prediction, reference=reference, threshold=0.0, collar=0.0, do_sorting=True
        )

        assert eou_metrics.num_predictions == 1
        assert eou_metrics.true_positives == 1
        assert eou_metrics.false_positives == 0
        assert eou_metrics.false_negatives == 0

    @pytest.mark.unit
    def test_get_seglst_from_frame_labels_multiple_eou(self):
        # Frames 4, 12 and 20 are labelled as EOU, i.e. utterances end at 0.32s, 0.96s and 1.6s.
        # Each segment spans from the end of the previous utterance to its own EOU frame.
        frame_len_in_secs = 0.08
        frame_labels = [0] * 26
        for frame_idx in [4, 12, 20]:
            frame_labels[frame_idx] = 1

        seg_lst = get_SegLST_from_frame_labels(frame_labels, frame_len_in_secs)

        assert len(seg_lst) == 3
        expected = [(0.0, 0.32), (0.32, 0.96), (0.96, 1.6)]
        for segment, (expected_start, expected_end) in zip(seg_lst, expected):
            assert np.isclose(segment["start_time"], expected_start)
            assert np.isclose(segment["end_time"], expected_end)

    @pytest.mark.unit
    def test_cal_eou_metrics_from_frame_labels_multiple_eou(self):
        # Every predicted EOU is 2 frames (0.16s) late, so every latency must be 0.16s,
        # regardless of how many utterances precede it.
        duration = 2.0
        frame_len_in_secs = 0.08
        delay = 2 * frame_len_in_secs
        ref_eou_times = [0.32, 0.96, 1.6]
        pred_eou_times = [t + delay for t in ref_eou_times]

        ref_labels = make_eou_frame_labels(duration, ref_eou_times, frame_len_in_secs)
        preds = make_eou_frame_labels(duration, pred_eou_times, frame_len_in_secs)

        eou_metrics: EOUResult = cal_eou_metrics_from_frame_labels(
            prediction=preds, reference=ref_labels, frame_len_in_secs=frame_len_in_secs
        )
        assert eou_metrics.num_utterances == 3
        assert eou_metrics.num_predictions == 3
        assert eou_metrics.false_negatives == 3
        assert eou_metrics.false_positives == 0
        assert eou_metrics.missing == 0
        assert eou_metrics.early_cutoff == []
        assert np.allclose(eou_metrics.latency, [delay] * len(ref_eou_times))
