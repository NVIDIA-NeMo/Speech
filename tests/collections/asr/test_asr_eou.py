# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

import numpy as np
import pytest

from nemo.collections.asr.parts.utils.eou_utils import EOUResult, cal_eou_metrics_from_frame_labels


def make_eou_frame_labels(duration: float, eou_time: float, frame_len_in_secs: float = 0.08) -> List[float]:
    """
    Make EOU frame labels.
    Args:
        duration (float): Duration of the audio in seconds.
        eou_time (float): Time of the EOU in seconds.
        frame_len_in_secs (float): Length of each frame in seconds.
    Returns:
        List[float]: List of EOU frame labels.
    """
    if eou_time < 0 or eou_time > duration:
        raise ValueError(f"EOU time ({eou_time}) is out of range for duration ({duration}).")

    labels = [0] * int(np.ceil(duration / frame_len_in_secs) + 1)
    labels[int(np.ceil(eou_time / frame_len_in_secs))] = 1
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
