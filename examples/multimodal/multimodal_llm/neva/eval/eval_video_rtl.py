# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#

"""
This script is used for evaluating RTL (Reasoning Temporal Localization) task.
It accepts one JSON file. The JSON file should have the following structure:
[
    {
    "video": "rY7eLyJF31M_6.mp4",
    "question_id": "rY7eLyJF31M_6_0",
    "question": "When is \"Apply mascara , false lashes on the lashes \" depicted in the video? Convey your answer using start and end timestamps exclusively.",
    "ref_answer": "<0> <53> Apply mascara , false lashes on the lashes ",
    "duration": 102.002002002002,
    "pred_answer": "<1> <53> Apply mascara , false lashes on the lashes ",
    },
  {
    "video": "rY7eLyJF31M_6.mp4",
    "question_id": "rY7eLyJF31M_6_1",
    "question": "When is \"Apply foundation on the face with a brush\" depicted in the video? Provide a response using only start and end timestamps.",
    "ref_answer": "<56> <97> Apply foundation on the face with a brush",
    "duration": 102.002002002002,
    "pred_answer": "<50> <97> Apply foundation on the face with a brush",
  },
]

The `xxx_answer` field should contain the start and end timestamps such as `<56>` and `<97>` of the event along with the sentence.
If not, the [0, duration] will be used as the predicted timestamps.

USAGE:
python eval_rtl.py --input_file <path_to_predictions.json> \
    --output_dir <path_to_output_dir> \
    --save_mid_result
"""
import argparse
import json
import os
import re
from collections import defaultdict


def iou(seg1, seg2):
    """Compute the intersection over union (IoU) between two segments.

    Args:
        seg1 (list): [start, end]
        seg2 (list): [start, end]

    Returns:
        float: IoU value
    """
    assert seg1[1] >= seg1[0] and seg2[1] >= seg2[0]

    x1 = max(seg1[0], seg2[0])
    x2 = min(seg1[1], seg2[1])
    inter = max(x2 - x1, 0)

    len1 = max(seg1[1] - seg1[0], 0)
    len2 = max(seg2[1] - seg2[0], 0)

    union = len1 + len2 - inter

    if union == 0:
        return 0.0
    else:
        return inter / union


def precision_func(thres):
    """calculate the precision based on the threshold.
    If the IoU value is greater than or equal to the threshold, \
    the precision is 1.0, otherwise 0.0.

    Args:
        thres (float): threshold value [0.0, 1.0]
    """

    def precision(seg1, seg2):
        return float(iou(seg1, seg2) >= thres)

    return precision


def parse_start_end_timestamps(outputs, duration, strict=False):
    timestamp_pattern = '\<(?: (?: \d* \.? \d+ ) | (?: \d+ \.? ) )\>'
    rx = re.compile(timestamp_pattern, re.VERBOSE)
    matches = list(rx.finditer(outputs))
    if strict:
        assert len(list(matches)) >= 2, "cannot find timestamps"
    elif len(list(matches)) < 2:
        return outputs, [0, duration]

    prev_end = 0
    sentence = ""
    timestamps = []
    for i in range(2):
        m = matches[i]
        start = m.start(0)
        end = m.end(0)
        timestamp = float(m.group(0)[1:-1])
        timestamp = min(max(timestamp, 0), duration)
        timestamps.append(timestamp)
        sentence += outputs[prev_end:start]
        prev_end = end
    sentence += outputs[prev_end:]
    sentence = sentence.strip()

    return sentence, [min(timestamps), max(timestamps)]


def eval(pred_file, output_dir, save_mid_result=True):
    """Evaluate the predictions against the ground truth.

    Args:
        pred_file (str): path to the predictions JSON file
        output_dir (str): path to the output directory,
            where the `answers.json` and `metrics.json` result will be saved.
    """
    metric_func = {'iou': iou, 'precision@0.5': precision_func(0.5)}
    metrics = {}
    for metric in metric_func:
        metrics[metric] = defaultdict(list)

    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    out_list = []
    for pred in pred_data:
        assert "pred_answer" in pred, "pred_answer field is missing"
        assert "ref_answer" in pred, "answer field is missing"
        duration = pred['duration']
        pred_answer, pred_timestamps = parse_start_end_timestamps(pred['pred_answer'], duration, strict=False)
        ref_answer, ref_timestamps = parse_start_end_timestamps(pred['ref_answer'], duration, strict=False)

        for metric in metric_func:
            metrics[metric][pred['video']].append(metric_func[metric](pred_timestamps, ref_timestamps))

        out_list.append(
            {
                'video': pred['video'],
                'question_id': pred['question_id'],
                'question': pred['question'],
                'pred_answer': pred_answer,
                'ref_answer': ref_answer,
                'pred_timestamps': pred_timestamps,
                'ref_timestamps': ref_timestamps,
            }
        )
    # save result
    os.makedirs(output_dir, exist_ok=True)
    if save_mid_result:
        output_file = os.path.join(output_dir, 'answers.json')
        print(f"Saving intermediate result to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(out_list, f, indent=2)

    final_result = {}
    for metric in metrics:
        values = []
        for vid in metrics[metric]:
            # get single video metric value
            cur_metric_values = metrics[metric][vid]
            values.append(sum(cur_metric_values) / len(cur_metric_values))
        # get global average video metric value
        values = sum(values) / len(values)
        final_result[metric] = values

    print(final_result)
    output_file = os.path.join(output_dir, 'metrics.json')
    with open(output_file, 'w') as f:
        json.dump(final_result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the predictions against the ground truth")
    parser.add_argument("--input_file", help="Path to the input JSON file", required=True)
    parser.add_argument("--output_dir", help="Path to the output directory", required=True)
    parser.add_argument("--save_mid_result", action="store_true", help="Save intermediate result")
    args = parser.parse_args()

    eval(args.input_file, args.output_dir, args.save_mid_result)


if __name__ == "__main__":
    main()
