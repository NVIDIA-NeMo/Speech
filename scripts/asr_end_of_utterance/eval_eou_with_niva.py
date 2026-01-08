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

"""
Example usage:

The NIVA_PRED_ROOT and REFERENCE_ROOT directories should have the following structure:

<NIVA_PRED_ROOT>:
->dataset1/
    eou/
    ctm/
->dataset2/
    eou/
    ctm/

<REFERENCE_ROOT>:
->dataset1/
->dataset2/


```bash
python eval_eou_with_niva.py \
    --prediction $NIVA_PRED_ROOT \
    --reference $REFERENCE_ROOT  \
    --multiple
```

"""


import argparse
import json
from pathlib import Path
from typing import List

from nemo.collections.asr.parts.utils.eou_utils import EOUResult, aggregate_eou_metrics, evaluate_eou

parser = argparse.ArgumentParser(description="Evaluate end of utterance predictions against reference labels.")
parser.add_argument(
    "-p",
    "--prediction",
    type=str,
    required=True,
    help="Path to the directory containing the predictions.",
)
parser.add_argument(
    "-r",
    "--reference",
    type=str,
    required=True,
    help="Path to the directory containing the groundtruth.",
)
parser.add_argument(
    "--eob",
    action="store_true",
    help="Whether to evaluate end of backchannel predictions.",
)
parser.add_argument(
    "--multiple",
    action="store_true",
    help="Whether to evaluate multiple datasets.",
)


def load_segLST(directory: str, use_eob: bool = False) -> dict:
    json_files = list(Path(directory).glob("*.json"))
    segLST = {}
    for json_file in json_files:
        key = json_file.stem
        with open(json_file, 'r') as f:
            data = json.load(f)
            is_backchannel = data[0].get("is_backchannel", False) if data else False
            if not isinstance(is_backchannel, list):
                is_backchannel = [is_backchannel]
            if any(is_backchannel) and not use_eob:
                continue
            segLST[key] = data
    return segLST


def evaluate_eou_predictions(prediction_dir: str, reference_dir: str, use_eob: bool = False) -> List[EOUResult]:
    prediction_dir = Path(prediction_dir) / "eou"
    prediction_segLST = load_segLST(prediction_dir, use_eob)
    reference_segLST = load_segLST(reference_dir, use_eob)

    eou_metrics = []
    for key, reference in reference_segLST.items():
        if key not in prediction_segLST:
            raise ValueError(f"Key {key} in reference not found in predictions.")
        prediction = prediction_segLST[key]
        eou_result = evaluate_eou(
            prediction=prediction, reference=reference, threshold=None, collar=0.0, do_sorting=True
        )
        eou_metrics.append(eou_result)

    results = aggregate_eou_metrics(eou_metrics)

    # add prefix to the keys of the results
    prefix = Path(reference_dir).stem
    prefix += "_eob" if use_eob else "_eou"
    results = {f"{prefix}_{k}": v for k, v in results.items()}

    return results


if __name__ == "__main__":
    args = parser.parse_args()

    prediction_dir = Path(args.prediction)
    reference_dir = Path(args.reference)

    if not prediction_dir.is_dir():
        raise ValueError(f"Prediction directory {prediction_dir} does not exist or is not a directory.")
    if not reference_dir.is_dir():
        raise ValueError(f"Reference directory {reference_dir} does not exist or is not a directory.")

    if args.multiple:
        # get all subdirectories in the prediction and reference directories
        prediction_dirs = sorted([x for x in prediction_dir.glob("*/") if x.is_dir()])
        reference_dirs = sorted([x for x in reference_dir.glob("*/") if x.is_dir()])
        if len(prediction_dirs) != len(reference_dirs):
            raise ValueError(
                f"Number of prediction directories {len(prediction_dirs)} must match number of reference directories {len(reference_dirs)}."
            )
    else:
        prediction_dirs = [prediction_dir]
        reference_dirs = [reference_dir]

    for ref_dir, pred_dir in zip(reference_dirs, prediction_dirs):
        if args.multiple and ref_dir.stem != pred_dir.stem:
            raise ValueError(
                f"Reference directory {ref_dir} and prediction directory {pred_dir} must have the same name."
            )
        results = evaluate_eou_predictions(prediction_dir=str(pred_dir), reference_dir=str(ref_dir), use_eob=args.eob)
        # Print the results
        print("==========================================")
        print(f"Evaluation Results for: {pred_dir} against {ref_dir}")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        print("==========================================")
