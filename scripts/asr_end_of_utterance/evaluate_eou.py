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
This script is deprecated !!!!
"""


import argparse
import json
from typing import List

import numpy as np

from nemo.collections.asr.parts.utils.eou_utils import evaluate_eou

parser = argparse.ArgumentParser(description="Evaluate end of utterance predictions against reference labels.")
parser.add_argument(
    "-p",
    "--predictions",
    type=str,
    required=True,
    help="Path to the JSON file containing the predictions.",
)
parser.add_argument(
    "-r",
    "--references",
    type=str,
    required=True,
    help="Path to the JSON file containing the reference labels.",
)
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help="Threshold for considering a prediction as EOU.",
)
parser.add_argument(
    "--drop_prefix",
    default="",
    type=str,
    help="Prefix to drop from the audio_filepath in the JSON file.",
)
parser.add_argument(
    "-c",
    "--collar",
    type=float,
    default=0.1,
    help="Collar time in seconds for matching predictions to references.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="eou_results/",
    help="Output directory to save the evaluation results.",
)


def load_json(file_path: str, drop_prefix: str = "") -> List[dict]:
    """Load a JSON file, then clean the audio_filepath."""
    with open(file_path, "r") as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        audio_filepath = item["audio_filepath"]
        if drop_prefix and audio_filepath.startswith(drop_prefix):
            audio_filepath = audio_filepath[len(drop_prefix) :]
        elif audio_filepath.startswith("./"):
            audio_filepath = audio_filepath[2:]
        item["audio_filepath"] = audio_filepath

        cleaned_data.append(item)
    return cleaned_data


def main():
    args = parser.parse_args()

    predictions = load_json(args.predictions, args.drop_prefix)
    references = load_json(args.references, args.drop_prefix)
    results = evaluate_eou(
        prediction=predictions,
        reference=references,
        threshold=args.threshold,
        collar=args.collar,
    )

    f1_score = (
        (2 * results.true_positives / (2 * results.true_positives + results.false_negatives + results.false_positives))
        if (results.true_positives + results.false_negatives + results.false_positives) > 0
        else 0
    )

    avg_cutoffs = len(results.early_cutoff) / len(results.num_utterances) if len(results.num_utterances) > 0 else 0

    p80_cutoff = np.percentile(results.early_cutoff, 80) if len(results.early_cutoff) > 0 else 0
    p90_cutoff = np.percentile(results.early_cutoff, 90) if len(results.early_cutoff) > 0 else 0
    p95_cutoff = np.percentile(results.early_cutoff, 95) if len(results.early_cutoff) > 0 else 0
    p99_cutoff = np.percentile(results.early_cutoff, 99) if len(results.early_cutoff) > 0 else 0

    p80_latency = np.percentile(results.latency, 80) if len(results.latency) > 0 else 0
    p90_latency = np.percentile(results.latency, 90) if len(results.latency) > 0 else 0
    p95_latency = np.percentile(results.latency, 95) if len(results.latency) > 0 else 0
    p99_latency = np.percentile(results.latency, 99) if len(results.latency) > 0 else 0

    # Print the results
    print("======================================")
    print("Evaluation Results:")
    print(f"Number of utterances: {results.num_utterances}")
    print(f"Number of predictions: {results.num_predictions}")
    print(f"F1 Score: {f1_score:.4f}")
    print("======================================")
    print(f"Early cutoff rate: {avg_cutoffs:.4f}")
    print(f"Early cutoff P80: {p80_cutoff:.4f} seconds")
    print(f"Early cutoff P90: {p90_cutoff:.4f} seconds")
    print(f"Early cutoff P95: {p95_cutoff:.4f} seconds")
    print(f"Early cutoff P99: {p99_cutoff:.4f} seconds")
    print("======================================")
    print(f"P80 Latency: {p80_latency:.4f} seconds")
    print(f"P90 Latency: {p90_latency:.4f} seconds")
    print(f"P95 Latency: {p95_latency:.4f} seconds")
    print(f"P99 Latency: {p99_latency:.4f} seconds")


if __name__ == "__main__":
    main()
