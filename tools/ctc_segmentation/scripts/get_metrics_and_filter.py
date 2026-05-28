# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import argparse
import json
import os
from glob import glob

import editdistance
from joblib import Parallel, delayed
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.utils import logging

parser = argparse.ArgumentParser("Calculate metrics and filters out samples based on thresholds")
parser.add_argument(
    "--manifest",
    required=True,
    help="Path .json manifest file with ASR predictions saved at `pred_text` field.",
)
parser.add_argument(
    "--edge_len", type=int, help="Number of characters to use for CER calculation at the edges", default=5
)
parser.add_argument("--audio_dir", type=str, help="Path to original .wav files", default=None)
parser.add_argument("--max_cer", type=int, help="Threshold CER value, %", default=30)
parser.add_argument("--max_wer", type=int, help="Threshold WER value, %", default=75)
parser.add_argument(
    "--max_len_diff_ratio",
    type=float,
    help="Threshold for len diff ratio between reference text "
    "length and predicted text length with respect to "
    "the reference text length (length measured "
    "in number of characters)",
    default=0.3,
)
parser.add_argument("--max_edge_cer", type=int, help="Threshold edge CER value, %", default=60)
parser.add_argument("--max_duration", type=int, help="Max duration of a segment, seconds", default=-1)
parser.add_argument("--min_duration", type=int, help="Min duration of a segment, seconds", default=1)
parser.add_argument(
    "--num_jobs",
    default=-2,
    type=int,
    help="The maximum number of concurrently running jobs, `-2` - all CPUs but one are used",
)
parser.add_argument(
    "--only_filter",
    action="store_true",
    help="Set to True to perform only filtering (when transcripts" "are already available)",
)


def _calculate(line: dict, edge_len: int):
    """
    Calculates metrics for every entry on manifest.json.

    Args:
        line - line of manifest.json (dict)
        edge_len - number of characters for edge Character Error Rate (CER) calculations

    Returns:
        line - line of manifest.json (dict) with the following metrics added:
        WER - word error rate
        CER - character error rate
        start_CER - CER at the beginning of the audio sample considering first 'edge_len' characters
        end_CER - CER at the end of the audio sample considering last 'edge_len' characters
        len_diff_ratio - ratio between reference text length and predicted text length with respect to
            the reference text length (length measured in number of characters)
    """
    eps = 1e-9

    text = line["text"].split()
    pred_text = line["pred_text"].split()

    num_words = max(len(text), eps)
    word_dist = editdistance.eval(text, pred_text)
    line["WER"] = word_dist / num_words * 100.0
    num_chars = max(len(line["text"]), eps)
    char_dist = editdistance.eval(line["text"], line["pred_text"])
    line["CER"] = char_dist / num_chars * 100.0

    line["start_CER"] = editdistance.eval(line["text"][:edge_len], line["pred_text"][:edge_len]) / edge_len * 100
    line["end_CER"] = editdistance.eval(line["text"][-edge_len:], line["pred_text"][-edge_len:]) / edge_len * 100
    line["len_diff_ratio"] = 1.0 * abs(len(text) - len(pred_text)) / max(len(text), eps)
    return line


def get_metrics(manifest, manifest_out):
    """Calculate metrics for sample in manifest and saves the results to manifest_out"""
    with open(manifest, "r") as f:
        lines = f.readlines()

    lines = Parallel(n_jobs=args.num_jobs)(
        delayed(_calculate)(json.loads(line), edge_len=args.edge_len) for line in tqdm(lines)
    )
    with open(manifest_out, "w") as f_out:
        for line in lines:
            f_out.write(json.dumps(line) + "\n")
    logging.info(f"Metrics save at {manifest_out}")


def _apply_filters(
    manifest,
    manifest_out,
    max_cer,
    max_wer,
    max_edge_cer,
    max_len_diff_ratio,
    max_dur=-1,
    min_dur=1,
    original_duration=0,
):
    """Filters out samples that do not satisfy specified threshold values and saves remaining samples to manifest_out"""
    remaining_duration = 0
    segmented_duration = 0
    with open(manifest, "r") as f, open(manifest_out, "w") as f_out:
        for line in f:
            item = json.loads(line)
            cer = item["CER"]
            wer = item["WER"]
            len_diff_ratio = item["len_diff_ratio"]
            duration = item["duration"]
            segmented_duration += duration
            if (
                cer <= max_cer
                and wer <= max_wer
                and len_diff_ratio <= max_len_diff_ratio
                and item["end_CER"] <= max_edge_cer
                and item["start_CER"] <= max_edge_cer
                and (max_dur == -1 or (max_dur > -1 and duration < max_dur))
                and duration > min_dur
            ):
                remaining_duration += duration
                f_out.write(json.dumps(item) + "\n")

    logging.info("-" * 50)
    logging.info("Threshold values:")
    logging.info(f"max WER, %: {max_wer}")
    logging.info(f"max CER, %: {max_cer}")
    logging.info(f"max edge CER, %: {max_edge_cer}")
    logging.info(f"max Word len diff: {max_len_diff_ratio}")
    logging.info(f"max Duration, s: {max_dur}")
    logging.info("-" * 50)

    remaining_duration = remaining_duration / 60
    original_duration = original_duration / 60
    segmented_duration = segmented_duration / 60
    logging.info(f"Original audio dur: {round(original_duration, 2)} min")
    logging.info(
        f"Segmented duration: {round(segmented_duration, 2)} min ({round(100 * segmented_duration / original_duration, 2)}% of original audio)"
    )
    logging.info(
        f"Retained {round(remaining_duration, 2)} min ({round(100*remaining_duration/original_duration, 2)}% of original or {round(100 * remaining_duration / segmented_duration, 2)}% of segmented audio)."
    )
    logging.info(f"Retained data saved to {manifest_out}")


def filter(manifest):
    """
    Filters out samples that do not satisfy specified threshold values.

    Args:
        manifest: path to .json manifest
    """
    original_duration = 0
    if args.audio_dir:
        audio_files = glob(f"{os.path.abspath(args.audio_dir)}/*")
        for audio in audio_files:
            try:
                audio_data = AudioSegment.from_file(audio)
                duration = len(audio_data._samples) / audio_data._sample_rate
                original_duration += duration
            except Exception as e:
                logging.info(f"Skipping {audio} -- {e}")

    _apply_filters(
        manifest=manifest,
        manifest_out=manifest.replace(".json", "_filtered.json"),
        max_cer=args.max_cer,
        max_wer=args.max_wer,
        max_edge_cer=args.max_edge_cer,
        max_len_diff_ratio=args.max_len_diff_ratio,
        max_dur=args.max_duration,
        min_dur=args.min_duration,
        original_duration=original_duration,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.only_filter:
        manifest_with_metrics = args.manifest.replace(".json", "_metrics.json")
        get_metrics(args.manifest, manifest_with_metrics)
    else:
        manifest_with_metrics = args.manifest
    filter(manifest_with_metrics)
