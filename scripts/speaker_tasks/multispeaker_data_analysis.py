# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import multiprocessing
import shutil
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sox
from scipy.stats import expon
from tqdm import tqdm

from nemo.collections.asr.parts.utils.vad_utils import (
    get_nonspeech_segments,
    load_speech_overlap_segments_from_rttm,
    plot_sample_from_rttm,
)

"""
This script analyzes multi-speaker speech dataset and generates statistics.
The input directory </path/to/rttm_and_wav_directory> is required to contain the following files:
    - rttm files (*.rttm)
    - wav files (*.wav)

Usage:
    python <NEMO_ROOT>/scripts/speaker_tasks/multispeaker_data_analysis.py \
        </path/to/rttm_and_wav_directory> \
        --session_dur 20 \
        --silence_mean 0.2 \
        --silence_var 100 \
        --overlap_mean 0.15 \
        --overlap_var 50 \
        --num_workers 8 \
        --num_samples 10 \
        --output_dir <path/to/output_directory>
"""


def process_sample(sess_dict: Dict) -> Dict:
    """
    Process each synthetic sample

    Args:
        sess_dict (dict): dictionary containing the following keys
            rttm_file (str): path to the rttm file
            session_dur (float): duration of the session (specified by argument)
            precise (bool): whether to measure the precise duration of the session using sox

    Returns:
        results (dict): dictionary containing the following keys
            session_dur (float): duration of the session
            silence_len_list (list): list of silence durations of each silence occurrence
            silence_dur (float): total silence duration in a session
            silence_ratio (float): ratio of silence duration to session duration
            overlap_len_list (list): list of overlap durations of each overlap occurrence
            overlap_dur (float): total overlap duration
            overlap_ratio (float): ratio of overlap duration to speech (non-silence) duration
    """

    rttm_file = sess_dict["rttm_file"]
    session_dur = sess_dict["session_dur"]
    precise = sess_dict["precise"]
    if precise or session_dur is None:
        wav_file = rttm_file.parent / Path(rttm_file.stem + ".wav")
        session_dur = sox.file_info.duration(str(wav_file))

    speech_seg, overlap_seg = load_speech_overlap_segments_from_rttm(rttm_file)
    speech_dur = sum([sess_dict[1] - sess_dict[0] for sess_dict in speech_seg])

    silence_seg = get_nonspeech_segments(speech_seg, session_dur)
    silence_len_list = [sess_dict[1] - sess_dict[0] for sess_dict in silence_seg]
    silence_dur = max(0, session_dur - speech_dur)
    silence_ratio = silence_dur / session_dur

    overlap_len_list = [sess_dict[1] - sess_dict[0] for sess_dict in overlap_seg]
    overlap_dur = sum(overlap_len_list) if len(overlap_len_list) else 0
    overlap_ratio = overlap_dur / speech_dur

    results = {
        "session_dur": session_dur,
        "silence_len_list": silence_len_list,
        "silence_dur": silence_dur,
        "silence_ratio": silence_ratio,
        "overlap_len_list": overlap_len_list,
        "overlap_dur": overlap_dur,
        "overlap_ratio": overlap_ratio,
    }

    return results


def run_multispeaker_data_analysis(
    input_dir,
    session_dur=None,
    silence_mean=None,
    silence_var=None,
    overlap_mean=None,
    overlap_var=None,
    precise=False,
    save_path=None,
    num_workers=1,
) -> Dict:
    rttm_list = list(Path(input_dir).glob("*.rttm"))
    """
    Analyze the multispeaker data and plot the distribution of silence and overlap durations.

    Args:
        input_dir (str): path to the directory containing the rttm files
        session_dur (float): duration of the session (specified by argument)
        silence_mean (float): mean of the silence duration distribution
        silence_var (float): variance of the silence duration distribution
        overlap_mean (float): mean of the overlap duration distribution
        overlap_var (float): variance of the overlap duration distribution
        precise (bool): whether to measure the precise duration of the session using sox
        save_path (str): path to save the plots

    Returns:
        stats (dict): dictionary containing the statistics of the analyzed data
    """

    print(f"Found {len(rttm_list)} files to be processed")
    if len(rttm_list) == 0:
        raise ValueError(f"No rttm files found in {input_dir}")

    silence_duration = 0.0
    total_duration = 0.0
    overlap_duration = 0.0

    silence_ratio_all = []
    overlap_ratio_all = []
    silence_length_all = []
    overlap_length_all = []

    queue = []
    for rttm_file in tqdm(rttm_list):
        queue.append(
            {"rttm_file": rttm_file, "session_dur": session_dur, "precise": precise,}
        )

    if num_workers <= 1:
        results = [process_sample(sess_dict) for sess_dict in tqdm(queue)]
    else:
        with multiprocessing.Pool(processes=num_workers) as p:
            results = list(tqdm(p.imap(process_sample, queue), total=len(queue), desc='Processing', leave=True,))

    for item in results:
        total_duration += item["session_dur"]
        silence_duration += item["silence_dur"]
        overlap_duration += item["overlap_dur"]

        silence_length_all += item["silence_len_list"]
        overlap_length_all += item["overlap_len_list"]

        silence_ratio_all.append(item["silence_ratio"])
        overlap_ratio_all.append(item["overlap_ratio"])

    actual_silence_mean = silence_duration / total_duration
    actual_silence_var = np.var(silence_ratio_all)
    actual_overlap_mean = overlap_duration / (total_duration - silence_duration)
    actual_overlap_var = np.var(overlap_ratio_all)

    stats = OrderedDict()
    stats["total duration (hours)"] = f"{total_duration / 3600:.2f}"
    stats["number of sessions"] = len(rttm_list)
    stats["average session duration (seconds)"] = f"{total_duration / len(rttm_list):.2f}"
    stats["actual silence ratio mean/var"] = f"{actual_silence_mean:.4f}/{actual_silence_var:.4f}"
    stats["actual overlap ratio mean/var"] = f"{actual_overlap_mean:.4f}/{actual_overlap_var:.4f}"
    stats["expected silence ratio mean/var"] = f"{silence_mean}/{silence_var}"
    stats["expected overlap ratio mean/var"] = f"{overlap_mean}/{overlap_var}"
    stats["save_path"] = save_path

    print("-----------------------------------------------")
    print("                    Results                    ")
    print("-----------------------------------------------")
    for k, v in stats.items():
        print(k, ": ", v)
    print("-----------------------------------------------")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(
        f"Average session={total_duration/len(rttm_list):.2f} seconds, num sessions={len(rttm_list)}, total={total_duration/3600:.2f} hours"
    )
    sns.histplot(silence_ratio_all, ax=ax1)
    ax1.set_xlabel("Silence ratio in a session")
    ax1.set_title(
        f"Target silence mean={silence_mean}, var={silence_var}. \nActual silence ratio={actual_silence_mean:.4f}, var={actual_silence_var:.4f}"
    )

    _, scale = expon.fit(silence_length_all, floc=0)
    sns.histplot(silence_length_all, ax=ax2)
    ax2.set_xlabel("Per-silence length in seconds")
    ax2.set_title(f"Per-silence length histogram, \nfitted exponential distribution with mean={scale:.4f}")

    sns.histplot(overlap_ratio_all, ax=ax3)
    ax3.set_title(
        f"Target overlap mean={overlap_mean}, var={overlap_var}. \nActual ratio={actual_overlap_mean:.4f}, var={actual_overlap_var:.4f}"
    )
    ax3.set_xlabel("Overlap ratio in a session")
    _, scale2 = expon.fit(overlap_length_all, floc=0)
    sns.histplot(overlap_length_all, ax=ax4)
    ax4.set_title(f"Per overlap length histogram, \nfitted exponential distribution with mean={scale2:.4f}")
    ax4.set_xlabel("Duration in seconds")

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved at: {save_path}")

    return stats


def visualize_multispeaker_data(input_dir: str, output_dir: str, num_samples: int = 10) -> None:
    """
    Visualize a set of randomly sampled data in the input directory

    Args:
        input_dir (str): Path to the input directory
        output_dir (str): Path to the output directory
        num_samples (int): Number of samples to visualize
    """
    rttm_list = list(Path(input_dir).glob("*.rttm"))
    idx_list = np.random.permutation(len(rttm_list))[:num_samples]
    print(f"Visualizing {num_samples} random samples")
    for idx in idx_list:
        rttm_file = rttm_list[idx]
        audio_file = rttm_file.parent / Path(rttm_file.stem + ".wav")
        output_file = Path(output_dir) / Path(rttm_file.stem + ".png")
        plot_sample_from_rttm(audio_file=audio_file, rttm_file=rttm_file, save_path=str(output_file), show=False)
    print(f"Sample plots saved at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", default="", help="Input directory")
    parser.add_argument("-sd", "--session_dur", default=None, type=float, help="Duration per session in seconds")
    parser.add_argument("-sm", "--silence_mean", default=None, type=float, help="Expected silence ratio mean")
    parser.add_argument("-sv", "--silence_var", default=None, type=float, help="Expected silence ratio variance")
    parser.add_argument("-om", "--overlap_mean", default=None, type=float, help="Expected overlap ratio mean")
    parser.add_argument("-ov", "--overlap_var", default=None, type=float, help="Expected overlap ratio variance")
    parser.add_argument("-w", "--num_workers", default=1, type=int, help="Number of CPU workers to use")
    parser.add_argument("-s", "--num_samples", default=10, type=int, help="Number of random samples to plot")
    parser.add_argument("-o", "--output_dir", default="analysis/", type=str, help="Directory for saving output figure")
    parser.add_argument(
        "--precise", action="store_true", help="Set to get precise duration, with significant time cost"
    )
    args = parser.parse_args()

    print("Running with params:")
    pprint(vars(args))

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True)

    run_multispeaker_data_analysis(
        input_dir=args.input_dir,
        session_dur=args.session_dur,
        silence_mean=args.silence_mean,
        silence_var=args.silence_var,
        overlap_mean=args.overlap_mean,
        overlap_var=args.overlap_var,
        precise=args.precise,
        save_path=str(Path(args.output_dir, "statistics.png")),
        num_workers=args.num_workers,
    )

    visualize_multispeaker_data(input_dir=args.input_dir, output_dir=args.output_dir, num_samples=args.num_samples)

    print("The multispeaker data analysis has been completed.")
    print(f"Please check the output directory: \n{args.output_dir}")
