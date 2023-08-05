# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import json
import math
import multiprocessing
import os
import shutil
from itertools import repeat
from math import ceil, floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from nemo.collections.asr.models import EncDecClassificationModel, EncDecFrameClassificationModel
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


"""
This file contains all the utility functions required for voice activity detection. 
"""


def prepare_manifest(config: dict) -> str:
    """
    Perform VAD on long audio snippet might cause CUDA out of memory issue. 
    Automatically split manifest entry by split_duration to avoid the potential memory issue.
    """
    if 'prepared_manifest_vad_input' in config and config['prepared_manifest_vad_input']:
        manifest_vad_input = config['prepared_manifest_vad_input']
    else:
        default_path = "manifest_vad_input.json"
        manifest_vad_input = os.path.join(config["out_dir"], default_path) if "out_dir" in config else default_path

    # input_list is a list of variable ['audio_filepath': i, "offset": xxx, "duration": xxx])
    if type(config['input']) == str:
        input_list = []
        with open(config['input'], 'r', encoding='utf-8') as manifest:
            for line in manifest.readlines():
                input_list.append(json.loads(line.strip()))
    elif type(config['input']) == list:
        input_list = config['input']
    else:
        raise ValueError(
            "The input for manifest preparation would either be a string of the filepath to manifest or a list of {'audio_filepath': i, 'offset': 0, 'duration': null} "
        )

    args_func = {
        'label': 'infer',
        'split_duration': config['split_duration'],
        'window_length_in_sec': config['window_length_in_sec'],
        'manifest_dir': Path(config['input']).parent if type(config['input']) == str else '',
    }

    if config.get('num_workers') is not None and config['num_workers'] > 1:
        with multiprocessing.Pool(processes=config['num_workers']) as p:
            inputs = zip(input_list, repeat(args_func))
            results = list(
                tqdm(
                    p.imap(write_vad_infer_manifest_star, inputs),
                    total=len(input_list),
                    desc='splitting manifest',
                    leave=True,
                )
            )
    else:
        results = [
            write_vad_infer_manifest(input_el, args_func)
            for input_el in tqdm(input_list, desc='splitting manifest', leave=True)
        ]

    if os.path.exists(manifest_vad_input):
        logging.info("The prepared manifest file exists. Overwriting!")
        os.remove(manifest_vad_input)

    with open(manifest_vad_input, 'a', encoding='utf-8') as fout:
        for res in results:
            for r in res:
                json.dump(r, fout)
                fout.write('\n')
                fout.flush()
    return manifest_vad_input


def write_vad_infer_manifest_star(args):
    """
    A workaround for tqdm with starmap of multiprocessing
    """
    return write_vad_infer_manifest(*args)


def write_vad_infer_manifest(file: dict, args_func: dict) -> list:
    """
    Used by prepare_manifest.
    Given a list of files, split them with maximum split_duration and write them to the manifest.
    Args:
        files (dict) : file to be processed
        args_func:
            label (str): label for audio snippet.y
            split_duration (float): max duration of each audio clip (each line in json)
            window_length_in_sec (float) : length of window for generating the frame. Used for taking care of joint. 
    Returns:
        res (list) : list of generated metadata line of json for file
    """
    res = []
    label = args_func['label']
    split_duration = args_func['split_duration']
    window_length_in_sec = args_func['window_length_in_sec']
    filepath = file['audio_filepath']
    in_duration = file.get('duration', None)
    in_offset = file.get('offset', 0)

    # if filepath is not found, try to find it in the dir of manifest
    if not Path(filepath).is_file():
        new_filepath = Path(args_func['manifest_dir']) / filepath
        if new_filepath.is_file():
            filepath = new_filepath.absolute().as_posix()

    try:
        sr = 16000
        x, _sr = librosa.load(filepath, sr=sr, offset=in_offset, duration=in_duration)
        duration = librosa.get_duration(y=x, sr=sr)
        left = duration
        current_offset = in_offset

        status = 'single'
        while left > 0:
            if left <= split_duration:
                if status == 'single':
                    write_duration = left
                    current_offset = 0
                else:
                    status = 'end'
                    write_duration = left + window_length_in_sec
                    current_offset -= window_length_in_sec
                offset_inc = left
                left = 0
            else:
                if status == 'start' or status == 'next':
                    status = 'next'
                else:
                    status = 'start'

                if status == 'start':
                    write_duration = split_duration
                    offset_inc = split_duration
                else:
                    write_duration = split_duration + window_length_in_sec
                    current_offset -= window_length_in_sec
                    offset_inc = split_duration + window_length_in_sec

                left -= split_duration

            metadata = {
                'audio_filepath': filepath,
                'duration': write_duration,
                'label': label,
                'text': '_',
                'offset': current_offset,
            }
            res.append(metadata)

            current_offset += offset_inc

    except Exception as e:
        err_file = "error.log"
        with open(err_file, 'w', encoding='utf-8') as fout:
            fout.write(filepath + ":" + str(e))
    return res


def get_vad_stream_status(data: list) -> list:
    """
    Generate a list of status for each snippet in manifest. A snippet should be in single, start, next or end status. 
    Used for concatenating to full audio file.
    Args:
        data (list): list of filepath of audio snippet
    Returns:
        status (list): list of status of each snippet.
    """
    if len(data) == 1:
        return ['single']

    status = [None] * len(data)
    for i in range(len(data)):
        if i == 0:
            status[i] = 'start' if data[i] == data[i + 1] else 'single'
        elif i == len(data) - 1:
            status[i] = 'end' if data[i] == data[i - 1] else 'single'
        else:
            if data[i] != data[i - 1] and data[i] == data[i + 1]:
                status[i] = 'start'
            elif data[i] == data[i - 1] and data[i] == data[i + 1]:
                status[i] = 'next'
            elif data[i] == data[i - 1] and data[i] != data[i + 1]:
                status[i] = 'end'
            else:
                status[i] = 'single'
    return status


def load_tensor_from_file(filepath: str) -> Tuple[torch.Tensor, str]:
    """
    Load torch.Tensor and the name from file
    """
    frame = []
    with open(filepath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            frame.append(float(line))

    name = Path(filepath).stem
    return torch.tensor(frame), name


def generate_overlap_vad_seq(
    frame_pred_dir: str,
    smoothing_method: str,
    overlap: float,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    num_workers: int,
    out_dir: str = None,
) -> str:
    """
    Generate predictions with overlapping input windows/segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple windows. 
    Two common smoothing filters are supported: majority vote (median) and average (mean).
    This function uses multiprocessing to speed up. 
    Args:
        frame_pred_dir (str): Directory of frame prediction file to be processed.
        smoothing_method (str): median or mean smoothing filter.
        overlap (float): amounts of overlap of adjacent windows.
        window_length_in_sec (float): length of window for generating the frame.
        shift_length_in_sec (float): amount of shift of window for generating the frame.
        out_dir (str): directory of generated predictions.
        num_workers(float): number of process for multiprocessing
    Returns:
        overlap_out_dir(str): directory of the generated predictions.
    """

    frame_filepathlist = glob.glob(frame_pred_dir + "/*.frame")
    if out_dir:
        overlap_out_dir = out_dir
    else:
        overlap_out_dir = os.path.join(
            frame_pred_dir, "overlap_smoothing_output" + "_" + smoothing_method + "_" + str(overlap)
        )

    if not os.path.exists(overlap_out_dir):
        os.mkdir(overlap_out_dir)

    per_args = {
        "overlap": overlap,
        "window_length_in_sec": window_length_in_sec,
        "shift_length_in_sec": shift_length_in_sec,
        "out_dir": overlap_out_dir,
        "smoothing_method": smoothing_method,
    }
    if num_workers is not None and num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as p:
            inputs = zip(frame_filepathlist, repeat(per_args))
            results = list(
                tqdm(
                    p.imap(generate_overlap_vad_seq_per_file_star, inputs),
                    total=len(frame_filepathlist),
                    desc='generating preds',
                    leave=True,
                )
            )

    else:
        for frame_filepath in tqdm(frame_filepathlist, desc='generating preds', leave=False):
            generate_overlap_vad_seq_per_file(frame_filepath, per_args)

    return overlap_out_dir


def generate_overlap_vad_seq_per_file_star(args):
    """
    A workaround for tqdm with starmap of multiprocessing
    """
    return generate_overlap_vad_seq_per_file(*args)


@torch.jit.script
def generate_overlap_vad_seq_per_tensor(
    frame: torch.Tensor, per_args: Dict[str, float], smoothing_method: str
) -> torch.Tensor:
    """
    Use generated frame prediction (generated by shifting window of shift_length_in_sec (10ms)) to generate prediction with overlapping input window/segments
    See description in generate_overlap_vad_seq.
    Use this for single instance pipeline. 
    """
    # This function will be refactor for vectorization but this is okay for now

    overlap = per_args['overlap']
    window_length_in_sec = per_args['window_length_in_sec']
    shift_length_in_sec = per_args['shift_length_in_sec']
    frame_len = per_args.get('frame_len', 0.01)

    shift = int(shift_length_in_sec / frame_len)  # number of units of shift
    seg = int((window_length_in_sec / frame_len + 1))  # number of units of each window/segment

    jump_on_target = int(seg * (1 - overlap))  # jump on target generated sequence
    jump_on_frame = int(jump_on_target / shift)  # jump on input frame sequence

    if jump_on_frame < 1:
        raise ValueError(
            f"Note we jump over frame sequence to generate overlapping input segments. \n \
        Your input makes jump_on_frame={jump_on_frame} < 1 which is invalid because it cannot jump and will stuck.\n \
        Please try different window_length_in_sec, shift_length_in_sec and overlap choices. \n \
        jump_on_target = int(seg * (1 - overlap)) \n \
        jump_on_frame  = int(jump_on_frame/shift) "
        )

    target_len = int(len(frame) * shift)

    if smoothing_method == 'mean':
        preds = torch.zeros(target_len)
        pred_count = torch.zeros(target_len)

        for i, og_pred in enumerate(frame):
            if i % jump_on_frame != 0:
                continue
            start = i * shift
            end = start + seg
            preds[start:end] = preds[start:end] + og_pred
            pred_count[start:end] = pred_count[start:end] + 1

        preds = preds / pred_count
        last_non_zero_pred = preds[pred_count != 0][-1]
        preds[pred_count == 0] = last_non_zero_pred

    elif smoothing_method == 'median':
        preds = [torch.empty(0) for _ in range(target_len)]
        for i, og_pred in enumerate(frame):
            if i % jump_on_frame != 0:
                continue

            start = i * shift
            end = start + seg
            for j in range(start, end):
                if j <= target_len - 1:
                    preds[j] = torch.cat((preds[j], og_pred.unsqueeze(0)), 0)

        preds = torch.stack([torch.nanquantile(l, q=0.5) for l in preds])
        nan_idx = torch.isnan(preds)
        last_non_nan_pred = preds[~nan_idx][-1]
        preds[nan_idx] = last_non_nan_pred

    else:
        raise ValueError("smoothing_method should be either mean or median")

    return preds


def generate_overlap_vad_seq_per_file(frame_filepath: str, per_args: dict) -> str:
    """
    A wrapper for generate_overlap_vad_seq_per_tensor.
    """

    out_dir = per_args['out_dir']
    smoothing_method = per_args['smoothing_method']
    frame, name = load_tensor_from_file(frame_filepath)

    per_args_float: Dict[str, float] = {}
    for i in per_args:
        if type(per_args[i]) == float or type(per_args[i]) == int:
            per_args_float[i] = per_args[i]

    preds = generate_overlap_vad_seq_per_tensor(frame, per_args_float, smoothing_method)

    overlap_filepath = os.path.join(out_dir, name + "." + smoothing_method)
    with open(overlap_filepath, "w", encoding='utf-8') as f:
        for pred in preds:
            f.write(f"{pred:.4f}\n")

    return overlap_filepath


@torch.jit.script
def merge_overlap_segment(segments: torch.Tensor) -> torch.Tensor:
    """
    Merged the given overlapped segments.
    For example:
    torch.Tensor([[0, 1.5], [1, 3.5]]) -> torch.Tensor([0, 3.5])
    """
    if (
        segments.shape == torch.Size([0])
        or segments.shape == torch.Size([0, 2])
        or segments.shape == torch.Size([1, 2])
    ):
        return segments

    segments = segments[segments[:, 0].sort()[1]]
    merge_boundary = segments[:-1, 1] >= segments[1:, 0]
    head_padded = torch.nn.functional.pad(merge_boundary, [1, 0], mode='constant', value=0.0)
    head = segments[~head_padded, 0]
    tail_padded = torch.nn.functional.pad(merge_boundary, [0, 1], mode='constant', value=0.0)
    tail = segments[~tail_padded, 1]
    merged = torch.stack((head, tail), dim=1)
    return merged


@torch.jit.script
def filter_short_segments(segments: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Remove segments which duration is smaller than a threshold.
    For example,
    torch.Tensor([[0, 1.5], [1, 3.5], [4, 7]]) and threshold = 2.0
    -> 
    torch.Tensor([[1, 3.5], [4, 7]])
    """
    return segments[segments[:, 1] - segments[:, 0] >= threshold]


def percentile(data: torch.Tensor, perc: int) -> float:
    """
    Calculate percentile given data
    """
    size = len(data)
    return float(sorted(data)[int(math.ceil((size * perc) / 100)) - 1])


def cal_vad_onset_offset(
    scale: str, onset: float, offset: float, sequence: torch.Tensor = None
) -> Tuple[float, float]:
    """
    Calculate onset and offset threshold given different scale.
    """
    if scale == "absolute":
        mini = 0
        maxi = 1
    elif scale == "relative":
        mini = min(sequence)
        maxi = max(sequence)
    elif scale == "percentile":
        mini = percentile(sequence, 1)
        maxi = percentile(sequence, 99)

    onset = mini + onset * (maxi - mini)
    offset = mini + offset * (maxi - mini)
    return float(onset), float(offset)


@torch.jit.script
def binarization(sequence: torch.Tensor, per_args: Dict[str, float]) -> torch.Tensor:
    """
    Binarize predictions to speech and non-speech

    Reference
    Paper: Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of RNN-based Voice Activity Detection", InterSpeech 2015. 
    Implementation: https://github.com/pyannote/pyannote-audio/blob/master/pyannote/audio/utils/signal.py 

    Args:
        sequence (torch.Tensor) : A tensor of frame level predictions.
        per_args:
            onset (float): onset threshold for detecting the beginning and end of a speech 
            offset (float): offset threshold for detecting the end of a speech. 
            pad_onset (float): adding durations before each speech segment
            pad_offset (float): adding durations after each speech segment;
            frame_length_in_sec (float): length of frame.
    
    Returns:
        speech_segments(torch.Tensor): A tensor of speech segment in torch.Tensor([[start1, end1], [start2, end2]]) format. 
    """
    frame_length_in_sec = per_args.get('frame_length_in_sec', 0.01)

    onset = per_args.get('onset', 0.5)
    offset = per_args.get('offset', 0.5)
    pad_onset = per_args.get('pad_onset', 0.0)
    pad_offset = per_args.get('pad_offset', 0.0)

    speech = False
    start = 0.0
    i = 0

    speech_segments = torch.empty(0)

    for i in range(0, len(sequence)):
        # Current frame is speech
        if speech:
            # Switch from speech to non-speech
            if sequence[i] < offset:
                if i * frame_length_in_sec + pad_offset > max(0, start - pad_onset):
                    new_seg = torch.tensor(
                        [max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]
                    ).unsqueeze(0)
                    speech_segments = torch.cat((speech_segments, new_seg), 0)

                start = i * frame_length_in_sec
                speech = False

        # Current frame is non-speech
        else:
            # Switch from non-speech to speech
            if sequence[i] > onset:
                start = i * frame_length_in_sec
                speech = True

    # if it's speech at the end, add final segment
    if speech:
        new_seg = torch.tensor([max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]).unsqueeze(0)
        speech_segments = torch.cat((speech_segments, new_seg), 0)

    # Merge the overlapped speech segments due to padding
    speech_segments = merge_overlap_segment(speech_segments)  # not sorted
    return speech_segments


@torch.jit.script
def remove_segments(original_segments: torch.Tensor, to_be_removed_segments: torch.Tensor) -> torch.Tensor:
    """
    Remove speech segments list in to_be_removed_segments from original_segments.
    For example, 
    remove torch.Tensor([[start2, end2],[start4, end4]]) from torch.Tensor([[start1, end1],[start2, end2],[start3, end3], [start4, end4]]),
    -> 
    torch.Tensor([[start1, end1],[start3, end3]])
    """
    for y in to_be_removed_segments:
        original_segments = original_segments[original_segments.eq(y).all(dim=1).logical_not()]
    return original_segments


@torch.jit.script
def get_gap_segments(segments: torch.Tensor) -> torch.Tensor:
    """
    Get the gap segments. 
    For example,
    torch.Tensor([[start1, end1], [start2, end2], [start3, end3]]) -> torch.Tensor([[end1, start2], [end2, start3]])
    """
    segments = segments[segments[:, 0].sort()[1]]
    return torch.column_stack((segments[:-1, 1], segments[1:, 0]))


@torch.jit.script
def filtering(speech_segments: torch.Tensor, per_args: Dict[str, float]) -> torch.Tensor:

    """
    Filter out short non_speech and speech segments.

    Reference
    Paper: Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of RNN-based Voice Activity Detection", InterSpeech 2015. 
    Implementation: https://github.com/pyannote/pyannote-audio/blob/master/pyannote/audio/utils/signal.py 
    Args:
        speech_segments (torch.Tensor):  A tensor of speech segment in torch.Tensor([[start1, end1], [start2, end2]]) format. 
        per_args:
            min_duration_on (float): threshold for small non_speech deletion
            min_duration_off (float): threshold for short speech segment deletion
            filter_speech_first (float): Whether to perform short speech segment deletion first. Use 1.0 to represent True. 

    Returns:
        speech_segments(torch.Tensor): A tensor of filtered speech segment in torch.Tensor([[start1, end1], [start2, end2]]) format. 
    """
    if speech_segments.shape == torch.Size([0]):
        return speech_segments

    min_duration_on = per_args.get('min_duration_on', 0.0)
    min_duration_off = per_args.get('min_duration_off', 0.0)
    filter_speech_first = per_args.get('filter_speech_first', 1.0)

    if filter_speech_first == 1.0:
        # Filter out the shorter speech segments
        if min_duration_on > 0.0:
            speech_segments = filter_short_segments(speech_segments, min_duration_on)
        # Filter out the shorter non-speech segments and return to be as speech segments
        if min_duration_off > 0.0:
            # Find non-speech segments
            non_speech_segments = get_gap_segments(speech_segments)
            # Find shorter non-speech segments
            short_non_speech_segments = remove_segments(
                non_speech_segments, filter_short_segments(non_speech_segments, min_duration_off)
            )
            # Return shorter non-speech segments to be as speech segments
            speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)

            # Merge the overlapped speech segments
            speech_segments = merge_overlap_segment(speech_segments)
    else:
        if min_duration_off > 0.0:
            # Find non-speech segments
            non_speech_segments = get_gap_segments(speech_segments)
            # Find shorter non-speech segments
            short_non_speech_segments = remove_segments(
                non_speech_segments, filter_short_segments(non_speech_segments, min_duration_off)
            )

            speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)

            # Merge the overlapped speech segments
            speech_segments = merge_overlap_segment(speech_segments)
        if min_duration_on > 0.0:
            speech_segments = filter_short_segments(speech_segments, min_duration_on)

    return speech_segments


def prepare_gen_segment_table(sequence: torch.Tensor, per_args: dict) -> Tuple[str, dict]:
    """
    Preparing for generating segment table. 
    """
    out_dir = per_args.get('out_dir', None)

    # calculate onset offset based on scale selection
    per_args['onset'], per_args['offset'] = cal_vad_onset_offset(
        per_args.get('scale', 'absolute'), per_args['onset'], per_args['offset'], sequence
    )

    # cast 'filter_speech_first' for torch.jit.script
    if 'filter_speech_first' in per_args:
        if per_args['filter_speech_first']:
            per_args['filter_speech_first'] = 1.0
        else:
            per_args['filter_speech_first'] = 0.0

    per_args_float: Dict[str, float] = {}
    for i in per_args:
        if type(per_args[i]) == float or type(per_args[i]) == int:
            per_args_float[i] = per_args[i]

    return out_dir, per_args_float


@torch.jit.script
def generate_vad_segment_table_per_tensor(sequence: torch.Tensor, per_args: Dict[str, float]) -> torch.Tensor:
    """
    See description in generate_overlap_vad_seq.
    Use this for single instance pipeline. 
    """
    UNIT_FRAME_LEN = 0.01

    speech_segments = binarization(sequence, per_args)
    speech_segments = filtering(speech_segments, per_args)

    if speech_segments.shape == torch.Size([0]):
        return speech_segments

    speech_segments, _ = torch.sort(speech_segments, 0)

    dur = speech_segments[:, 1:2] - speech_segments[:, 0:1] + UNIT_FRAME_LEN
    speech_segments = torch.column_stack((speech_segments, dur))

    return speech_segments


def generate_vad_segment_table_per_file(pred_filepath: str, per_args: dict) -> str:
    """
    A wrapper for generate_vad_segment_table_per_tensor
    """
    sequence, name = load_tensor_from_file(pred_filepath)
    out_dir, per_args_float = prepare_gen_segment_table(sequence, per_args)

    preds = generate_vad_segment_table_per_tensor(sequence, per_args_float)
    ext = ".rttm" if per_args.get("use_rttm", False) else ".txt"
    save_name = name + ext
    save_path = os.path.join(out_dir, save_name)

    if preds.shape[0] == 0:
        with open(save_path, "w", encoding='utf-8') as fp:
            if per_args.get("use_rttm", False):
                fp.write(f"SPEAKER <NA> 1 0 0 <NA> <NA> speech <NA> <NA>\n")
            else:
                fp.write(f"0 0 speech\n")
    else:
        with open(save_path, "w", encoding='utf-8') as fp:
            for i in preds:
                if per_args.get("use_rttm", False):
                    fp.write(f"SPEAKER {name} 1 {i[0]:.4f} {i[2]:.4f} <NA> <NA> speech <NA> <NA>\n")
                else:
                    fp.write(f"{i[0]:.4f} {i[2]:.4f} speech\n")

    return save_path


def generate_vad_segment_table(
    vad_pred_dir: str,
    postprocessing_params: dict,
    frame_length_in_sec: float,
    num_workers: int,
    out_dir: str = None,
    use_rttm: bool = False,
) -> str:
    """
    Convert frame level prediction to speech segment in start and end times format.
    And save to csv file  in rttm-like format
            0, 10, speech
            17,18, speech
    Args:
        vad_pred_dir (str): directory of prediction files to be processed.
        postprocessing_params (dict): dictionary of thresholds for prediction score. See details in binarization and filtering.
        frame_length_in_sec (float): frame length. 
        out_dir (str): output dir of generated table/csv file.
        num_workers(float): number of process for multiprocessing
    Returns:
        out_dir(str): directory of the generated table.
    """

    suffixes = ("frame", "mean", "median")
    vad_pred_filepath_list = [os.path.join(vad_pred_dir, x) for x in os.listdir(vad_pred_dir) if x.endswith(suffixes)]

    if not out_dir:
        out_dir_name = "seg_output"
        for key in postprocessing_params:
            out_dir_name = out_dir_name + "-" + str(key) + str(postprocessing_params[key])

        out_dir = os.path.join(vad_pred_dir, out_dir_name)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    per_args = {
        "frame_length_in_sec": frame_length_in_sec,
        "out_dir": out_dir,
        "use_rttm": use_rttm,
    }
    per_args = {**per_args, **postprocessing_params}
    num_workers = None
    if num_workers is not None and num_workers > 1:
        with multiprocessing.Pool(num_workers) as p:
            inputs = zip(vad_pred_filepath_list, repeat(per_args))
            list(
                tqdm(
                    p.imap(generate_vad_segment_table_per_file_star, inputs),
                    total=len(vad_pred_filepath_list),
                    desc='creating speech segments',
                    leave=True,
                )
            )
    else:
        for vad_pred_filepath in tqdm(vad_pred_filepath_list, desc='creating speech segments', leave=True):
            generate_vad_segment_table_per_file(vad_pred_filepath, per_args)

    return out_dir


def generate_vad_segment_table_per_file_star(args):
    """
    A workaround for tqdm with starmap of multiprocessing
    """
    return generate_vad_segment_table_per_file(*args)


def vad_construct_pyannote_object_per_file(
    vad_table_filepath: str, groundtruth_RTTM_file: str
) -> Tuple[Annotation, Annotation]:
    """
    Construct a Pyannote object for evaluation.
    Args:
        vad_table_filepath(str) : path of vad rttm-like table.
        groundtruth_RTTM_file(str): path of groundtruth rttm file.
    Returns:
        reference(pyannote.Annotation): groundtruth
        hypothesis(pyannote.Annotation): prediction
    """

    pred = pd.read_csv(vad_table_filepath, sep=" ", header=None)
    label = pd.read_csv(groundtruth_RTTM_file, sep=" ", delimiter=None, header=None)
    label = label.rename(columns={3: "start", 4: "dur", 7: "speaker"})

    # construct reference
    reference = Annotation()
    for index, row in label.iterrows():
        reference[Segment(row['start'], row['start'] + row['dur'])] = row['speaker']

    # construct hypothsis
    hypothesis = Annotation()
    for index, row in pred.iterrows():
        hypothesis[Segment(float(row[0]), float(row[0]) + float(row[1]))] = 'Speech'
    return reference, hypothesis


def get_parameter_grid(params: dict) -> list:
    """
    Get the parameter grid given a dictionary of parameters.
    """
    has_filter_speech_first = False
    if 'filter_speech_first' in params:
        filter_speech_first = params['filter_speech_first']
        has_filter_speech_first = True
        params.pop("filter_speech_first")

    params_grid = list(ParameterGrid(params))

    if has_filter_speech_first:
        for i in params_grid:
            i['filter_speech_first'] = filter_speech_first
    return params_grid


def vad_tune_threshold_on_dev(
    params: dict,
    vad_pred: str,
    groundtruth_RTTM: str,
    result_file: str = "res",
    vad_pred_method: str = "frame",
    focus_metric: str = "DetER",
    frame_length_in_sec: float = 0.01,
    num_workers: int = 20,
) -> Tuple[dict, dict]:
    """
    Tune thresholds on dev set. Return best thresholds which gives the lowest detection error rate (DetER) in thresholds.
    Args:
        params (dict): dictionary of parameters to be tuned on.
        vad_pred_method (str): suffix of prediction file. Use to locate file. Should be either in "frame", "mean" or "median".
        groundtruth_RTTM_dir (str): directory of ground-truth rttm files or a file contains the paths of them.
        focus_metric (str): metrics we care most when tuning threshold. Should be either in "DetER", "FA", "MISS"
        frame_length_in_sec (float): frame length.
        num_workers (int): number of workers.
    Returns:
        best_threshold (float): threshold that gives lowest DetER.
    """
    min_score = 100
    all_perf = {}
    try:
        check_if_param_valid(params)
    except:
        raise ValueError("Please check if the parameters are valid")

    paired_filenames, groundtruth_RTTM_dict, vad_pred_dict = pred_rttm_map(vad_pred, groundtruth_RTTM, vad_pred_method)
    metric = detection.DetectionErrorRate()
    params_grid = get_parameter_grid(params)

    for param in params_grid:
        for i in param:
            if type(param[i]) == np.float64 or type(param[i]) == np.int64:
                param[i] = float(param[i])
        try:
            # Generate speech segments by performing binarization on the VAD prediction according to param.
            # Filter speech segments according to param and write the result to rttm-like table.
            vad_table_dir = generate_vad_segment_table(
                vad_pred, param, frame_length_in_sec=frame_length_in_sec, num_workers=num_workers
            )
            # add reference and hypothesis to metrics
            for filename in paired_filenames:
                groundtruth_RTTM_file = groundtruth_RTTM_dict[filename]
                vad_table_filepath = os.path.join(vad_table_dir, filename + ".txt")
                reference, hypothesis = vad_construct_pyannote_object_per_file(
                    vad_table_filepath, groundtruth_RTTM_file
                )
                metric(reference, hypothesis)  # accumulation

            # delete tmp table files
            shutil.rmtree(vad_table_dir, ignore_errors=True)

            report = metric.report(display=False)
            DetER = report.iloc[[-1]][('detection error rate', '%')].item()
            FA = report.iloc[[-1]][('false alarm', '%')].item()
            MISS = report.iloc[[-1]][('miss', '%')].item()

            assert (
                focus_metric == "DetER" or focus_metric == "FA" or focus_metric == "MISS"
            ), "Metric we care most should be only in 'DetER', 'FA' or 'MISS'!"
            all_perf[str(param)] = {'DetER (%)': DetER, 'FA (%)': FA, 'MISS (%)': MISS}
            logging.info(f"parameter {param}, {all_perf[str(param)] }")

            score = all_perf[str(param)][focus_metric + ' (%)']

            del report
            metric.reset()  # reset internal accumulator

            # save results for analysis
            with open(result_file + ".txt", "a", encoding='utf-8') as fp:
                fp.write(f"{param}, {all_perf[str(param)] }\n")

            if score < min_score:
                best_threshold = param
                optimal_scores = all_perf[str(param)]
                min_score = score
            print("Current best", best_threshold, optimal_scores)

        except RuntimeError as e:
            print(f"Pass {param}, with error {e}")
        except pd.errors.EmptyDataError as e1:
            print(f"Pass {param}, with error {e1}")

    return best_threshold, optimal_scores


def check_if_param_valid(params: dict) -> bool:
    """
    Check if the parameters are valid.
    """
    for i in params:
        if i == "filter_speech_first":
            if not type(params["filter_speech_first"]) == bool:
                raise ValueError("Invalid inputs! filter_speech_first should be either True or False!")
        elif i == "pad_onset":
            continue
        elif i == "pad_offset":
            continue
        else:
            for j in params[i]:
                if not j >= 0:
                    raise ValueError(
                        "Invalid inputs! All float parameters except pad_onset and pad_offset should be larger than 0!"
                    )

    if not (all(i <= 1 for i in params['onset']) and all(i <= 1 for i in params['offset'])):
        raise ValueError("Invalid inputs! The onset and offset thresholds should be in range [0, 1]!")

    return True


def pred_rttm_map(vad_pred: str, groundtruth_RTTM: str, vad_pred_method: str = "frame") -> Tuple[set, dict, dict]:
    """
    Find paired files in vad_pred and groundtruth_RTTM
    """
    groundtruth_RTTM_dict = {}
    if os.path.isfile(groundtruth_RTTM):
        with open(groundtruth_RTTM, "r", encoding='utf-8') as fp:
            groundtruth_RTTM_files = fp.read().splitlines()
    elif os.path.isdir(groundtruth_RTTM):
        groundtruth_RTTM_files = glob.glob(os.path.join(groundtruth_RTTM, "*.rttm"))
    else:
        raise ValueError(
            "groundtruth_RTTM should either be a directory contains rttm files or a file contains paths to them!"
        )
    for f in groundtruth_RTTM_files:
        filename = os.path.basename(f).rsplit(".", 1)[0]
        groundtruth_RTTM_dict[filename] = f

    vad_pred_dict = {}
    if os.path.isfile(vad_pred):
        with open(vad_pred, "r", encoding='utf-8') as fp:
            vad_pred_files = fp.read().splitlines()
    elif os.path.isdir(vad_pred):
        vad_pred_files = glob.glob(os.path.join(vad_pred, "*." + vad_pred_method))
    else:
        raise ValueError(
            "vad_pred should either be a directory containing vad pred files or a file contains paths to them!"
        )
    for f in vad_pred_files:
        filename = os.path.basename(f).rsplit(".", 1)[0]
        vad_pred_dict[filename] = f

    paired_filenames = groundtruth_RTTM_dict.keys() & vad_pred_dict.keys()
    return paired_filenames, groundtruth_RTTM_dict, vad_pred_dict


def plot(
    path2audio_file: str,
    path2_vad_pred: Optional[str] = None,
    path2groundtruth_rttm: Optional[str] = None,
    groundtruth_labels: Optional[str] = None,
    sample_rate: int = 16000,
    offset: float = 0,
    duration: float = None,
    threshold: float = None,
    per_args: dict = None,
    unit_frame_len: float = 0.01,
    label_repeat: int = 1,
    xticks_step: int = 5,
) -> ipd.Audio:
    """
    Plot Audio and/or VAD output and/or groundtruth labels for visualization
    Args:
        path2audio_file (str):  path to audio file.
        path2_vad_pred (str): path to vad prediction file,
        path2groundtruth_rttm(str): path to groundtruth RTTM file.
        ground_truth_labels(str): a list of groundtruth label.
        sample_rate (int): sample rate of audio file.
        offset (float): offset in seconds.
        duration (float): duration in seconds.
        threshold (float): threshold for prediction score (from 0 to 1).
        per_args(dict): a dict that stores the thresholds for postprocessing.
        unit_frame_len (float): unit frame length in seconds for VAD predictions.
        label_repeat (int): repeat the label for this number of times to match different frame lengths in preds and labels.
        xticks_step (int): step size for xticks.
    """
    plt.figure(figsize=[20, 2])

    audio, sample_rate = librosa.load(
        path=path2audio_file, sr=sample_rate, mono=True, offset=offset, duration=duration
    )
    dur = librosa.get_duration(y=audio, sr=sample_rate)

    time = np.arange(offset, offset + dur, unit_frame_len)
    len_pred = int(dur / unit_frame_len) + 1

    frame_snippet = None
    if path2_vad_pred:
        frame, _ = load_tensor_from_file(path2_vad_pred)
        frame_snippet = frame[int(offset / unit_frame_len) : int((offset + dur) / unit_frame_len)]
        len_pred = len(frame_snippet)

    ax1 = plt.subplot()
    ax1.plot(np.arange(audio.size) / sample_rate, audio, 'gray')
    ax1.set_xlim([0, int(dur) + 1])
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Signal')
    ax1.set_ylim([-1, 1])
    ax2 = ax1.twinx()

    if threshold and per_args:
        raise ValueError("threshold and per_args cannot be used at same time!")
    if not threshold and not per_args:
        raise ValueError("One and only one of threshold and per_args must have been used!")

    if threshold and frame_snippet is not None:
        pred_snippet = np.where(frame_snippet >= threshold, 1, 0)
    elif per_args and frame_snippet is not None:
        _, per_args_float = prepare_gen_segment_table(
            frame, per_args
        )  # take whole frame here for calculating onset and offset
        speech_segments = generate_vad_segment_table_per_tensor(frame, per_args_float)
        pred = gen_pred_from_speech_segments(speech_segments, frame)
        pred_snippet = pred[int(offset / unit_frame_len) : int((offset + dur) / unit_frame_len)]
    else:
        pred_snippet = None

    if path2groundtruth_rttm and path2groundtruth_rttm.endswith('.rttm'):
        label = extract_labels(path2groundtruth_rttm, time)
    elif groundtruth_labels:
        label = [float(x) for x in groundtruth_labels]
        if label_repeat > 1:
            label = np.repeat(label, label_repeat)
        label = label[int(offset / unit_frame_len) : int((offset + dur) / unit_frame_len)]
    else:
        label = None

    if label is not None:
        ax2.plot(np.arange(len_pred) * unit_frame_len, label, 'r', label='label')
    if pred_snippet is not None:
        ax2.plot(np.arange(len_pred) * unit_frame_len, pred_snippet, 'b', label='pred')
    if frame_snippet is not None:
        ax2.plot(np.arange(len_pred) * unit_frame_len, frame_snippet, 'g--', label='speech prob')

    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='lower right', shadow=True)
    ax2.set_ylabel('Preds and Probas')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_xticks(np.arange(0, int(dur) + 1, xticks_step))
    return ipd.Audio(audio, rate=sample_rate)


def gen_pred_from_speech_segments(
    speech_segments: torch.Tensor, prob: float, shift_length_in_sec: float = 0.01
) -> np.array:
    """
    Generate prediction arrays like 000111000... from speech segments {[0,1][2,4]} 
    """
    pred = np.zeros(prob.shape)
    speech_segments = [list(i) for i in speech_segments]
    speech_segments.sort(key=lambda x: x[0])

    for seg in speech_segments:
        start = int(seg[0] / shift_length_in_sec)
        end = int(seg[1] / shift_length_in_sec)
        pred[start:end] = 1
    return pred


def extract_labels(path2ground_truth_label: str, time: list) -> list:
    """
    Extract ground-truth label for given time period.
    path2ground_truth_label (str): path of groundtruth RTTM file 
    time (list) : a list of array representing time period.
    """

    data = pd.read_csv(path2ground_truth_label, sep="\s+", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})
    labels = []
    for pos in time:
        line = data[(data["start"] <= pos) & (data["start"] + data["dur"] > pos)]
        if len(line) >= 1:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def generate_vad_frame_pred(
    vad_model,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    manifest_vad_input: str,
    out_dir: str,
    use_feat: bool = False,
) -> str:
    """
    Generate VAD frame level prediction and write to out_dir
    """
    time_unit = int(window_length_in_sec / shift_length_in_sec)
    trunc = int(time_unit / 2)
    trunc_l = time_unit - trunc
    all_len = 0

    data = []
    with open(manifest_vad_input, 'r', encoding='utf-8') as f:
        for line in f:
            file = json.loads(line)['audio_filepath'].split("/")[-1]
            data.append(file.split(".wav")[0])
    logging.info(f"Inference on {len(data)} audio files/json lines!")

    status = get_vad_stream_status(data)
    for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
        test_batch = [x.to(vad_model.device) for x in test_batch]
        with autocast():
            if use_feat:
                log_probs = vad_model(processed_signal=test_batch[0], processed_signal_length=test_batch[1])
            else:
                log_probs = vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
            probs = torch.softmax(log_probs, dim=-1)
            if len(probs.shape) == 3 and probs.shape[0] == 1:
                # squeeze the batch dimension, since batch size is 1 for frame-VAD
                probs = probs.squeeze(0)  # [1,T,C] -> [T,C]
            pred = probs[:, 1]

            if window_length_in_sec == 0:
                to_save = pred
            elif status[i] == 'start':
                to_save = pred[:-trunc]
            elif status[i] == 'next':
                to_save = pred[trunc:-trunc_l]
            elif status[i] == 'end':
                to_save = pred[trunc_l:]
            else:
                to_save = pred

            to_save = to_save.cpu().tolist()
            all_len += len(to_save)
            outpath = os.path.join(out_dir, data[i] + ".frame")
            with open(outpath, "a", encoding='utf-8') as fout:
                for f in range(len(to_save)):
                    fout.write('{0:0.4f}\n'.format(to_save[f]))

        del test_batch
        if status[i] == 'end' or status[i] == 'single':
            logging.debug(f"Overall length of prediction of {data[i]} is {all_len}!")
            all_len = 0
    return out_dir


def init_vad_model(model_path: str):
    """
    Initiate VAD model with model path
    """
    if model_path.endswith('.nemo'):
        logging.info(f"Using local VAD model from {model_path}")
        vad_model = EncDecClassificationModel.restore_from(restore_path=model_path)
    elif model_path.endswith('.ckpt'):
        vad_model = EncDecClassificationModel.load_from_checkpoint(checkpoint_path=model_path)
    else:
        logging.info(f"Using NGC cloud VAD model {model_path}")
        vad_model = EncDecClassificationModel.from_pretrained(model_name=model_path)
    return vad_model


def init_frame_vad_model(model_path: str):
    """
    Initiate VAD model with model path
    """
    if model_path.endswith('.nemo'):
        logging.info(f"Using local VAD model from {model_path}")
        vad_model = EncDecFrameClassificationModel.restore_from(restore_path=model_path)
    elif model_path.endswith('.ckpt'):
        vad_model = EncDecFrameClassificationModel.load_from_checkpoint(checkpoint_path=model_path)
    else:
        logging.info(f"Using NGC cloud VAD model {model_path}")
        vad_model = EncDecFrameClassificationModel.from_pretrained(model_name=model_path)
    return vad_model


def stitch_segmented_asr_output(
    segmented_output_manifest: str,
    speech_segments_tensor_dir: str = "speech_segments",
    stitched_output_manifest: str = "asr_stitched_output_manifest.json",
) -> str:
    """
    Stitch the prediction of speech segments.
    """
    if not os.path.exists(speech_segments_tensor_dir):
        os.mkdir(speech_segments_tensor_dir)

    segmented_output = []
    with open(segmented_output_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            file = json.loads(line)
            segmented_output.append(file)

    with open(stitched_output_manifest, 'w', encoding='utf-8') as fout:
        speech_segments = torch.Tensor()
        all_pred_text = ""
        if len(segmented_output) > 1:
            for i in range(1, len(segmented_output)):
                start, end = (
                    segmented_output[i - 1]['offset'],
                    segmented_output[i - 1]['offset'] + segmented_output[i - 1]['duration'],
                )
                new_seg = torch.tensor([start, end]).unsqueeze(0)
                speech_segments = torch.cat((speech_segments, new_seg), 0)
                pred_text = segmented_output[i - 1]['pred_text']
                all_pred_text += pred_text
                name = segmented_output[i - 1]['audio_filepath'].split("/")[-1].rsplit(".", 1)[0]

                if segmented_output[i - 1]['audio_filepath'] != segmented_output[i]['audio_filepath']:

                    speech_segments_tensor_path = os.path.join(speech_segments_tensor_dir, name + '.pt')
                    torch.save(speech_segments, speech_segments_tensor_path)
                    meta = {
                        'audio_filepath': segmented_output[i - 1]['audio_filepath'],
                        'speech_segments_filepath': speech_segments_tensor_path,
                        'pred_text': all_pred_text,
                    }

                    json.dump(meta, fout)
                    fout.write('\n')
                    fout.flush()
                    speech_segments = torch.Tensor()
                    all_pred_text = ""
                else:
                    all_pred_text += " "
        else:
            i = -1

        start, end = segmented_output[i]['offset'], segmented_output[i]['offset'] + segmented_output[i]['duration']
        new_seg = torch.tensor([start, end]).unsqueeze(0)
        speech_segments = torch.cat((speech_segments, new_seg), 0)
        pred_text = segmented_output[i]['pred_text']
        all_pred_text += pred_text
        name = segmented_output[i]['audio_filepath'].split("/")[-1].rsplit(".", 1)[0]
        speech_segments_tensor_path = os.path.join(speech_segments_tensor_dir, name + '.pt')
        torch.save(speech_segments, speech_segments_tensor_path)

        meta = {
            'audio_filepath': segmented_output[i]['audio_filepath'],
            'speech_segments_filepath': speech_segments_tensor_path,
            'pred_text': all_pred_text,
        }
        json.dump(meta, fout)
        fout.write('\n')
        fout.flush()

        logging.info(
            f"Finish stitch segmented ASR output to {stitched_output_manifest}, the speech segments info has been stored in directory {speech_segments_tensor_dir}"
        )
        return stitched_output_manifest


def construct_manifest_eval(
    input_manifest: str, stitched_output_manifest: str, aligned_vad_asr_output_manifest: str = "vad_asr_out.json"
) -> str:

    """
    Generate aligned manifest for evaluation.
    Because some pure noise samples might not appear in stitched_output_manifest.
    """
    stitched_output = dict()
    with open(stitched_output_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            file = json.loads(line)
            stitched_output[file["audio_filepath"]] = file

    out = []
    with open(input_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            file = json.loads(line)
            sample = file["audio_filepath"]
            if sample in stitched_output:
                file["pred_text"] = stitched_output[sample]["pred_text"]
                file["speech_segments_filepath"] = stitched_output[sample]["speech_segments_filepath"]
            else:
                file["pred_text"] = ""
                file["speech_segments_filepath"] = ""

            out.append(file)

    with open(aligned_vad_asr_output_manifest, 'w', encoding='utf-8') as fout:
        for i in out:
            json.dump(i, fout)
            fout.write('\n')
            fout.flush()

    return aligned_vad_asr_output_manifest


def load_rttm_file(filepath: str) -> pd.DataFrame:
    """
    Load rttm file and extract speech segments
    """
    if not Path(filepath).exists():
        raise ValueError(f"File not found: {filepath}")
    data = pd.read_csv(filepath, sep="\s+", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})

    data['start'] = data['start'].astype(float)
    data['dur'] = data['dur'].astype(float)
    data['end'] = data['start'] + data['dur']

    data = data.sort_values(by=['start'])
    data['segment'] = list(zip(data['start'], data['end']))

    return data


def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
    """
    Merge speech segments into non-overlapping segments
    """
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def load_speech_segments_from_rttm(rttm_file: str) -> List[List[float]]:
    """
    load speech segments from rttm file, where each segment is represented
    as [start, end] interval
    """
    speech_segments = list(load_rttm_file(rttm_file)['segment'])
    speech_segments = [list(x) for x in speech_segments]
    speech_segments = merge_intervals(speech_segments)
    return speech_segments


def load_speech_overlap_segments_from_rttm(rttm_file: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Load speech segments from RTTM file, merge and extract possible overlaps

    Args:
        rttm_file (str): Path to RTTM file

    Returns:
        merged (List[List[float]]): merged speech intervals without overlaps
        overlaps (List[List[float]]): intervals with overlap speech
    """
    speech_segments = list(load_rttm_file(rttm_file)['segment'])
    speech_segments = [list(x) for x in speech_segments]
    speech_segments.sort(key=lambda x: x[0])  # sort by start time
    merged = []
    overlaps = []
    for interval in speech_segments:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            overlaps.append([interval[0], min(merged[-1][1], interval[1])])
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged, overlaps


def get_nonspeech_segments(
    speech_segments: List[List[float]], max_duration: Optional[float] = None
) -> List[List[float]]:
    """
    Get non-speech segments from given speech segments and maximum duration

    Args:
        speech_segments (List[List[float]]): speech segment intervals loaded by load_speech_segments()
        max_duration (Optional[float]): maximum duration of the audio, used to calculate the last silence segment
    
    Returns:
        nonspeech_segments (List[List[float]]): intervals of non-speech segments
    """
    nonspeech_segments = []
    start = 0.0
    for sp_seg in speech_segments:
        end = sp_seg[0]
        nonspeech_segments.append([start, end])
        start = sp_seg[1]

    if max_duration is not None and start < max_duration:
        nonspeech_segments.append([start, max_duration])

    return nonspeech_segments


def get_frame_labels(
    segments: List[List[float]], frame_length: float, offset: float, duration: float, as_str: bool = True
) -> str:
    """
    Generate frame-level binary labels for audio, '0' for non-speech and '1' for speech

    Args:
        segments (List[List[float]]): speech segments loaded by load_speech_segments_from_rttm
        frame_length (float): frame length in seconds, e.g. 0.01 for 10ms frames
        offset (float): Offset of the audio clip
        duration (float): duration of the audio clip
    """
    labels = []
    n_frames = int(np.ceil(duration / frame_length))
    sid = 0
    for i in range(n_frames):
        t = offset + i * frame_length
        while sid < len(segments) - 1 and segments[sid][1] < t:
            sid += 1
        if segments[sid][1] != 0 and segments[sid][0] <= t <= segments[sid][1]:
            labels.append(1)
        else:
            labels.append(0)
    if as_str:
        return ' '.join([str(x) for x in labels])
    return [float(x) for x in labels]


def plot_sample_from_rttm(
    audio_file: str,
    rttm_file: str,
    max_duration: Optional[float] = None,
    save_path: str = "",
    show: bool = True,
    offset: float = 0.0,
    unit_frame_len: float = 0.01,
):
    """
    Plot audio signal and frame-level labels from RTTM file
    """
    plt.figure(figsize=[20, 2])

    audio, sample_rate = librosa.load(path=audio_file, sr=16000, mono=True, offset=offset, duration=max_duration)
    dur = librosa.get_duration(y=audio, sr=sample_rate)

    segments = load_speech_segments_from_rttm(rttm_file)
    labels = get_frame_labels(segments, unit_frame_len, offset, dur)
    labels = [float(x) for x in labels.split()]

    length = len(labels)
    ax1 = plt.subplot()
    ax1.set_title(audio_file)
    ax1.plot(np.arange(audio.size) / sample_rate, audio, 'gray')
    ax1.set_xlim([0, int(dur) + 1])
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Signal')
    ax1.set_ylim([-1, 1])
    ax2 = ax1.twinx()

    ax2.plot(np.arange(length) * unit_frame_len, labels, 'r', label='label')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='lower right', shadow=True)
    ax2.set_ylabel('Labels')
    ax2.set_ylim([-0.1, 1.1])
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    return ipd.Audio(audio, rate=16000)


def align_labels_to_frames(probs, labels, threshold=0.2):
    """
    Aligns labels to frames when the frame length (e.g., 10ms) is different from the label length (e.g., 20ms). 
    The threshold 0.2 is not important, since the actual ratio will always be close to an integer unless using frame/label 
    lengths that are not multiples of each other (e.g., 15ms frame length and 20ms label length), which is not valid.
    The value 0.2 here is just for easier unit testing.
    Args:
        probs (List[float]): list of probabilities
        labels (List[int]): list of labels
        threshold (float): threshold for rounding ratio to integer
    Returns:
        labels (List[int]): list of labels aligned to frames
    """
    frames_len = len(probs)
    labels_len = len(labels)
    probs = torch.tensor(probs).float()
    labels = torch.tensor(labels).long()

    if frames_len < labels_len:
        # pad labels with zeros until labels_len is a multiple of frames_len
        ratio = labels_len / frames_len
        res = labels_len % frames_len
        if (
            ceil(ratio) - ratio < threshold
        ):  # e.g., ratio = 2.9, ceil(ratio) = 3, then we pad labels to make it a multiple of 3
            # pad labels with zeros until labels_max_len is a multiple of logits_max_len
            labels = labels.tolist()
            if len(labels) % ceil(ratio) != 0:
                labels += [0] * (ceil(ratio) - len(labels) % ceil(ratio))
            labels = torch.tensor(labels).long()
            labels = labels.view(-1, ceil(ratio)).amax(1)
            return align_labels_to_frames(probs.tolist(), labels.long().tolist())
        # otherwise, truncate additional labels until labels_max_len is a multiple of logits_max_len
        if res > 0:
            labels = labels[:-res]
        labels = labels.view(-1, floor(ratio)).amax(1)
        return labels.long().tolist()
    elif frames_len > labels_len:
        # repeat labels until labels_len is a multiple of frames_len
        ratio = frames_len / labels_len
        res = frames_len % labels_len
        if ceil(ratio) - ratio < threshold:
            # e.g., ratio is 1.83, ceil(ratio) = 2, then we repeat labels to make it a multiple of 2, and discard the redundant labels
            labels = labels.repeat_interleave(ceil(ratio), dim=0).long().tolist()
            labels = labels[:frames_len]
        else:
            # e.g., ratio is 2.02, floor(ratio) = 2, then we repeat labels to make it a multiple of 2 and add additional labels
            labels = labels.repeat_interleave(floor(ratio), dim=0).long().tolist()
            if res > 0:
                labels += labels[-res:]
        return labels
    else:
        return labels.long().tolist()


def read_rttm_as_pyannote_object(rttm_file: str, speaker_override: Optional[str] = None) -> Annotation:
    """
    Read rttm file and construct a Pyannote object.
    Args:
        rttm_file(str) : path of rttm file.
        speaker_override(str) : if not None, all speakers will be replaced by this value.
    Returns:
        annotation(pyannote.Annotation): annotation object
    """
    annotation = Annotation()
    data = pd.read_csv(rttm_file, sep="\s+", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})
    for index, row in data.iterrows():
        if speaker_override is not None:
            annotation[Segment(row['start'], row['start'] + row['dur'])] = speaker_override
        else:
            annotation[Segment(row['start'], row['start'] + row['dur'])] = row['speaker']
    return annotation


def convert_labels_to_speech_segments(labels: List[float], frame_length_in_sec: float = 0.01):
    """
    Convert a list of labels to a list of speech segments.
    Args:
        labels (List[float]): list of labels
        frame_length_in_sec (float): frame length in seconds
    Returns:
        segments (List[Tuple[float, float]]): list of speech segments
    """
    segments = []
    start = -1
    for i, label in enumerate(labels):
        if label == 1:
            if start == -1:
                start = i * frame_length_in_sec
        else:
            if start > -1:
                segments.append([start, (i - 1) * frame_length_in_sec])
                start = -1
    if start != -1:
        segments.append([start, (len(labels) - 1) * frame_length_in_sec])
    return segments


def frame_vad_construct_pyannote_object_per_file(
    prediction: Union[str, List[float]], groundtruth: Union[str, List[float]], frame_length_in_sec: float = 0.01
) -> Tuple[Annotation, Annotation]:
    """
    Construct a Pyannote object for evaluation.
    Args:
        prediction (str) : path of VAD predictions stored as RTTM or CSV-like txt.
        groundtruth (str): path of groundtruth rttm file.
        frame_length_in_sec(float): frame length in seconds
    Returns:
        reference(pyannote.Annotation): groundtruth
        hypothesis(pyannote.Annotation): prediction
    """

    hypothesis = Annotation()
    if isinstance(groundtruth, str) and prediction.endswith('.rttm'):
        hypothesis = read_rttm_as_pyannote_object(prediction, speaker_override='speech')
    elif isinstance(groundtruth, str) and prediction.endswith('.txt'):
        pred = pd.read_csv(prediction, sep=" ", header=None)
        for index, row in pred.iterrows():
            hypothesis[Segment(float(row[0]), float(row[0]) + float(row[1]))] = 'speech'
    elif isinstance(groundtruth, list):
        segments = convert_labels_to_speech_segments(prediction, frame_length_in_sec)
        for segment in segments:
            hypothesis[Segment(segment[0], segment[1])] = 'speech'
    else:
        raise ValueError('prediction must be a path to rttm file or a list of frame labels.')

    reference = Annotation()
    if isinstance(groundtruth, str) and groundtruth.endswith('.rttm'):
        reference = read_rttm_as_pyannote_object(groundtruth, speaker_override='speech')
    elif isinstance(groundtruth, list):
        segments = convert_labels_to_speech_segments(groundtruth, frame_length_in_sec)
        for segment in segments:
            reference[Segment(segment[0], segment[1])] = 'speech'
    else:
        raise ValueError('groundtruth must be a path to rttm file or a list of frame labels.')
    return reference, hypothesis


def frame_vad_infer_load_manifest(cfg: DictConfig):
    """
    Load manifest file and prepare label/rttm mapping
    Args:
        cfg: config file
    Returns:
        manifest_orig (List[Dict]): original manifest data 
        key_labels_map (Dict): mapping from unique_audio_name to its labels
        key_rttm_map (Dict): mapping from unique_audio_name to its rttm file
    """
    unique_audio_names = set()
    key_labels_map = {}
    key_rttm_map = {}
    manifest_orig = []
    manifest_file = Path(cfg.dataset).absolute().as_posix()
    with open(manifest_file, 'r') as fin:
        for line in fin.readlines():
            entry = json.loads(line.strip())
            audio_filepath = get_full_path(audio_file=entry['audio_filepath'], manifest_file=manifest_file)
            entry['audio_filepath'] = str(audio_filepath)
            uniq_audio_name = Path(audio_filepath).stem

            if uniq_audio_name in unique_audio_names:
                raise ValueError("Please make sure each line is with different audio_filepath! ")
            else:
                unique_audio_names.add(uniq_audio_name)

            manifest_orig.append(entry)

            # always prefer RTTM labels if exist
            if "label" not in entry and ("rttm_filepath" in entry or "rttm_file" in entry):
                rttm_key = "rttm_filepath" if "rttm_filepath" in entry else "rttm_file"
                segments = load_speech_segments_from_rttm(entry[rttm_key])
                label_str = get_frame_labels(
                    segments=segments,
                    frame_length=cfg.vad.parameters.shift_length_in_sec,
                    duration=entry['duration'],
                    offset=entry['offset'],
                )
                key_rttm_map[uniq_audio_name] = entry[rttm_key]
                key_labels_map[uniq_audio_name] = [float(x) for x in label_str.split()]
            elif entry.get("label", None) is not None:
                key_labels_map[uniq_audio_name] = [float(x) for x in entry["label"].split()]
            elif cfg.evaluate:
                raise ValueError("Must have either `label` or `rttm_filepath` in manifest when evaluate=True")

    return manifest_orig, key_labels_map, key_rttm_map


def frame_vad_eval_detection_error(
    pred_dir: str, key_labels_map: dict, key_rttm_map: dict, key_pred_rttm_map: dict, frame_length_in_sec: float
):
    """
    Perform evaluation on frame-VAD results
    Args:
        pred_dir: directory of frame-VAD prediction files with in `<unique_audio_name>.frame` format
        key_labels_map: dictionary of mapping each <unique_audio_name> to its labels
        key_rttm_map: dictionary of mapping each <unique_audio_name> to its GROUNDTRUTH rttm file
        key_pred_rttm_map: dictionary of mapping each <unique_audio_name> to its PREDICTED rttm file
        frame_length_in_sec: frame length in seconds, e.g. 0.02s
    Returns:
        auroc: AUROC score in 0~100%
        report: Pyannote detection.DetectionErrorRate() report
    """
    all_probs = []
    all_labels = []
    metric = detection.DetectionErrorRate()
    key_probs_map = {}
    predictions_list = list(Path(pred_dir).glob("*.frame"))
    for frame_pred in tqdm(predictions_list, desc="Evaluating VAD results", total=len(predictions_list)):
        pred_probs = []
        with frame_pred.open("r") as fin:
            for line in fin.readlines():
                line = line.strip()
                if not line:
                    continue
                pred_probs.append(float(line))
        key = frame_pred.stem
        key_probs_map[key] = pred_probs
        key_labels_map[key] = align_labels_to_frames(probs=pred_probs, labels=key_labels_map[key])
        all_probs.extend(key_probs_map[key])
        all_labels.extend(key_labels_map[key])

        if key in key_rttm_map:
            groundtruth = key_rttm_map[key]
        else:
            groundtruth = key_labels_map[key]

        reference, hypothesis = frame_vad_construct_pyannote_object_per_file(
            prediction=key_pred_rttm_map[key], groundtruth=groundtruth, frame_length_in_sec=frame_length_in_sec,
        )
        metric(reference, hypothesis)

    auroc = roc_auc_score(y_true=all_labels, y_score=all_probs)
    report = metric.report(display=False)
    return auroc, report
