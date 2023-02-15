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

import logging
import logging.handlers
import math
import os
import sys
from pathlib import PosixPath
from typing import List, Tuple, Union

import ctc_segmentation as cs
import numpy as np
from tqdm import tqdm

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


def get_segments(
    log_probs: np.ndarray,
    path_wav: Union[PosixPath, str],
    transcript_file: Union[PosixPath, str],
    output_file: str,
    vocabulary: List[str],
    tokenizer: SentencePieceTokenizer,
    bpe_model: bool,
    index_duration: float,
    window_size: int = 8000,
    log_file: str = "log.log",
    debug: bool = False,
) -> None:
    """
    Segments the audio into segments and saves segments timings to a file

    Args:
        log_probs: Log probabilities for the original audio from an ASR model, shape T * |vocabulary|.
                   values for blank should be at position 0
        path_wav: path to the audio .wav file
        transcript_file: path to
        output_file: path to the file to save timings for segments
        vocabulary: vocabulary used to train the ASR model, note blank is at position len(vocabulary) - 1
        tokenizer: ASR model tokenizer (for BPE models, None for char-based models)
        bpe_model: Indicates whether the model uses BPE
        window_size: the length of each utterance (in terms of frames of the CTC outputs) fits into that window.
        index_duration: corresponding time duration of one CTC output index (in seconds)
    """
    level = "DEBUG" if debug else "INFO"
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(handlers=handlers, level=level)

    try:
        with open(transcript_file, "r") as f:
            text = f.readlines()
            text = [t.strip() for t in text if t.strip()]

        # add corresponding original text without pre-processing
        transcript_file_no_preprocessing = transcript_file.replace(".txt", "_with_punct.txt")
        if not os.path.exists(transcript_file_no_preprocessing):
            raise ValueError(f"{transcript_file_no_preprocessing} not found.")

        with open(transcript_file_no_preprocessing, "r") as f:
            text_no_preprocessing = f.readlines()
            text_no_preprocessing = [t.strip() for t in text_no_preprocessing if t.strip()]

        # add corresponding normalized original text
        transcript_file_normalized = transcript_file.replace(".txt", "_with_punct_normalized.txt")
        if not os.path.exists(transcript_file_normalized):
            raise ValueError(f"{transcript_file_normalized} not found.")

        with open(transcript_file_normalized, "r") as f:
            text_normalized = f.readlines()
            text_normalized = [t.strip() for t in text_normalized if t.strip()]

        if len(text_no_preprocessing) != len(text):
            raise ValueError(f"{transcript_file} and {transcript_file_no_preprocessing} do not match")

        if len(text_normalized) != len(text):
            raise ValueError(f"{transcript_file} and {transcript_file_normalized} do not match")

        config = cs.CtcSegmentationParameters()
        config.char_list = vocabulary
        config.min_window_size = window_size
        config.index_duration = index_duration

        if bpe_model:
            ground_truth_mat, utt_begin_indices = _prepare_tokenized_text_for_bpe_model(text, tokenizer, vocabulary, 0)
        else:
            config.excluded_characters = ".,-?!:»«;'›‹()"
            config.blank = vocabulary.index(" ")
            ground_truth_mat, utt_begin_indices = cs.prepare_text(config, text)

        _print(ground_truth_mat, config.char_list)

        # set this after text prepare_text()
        config.blank = 0
        logging.debug(f"Syncing {transcript_file}")
        logging.debug(
            f"Audio length {os.path.basename(path_wav)}: {log_probs.shape[0]}. "
            f"Text length {os.path.basename(transcript_file)}: {len(ground_truth_mat)}"
        )

        timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
        _print(ground_truth_mat, vocabulary)
        segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text, char_list)

        write_output(output_file, path_wav, segments, text, text_no_preprocessing, text_normalized)

        # Also writes labels in audacity format
        output_file_audacity = output_file[:-4] + "_audacity.txt"
        write_labels_for_audacity(output_file_audacity, segments, text_no_preprocessing)
        logging.info(f"Label file for Audacity written to {output_file_audacity}.")

        for i, (word, segment) in enumerate(zip(text, segments)):
            if i < 5:
                logging.debug(f"{segment[0]:.2f} {segment[1]:.2f} {segment[2]:3.4f} {word}")
        logging.info(f"segmentation of {transcript_file} complete.")

    except Exception as e:
        logging.info(f"{e} -- segmentation of {transcript_file} failed")


def _prepare_tokenized_text_for_bpe_model(text: List[str], tokenizer, vocabulary: List[str], blank_idx: int = 0):
    """ Creates a transition matrix for BPE-based models"""
    space_idx = vocabulary.index("▁")
    ground_truth_mat = [[-1, -1]]
    utt_begin_indices = []
    for uttr in text:
        ground_truth_mat += [[blank_idx, space_idx]]
        utt_begin_indices.append(len(ground_truth_mat))
        token_ids = tokenizer.text_to_ids(uttr)
        # blank token is moved from the last to the first (0) position in the vocabulary
        token_ids = [idx + 1 for idx in token_ids]
        ground_truth_mat += [[t, -1] for t in token_ids]

    utt_begin_indices.append(len(ground_truth_mat))
    ground_truth_mat += [[blank_idx, space_idx]]
    ground_truth_mat = np.array(ground_truth_mat, np.int64)
    return ground_truth_mat, utt_begin_indices


def _print(ground_truth_mat, vocabulary, limit=20):
    """Prints transition matrix"""
    chars = []
    for row in ground_truth_mat:
        chars.append([])
        for ch_id in row:
            if ch_id != -1:
                chars[-1].append(vocabulary[int(ch_id)])

    for x in chars[:limit]:
        logging.debug(x)


def _get_blank_spans(char_list, blank="ε"):
    """
    Returns a list of tuples:
        (start index, end index (exclusive), count)

    ignores blank symbols at the beginning and end of the char_list
    since they're not suitable for split in between
    """
    blanks = []
    start = None
    end = None
    for i, ch in enumerate(char_list):
        if ch == blank:
            if start is None:
                start, end = i, i
            else:
                end = i
        else:
            if start is not None:
                # ignore blank tokens at the beginning
                if start > 0:
                    end += 1
                    blanks.append((start, end, end - start))
                start = None
                end = None
    return blanks


def _compute_time(index, align_type, timings):
    """Compute start and end time of utterance.
    Adapted from https://github.com/lumaku/ctc-segmentation

    Args:
        index:  frame index value
        align_type:  one of ["begin", "end"]

    Return:
        start/end time of utterance in seconds
    """
    middle = (timings[index] + timings[index - 1]) / 2
    if align_type == "begin":
        return max(timings[index + 1] - 0.5, middle)
    elif align_type == "end":
        return min(timings[index - 1] + 0.5, middle)


def determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text, char_list):
    """Utterance-wise alignments from char-wise alignments.
    Adapted from https://github.com/lumaku/ctc-segmentation

    Args:
        config: an instance of CtcSegmentationParameters
        utt_begin_indices: list of time indices of utterance start
        char_probs:  character positioned probabilities obtained from backtracking
        timings: mapping of time indices to seconds
        text: list of utterances
    Return:
        segments, a list of: utterance start and end [s], and its confidence score
    """
    segments = []
    min_prob = np.float64(-10000000000.0)
    for i in tqdm(range(len(text))):
        start = _compute_time(utt_begin_indices[i], "begin", timings)
        end = _compute_time(utt_begin_indices[i + 1], "end", timings)

        start_t = start / config.index_duration_in_seconds
        start_t_floor = math.floor(start_t)

        # look for the left most blank symbol and split in the middle to fix start utterance segmentation
        if char_list[start_t_floor] == config.char_list[config.blank]:
            start_blank = None
            j = start_t_floor - 1
            while char_list[j] == config.char_list[config.blank] and j > start_t_floor - 20:
                start_blank = j
                j -= 1
            if start_blank:
                start_t = int(round(start_blank + (start_t_floor - start_blank) / 2))
            else:
                start_t = start_t_floor
            start = start_t * config.index_duration_in_seconds

        else:
            start_t = int(round(start_t))

        end_t = int(round(end / config.index_duration_in_seconds))

        # Compute confidence score by using the min mean probability after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t <= start_t:
            min_avg = min_prob
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = np.float64(0.0)
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments


def write_output(
    out_path: str,
    path_wav: str,
    segments: List[Tuple[float]],
    text: str,
    text_no_preprocessing: str,
    text_normalized: str,
):
    """
    Write the segmentation output to a file

    out_path: Path to output file
    path_wav: Path to the original audio file
    segments: Segments include start, end and alignment score
    text: Text used for alignment
    text_no_preprocessing: Reference txt without any pre-processing
    text_normalized: Reference text normalized
    """
    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:
        outfile.write(str(path_wav) + "\n")

        for i, segment in enumerate(segments):
            if isinstance(segment, list):
                for j, x in enumerate(segment):
                    start, end, score = x
                    outfile.write(
                        f"{start} {end} {score} | {text[i][j]} | {text_no_preprocessing[i][j]} | {text_normalized[i][j]}\n"
                    )
            else:
                start, end, score = segment
                outfile.write(
                    f"{start} {end} {score} | {text[i]} | {text_no_preprocessing[i]} | {text_normalized[i]}\n"
                )


def write_labels_for_audacity(
    out_path: str, segments: List[Tuple[float]], text_no_preprocessing: str,
):
    """
    Write the segmentation output to a file ready to be imported in Audacity with the unprocessed text as labels

    out_path: Path to output file
    segments: Segments include start, end and alignment score
    text_no_preprocessing: Reference txt without any pre-processing
    """
    # Audacity uses tab to separate each field (start end text)
    TAB_CHAR = "	"

    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:

        for i, segment in enumerate(segments):
            if isinstance(segment, list):
                for j, x in enumerate(segment):
                    start, end, _ = x
                    outfile.write(f"{start}{TAB_CHAR}{end}{TAB_CHAR}{text_no_preprocessing[i][j]} \n")
            else:
                start, end, _ = segment
                outfile.write(f"{start}{TAB_CHAR}{end}{TAB_CHAR}{text_no_preprocessing[i]} \n")
