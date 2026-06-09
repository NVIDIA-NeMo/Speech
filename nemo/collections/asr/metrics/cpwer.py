# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from itertools import permutations
from typing import List, Tuple

import numpy as np
from kaldialign import edit_distance
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

__all__ = [
    'calculate_session_cpWER',
    'calculate_session_cpWER_bruteforce',
    'concat_perm_word_error_rate',
]


def calculate_session_cpWER_bruteforce(spk_hypothesis: List[str], spk_reference: List[str]) -> Tuple[float, str, str]:
    """
    Calculate cpWER with brute-force permutation search. Matches MeetEval's cpWER algorithm:
    each (ref_speaker, hyp_speaker) pair is scored independently via edit distance, then
    cpWER = sum(errors) / sum(ref_word_counts).

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    num_hyp = len(spk_hypothesis)
    num_ref = len(spk_reference)
    num_speakers_padded = max(num_hyp, num_ref)

    ref_word_lists = [
        spk_reference[ref_idx].split() if ref_idx < num_ref else [] for ref_idx in range(num_speakers_padded)
    ]
    hyp_word_lists = [
        spk_hypothesis[hyp_idx].split() if hyp_idx < num_hyp else [] for hyp_idx in range(num_speakers_padded)
    ]

    best_total_errors = float('inf')
    best_hyp_trans = ""
    total_ref_length = sum(len(word_list) for word_list in ref_word_lists)

    for perm in permutations(range(num_speakers_padded)):
        total_errors = 0
        hyp_texts = []
        for ref_idx, hyp_idx in enumerate(perm):
            total_errors += edit_distance(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])['total']
            hyp_texts.append(spk_hypothesis[hyp_idx] if hyp_idx < num_hyp else "")
        if total_errors < best_total_errors:
            best_total_errors = total_errors
            best_hyp_trans = " ".join(hyp_texts)

    cpWER = best_total_errors / total_ref_length if total_ref_length > 0 else float('inf')
    ref_trans = " ".join(spk_reference)
    return cpWER, best_hyp_trans, ref_trans


def calculate_session_cpWER(spk_hypothesis: List[str], spk_reference: List[str]) -> Tuple[float, str, str]:
    """
    Calculate a session-level concatenated minimum-permutation word error rate (cpWER) value,
    matching MeetEval's cpWER algorithm (https://github.com/fgnt/meeteval).

    Algorithm (identical to MeetEval):
        1. Build a square cost matrix of size max(num_hyp, num_ref) using raw edit distance
           counts between every (ref_speaker, hyp_speaker) pair. Missing speakers are padded
           with empty word lists.
        2. Use the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find the
           speaker assignment that minimizes total edit distance.
        3. Compute per-pair edit distance independently for the optimal assignment.
        4. cpWER = sum(errors_per_pair) / sum(ref_word_counts_per_pair).

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    num_hyp = len(spk_hypothesis)
    num_ref = len(spk_reference)

    if num_hyp == 0 and num_ref == 0:
        return 0.0, "", ""

    num_speakers_padded = max(num_hyp, num_ref)

    ref_word_lists = [
        spk_reference[ref_idx].split() if ref_idx < num_ref else [] for ref_idx in range(num_speakers_padded)
    ]
    hyp_word_lists = [
        spk_hypothesis[hyp_idx].split() if hyp_idx < num_hyp else [] for hyp_idx in range(num_speakers_padded)
    ]

    cost_matrix = np.zeros((num_speakers_padded, num_speakers_padded), dtype=np.float64)
    for ref_idx in range(num_speakers_padded):
        for hyp_idx in range(num_speakers_padded):
            cost_matrix[ref_idx, hyp_idx] = edit_distance(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])['total']

    row_ind, col_ind = scipy_linear_sum_assignment(cost_matrix)

    total_errors = 0
    total_ref_length = 0
    hyp_texts = []
    for ref_idx, hyp_idx in zip(row_ind, col_ind):
        total_errors += int(cost_matrix[ref_idx, hyp_idx])
        total_ref_length += len(ref_word_lists[ref_idx])
        hyp_texts.append(spk_hypothesis[hyp_idx] if hyp_idx < num_hyp else "")

    cpWER = total_errors / total_ref_length if total_ref_length > 0 else float('inf')

    min_perm_hyp_trans = " ".join(hyp_texts)
    ref_trans = " ".join(spk_reference)

    return cpWER, min_perm_hyp_trans, ref_trans


def concat_perm_word_error_rate(
    spk_hypotheses: List[List[str]], spk_references: List[List[str]]
) -> Tuple[List[float], List[str], List[str]]:
    """
    Launcher function for `calculate_session_cpWER`. Calculate session-level cpWER and average cpWER.
    For detailed information about cpWER, see docstrings of `calculate_session_cpWER` function.

    As opposed to `cpWER`, `WER` is the regular WER value where the hypothesis transcript contains
    words in temporal order regardless of the speakers. `WER` value can be different from cpWER value,
    depending on the speaker diarization results.

    Args:
        spk_hypotheses (list):
            List containing the lists of speaker-separated hypothesis transcripts.
        spk_references (list):
            List containing the lists of speaker-separated reference transcripts.

    Returns:
        cpWER (float):
            List containing cpWER values for each session
        min_perm_hyp_trans (list):
            List containing transcripts that lead to the minimum WER in string format
        ref_trans (list):
            List containing concatenated reference transcripts
    """
    if len(spk_hypotheses) != len(spk_references):
        raise ValueError(
            "In concatenated-minimum permutation word error rate calculation, "
            "hypotheses and reference lists must have the same number of elements. But got arguments:"
            f"{len(spk_hypotheses)} and {len(spk_references)} correspondingly"
        )
    cpWER_values, hyps_spk, refs_spk = [], [], []
    for spk_hypothesis, spk_reference in zip(spk_hypotheses, spk_references):
        cpWER, min_hypothesis, concat_reference = calculate_session_cpWER(spk_hypothesis, spk_reference)
        cpWER_values.append(cpWER)
        hyps_spk.append(min_hypothesis)
        refs_spk.append(concat_reference)
    return cpWER_values, hyps_spk, refs_spk
