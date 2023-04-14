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

import itertools
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.optimization_utils import linear_sum_assignment

from nemo.utils import logging

__all__ = [
    'score_labels',
    'calculate_session_cpWER',
    'calculate_session_cpWER_bruteforce',
    'concat_perm_word_error_rate',
]


def get_partial_ref_labels(pred_labels: List[str], ref_labels: List[str]) -> List[str]:
    """
    For evaluation of online diarization performance, generate partial reference labels 
    from the last prediction time.

    Args:
        pred_labels (list[str]): list of partial prediction labels
        ref_labels (list[str]): list of full reference labels 

    Returns:
        ref_labels_out (list[str]): list of partial reference labels
    """
    # If there is no reference, return empty list
    if len(ref_labels) == 0:
        return []

    # If there is no prediction, set the last prediction time to 0
    if len(pred_labels) == 0:
        last_pred_time = 0
    else:
        # The lastest prediction time in the prediction labels
        last_pred_time = max([float(labels.split()[1]) for labels in pred_labels])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        # If the current [start, end] interval extends beyond the end of hypothesis time stamps
        if start < last_pred_time:
            end_time = min(end, last_pred_time)
            label = f"{start} {end_time} {speaker}"
            ref_labels_out.append(label)
        # Other cases where the current [start, end] interval is before the last prediction time
        elif end < last_pred_time:
            ref_labels_out.append(label)
    return ref_labels_out


def get_online_DER_stats(
    DER: float,
    CER: float,
    FA: float,
    MISS: float,
    diar_eval_count: int,
    der_stat_dict: Dict[str, float],
    deci: int = 3,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    For evaluation of online diarization performance, add cumulative, average, and maximum DER/CER.

    Args:
        DER (float): Diarization Error Rate from the start to the current point 
        CER (float): Confusion Error Rate from the start to the current point 
        FA (float): False Alarm from the start to the current point
        MISS (float): Miss rate from the start to the current point
        diar_eval_count (int): Number of evaluation sessions
        der_stat_dict (dict): Dictionary containing cumulative, average, and maximum DER/CER
        deci (int): Number of decimal places to round

    Returns:
        der_dict (dict): Dictionary containing DER, CER, FA, and MISS
        der_stat_dict (dict): Dictionary containing cumulative, average, and maximum DER/CER
    """
    der_dict = {
        "DER": round(100 * DER, deci),
        "CER": round(100 * CER, deci),
        "FA": round(100 * FA, deci),
        "MISS": round(100 * MISS, deci),
    }
    der_stat_dict['cum_DER'] += DER
    der_stat_dict['cum_CER'] += CER
    der_stat_dict['avg_DER'] = round(100 * der_stat_dict['cum_DER'] / diar_eval_count, deci)
    der_stat_dict['avg_CER'] = round(100 * der_stat_dict['cum_CER'] / diar_eval_count, deci)
    der_stat_dict['max_DER'] = round(max(der_dict['DER'], der_stat_dict['max_DER']), deci)
    der_stat_dict['max_CER'] = round(max(der_dict['CER'], der_stat_dict['max_CER']), deci)
    return der_dict, der_stat_dict


def uem_timeline_from_file(uem_file, uniq_name=''):
    """
    Generate pyannote timeline segments for uem file

     <UEM> file format
     UNIQ_SPEAKER_ID CHANNEL START_TIME END_TIME
    """
    timeline = Timeline(uri=uniq_name)
    with open(uem_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            speaker_id, channel, start_time, end_time = line.split()
            timeline.add(Segment(float(start_time), float(end_time)))

    return timeline


def score_labels(
    AUDIO_RTTM_MAP, all_reference, all_hypothesis, collar=0.25, ignore_overlap=True, verbose: bool = True
) -> Optional[Tuple[DiarizationErrorRate, Dict]]:
    """
    Calculate DER, CER, FA and MISS rate from hypotheses and references. Hypothesis results are
    coming from Pyannote-formatted speaker diarization results and References are coming from
    Pyannote-formatted RTTM data.


    Args:
        AUDIO_RTTM_MAP (dict): Dictionary containing information provided from manifestpath
        all_reference (list[uniq_name,Annotation]): reference annotations for score calculation
        all_hypothesis (list[uniq_name,Annotation]): hypothesis annotations for score calculation
        verbose (bool): Warns if RTTM file is not found.

    Returns:
        metric (pyannote.DiarizationErrorRate): Pyannote Diarization Error Rate metric object. This object contains detailed scores of each audiofile.
        mapping (dict): Mapping dict containing the mapping speaker label for each audio input

    < Caveat >
    Unlike md-eval.pl, "no score" collar in pyannote.metrics is the maximum length of
    "no score" collar from left to right. Therefore, if 0.25s is applied for "no score"
    collar in md-eval.pl, 0.5s should be applied for pyannote.metrics.
    """
    metric = None
    if len(all_reference) == len(all_hypothesis):
        metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=ignore_overlap)

        mapping_dict = {}
        for (reference, hypothesis) in zip(all_reference, all_hypothesis):
            ref_key, ref_labels = reference
            _, hyp_labels = hypothesis
            uem = AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None)
            if uem is not None:
                uem = uem_timeline_from_file(uem_file=uem, uniq_name=ref_key)
            metric(ref_labels, hyp_labels, uem=uem, detailed=True)
            mapping_dict[ref_key] = metric.optimal_mapping(ref_labels, hyp_labels)

        DER = abs(metric)
        CER = metric['confusion'] / metric['total']
        FA = metric['false alarm'] / metric['total']
        MISS = metric['missed detection'] / metric['total']
        itemized_errors = (DER, CER, FA, MISS)

        logging.info(
            "Cumulative Results for collar {} sec and ignore_overlap {}: \n FA: {:.4f}\t MISS {:.4f}\t \
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                collar, ignore_overlap, FA, MISS, DER, CER
            )
        )

        return metric, mapping_dict, itemized_errors
    elif verbose:
        logging.warning(
            "Check if each ground truth RTTMs were present in the provided manifest file. Skipping calculation of Diariazation Error Rate"
        )
    return None


def evaluate_der(audio_rttm_map_dict, all_reference, all_hypothesis, diar_eval_mode='all'):
    """
    Evaluate with a selected diarization evaluation scheme

    AUDIO_RTTM_MAP (dict):
        Dictionary containing information provided from manifestpath
    all_reference (list[uniq_name,annotation]):
        reference annotations for score calculation
    all_hypothesis (list[uniq_name,annotation]):
        hypothesis annotations for score calculation
    diar_eval_mode (str):
        Diarization evaluation modes

        diar_eval_mode == "full":
            DIHARD challenge style evaluation, the most strict way of evaluating diarization
            (collar, ignore_overlap) = (0.0, False)
        diar_eval_mode == "fair":
            Evaluation setup used in VoxSRC challenge
            (collar, ignore_overlap) = (0.25, False)
        diar_eval_mode == "forgiving":
            Traditional evaluation setup
            (collar, ignore_overlap) = (0.25, True)
        diar_eval_mode == "all":
            Compute all three modes (default)
    """
    eval_settings = []
    if diar_eval_mode == "full":
        eval_settings = [(0.0, False)]
    elif diar_eval_mode == "fair":
        eval_settings = [(0.25, False)]
    elif diar_eval_mode == "forgiving":
        eval_settings = [(0.25, True)]
    elif diar_eval_mode == "all":
        eval_settings = [(0.0, False), (0.25, False), (0.25, True)]
    else:
        raise ValueError("`diar_eval_mode` variable contains an unsupported value")

    for collar, ignore_overlap in eval_settings:
        diar_score = score_labels(
            AUDIO_RTTM_MAP=audio_rttm_map_dict,
            all_reference=all_reference,
            all_hypothesis=all_hypothesis,
            collar=collar,
            ignore_overlap=ignore_overlap,
        )
    return diar_score


def calculate_session_cpWER_bruteforce(spk_hypothesis: List[str], spk_reference: List[str]) -> Tuple[float, str, str]:
    """
    Calculate cpWER with actual permutations in brute-force way when LSA algorithm cannot deliver the correct result.

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

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
    p_wer_list, permed_hyp_lists = [], []
    ref_word_list = []

    # Concatenate the hypothesis transcripts into a list
    for spk_id, word_list in enumerate(spk_reference):
        ref_word_list.append(word_list)
    ref_trans = " ".join(ref_word_list)

    # Calculate WER for every permutation
    for hyp_word_list in permutations(spk_hypothesis):
        hyp_trans = " ".join(hyp_word_list)
        permed_hyp_lists.append(hyp_trans)

        # Calculate a WER value of the permuted and concatenated transcripts
        p_wer = word_error_rate(hypotheses=[hyp_trans], references=[ref_trans])
        p_wer_list.append(p_wer)

    # Find the lowest WER and its hypothesis transcript
    argmin_idx = np.argmin(p_wer_list)
    min_perm_hyp_trans = permed_hyp_lists[argmin_idx]
    cpWER = p_wer_list[argmin_idx]
    return cpWER, min_perm_hyp_trans, ref_trans


def calculate_session_cpWER(
    spk_hypothesis: List[str], spk_reference: List[str], use_lsa_only: bool = False
) -> Tuple[float, str, str]:
    """
    Calculate a session-level concatenated minimum-permutation word error rate (cpWER) value. cpWER is
    a scoring method that can evaluate speaker diarization and speech recognition performance at the same time.
    cpWER is calculated by going through the following steps.

    1. Concatenate all utterances of each speaker for both reference and hypothesis files.
    2. Compute the WER between the reference and all possible speaker permutations of the hypothesis.
    3. Pick the lowest WER among them (this is assumed to be the best permutation: `min_perm_hyp_trans`).

    cpWER was proposed in the following article:
        CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings
        https://arxiv.org/pdf/2004.09249.pdf

    Implementation:
        - Brute force permutation method for calculating cpWER has a time complexity of `O(n!)`.
        - To reduce the computational burden, linear sum assignment (LSA) algorithm is applied
          (also known as Hungarian algorithm) to find the permutation that leads to the lowest WER.
        - In this implementation, instead of calculating all WER values for all permutation of hypotheses,
          we only calculate WER values of (estimated number of speakers) x (reference number of speakers)
          combinations with `O(n^2)`) time complexity and then select the permutation that yields the lowest
          WER based on LSA algorithm.
        - LSA algorithm has `O(n^3)` time complexity in the worst case.
        - We cannot use LSA algorithm to find the best permutation when there are more hypothesis speakers
          than reference speakers. In this case, we use the brute-force permutation method instead.

          Example:
              >>> transcript_A = ['a', 'b', 'c', 'd', 'e', 'f'] # 6 speakers
              >>> transcript_B = ['a c b d', 'e f'] # 2 speakers

              [case1] hypothesis is transcript_A, reference is transcript_B
              [case2] hypothesis is transcript_B, reference is transcript_A

              LSA algorithm based cpWER is:
                [case1] 4/6 (4 deletion)
                [case2] 2/6 (2 substitution)
              brute force permutation based cpWER is:
                [case1] 0
                [case2] 2/6 (2 substitution)

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

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
    # Get all pairs of (estimated num of spks) x (reference num of spks) combinations
    hyp_ref_pair = [spk_hypothesis, spk_reference]
    all_pairs = list(itertools.product(*hyp_ref_pair))

    num_hyp_spks, num_ref_spks = len(spk_hypothesis), len(spk_reference)

    if not use_lsa_only and num_ref_spks < num_hyp_spks:
        # Brute force algorithm when there are more speakers in the hypothesis
        cpWER, min_perm_hyp_trans, ref_trans = calculate_session_cpWER_bruteforce(spk_hypothesis, spk_reference)
    else:
        # Calculate WER for each speaker in hypothesis with reference
        # There are (number of hyp speakers) x (number of ref speakers) combinations
        lsa_wer_list = []
        for (spk_hyp_trans, spk_ref_trans) in all_pairs:
            spk_wer = word_error_rate(hypotheses=[spk_hyp_trans], references=[spk_ref_trans])
            lsa_wer_list.append(spk_wer)

        # Make a cost matrix and calculate a linear sum assignment on the cost matrix.
        # Row is hypothesis index and column is reference index
        cost_wer = torch.tensor(lsa_wer_list).reshape([len(spk_hypothesis), len(spk_reference)])
        row_hyp_ind, col_ref_ind = linear_sum_assignment(cost_wer)

        # In case where hypothesis has more speakers, add words from residual speakers
        hyp_permed = [spk_hypothesis[k] for k in np.argsort(col_ref_ind)]
        min_perm_hyp_trans = " ".join(hyp_permed)

        # Concatenate the reference transcripts into a string variable
        ref_trans = " ".join(spk_reference)

        # Calculate a WER value from the permutation that yields the lowest WER.
        cpWER = word_error_rate(hypotheses=[min_perm_hyp_trans], references=[ref_trans])

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
    for (spk_hypothesis, spk_reference) in zip(spk_hypotheses, spk_references):
        cpWER, min_hypothesis, concat_reference = calculate_session_cpWER(spk_hypothesis, spk_reference)
        cpWER_values.append(cpWER)
        hyps_spk.append(min_hypothesis)
        refs_spk.append(concat_reference)
    return cpWER_values, hyps_spk, refs_spk
