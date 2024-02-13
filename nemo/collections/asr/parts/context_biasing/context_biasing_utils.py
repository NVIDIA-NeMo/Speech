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

import os
from typing import List, Union

import numpy as np
import texterrors

from nemo.collections.asr.parts.context_biasing.ctc_based_word_spotter import WSHyp
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.utils import logging


def merge_alignment_with_ws_hyps(
    candidate: Union[np.ndarray, rnnt_utils.Hypothesis],
    asr_model,
    cb_results: List[WSHyp],
    decoder_type: str = "ctc",
    intersection_threshold: float = 30.0,
    blank_idx: int = 0,
    print_stats: bool = False,
    bow: str = "▁",
) -> tuple[str, str]:
    """
    Merge context biasing predictions with ctc/rnnt word-level alignment.
    Words from alignment will be replaced by spotted words if intersection between them is greater than threshold.

    Args:
        candidate: argmax predictions per frame (for ctc) or rnnt hypothesis (for rnnt)
        asr_model: ctc or hybrid transducer-ctc model
        cb_results: list of context biasing predictions (spotted words)
        decoder_type: ctc or rnnt
        intersection_threshold: threshold for intersection between spotted word and word from alignment (in percentage)
        blank_idx: blank index for ctc/rnnt decoding
        print_stats: if True, print word alignment and spotted words
        bow: symbol for begin of word (bow) in BPE tokenizer
    Returns:
        boosted_text: final text with context biasing predictions
    """

    # step 1: get token-level alignment [frame, token]
    if decoder_type == "ctc":
        alignment_tokens = []
        prev_token = None
        for idx, token in enumerate(candidate):
            if token != blank_idx:
                if token == prev_token:
                    alignment_tokens[-1] = [idx, asr_model.tokenizer.ids_to_tokens([int(token)])[0]]
                else:
                    alignment_tokens.append([idx, asr_model.tokenizer.ids_to_tokens([int(token)])[0]])
            prev_token = token

    elif decoder_type == "rnnt":
        alignment_tokens = []
        if not isinstance(candidate.y_sequence, list):
            candidate.y_sequence = candidate.y_sequence.tolist()
        tokens = asr_model.tokenizer.ids_to_tokens(candidate.y_sequence)
        for idx, token in enumerate(tokens):
            # bow symbol may be predicted separately from token
            if token == bow:
                if idx + 1 < len(tokens) and not tokens[idx + 1].startswith(bow):
                    tokens[idx + 1] = bow + tokens[idx + 1]
                    continue
            alignment_tokens.append([candidate.timestep[idx].item(), token])
    else:
        raise ValueError(f"decoder_type {decoder_type} is not supported")

    if not alignment_tokens:
        # ctc/rnnt decoding results are empty, return context biasing results only
        return " ".join([ws_hyp.word for ws_hyp in cb_results]), ""

    # step 2: get word-level alignment [word, start_frame, end_frame]
    word_alignment = []
    word = ""
    l, r, = None, None
    for item in alignment_tokens:
        if not word:
            word = item[1][1:]
            l = r = item[0]
        else:
            if item[1].startswith(bow):
                word_alignment.append((word, l, r))
                word = item[1][1:]
                l = r = item[0]
            else:
                word += item[1]
                r = item[0]
    word_alignment.append((word, l, r))
    initial_text_transcript = " ".join([item[0] for item in word_alignment])
    if print_stats:
        logging.info(f"Word alignment: {word_alignment}")

    # step 3: merge spotted words with word alignment
    for ws_hyp in cb_results:
        # extend ws_hyp start frame in case of rnnt (rnnt tends to predict labels one frame earlier sometimes)
        if ws_hyp.start_frame > 0 and decoder_type == "rnnt":
            ws_hyp.start_frame -= 1
        new_word_alignment = []
        already_inserted = False
        # get interval of spotted word
        ws_interval = set(range(ws_hyp.start_frame, ws_hyp.end_frame + 1))
        for item in word_alignment:
            # get interval if word from alignment
            li, ri = item[1], item[2]
            item_interval = set(range(li, ri + 1))
            if ws_hyp.start_frame < li:
                # spotted word starts before first word from alignment
                if not already_inserted:
                    new_word_alignment.append((ws_hyp.word, ws_hyp.start_frame, ws_hyp.end_frame))
                    already_inserted = True
            # compute intersection between spotted word and word from alignment in percentage
            intersection_part = 100 / len(item_interval) * len(ws_interval & item_interval)
            if intersection_part <= intersection_threshold:
                new_word_alignment.append(item)
            elif not already_inserted:
                # word from alignment will be replaced by spotted word
                new_word_alignment.append((ws_hyp.word, ws_hyp.start_frame, ws_hyp.end_frame))
                already_inserted = True
        # insert last spotted word if not yet
        if not already_inserted:
            new_word_alignment.append((ws_hyp.word, ws_hyp.start_frame, ws_hyp.end_frame))
        word_alignment = new_word_alignment
        if print_stats:
            logging.info(f"Spotted word: {ws_hyp.word} [{ws_hyp.start_frame}, {ws_hyp.end_frame}]")

    boosted_text_list = [item[0] for item in new_word_alignment]
    boosted_text = " ".join(boosted_text_list)

    return boosted_text, initial_text_transcript


def compute_fscore(
    recognition_results_manifest: str, key_words_list: List, eps: str = "<eps>"
) -> tuple[float, float, float]:
    """
    Compute fscore for list of context biasing words/phrases.
    The idea is to get a word-level alignment for ground truth text and prediction results from manifest file.
    Then compute f-score for each word/phrase from key_words_list according to obtained word alignment.

    Args:
        recognition_results_manifest: path to nemo manifest file with recognition results in pred_text field.
        key_words_list: list of context biasing words/phrases.
        return_scores: if True, return precision, recall and fscore (not only print).
        eps: epsilon symbol for alignment ('<eps>' in case of texterrors aligner).
    Returns:
        Returns tuple of precision, recall and fscore.
    """

    assert key_words_list, "key_words_list is empty"

    # get data from manifest
    assert os.path.isfile(recognition_results_manifest), f"manifest file {recognition_results_manifest} doesn't exist"
    data = read_manifest(recognition_results_manifest)
    assert len(data) > 0, "manifest file is empty"
    assert data[0].get('text', None), "manifest file should contain text field"
    assert data[0].get('pred_text', None), "manifest file should contain pred_text field"

    # compute max number of words in one context biasing phrase
    max_ngram_order = max([len(item.split()) for item in key_words_list])
    key_words_stat = {}  # a word here can be single word or phareses
    for word in key_words_list:
        key_words_stat[word] = [0, 0, 0]  # [true positive (tp), groud truth (gt), false positive (fp)]

    for item in data:
        # get alignment by texterrors
        ref = item['text'].split()
        hyp = item['pred_text'].split()
        texterrors_ali = texterrors.align_texts(ref, hyp, False)
        ali = []
        for i in range(len(texterrors_ali[0])):
            ali.append((texterrors_ali[0][i], texterrors_ali[1][i]))

        # 1-grams
        for idx in range(len(ali)):
            word_ref = ali[idx][0]
            word_hyp = ali[idx][1]
            if word_ref in key_words_stat:
                key_words_stat[word_ref][1] += 1  # add to gt
                if word_ref == word_hyp:
                    key_words_stat[word_ref][0] += 1  # add to tp
            elif word_hyp in key_words_stat:
                key_words_stat[word_hyp][2] += 1  # add to fp

        # 2-grams and higher (takes into account epsilons in alignment)
        for ngram_order in range(2, max_ngram_order + 1):
            # for reference phrase
            idx = 0
            item_ref = []
            while idx < len(ali):
                if item_ref:
                    item_ref = [item_ref[1]]
                    idx = item_ref[0][1] + 1  # idex of second non eps word + 1
                while len(item_ref) != ngram_order and idx < len(ali):
                    word = ali[idx][0]
                    idx += 1
                    if word == eps:
                        continue
                    else:
                        item_ref.append((word, idx - 1))
                if len(item_ref) == ngram_order:
                    phrase_ref = " ".join([item[0] for item in item_ref])
                    phrase_hyp = " ".join([ali[item[1]][1] for item in item_ref])
                    if phrase_ref in key_words_stat:
                        key_words_stat[phrase_ref][1] += 1  # add to gt
                        if phrase_ref == phrase_hyp:
                            key_words_stat[phrase_ref][0] += 1  # add to tp
            # in case of false positive hypothesis phrase
            idx = 0
            item_hyp = []
            while idx < len(ali):
                if item_hyp:
                    item_hyp = [item_hyp[1]]
                    idx = item_hyp[0][1] + 1  # idex of first non eps word in previous ngram + 1
                while len(item_hyp) != ngram_order and idx < len(ali):
                    word = ali[idx][1]
                    idx += 1
                    if word == eps:
                        continue
                    else:
                        item_hyp.append((word, idx - 1))
                if len(item_hyp) == ngram_order:
                    phrase_hyp = " ".join([item[0] for item in item_hyp])
                    phrase_ref = " ".join([ali[item[1]][0] for item in item_hyp])
                    if phrase_hyp in key_words_stat and phrase_hyp != phrase_ref:
                        key_words_stat[phrase_hyp][2] += 1  # add to fp

    tp = sum([key_words_stat[x][0] for x in key_words_stat])
    gt = sum([key_words_stat[x][1] for x in key_words_stat])
    fp = sum([key_words_stat[x][2] for x in key_words_stat])

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-8)

    logging.info("=" * 60)
    logging.info("Per words statistic (word: correct/totall | false positive):\n")
    max_len = max([len(x) for x in key_words_stat if key_words_stat[x][1] > 0 or key_words_stat[x][2] > 0])
    for word in key_words_list:
        if key_words_stat[word][1] > 0 or key_words_stat[word][2] > 0:
            false_positive = ""
            if key_words_stat[word][2] > 0:
                false_positive = key_words_stat[word][2]
            logging.info(
                f"{word:>{max_len}}: {key_words_stat[word][0]:3}/{key_words_stat[word][1]:<3} |{false_positive:>3}"
            )
    logging.info("=" * 60)
    logging.info("=" * 60)
    logging.info(f"Precision: {precision:.4f} ({tp}/{tp + fp}) fp:{fp}")
    logging.info(f"Recall:    {recall:.4f} ({tp}/{gt})")
    logging.info(f"Fscore:    {fscore:.4f}")
    logging.info("=" * 60)

    return (precision, recall, fscore)
