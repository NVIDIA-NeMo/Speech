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


import itertools
from typing import Callable

from joblib import Parallel, delayed

from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
from nemo.collections.asr.inference.utils.text_segment import Word


def merge_punctuation_and_itn_tags(
    input_words: list[str],
    output_words: list[str],
    word_alignment: list[tuple],
    pnc_words: list[Word],
    punct_marks: set,
    sep: str,
    conf_aggregate_fn: Callable,
) -> list[Word]:
    """
    Merge the punctuation marks and ITN tags to the final text.
    It will also preserve first letter capitalization, start and end time of the span.
    Args:
        input_words: (list[str]) List of input words
        output_words: (list[str]) List of output words
        word_alignment: (list[tuple[list[int], list[int]]]) Word alignment between the input and output words
        pnc_words: (list[Word]) List of words with punctuation marks
        punct_marks: (set) Punctuation marks
        sep: (str) Separator
        conf_aggregate_fn: (Callable) Confidence aggregation function
    Returns:
        (list[Word]) Final words after merging the punctuation marks and ITN tags
    """
    assert len(input_words) == len(pnc_words)
    spans = []
    for s_idx, t_idx, semiotic_class in word_alignment:
        if len(t_idx) == 1 and len(s_idx) == 1 and input_words[s_idx[0]] == output_words[t_idx[0]]:
            span = pnc_words[s_idx[0]]
            span.semiotic_class = semiotic_class
        else:
            span_text = sep.join([output_words[i] for i in t_idx])
            last_char = pnc_words[s_idx[-1]].text[-1]
            first_char = pnc_words[s_idx[0]].text[0]

            # preserve the first char capitalization
            first_word = pnc_words[s_idx[0]].copy()
            first_char_is_upper = first_word.text[0].isupper()
            first_word.normalize_text_inplace(punct_marks, sep)
            if span_text.startswith(first_word.text):
                if first_char_is_upper:
                    span_text = span_text[0].upper() + span_text[1:]

            # preserve the last punctuation mark
            if last_char in punct_marks:
                span_text += last_char

            # preserve the first punctuation mark
            if first_char in punct_marks:
                span_text = first_char + span_text

            scores = [pnc_words[i].conf for i in s_idx]
            conf = conf_aggregate_fn(scores) if len(scores) > 0 else 0.0
            span = Word(
                text=span_text,
                start=pnc_words[s_idx[0]].start,
                end=pnc_words[s_idx[-1]].end,
                semiotic_class=semiotic_class,
                conf=conf,
            )
        spans.append(span)
    return spans


class BatchAlignmentPreservingInverseNormalizer:
    """
    Batch Alignment Preserving Inverse Text Normalizer. It is used to apply ITN to a batch of texts.
    joblib.Parallel is used to parallelize the processing.
    """

    def __init__(
        self,
        itn_model: AlignmentPreservingInverseNormalizer,
        sep: str,
        asr_supported_puncts: set[str],
        post_word_punctuation: set[str],
        conf_aggregate_fn: Callable,
    ):
        """
        Batch Alignment Preserving Inverse Text Normalizer. It is used to apply ITN to a batch of texts.
        Args:
            itn_model: (AlignmentPreservingInverseNormalizer) Alignment Preserving Inverse Text Normalizer
            sep: (str) Separator
            asr_supported_puncts: (Set[str]) Punctuation marks supported by ASR model
            post_word_punctuation: (Set[str]) Punctuation marks which usually appear after a word
            conf_aggregate_fn: (Callable) Confidence aggregation function
        """
        self.itn_model = itn_model
        self.sep = sep
        self.asr_supported_puncts = asr_supported_puncts
        self.conf_aggregate_fn = conf_aggregate_fn
        self.punct_marks = self.asr_supported_puncts | post_word_punctuation

    def apply_itn(
        self, asr_words: list[Word], pnc_words: list[Word], return_alignment: bool = False
    ) -> list[Word] | tuple[list[Word], list]:
        """
        Apply Alignment Preserving Inverse Text Normalization.
        Args:
            asr_words: (list[Word]) List of ASR words
            pnc_words: (list[Word]) List of words with punctuation/capitalization
            return_alignment: (bool) Flag to return the word alignment
        Returns:
            (list[Word]) List of words after applying ITN
        """
        input_words = []
        for word in asr_words:
            word.normalize_text_inplace(self.asr_supported_puncts, self.sep)
            input_words.append(word.text)

        input_words, output_words, word_alignment = self.itn_model.get_word_alignment(input_words, sep=self.sep)
        spans = merge_punctuation_and_itn_tags(
            input_words, output_words, word_alignment, pnc_words, self.punct_marks, self.sep, self.conf_aggregate_fn
        )

        if return_alignment:
            # word alignment is needed for streaming inference
            return spans, word_alignment
        return spans

    def __call__(
        self,
        asr_words_list: list[list[Word]],
        pnc_words_list: list[list[Word]],
        itn_params: dict,
        return_alignment: bool = False,
    ) -> list[list[Word]] | list[tuple]:
        """
        Batch Alignment Preserving Inverse Text Normalization.
        Args:
            asr_words_list: (list[list[Word]]) List of ASR words
            pnc_words_list: (list[list[Word]]) List of words with punctuation/capitalization
            itn_params: (dict) Parameters for the ITN model
            return_alignment: (bool) Flag to return the word alignment
        Returns:
            (list[list[Word]]) List of words after applying ITN
        """
        if len(asr_words_list) == 0:
            return []

        batch_size = itn_params.get("batch_size", 1)
        n_texts = len(asr_words_list)
        batch_size = min(n_texts, batch_size)

        def process_batch(batch_words, batch_words_with_pnc):
            return [
                self.apply_itn(words, words_with_pnc, return_alignment)
                for words, words_with_pnc in zip(batch_words, batch_words_with_pnc)
            ]

        if n_texts <= 3 * batch_size or n_texts == 1:
            # If the number of texts is less than 3 * batch_size, process the batch sequentially
            # For small batch size, it is faster to process the batch sequentially
            return process_batch(asr_words_list, pnc_words_list)

        n_jobs = itn_params.get("n_jobs", 1)
        itn_words_list = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(asr_words_list[i : i + batch_size], pnc_words_list[i : i + batch_size])
            for i in range(0, n_texts, batch_size)
        )
        itn_words_list = list(itertools.chain(*itn_words_list))
        return itn_words_list
