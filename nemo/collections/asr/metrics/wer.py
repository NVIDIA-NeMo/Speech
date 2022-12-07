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

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import editdistance
import jiwer
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric

from nemo.collections.asr.parts.submodules import ctc_greedy_decoding
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMixin
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.utils import logging

__all__ = ['word_error_rate', 'word_error_rate_detail', 'WER', 'move_dimension_to_the_front']


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        # May deprecate using editdistance in future release for here and rest of codebase
        # once we confirm jiwer is reliable.
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


def word_error_rate_detail(
    hypotheses: List[str], references: List[str], use_cer=False
) -> Tuple[float, int, float, float, float]:
    """
    Computes Average Word Error Rate with details (insertion rate, deletion rate, substitution rate)
    between two texts represented as corresponding lists of string. 
    
    Hypotheses and references must have same length.
    Args:
      hypotheses (list): list of hypotheses
      references(list) : list of references
      use_cer (bool): set True to enable cer
    Returns:
      wer (float): average word error rate
      words (int):  Total number of words/charactors of given reference texts
      ins_rate (float): average insertion error rate
      del_rate (float): average deletion error rate
      sub_rate (float): average substitution error rate
      
    """
    scores = 0
    words = 0
    ops_count = {'substitutions': 0, 'insertions': 0, 'deletions': 0}

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                ops_count['insertions'] += errors
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
            else:
                measures = jiwer.compute_measures(r, h)

            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            ops_count['insertions'] += measures['insertions']
            ops_count['deletions'] += measures['deletions']
            ops_count['substitutions'] += measures['substitutions']

        scores += errors
        words += len(r_list)

    if words != 0:
        wer = 1.0 * scores / words
        ins_rate = 1.0 * ops_count['insertions'] / words
        del_rate = 1.0 * ops_count['deletions'] / words
        sub_rate = 1.0 * ops_count['substitutions'] / words
    else:
        wer, ins_rate, del_rate, sub_rate = float('inf'), float('inf'), float('inf'), float('inf')

    return wer, words, ins_rate, del_rate, sub_rate


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class AbstractCTCDecoding(ConfidenceMixin):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy (for greedy decoding).

            compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            ctc_timestamp_type: A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator: Str token representing the seperator between words.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `ctc_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding. When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.
                preserve_token_confidence: Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.
                preserve_word_confidence: Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.
                exclude_blank: Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name: The method name (str).
                        Supported values:
                            - 'max_prob' for using the maximum token probability as a confidence.
                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type: Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:
                            - 'gibbs' for the (standard) Gibbs entropy. If the temperature α is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the temperature should comply the following inequality:
                                1/log(V) <= α <= -1/log(1-1/V) where V is the model vocabulary size.
                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy
                            - 'renui' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    temperature: Temperature scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the temperature equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm: A mapping of the entropy value to the interval [0,1].
                        Supported values:
                            - 'lin' for using the linear mapping.
                            - 'exp' for using exponential mapping with linear shift.

            batch_dim_index: Index of the batch dimension of ``targets`` and ``predictions`` parameters of
                ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.

            The config may further contain the following sub-dictionaries:
            "greedy":
                preserve_alignments: Same as above, overrides above value.
                compute_timestamps: Same as above, overrides above value.
                preserve_frame_confidence: Same as above, overrides above value.

        blank_id: The id of the RNNT blank token.
    """

    def __init__(self, decoding_cfg, blank_id: int):
        super().__init__()

        # Convert dataclas to config
        if is_dataclass(decoding_cfg):
            decoding_cfg = OmegaConf.structured(decoding_cfg)

        if not isinstance(decoding_cfg, DictConfig):
            decoding_cfg = OmegaConf.create(decoding_cfg)

        OmegaConf.set_struct(decoding_cfg, False)

        # update minimal config
        minimal_cfg = ['greedy']
        for item in minimal_cfg:
            if item not in decoding_cfg:
                decoding_cfg[item] = OmegaConf.create({})

        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.compute_timestamps = self.cfg.get('compute_timestamps', None)
        self.batch_dim_index = self.cfg.get('batch_dim_index', 0)
        self.word_seperator = self.cfg.get('word_seperator', ' ')

        possible_strategies = ['greedy']
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}. Given {self.cfg.strategy}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if self.cfg.strategy in ['greedy']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)

        # Update compute timestamps
        if self.compute_timestamps is None:
            if self.cfg.strategy in ['greedy']:
                self.compute_timestamps = self.cfg.greedy.get('compute_timestamps', False)

        # initialize confidence-related fields
        self._init_confidence(self.cfg.get('confidence_cfg', None))

        # we need timestamps to extract non-blank per-frame confidence
        self.compute_timestamps |= self.preserve_frame_confidence

        if self.cfg.strategy == 'greedy':

            self.decoding = ctc_greedy_decoding.GreedyCTCInfer(
                blank_id=self.blank_id,
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                preserve_frame_confidence=self.preserve_frame_confidence,
                confidence_method_cfg=self.confidence_method_cfg,
            )

        else:
            raise ValueError(
                f"Incorrect decoding strategy supplied. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )

    def ctc_decoder_predictions_tensor(
        self,
        decoder_outputs: torch.Tensor,
        decoder_lengths: torch.Tensor = None,
        fold_consecutive: bool = True,
        return_hypotheses: bool = False,
    ) -> Tuple[List[str], Optional[List[List[str]]], Optional[Union[Hypothesis, NBestHypotheses]]]:
        """
        Decodes a sequence of labels to words

        Args:
            decoder_outputs: An integer torch.Tensor of shape [Batch, Time, {Vocabulary}] (if ``batch_index_dim == 0``) or [Time, Batch]
                (if ``batch_index_dim == 1``) of integer indices that correspond to the index of some character in the
                label set.
            decoder_lengths: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            fold_consecutive: Bool, determine whether to perform "ctc collapse", folding consecutive tokens
                into a single token.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe())

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """

        if isinstance(decoder_outputs, torch.Tensor):
            decoder_outputs = move_dimension_to_the_front(decoder_outputs, self.batch_dim_index)

        with torch.inference_mode():
            # Resolve the forward step of the decoding strategy
            hypotheses_list = self.decoding(
                decoder_output=decoder_outputs, decoder_lengths=decoder_lengths
            )  # type: List[List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        if isinstance(hypotheses_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []

            for nbest_hyp in hypotheses_list:  # type: NBestHypotheses
                n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(
                    n_hyps, fold_consecutive
                )  # type: List[Union[Hypothesis, NBestHypotheses]]

                # If computing timestamps
                if self.compute_timestamps is True:
                    timestamp_type = self.cfg.get('ctc_timestamp_type', 'all')
                    for hyp_idx in range(len(decoded_hyps)):
                        decoded_hyps[hyp_idx] = self.compute_ctc_timestamps(decoded_hyps[hyp_idx], timestamp_type)

                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return hypotheses, all_hypotheses

            best_hyp_text = [h.text for h in hypotheses]
            all_hyp_text = [h.text for hh in all_hypotheses for h in hh]
            return best_hyp_text, all_hyp_text

        else:
            hypotheses = self.decode_hypothesis(
                hypotheses_list, fold_consecutive
            )  # type: List[Union[Hypothesis, NBestHypotheses]]

            # If computing timestamps
            if self.compute_timestamps is True:
                # greedy decoding, can get high-level confidence scores
                if return_hypotheses and (self.preserve_word_confidence or self.preserve_token_confidence):
                    hypotheses = self.compute_confidence(hypotheses)
                else:
                    # remove unused token_repetitions from Hypothesis.text
                    for hyp in hypotheses:
                        hyp.text = hyp.text[:2]
                timestamp_type = self.cfg.get('ctc_timestamp_type', 'all')
                for hyp_idx in range(len(hypotheses)):
                    hypotheses[hyp_idx] = self.compute_ctc_timestamps(hypotheses[hyp_idx], timestamp_type)

            if return_hypotheses:
                return hypotheses, None

            best_hyp_text = [h.text for h in hypotheses]
            return best_hyp_text, None

    def decode_hypothesis(
        self, hypotheses_list: List[Hypothesis], fold_consecutive: bool
    ) -> List[Union[Hypothesis, NBestHypotheses]]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.
            fold_consecutive: Whether to collapse the ctc blank tokens or not.

        Returns:
            A list of strings.
        """
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            hyp = hypotheses_list[ind]
            prediction = hyp.y_sequence
            predictions_len = hyp.length if hyp.length > 0 else None

            if fold_consecutive:
                if type(prediction) != list:
                    prediction = prediction.numpy().tolist()

                if predictions_len is not None:
                    prediction = prediction[:predictions_len]

                # CTC decoding procedure
                decoded_prediction = []
                token_lengths = []  # preserve token lengths
                token_repetitions = []  # preserve number of repetitions per token

                previous = self.blank_id
                last_length = 0
                last_repetition = 1

                for pidx, p in enumerate(prediction):
                    if (p != previous or previous == self.blank_id) and p != self.blank_id:
                        decoded_prediction.append(p)

                        token_lengths.append(pidx - last_length)
                        last_length = pidx
                        token_repetitions.append(last_repetition)
                        last_repetition = 1

                    if p == previous and previous != self.blank_id:
                        last_repetition += 1

                    previous = p

                if len(token_repetitions) > 0:
                    token_repetitions = token_repetitions[1:] + [last_repetition]

            else:
                if predictions_len is not None:
                    prediction = prediction[:predictions_len]
                decoded_prediction = prediction[prediction != self.blank_id].tolist()
                token_lengths = [1] * len(decoded_prediction)  # preserve number of repetitions per token
                token_repetitions = [1] * len(decoded_prediction)  # preserve number of repetitions per token

            # De-tokenize the integer tokens; if not computing timestamps
            if self.compute_timestamps is True:
                # keep the original predictions, wrap with the number of repetitions per token
                # this is done so that `ctc_decoder_predictions_tensor()` can process this hypothesis
                # in order to compute exact time stamps.
                hypothesis = (decoded_prediction, token_lengths, token_repetitions)
            else:
                hypothesis = self.decode_tokens_to_str(decoded_prediction)

            # Preserve this wrapped hypothesis or decoded text tokens.
            hypotheses_list[ind].text = hypothesis

        return hypotheses_list

    def compute_confidence(self, hypotheses_list: List[Hypothesis]) -> List[Hypothesis]:
        """
        Computes high-level (per-token and/or per-word) confidence scores for a list of hypotheses.
        Assumes that `frame_confidence` is present in the hypotheses.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of hypotheses with high-level confidence scores.
        """
        for hyp in hypotheses_list:
            if not isinstance(hyp.text, tuple) or len(hyp.text) != 3:
                # the method must have been called in the wrong place
                raise ValueError(
                    """Wrong format of the `text` attribute of a hypothesis.\n
                    Expected: (decoded_prediction, token_repetitions)\n
                    The method invocation is expected between .decode_hypothesis() and .compute_ctc_timestamps()"""
                )
            token_repetitions = hyp.text[2]
            hyp.text = hyp.text[:2]
            token_confidence = []
            if self.exclude_blank_from_confidence:
                non_blank_frame_confidence = hyp.non_blank_frame_confidence
                i = 0
                for tr in token_repetitions:
                    # token repetition can be zero
                    j = i + tr
                    token_confidence.append(self._aggregate_confidence(non_blank_frame_confidence[i:j]))
                    i = j
            else:
                # <blank> tokens are considered to belong to the last non-blank token, if any.
                token_lengths = hyp.text[1]
                if len(token_lengths) > 0:
                    ts = token_lengths[0]
                    for tl in token_lengths[1:] + [len(hyp.frame_confidence)]:
                        token_confidence.append(self._aggregate_confidence(hyp.frame_confidence[ts : ts + tl]))
                        ts += tl
            hyp.token_confidence = token_confidence
        if self.preserve_word_confidence:
            for hyp in hypotheses_list:
                hyp.word_confidence = self._aggregate_token_confidence(hyp)
        return hypotheses_list

    @abstractmethod
    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token id list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        raise NotImplementedError()

    def compute_ctc_timestamps(self, hypothesis: Hypothesis, timestamp_type: str = "all"):
        """
        Method to compute time stamps at char/subword, and word level given some hypothesis.
        Requires the input hypothesis to contain a `text` field that is the tuple. The tuple contains -
        the ctc collapsed integer ids, and the number of repetitions of each token.

        Args:
            hypothesis: A Hypothesis object, with a wrapped `text` field.
                The `text` field must contain a tuple with two values -
                The ctc collapsed integer ids
                A list of integers that represents the number of repetitions per token.
            timestamp_type: A str value that represents the type of time stamp calculated.
                Can be one of "char", "word" or "all"

        Returns:
            A Hypothesis object with a modified `timestep` value, which is now a dictionary containing
            the time stamp information.
        """
        assert timestamp_type in ['char', 'word', 'all']

        # Unpack the temporary storage, and set the decoded predictions
        decoded_prediction, token_lengths = hypothesis.text
        hypothesis.text = decoded_prediction

        # Retrieve offsets
        char_offsets = word_offsets = None
        char_offsets = self._compute_offsets(hypothesis, token_lengths, self.blank_id)

        # Assert number of offsets and hypothesis tokens are 1:1 match.
        if len(char_offsets) != len(hypothesis.text):
            raise ValueError(
                f"`char_offsets`: {char_offsets} and `processed_tokens`: {hypothesis.text}"
                " have to be of the same length, but are: "
                f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                f" {len(hypothesis.text)}"
            )

        # Correctly process the token ids to chars/subwords.
        for i, char in enumerate(hypothesis.text):
            char_offsets[i]["char"] = self.decode_tokens_to_str([char])

        # detect char vs subword models
        lens = [len(list(v["char"])) > 1 for v in char_offsets]
        if any(lens):
            text_type = 'subword'
        else:
            text_type = 'char'

        # retrieve word offsets from character offsets
        word_offsets = None
        if timestamp_type in ['word', 'all']:
            if text_type == 'char':
                word_offsets = self._get_word_offsets_chars(char_offsets, word_delimiter_char=self.word_seperator)
            else:
                word_offsets = self._get_word_offsets_subwords_sentencepiece(
                    char_offsets,
                    hypothesis,
                    decode_ids_to_tokens=self.decode_ids_to_tokens,
                    decode_tokens_to_str=self.decode_tokens_to_str,
                )

        # attach results
        if len(hypothesis.timestep) > 0:
            timestep_info = hypothesis.timestep
        else:
            timestep_info = []

        # Setup defaults
        hypothesis.timestep = {"timestep": timestep_info}

        # Add char / subword time stamps
        if char_offsets is not None and timestamp_type in ['char', 'all']:
            hypothesis.timestep['char'] = char_offsets

        # Add word time stamps
        if word_offsets is not None and timestamp_type in ['word', 'all']:
            hypothesis.timestep['word'] = word_offsets

        # Convert the token indices to text
        hypothesis.text = self.decode_tokens_to_str(hypothesis.text)

        return hypothesis

    @staticmethod
    def _compute_offsets(
        hypothesis: Hypothesis, token_lengths: List[int], ctc_token: int
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Utility method that calculates the indidual time indices where a token starts and ends.

        Args:
            hypothesis: A Hypothesis object that contains `text` field that holds the character / subword token
                emitted at every time step after ctc collapse.
            token_lengths: A list of ints representing the lengths of each emitted token.
            ctc_token: The integer of the ctc blank token used during ctc collapse.

        Returns:

        """
        start_index = 0

        # If the exact timestep information is available, utilize the 1st non-ctc blank token timestep
        # as the start index.
        if hypothesis.timestep is not None and len(hypothesis.timestep) > 0:
            start_index = max(0, hypothesis.timestep[0] - 1)

        # Construct the start and end indices brackets
        end_indices = np.asarray(token_lengths).cumsum()
        start_indices = np.concatenate(([start_index], end_indices[:-1]))

        # Merge the results per token into a list of dictionaries
        offsets = [
            {"char": t, "start_offset": s, "end_offset": e}
            for t, s, e in zip(hypothesis.text, start_indices, end_indices)
        ]

        # Filter out CTC token
        offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
        return offsets

    @staticmethod
    def _get_word_offsets_chars(
        offsets: Dict[str, Union[str, float]], word_delimiter_char: str = " "
    ) -> Dict[str, Union[str, float]]:
        """
        Utility method which constructs word time stamps out of character time stamps.

        References:
            This code is a port of the Hugging Face code for word time stamp construction.

        Args:
            offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset".
            word_delimiter_char: Character token that represents the word delimiter. By default, " ".

        Returns:
            A list of dictionaries containing the word offsets. Each item contains "word", "start_offset" and
            "end_offset".
        """
        word_offsets = []

        last_state = "SPACE"
        word = ""
        start_offset = 0
        end_offset = 0
        for i, offset in enumerate(offsets):
            char = offset["char"]
            state = "SPACE" if char == word_delimiter_char else "WORD"

            if state == last_state:
                # If we are in the same state as before, we simply repeat what we've done before
                end_offset = offset["end_offset"]
                word += char
            else:
                # Switching state
                if state == "SPACE":
                    # Finishing a word
                    word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                else:
                    # Starting a new word
                    start_offset = offset["start_offset"]
                    end_offset = offset["end_offset"]
                    word = char

            last_state = state
        if last_state == "WORD":
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        return word_offsets

    @staticmethod
    def _get_word_offsets_subwords_sentencepiece(
        offsets: Dict[str, Union[str, float]],
        hypothesis: Hypothesis,
        decode_ids_to_tokens: Callable[[List[int]], str],
        decode_tokens_to_str: Callable[[List[int]], str],
    ) -> Dict[str, Union[str, float]]:
        """
        Utility method which constructs word time stamps out of sub-word time stamps.

        **Note**: Only supports Sentencepiece based tokenizers !

        Args:
            offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset".
            hypothesis: Hypothesis object that contains `text` field, where each token is a sub-word id
                after ctc collapse.
            decode_ids_to_tokens: A Callable function that accepts a list of integers and maps it to a sub-word.
            decode_tokens_to_str: A Callable function that accepts a list of integers and maps it to text / str.

        Returns:
            A list of dictionaries containing the word offsets. Each item contains "word", "start_offset" and
            "end_offset".
        """
        word_offsets = []
        built_token = []
        previous_token_index = 0
        # For every collapsed sub-word token
        for i, char in enumerate(hypothesis.text):
            # Compute the sub-word text representation, and the decoded text (stripped of sub-word markers).
            token = decode_ids_to_tokens([char])[0]
            token_text = decode_tokens_to_str([char])

            # It is a sub-word token, or contains an identifier at the beginning such as _ or ## that was stripped
            # after forcing partial text conversion of the token.
            if token != token_text:
                # If there are any partially or fully built sub-word token ids, construct to text.
                # Note: This is "old" subword, that occurs *after* current sub-word has started.
                if len(built_token) > 0:
                    word_offsets.append(
                        {
                            "word": decode_tokens_to_str(built_token),
                            "start_offset": offsets[previous_token_index]["start_offset"],
                            "end_offset": offsets[i]["start_offset"],
                        }
                    )

                # Prepare list of new sub-word ids
                built_token.clear()
                built_token.append(char)
                previous_token_index = i
            else:
                # If the token does not contain any sub-word start mark, then the sub-word has not completed yet
                # Append to current sub-word list.
                built_token.append(char)

        # Inject the start offset of the first token to word offsets
        # This is because we always skip the delay the injection of the first sub-word due to the loop
        # condition and check whether built token is ready or not.
        # Therefore without this forced injection, the start_offset appears as off by 1.
        if len(word_offsets) == 0:
            # alaptev: sometimes word_offsets can be empty
            if len(built_token) > 0:
                word_offsets.append(
                    {
                        "word": decode_tokens_to_str(built_token),
                        "start_offset": offsets[0]["start_offset"],
                        "end_offset": offsets[-1]["end_offset"],
                    }
                )
                built_token.clear()
        else:
            word_offsets[0]["start_offset"] = offsets[0]["start_offset"]

            # If there are any remaining tokens left, inject them all into the final word offset.
            # Note: The start offset of this token is the start time of the first token inside build_token.
            # Note: The end offset of this token is the end time of the last token inside build_token
            if len(built_token) > 0:
                word_offsets.append(
                    {
                        "word": decode_tokens_to_str(built_token),
                        "start_offset": offsets[-(len(built_token))]["start_offset"],
                        "end_offset": offsets[-1]["end_offset"],
                    }
                )
            built_token.clear()

        return word_offsets

    @property
    def preserve_alignments(self):
        return self._preserve_alignments

    @preserve_alignments.setter
    def preserve_alignments(self, value):
        self._preserve_alignments = value

        if hasattr(self, 'decoding'):
            self.decoding.preserve_alignments = value

    @property
    def compute_timestamps(self):
        return self._compute_timestamps

    @compute_timestamps.setter
    def compute_timestamps(self, value):
        self._compute_timestamps = value

        if hasattr(self, 'decoding'):
            self.decoding.compute_timestamps = value

    @property
    def preserve_frame_confidence(self):
        return self._preserve_frame_confidence

    @preserve_frame_confidence.setter
    def preserve_frame_confidence(self, value):
        self._preserve_frame_confidence = value

        if hasattr(self, 'decoding'):
            self.decoding.preserve_frame_confidence = value


class CTCDecoding(AbstractCTCDecoding):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy (for greedy decoding).

            compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            ctc_timestamp_type: A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator: Str token representing the seperator between words.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `ctc_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding. When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.
                preserve_token_confidence: Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.
                preserve_word_confidence: Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.
                exclude_blank: Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name: The method name (str).
                        Supported values:
                            - 'max_prob' for using the maximum token probability as a confidence.
                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type: Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:
                            - 'gibbs' for the (standard) Gibbs entropy. If the temperature α is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the temperature should comply the following inequality:
                                1/log(V) <= α <= -1/log(1-1/V) where V is the model vocabulary size.
                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy
                            - 'renui' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    temperature: Temperature scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the temperature equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm: A mapping of the entropy value to the interval [0,1].
                        Supported values:
                            - 'lin' for using the linear mapping.
                            - 'exp' for using exponential mapping with linear shift.

            batch_dim_index: Index of the batch dimension of ``targets`` and ``predictions`` parameters of
                ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.

            The config may further contain the following sub-dictionaries:
            "greedy":
                preserve_alignments: Same as above, overrides above value.
                compute_timestamps: Same as above, overrides above value.
                preserve_frame_confidence: Same as above, overrides above value.
                confidence_method: Same as above, overrides confidence_cfg.method.

        blank_id: The id of the RNNT blank token.
    """

    def __init__(
        self, decoding_cfg, vocabulary,
    ):
        blank_id = len(vocabulary)
        self.vocabulary = vocabulary
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        super().__init__(decoding_cfg=decoding_cfg, blank_id=blank_id)

    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """
        Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_chars(
            self.decode_tokens_to_str(hypothesis.text[0]).split(), hypothesis.token_confidence
        )

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list


class WER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference
    texts. When doing distributed training/evaluation the result of ``res=WER(predictions, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations. Here ``res`` contains three numbers
    ``res=[wer, total_levenstein_distance, total_number_of_words]``.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, transcript, transcript_len)
            return {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom}

        def validation_epoch_end(self, outputs):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denom}
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        decoding: An instance of CTCDecoding.
        use_cer: Whether to use Character Error Rate instead of Word Error Rate.
        log_prediction: Whether to log a single decoded sample per call.
        fold_consecutive: Whether repeated consecutive characters should be folded into one when decoding.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: CTCDecoding,
        use_cer=False,
        log_prediction=True,
        fold_consecutive=True,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0
        scores = 0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)

            hypotheses, _ = self.decoding.ctc_decoder_predictions_tensor(
                predictions, predictions_lengths, fold_consecutive=self.fold_consecutive
            )

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)

        self.scores = torch.tensor(scores, device=self.scores.device, dtype=self.scores.dtype)
        self.words = torch.tensor(words, device=self.words.device, dtype=self.words.dtype)
        # return torch.tensor([scores, words]).to(predictions.device)

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words


@dataclass
class CTCDecodingConfig:
    strategy: str = "greedy"

    # preserve decoding alignments
    preserve_alignments: Optional[bool] = None

    # compute ctc time stamps
    compute_timestamps: Optional[bool] = None

    #  confidence config
    confidence_cfg: ConfidenceConfig = ConfidenceConfig()

    # token representing word seperator
    word_seperator: str = " "

    # type of timestamps to calculate
    ctc_timestamp_type: str = "all"  # can be char, word or all for both

    # batch dimension
    batch_dim_index: int = 0

    # greedy decoding config
    greedy: ctc_greedy_decoding.GreedyCTCInferConfig = ctc_greedy_decoding.GreedyCTCInferConfig()
