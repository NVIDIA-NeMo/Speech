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

import copy
import re
import unicodedata
from abc import abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.submodules import ctc_beam_decoding, ctc_greedy_decoding
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMixin
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.aggregate_tokenizer import DummyTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging, logging_mode


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class AbstractCTCDecoding(ConfidenceMixin):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy:
                str value which represents the type of decoding that can occur.
                Possible values are :

                    greedy (for greedy decoding).

                    beam (for DeepSpeed KenLM based decoding).

            compute_timestamps:
                A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            ctc_timestamp_type:
                A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator:
                Str token representing the seperator between words.

            segment_seperators:
                List containing tokens representing the seperator(s) between segments.

            segment_gap_threshold:
                The threshold (in frames) that caps the gap between two words necessary for forming the segments.

            preserve_alignments:
                Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.

            confidence_cfg:
                A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `ctc_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence:
                    Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding. When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.

                preserve_token_confidence:
                    Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.

                preserve_word_confidence:
                    Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.

                exclude_blank:
                    Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.

                aggregation:
                    Which aggregation type to use for collapsing per-token confidence into per-word confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.

                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).

                method_cfg:
                    A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name:
                        The method name (str).
                        Supported values:

                            'max_prob' for using the maximum token probability as a confidence.

                            'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type:
                        Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:

                            - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the alpha should comply the following inequality:
                                (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                                where V is the model vocabulary size.

                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy

                            - 'renyi' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    alpha:
                        Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm:
                        A mapping of the entropy value to the interval [0,1].
                        Supported values:

                            - 'lin' for using the linear mapping.

                            - 'exp' for using exponential mapping with linear shift.

            batch_dim_index:
                Index of the batch dimension of ``targets`` and ``predictions`` parameters of
                ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.

            The config may further contain the following sub-dictionaries:

                "greedy":
                    preserve_alignments: Same as above, overrides above value.
                    compute_timestamps: Same as above, overrides above value.
                    preserve_frame_confidence: Same as above, overrides above value.
                    confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

                "beam":
                    beam_size:
                        int, defining the beam size for beam search. Must be >= 1.
                        If beam_size == 1, will perform cached greedy search. This might be slightly different
                        results compared to the greedy search above.

                    return_best_hypothesis:
                        optional bool, whether to return just the best hypothesis or all of the
                        hypotheses after beam search has concluded. This flag is set by default.

                    ngram_lm_alpha:
                        float, the strength of the Language model on the final score of a token.
                        final_score = acoustic_score + ngram_lm_alpha * lm_score + beam_beta * seq_length.

                    beam_beta:
                        float, the strength of the sequence length penalty on the final score of a token.
                        final_score = acoustic_score + ngram_lm_alpha * lm_score + beam_beta * seq_length.

                    ngram_lm_model:
                        str, path to a KenLM ARPA or .binary file (depending on the strategy chosen).
                        If the path is invalid (file is not found at path), will raise a deferred error at the moment
                        of calculation of beam search, so that users may update / change the decoding strategy
                        to point to the correct file.

        blank_id:
            The id of the RNNT blank token.
        supported_punctuation:
            Set of punctuation marks in the vocabulary.
    """

    def __init__(self, decoding_cfg, blank_id: int, supported_punctuation: Optional[Set] = None):
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
        self.supported_punctuation = supported_punctuation
        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.compute_timestamps = self.cfg.get('compute_timestamps', None)
        self.batch_dim_index = self.cfg.get('batch_dim_index', 0)
        self.word_seperator = self.cfg.get('word_seperator', ' ')
        self.segment_seperators = self.cfg.get('segment_seperators', ['.', '?', '!'])
        self.segment_gap_threshold = self.cfg.get('segment_gap_threshold', None)

        possible_strategies = ['greedy', 'greedy_batch', 'beam', 'pyctcdecode', 'flashlight', 'wfst', 'beam_batch']
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}. Given {self.cfg.strategy}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)
            else:
                self.preserve_alignments = self.cfg.beam.get('preserve_alignments', False)

        # Update compute timestamps
        if self.compute_timestamps is None:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.compute_timestamps = self.cfg.greedy.get('compute_timestamps', False)
            elif self.cfg.strategy in ['beam']:
                self.compute_timestamps = self.cfg.beam.get('compute_timestamps', False)

        # Check if the model supports punctuation
        # and compile regex pattern to remove A space before supported punctuation marks if applicable
        # We remove only one space before punctuation marks as for some models punctuation marks are included in the vocabulary with a space.
        # The presence of multiple spaces before punctuation marks is a result of erroneous prediction of the ASR model, which should not be fixed during the decoding process.
        if self.supported_punctuation:
            punct_pattern = '|'.join([re.escape(p) for p in self.supported_punctuation])
            self.space_before_punct_pattern = re.compile(r'(\s)(' + punct_pattern + ')')

        # initialize confidence-related fields
        self._init_confidence(self.cfg.get('confidence_cfg', None))

        # Confidence estimation is not implemented for strategies other than `greedy` and `greedy_batch`
        if (
            not self.preserve_frame_confidence
            and self.cfg.strategy not in ('greedy', 'greedy_batch')
            and self.cfg.beam.get('preserve_frame_confidence', False)
        ):
            raise NotImplementedError(f"Confidence calculation is not supported for strategy `{self.cfg.strategy}`")

        # we need timestamps to extract non-blank per-frame confidence
        if self.compute_timestamps is not None:
            self.compute_timestamps |= self.preserve_frame_confidence

        if self.cfg.strategy in ['flashlight', 'wfst', 'beam_batch', 'pyctcdecode', 'beam']:
            if self.cfg.beam.beam_alpha is not None:
                logging.warning(
                    "`beam_alpha` is deprecated and will be removed in a future release. "
                    "Please use `ngram_lm_alpha` instead."
                )
                self.cfg.beam.ngram_lm_alpha = self.cfg.beam.beam_alpha
            if self.cfg.beam.kenlm_path is not None:
                logging.warning(
                    "`kenlm_path` is deprecated and will be removed in a future release. "
                    "Please use `ngram_lm_model` instead."
                )
                self.cfg.beam.ngram_lm_model = self.cfg.beam.kenlm_path

        if self.cfg.strategy == 'greedy':
            self.decoding = ctc_greedy_decoding.GreedyCTCInfer(
                blank_id=self.blank_id,
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                preserve_frame_confidence=self.preserve_frame_confidence,
                confidence_method_cfg=self.confidence_method_cfg,
            )

        elif self.cfg.strategy == "greedy_batch":
            self.decoding = ctc_greedy_decoding.GreedyBatchedCTCInfer(
                blank_id=self.blank_id,
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                preserve_frame_confidence=self.preserve_frame_confidence,
                confidence_method_cfg=self.confidence_method_cfg,
            )

        elif self.cfg.strategy == 'beam':

            self.decoding = ctc_beam_decoding.BeamCTCInfer(
                blank_id=blank_id,
                beam_size=self.cfg.beam.get('beam_size', 1),
                search_type='default',
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 1.0),
                beam_beta=self.cfg.beam.get('beam_beta', 0.0),
                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
            )

            self.decoding.override_fold_consecutive_value = False

        elif self.cfg.strategy == 'pyctcdecode':

            self.decoding = ctc_beam_decoding.BeamCTCInfer(
                blank_id=blank_id,
                beam_size=self.cfg.beam.get('beam_size', 1),
                search_type='pyctcdecode',
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 1.0),
                beam_beta=self.cfg.beam.get('beam_beta', 0.0),
                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                pyctcdecode_cfg=self.cfg.beam.get('pyctcdecode_cfg', None),
            )

            self.decoding.override_fold_consecutive_value = False

        elif self.cfg.strategy == 'flashlight':

            self.decoding = ctc_beam_decoding.BeamCTCInfer(
                blank_id=blank_id,
                beam_size=self.cfg.beam.get('beam_size', 1),
                search_type='flashlight',
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 1.0),
                beam_beta=self.cfg.beam.get('beam_beta', 0.0),
                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                flashlight_cfg=self.cfg.beam.get('flashlight_cfg', None),
            )

            self.decoding.override_fold_consecutive_value = False

        elif self.cfg.strategy == 'wfst':

            self.decoding = ctc_beam_decoding.WfstCTCInfer(
                blank_id=blank_id,
                beam_size=self.cfg.wfst.get('beam_size', 1),
                search_type=self.cfg.wfst.get('search_type', 'riva'),
                return_best_hypothesis=self.cfg.wfst.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                decoding_mode=self.cfg.wfst.get('decoding_mode', 'nbest'),
                open_vocabulary_decoding=self.cfg.wfst.get('open_vocabulary_decoding', False),
                beam_width=self.cfg.wfst.get('beam_width', 10.0),
                lm_weight=self.cfg.wfst.get('lm_weight', 1.0),
                device=self.cfg.wfst.get('device', 'cuda'),
                arpa_lm_path=self.cfg.wfst.get('arpa_lm_path', None),
                wfst_lm_path=self.cfg.wfst.get('wfst_lm_path', None),
                riva_decoding_cfg=self.cfg.wfst.get('riva_decoding_cfg', None),
                k2_decoding_cfg=self.cfg.wfst.get('k2_decoding_cfg', None),
            )

            self.decoding.override_fold_consecutive_value = False

        elif self.cfg.strategy == 'beam_batch':
            self.decoding = ctc_beam_decoding.BeamBatchedCTCInfer(
                blank_index=blank_id,
                beam_size=self.cfg.beam.get('beam_size', 1),
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 1.0),
                beam_beta=self.cfg.beam.get('beam_beta', 0.0),
                beam_threshold=self.cfg.beam.get('beam_threshold', 20.0),
                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', True),
            )

            self.decoding.override_fold_consecutive_value = False

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
    ) -> Union[List[Hypothesis], List[List[Hypothesis]]]:
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
            A list of Hypothesis objects containing additional information.
        """

        if isinstance(decoder_outputs, torch.Tensor):
            decoder_outputs = move_dimension_to_the_front(decoder_outputs, self.batch_dim_index)

        if (
            hasattr(self.decoding, 'override_fold_consecutive_value')
            and self.decoding.override_fold_consecutive_value is not None
        ):
            logging.info(
                f"Beam search requires that consecutive ctc tokens are not folded. \n"
                f"Overriding provided value of `fold_consecutive` = {fold_consecutive} to "
                f"{self.decoding.override_fold_consecutive_value}",
                mode=logging_mode.ONCE,
            )
            fold_consecutive = self.decoding.override_fold_consecutive_value

        with torch.inference_mode():
            # Resolve the forward step of the decoding strategy
            hypotheses_list = self.decoding(
                decoder_output=decoder_outputs, decoder_lengths=decoder_lengths
            )  # type: List[List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        if isinstance(hypotheses_list[0], NBestHypotheses):
            if self.cfg.strategy == 'wfst':
                all_hypotheses = [hyp.n_best_hypotheses for hyp in hypotheses_list]
            else:
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

                    all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return all_hypotheses  # type: list[list[Hypothesis]]

            # alaptev: The line below might contain a bug. Do we really want all_hyp_text to be flat?
            all_hyp = [[Hypothesis(h.score, h.y_sequence, h.text) for h in hh] for hh in all_hypotheses]
            return all_hyp

        else:
            if self.cfg.strategy == 'wfst':
                hypotheses = hypotheses_list
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
                return hypotheses

            return [Hypothesis(h.score, h.y_sequence, h.text) for h in hypotheses]

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
                hypothesis = self.decode_tokens_to_str_with_strip_punctuation(decoded_prediction)

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
    def get_words_offsets(
        self,
        char_offsets: List[Dict[str, Union[str, float]]],
        encoded_char_offsets: List[Dict[str, Union[str, float]]],
        word_delimiter_char: str,
        supported_punctuation: Optional[Set],
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Implemented by subclass in order to get the words offsets.
        """
        raise NotImplementedError()

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

    def decode_tokens_to_str_with_strip_punctuation(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens to a string and removes a space before supported punctuation marks.
        """
        text = self.decode_tokens_to_str(tokens)

        if self.supported_punctuation:
            text = self.space_before_punct_pattern.sub(r'\2', text)
        return text

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
                Can be one of "char", "word" "segment" or "all"

        Returns:
            A Hypothesis object with a modified `timestep` value, which is now a dictionary containing
            the time stamp information.
        """
        assert timestamp_type in ['char', 'word', 'segment', 'all']

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

        encoded_char_offsets = copy.deepcopy(char_offsets)

        # Correctly process the token ids to chars/subwords.
        for i, char in enumerate(hypothesis.text):
            char_offsets[i]["char"] = self.decode_tokens_to_str([char])

        encoded_char_offsets, char_offsets = self._refine_timestamps(
            encoded_char_offsets=encoded_char_offsets,
            char_offsets=char_offsets,
            supported_punctuation=self.supported_punctuation,
        )

        # retrieve word offsets from character offsets
        word_offsets = None
        if timestamp_type in ['word', 'segment', 'all']:
            word_offsets = self.get_words_offsets(
                char_offsets=char_offsets,
                encoded_char_offsets=encoded_char_offsets,
                word_delimiter_char=self.word_seperator,
                supported_punctuation=self.supported_punctuation,
            )

        segment_offsets = None
        if timestamp_type in ['segment', 'all']:
            segment_offsets = segment_offsets = self._get_segment_offsets(
                word_offsets,
                segment_delimiter_tokens=self.segment_seperators,
                supported_punctuation=self.supported_punctuation,
                segment_gap_threshold=self.segment_gap_threshold,
            )

        # attach results
        if len(hypothesis.timestamp) > 0:
            timestep_info = hypothesis.timestamp
        else:
            timestep_info = []

        # Setup defaults
        hypothesis.timestamp = {"timestep": timestep_info}

        # Add char / subword time stamps
        if char_offsets is not None and timestamp_type in ['char', 'all']:
            hypothesis.timestamp['char'] = char_offsets

        # Add word time stamps
        if word_offsets is not None and timestamp_type in ['word', 'all']:
            hypothesis.timestamp['word'] = word_offsets

        # Add segment time stamps
        if segment_offsets is not None and timestamp_type in ['segment', 'all']:
            hypothesis.timestamp['segment'] = segment_offsets

        # Convert the token indices to text
        hypothesis.text = self.decode_tokens_to_str_with_strip_punctuation(hypothesis.text)

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
        if hypothesis.timestamp is not None and len(hypothesis.timestamp) > 0:
            start_index = max(0, hypothesis.timestamp[0] - 1)

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
    def _refine_timestamps(
        encoded_char_offsets: List[Dict[str, Union[str, int]]],
        char_offsets: List[Dict[str, Union[str, int]]],
        supported_punctuation: Optional[Set] = None,
    ) -> List[Dict[str, Union[str, int]]]:

        if not supported_punctuation:
            return encoded_char_offsets, char_offsets

        for i, offset in enumerate(char_offsets):
            # Check if token is a punctuation mark
            # If so, set its end offset as its start offset
            # This is done because there was observed a behaviour for CTC decoding,
            # when punctuation marks are predicted for long frames
            if offset['char'] and offset['char'][0] in supported_punctuation and i > 0:
                encoded_char_offsets[i]['end_offset'] = offset['end_offset'] = offset['start_offset']

        return encoded_char_offsets, char_offsets

    @staticmethod
    def _get_segment_offsets(
        offsets: List[Dict[str, Union[str, float]]],
        segment_delimiter_tokens: List[str],
        supported_punctuation: Optional[Set] = None,
        segment_gap_threshold: Optional[int] = None,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Utility method which constructs segment time stamps out of word time stamps.

        Args:
            offsets: A list of dictionaries, each containing "word", "start_offset" and "end_offset".
            segments_delimiter_tokens: List containing tokens representing the seperator(s) between segments.
            supported_punctuation: Set containing punctuation marks in the vocabulary.
            segment_gap_threshold: Number of frames between 2 consecutive words necessary to form segments out of plain text.

        Returns:
            A list of dictionaries containing the segment offsets. Each item contains "segment", "start_offset" and
            "end_offset".
        """
        if (
            supported_punctuation
            and not set(segment_delimiter_tokens).intersection(supported_punctuation)
            and not segment_gap_threshold
        ):
            logging.warning(
                f"Specified segment seperators are not in supported punctuation {supported_punctuation}. "
                "If the seperators are not punctuation marks, ignore this warning. "
                "Otherwise, specify 'segment_gap_threshold' parameter in decoding config to form segments.",
                mode=logging_mode.ONCE,
            )

        segment_offsets = []
        segment_words = []
        previous_word_index = 0

        # For every offset word
        for i, offset in enumerate(offsets):

            word = offset['word']
            if segment_gap_threshold and segment_words:
                gap_between_words = offset['start_offset'] - offsets[i - 1]['end_offset']

                if gap_between_words >= segment_gap_threshold:
                    segment_offsets.append(
                        {
                            "segment": ' '.join(segment_words),
                            "start_offset": offsets[previous_word_index]["start_offset"],
                            "end_offset": offsets[i - 1]["end_offset"],
                        }
                    )

                    segment_words = [word]
                    previous_word_index = i
                    continue

            # check if the word ends with any delimeter token or the word itself is a delimeter
            elif word and (word[-1] in segment_delimiter_tokens or word in segment_delimiter_tokens):
                segment_words.append(word)
                if segment_words:
                    segment_offsets.append(
                        {
                            "segment": ' '.join(segment_words),
                            "start_offset": offsets[previous_word_index]["start_offset"],
                            "end_offset": offset["end_offset"],
                        }
                    )

                segment_words = []
                previous_word_index = i + 1
                continue

            segment_words.append(word)

        if segment_words:
            start_offset = offsets[previous_word_index]["start_offset"]
            segment_offsets.append(
                {
                    "segment": ' '.join(segment_words),
                    "start_offset": start_offset,
                    "end_offset": offsets[-1]["end_offset"],
                }
            )
        segment_words.clear()

        return segment_offsets

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
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs for character
    based models.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.

            strategy:
                str value which represents the type of decoding that can occur.
                Possible values are :

                    -   greedy (for greedy decoding).

                    -   beam (for DeepSpeed KenLM based decoding).

            compute_timestamps:
                A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            ctc_timestamp_type:
                A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator:
                Str token representing the seperator between words.

            segment_seperators:
                List containing tokens representing the seperator(s) between segments.

            segment_gap_threshold:
                The threshold (in frames) that caps the gap between two words necessary for forming the segments.

            preserve_alignments:
                Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.

            confidence_cfg:
                A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `ctc_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence:
                    Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding. When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.

                preserve_token_confidence:
                    Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.

                preserve_word_confidence:
                    Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.

                exclude_blank:
                    Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.

                aggregation:
                    Which aggregation type to use for collapsing per-token confidence into per-word confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.

                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).

                method_cfg:
                    A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name:
                        The method name (str).
                        Supported values:

                            - 'max_prob' for using the maximum token probability as a confidence.

                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type:
                        Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:

                            - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the alpha should comply the following inequality:
                                (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                                where V is the model vocabulary size.

                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy

                            - 'renyi' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    alpha:
                        Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm:
                        A mapping of the entropy value to the interval [0,1].
                        Supported values:

                            - 'lin' for using the linear mapping.

                            - 'exp' for using exponential mapping with linear shift.

            batch_dim_index:
                Index of the batch dimension of ``targets`` and ``predictions`` parameters of
                ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.

            The config may further contain the following sub-dictionaries:

                "greedy":
                    preserve_alignments: Same as above, overrides above value.
                    compute_timestamps: Same as above, overrides above value.
                    preserve_frame_confidence: Same as above, overrides above value.
                    confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

                "beam":
                    beam_size:
                        int, defining the beam size for beam search. Must be >= 1.
                        If beam_size == 1, will perform cached greedy search. This might be slightly different
                        results compared to the greedy search above.

                    return_best_hypothesis:
                        optional bool, whether to return just the best hypothesis or all of the
                        hypotheses after beam search has concluded. This flag is set by default.

                    ngram_lm_alpha:
                        float, the strength of the Language model on the final score of a token.
                        final_score = acoustic_score + ngram_lm_alpha * lm_score + beam_beta * seq_length.

                    beam_beta:
                        float, the strength of the sequence length penalty on the final score of a token.
                        final_score = acoustic_score + ngram_lm_alpha * lm_score + beam_beta * seq_length.

                    ngram_lm_model:
                        str, path to a KenLM ARPA or .binary file (depending on the strategy chosen).
                        If the path is invalid (file is not found at path), will raise a deferred error at the moment
                        of calculation of beam search, so that users may update / change the decoding strategy
                        to point to the correct file.

        blank_id: The id of the RNNT blank token.
    """

    def __init__(
        self,
        decoding_cfg,
        vocabulary,
    ):
        blank_id = len(vocabulary)
        self.vocabulary = vocabulary
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        supported_punctuation = {
            char for token in vocabulary for char in token if unicodedata.category(char).startswith('P')
        }

        super().__init__(decoding_cfg=decoding_cfg, blank_id=blank_id, supported_punctuation=supported_punctuation)

        # Finalize Beam Search Decoding framework
        if isinstance(self.decoding, ctc_beam_decoding.AbstractBeamCTCInfer):
            self.decoding.set_vocabulary(self.vocabulary)
            self.decoding.set_decoding_type('char')

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

    @staticmethod
    def get_words_offsets(
        char_offsets: List[Dict[str, Union[str, float]]],
        encoded_char_offsets: List[Dict[str, Union[str, float]]],
        word_delimiter_char: str = " ",
        supported_punctuation: Optional[Set] = None,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Utility method which constructs word time stamps out of character time stamps.

        References:
            This code is a port of the Hugging Face code for word time stamp construction.

        Args:
            char_offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset",
                        where "char" is decoded with the tokenizer.
            encoded_char_offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset",
                        where "char" is the original id/ids from the hypotheses (not decoded with the tokenizer).
                        As we are working with char-based models here, we are using the `char_offsets` to get the word offsets.
                        `encoded_char_offsets` is passed for keeping the consistency with `AbstractRNNTDecoding`'s abstract method.
            word_delimiter_char: Character token that represents the word delimiter. By default, " ".
            supported_punctuation: Set containing punctuation marks in the vocabulary.

        Returns:
            A list of dictionaries containing the word offsets. Each item contains "word", "start_offset" and
            "end_offset".
        """

        word_offsets = []

        last_state = "DELIMITER"
        word = ""
        start_offset = 0
        end_offset = 0
        for i, offset in enumerate(char_offsets):
            char = offset["char"]
            state = "DELIMITER" if char == word_delimiter_char else "WORD"

            next_char = char_offsets[i + 1]['char'] if i < len(char_offsets) - 1 else None

            if next_char:
                next_punctuation = next_char in supported_punctuation and next_char != word_delimiter_char
            else:
                next_punctuation = False

            # If we have a space and the next character is a punctuation, we skip adding the space to the word
            # This is for being consistent with the final hypothesis text,
            # For which we are removing a space before a punctuation.
            if char == " " and next_punctuation:
                continue

            if state == last_state:
                # If we are in the same state as before, we simply repeat what we've done before
                end_offset = offset["end_offset"]
                word += char
            else:
                # Switching state
                if state == "DELIMITER":
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


class CTCBPEDecoding(AbstractCTCDecoding):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs for subword based
    models.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.

            strategy:
                str value which represents the type of decoding that can occur.
                Possible values are :

                    -   greedy (for greedy decoding).

                    -   beam (for DeepSpeed KenLM based decoding).

            compute_timestamps:
                A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            ctc_timestamp_type:
                A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator:
                Str token representing the seperator between words.

            preserve_alignments:
                Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.

            confidence_cfg:
                A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `ctc_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence:
                    Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding. When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.

                preserve_token_confidence:
                    Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `token_confidence` in it. Here, `token_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized tokens.

                preserve_word_confidence:
                    Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `word_confidence` in it. Here, `word_confidence` is a List of floats.

                    The length of the list corresponds to the number of recognized words.

                exclude_blank:
                    Bool flag indicating that blank token confidence scores are to be excluded
                    from the `token_confidence`.

                aggregation:
                    Which aggregation type to use for collapsing per-token confidence into per-word confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.

                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).

                method_cfg:
                    A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name:
                        The method name (str).
                        Supported values:

                            - 'max_prob' for using the maximum token probability as a confidence.

                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type:
                        Which type of entropy to use (str).
                        Used if confidence_method_cfg.name is set to `entropy`.
                        Supported values:

                            - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                                the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                                Note that for this entropy, the alpha should comply the following inequality:
                                (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                                where V is the model vocabulary size.

                            - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                                Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/Tsallis_entropy

                            - 'renyi' for the Rényi entropy.
                                Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                                where α is a parameter. When α == 1, it works like the Gibbs entropy.
                                More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

                    alpha:
                        Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm:
                        A mapping of the entropy value to the interval [0,1].
                        Supported values:

                            - 'lin' for using the linear mapping.

                            - 'exp' for using exponential mapping with linear shift.

            batch_dim_index:
                Index of the batch dimension of ``targets`` and ``predictions`` parameters of
                ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.

            The config may further contain the following sub-dictionaries:

                "greedy":
                    preserve_alignments: Same as above, overrides above value.
                    compute_timestamps: Same as above, overrides above value.
                    preserve_frame_confidence: Same as above, overrides above value.
                    confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

                "beam":
                    beam_size:
                        int, defining the beam size for beam search. Must be >= 1.
                        If beam_size == 1, will perform cached greedy search. This might be slightly different
                        results compared to the greedy search above.

                    return_best_hypothesis:
                        optional bool, whether to return just the best hypothesis or all of the
                        hypotheses after beam search has concluded. This flag is set by default.

                    ngram_lm_alpha:
                        float, the strength of the Language model on the final score of a token.
                        final_score = acoustic_score + ngram_lm_alpha * lm_score + beam_beta * seq_length.

                    beam_beta:
                        float, the strength of the sequence length penalty on the final score of a token.
                        final_score = acoustic_score + ngram_lm_alpha * lm_score + beam_beta * seq_length.

                    ngram_lm_model:
                        str, path to a KenLM ARPA or .binary file (depending on the strategy chosen).
                        If the path is invalid (file is not found at path), will raise a deferred error at the moment
                        of calculation of beam search, so that users may update / change the decoding strategy
                        to point to the correct file.

        tokenizer: NeMo tokenizer object, which inherits from TokenizerSpec.
    """

    def __init__(self, decoding_cfg, tokenizer: TokenizerSpec):
        blank_id = tokenizer.tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.tokenizer_type = self.define_tokenizer_type(tokenizer.vocab)

        if hasattr(tokenizer, 'supported_punctuation'):
            supported_punctuation = tokenizer.supported_punctuation
        else:
            supported_punctuation = {
                char for token in tokenizer.vocab for char in token if unicodedata.category(char).startswith('P')
            }

        super().__init__(decoding_cfg=decoding_cfg, blank_id=blank_id, supported_punctuation=supported_punctuation)

        # Finalize Beam Search Decoding framework
        if isinstance(self.decoding, ctc_beam_decoding.AbstractBeamCTCInfer):
            if hasattr(self.tokenizer.tokenizer, 'get_vocab'):
                vocab_dict = self.tokenizer.tokenizer.get_vocab()
                if isinstance(self.tokenizer.tokenizer, DummyTokenizer):  # AggregateTokenizer.DummyTokenizer
                    vocab = vocab_dict
                else:
                    vocab = list(vocab_dict.keys())
                self.decoding.set_vocabulary(vocab)
                self.decoding.set_tokenizer(tokenizer)
            else:
                logging.warning("Could not resolve the vocabulary of the tokenizer !")

            self.decoding.set_decoding_type('subword')

    @staticmethod
    def define_tokenizer_type(vocabulary: List[str]) -> str:
        """
        Define the tokenizer type based on the vocabulary.
        """
        if any(token.startswith("##") for token in vocabulary):
            return "wpe"
        return "bpe"

    @staticmethod
    def define_word_start_condition(tokenizer_type: str, word_delimiter_char: str) -> Callable[[str, str], bool]:
        """
        Define the word start condition based on the tokenizer type and word delimiter character.
        """
        if word_delimiter_char == " ":
            if tokenizer_type == "wpe":
                return lambda token, token_text: token_text and not token_text.startswith("##")
            return lambda token, token_text: token != token_text
        else:
            return lambda token, token_text: token_text == word_delimiter_char

    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """
        Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        **Note**: Only supports Sentencepiece based tokenizers!

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_subwords_sentencepiece(
            self.decode_tokens_to_str(hypothesis.text[0]).split(), hypothesis.token_confidence, hypothesis.text[0]
        )

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = self.tokenizer.ids_to_text(tokens)
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
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list

    def get_words_offsets(
        self,
        char_offsets: List[Dict[str, Union[str, float]]],
        encoded_char_offsets: List[Dict[str, Union[str, float]]],
        word_delimiter_char: str = " ",
        supported_punctuation: Optional[Set] = None,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Utility method which constructs word time stamps out of sub-word time stamps.

        **Note**: Only supports Sentencepiece based tokenizers !

        Args:
            char_offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset",
                        where "char" is decoded with the tokenizer.
            encoded_char_offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset",
                        where "char" is the original id/ids from the hypotheses (not decoded with the tokenizer).
                        This is needed for subword tokenization models.
            word_delimiter_char: Character token that represents the word delimiter. By default, " ".
            supported_punctuation: Set containing punctuation marks in the vocabulary.

        Returns:
            A list of dictionaries containing the word offsets. Each item contains "word", "start_offset" and
            "end_offset".
        """
        char_offsets = encoded_char_offsets.copy()
        word_offsets = []
        previous_token_index = 0

        # Built tokens should be list here as when dealing with wpe tokenizer,
        # ids should be decoded together to ensure tokens starting with ## are not split
        built_tokens = []

        condition_for_word_start = self.define_word_start_condition(self.tokenizer_type, word_delimiter_char)

        # For every collapsed sub-word token
        for i, offset in enumerate(char_offsets):

            char = offset['char']

            # Compute the sub-word text representation, and the decoded text (stripped of sub-word markers).
            token = self.decode_ids_to_tokens([char])[0]
            token_text = self.decode_tokens_to_str([char]).strip()

            curr_punctuation = supported_punctuation and token_text in supported_punctuation

            # It is a sub-word token, or contains an identifier at the beginning such as _ or ## that was stripped
            # after forcing partial text conversion of the token.
            # AND it is not a supported punctuation mark, which needs to be added to the built word regardless of its identifier.
            if condition_for_word_start(token, token_text) and not curr_punctuation:
                # If there are any partially or fully built sub-word token ids, construct to text.
                # Note: This is "old" subword, that occurs *after* current sub-word has started.
                if built_tokens:
                    built_word = self.decode_tokens_to_str(built_tokens)
                    if built_word:
                        word_offsets.append(
                            {
                                "word": built_word,
                                "start_offset": char_offsets[previous_token_index]["start_offset"],
                                "end_offset": char_offsets[i - 1]["end_offset"],
                            }
                        )

                # Prepare new built_tokens
                built_tokens.clear()

                if token_text != word_delimiter_char:
                    built_tokens.append(char)
                    previous_token_index = i

            # If the token is a punctuation mark and there is no built word, then the previous word is complete
            # and lacks the punctuation mark. We need to add the punctuation mark to the previous formed word.
            elif curr_punctuation and not built_tokens:
                last_built_word = word_offsets[-1]
                last_built_word['end_offset'] = offset['end_offset']
                if last_built_word['word'][-1] == ' ':
                    last_built_word['word'] = last_built_word['word'][:-1]
                last_built_word['word'] += token_text
            else:
                # If the token does not contain any sub-word start mark, then the sub-word has not completed yet
                # Append to current built word.
                # If this token is the first in the built_tokens, we should save its index as the previous token index
                # because it will be used to calculate the start offset of the word.
                if not built_tokens:
                    previous_token_index = i
                built_tokens.append(char)

        # Inject the start offset of the first token to word offsets
        # This is because we always skip the delay the injection of the first sub-word due to the loop
        # condition and check whether built token is ready or not.
        # Therefore without this forced injection, the start_offset appears as off by 1.
        if len(word_offsets) == 0:
            # alaptev: sometimes word_offsets can be empty
            if built_tokens:
                built_word = self.decode_tokens_to_str(built_tokens)
                if built_word:
                    word_offsets.append(
                        {
                            "word": built_word,
                            "start_offset": char_offsets[0]["start_offset"],
                            "end_offset": char_offsets[-1]["end_offset"],
                        }
                    )
        else:
            word_offsets[0]["start_offset"] = char_offsets[0]["start_offset"]

            # If there are any remaining tokens left, inject them all into the final word offset.
            # Note: The start offset of this token is the start time of the first token inside build_token.
            # Note: The end offset of this token is the end time of the last token inside build_token
            if built_tokens:
                built_word = self.decode_tokens_to_str(built_tokens)
                if built_word:
                    word_offsets.append(
                        {
                            "word": built_word,
                            "start_offset": char_offsets[previous_token_index]["start_offset"],
                            "end_offset": char_offsets[-1]["end_offset"],
                        }
                    )

        return word_offsets


@dataclass
class CTCDecodingConfig:
    strategy: str = "greedy_batch"

    # preserve decoding alignments
    preserve_alignments: Optional[bool] = None

    # compute ctc time stamps
    compute_timestamps: Optional[bool] = None

    # token representing word seperator
    word_seperator: str = " "

    # tokens representing segments seperators
    segment_seperators: Optional[List[str]] = field(default_factory=lambda: [".", "!", "?"])

    # threshold (in frames) that caps the gap between two words necessary for forming the segments
    segment_gap_threshold: Optional[int] = None

    # type of timestamps to calculate
    ctc_timestamp_type: str = "all"  # can be char, word or all for both

    # batch dimension
    batch_dim_index: int = 0

    # greedy decoding config
    greedy: ctc_greedy_decoding.GreedyCTCInferConfig = field(
        default_factory=lambda: ctc_greedy_decoding.GreedyCTCInferConfig()
    )

    # beam decoding config
    beam: ctc_beam_decoding.BeamCTCInferConfig = field(
        default_factory=lambda: ctc_beam_decoding.BeamCTCInferConfig(beam_size=4)
    )

    # wfst decoding config
    wfst: ctc_beam_decoding.WfstCTCInferConfig = field(
        default_factory=lambda: ctc_beam_decoding.WfstCTCInferConfig(beam_size=4)
    )

    # confidence config
    confidence_cfg: ConfidenceConfig = field(default_factory=lambda: ConfidenceConfig())

    # can be used to change temperature for decoding
    temperature: float = 1.0


@dataclass
class CTCBPEDecodingConfig(CTCDecodingConfig):
    pass
