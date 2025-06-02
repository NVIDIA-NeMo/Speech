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
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules import rnnt_beam_decoding, rnnt_greedy_decoding, tdt_beam_decoding
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMixin
from nemo.collections.asr.parts.utils.rnnt_batched_beam_utils import BlankLMScoreMode, PruningMode
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging, logging_mode


class AbstractRNNTDecoding(ConfidenceMixin):
    """
    Used for performing RNN-T auto-regressive decoding of the Decoder+Joint network given the encoder state.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy, greedy_batch (for greedy decoding).
                -   beam, tsd, alsd (for beam search decoding).

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `alignments` in it. Here, `alignments` is a List of List of
                Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

                The length of the list corresponds to the Acoustic Length (T).
                Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
                U is the number of target tokens for the current timestep Ti.

            tdt_include_token_duration: Bool flag, which determines whether predicted durations for each token
            need to be included in the Hypothesis object. Defaults to False.

            compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrete intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            rnnt_timestamp_type: A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps, "segment" for segment level time stamps and "all" (default), for character, word and
                segment level time stamps.

            word_seperator: Str token representing the seperator between words.

            segment_seperators: List containing tokens representing the seperator(s) between segments.

            segment_gap_threshold: The threshold (in frames) that caps the gap between two words necessary for forming
            the segments.

            preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of ints.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `rnnt_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of floats.

                    The length of the list corresponds to the Acoustic Length (T).
                    Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
                    U is the number of target tokens for the current timestep Ti.
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
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word
                    confidence. Valid options are `mean`, `min`, `max`, `prod`.
                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name: The method name (str).
                        Supported values:
                            - 'max_prob' for using the maximum token probability as a confidence.
                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type: Which type of entropy to use (str).
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

                    alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm: A mapping of the entropy value to the interval [0,1].
                        Supported values:
                            - 'lin' for using the linear mapping.
                            - 'exp' for using exponential mapping with linear shift.

            The config may further contain the following sub-dictionaries:
            "greedy":
                max_symbols: int, describing the maximum number of target tokens to decode per
                    timestep during greedy decoding. Setting to larger values allows longer sentences
                    to be decoded, at the cost of increased execution time.
                preserve_frame_confidence: Same as above, overrides above value.
                confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                score_norm: optional bool, whether to normalize the returned beam score in the hypotheses.
                    Set to True by default.

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.

                tsd_max_sym_exp: optional int, determines number of symmetric expansions of the target symbols
                    per timestep of the acoustic model. Larger values will allow longer sentences to be decoded,
                    at increased cost to execution time.

                alsd_max_target_len: optional int or float, determines the potential maximum target sequence length.
                    If an integer is provided, it can decode sequences of that particular maximum length.
                    If a float is provided, it can decode sequences of int(alsd_max_target_len * seq_len),
                    where seq_len is the length of the acoustic model output (T).

                    NOTE:
                        If a float is provided, it can be greater than 1!
                        By default, a float of 2.0 is used so that a target sequence can be at most twice
                        as long as the acoustic model output length T.

                maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient,
                    and can be reduced to 1 to improve decoding speed while sacrificing some accuracy. int > 0.

                maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to keep
                    this as 1 in order to reduce expensive beam search cost later. int >= 0.

                maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                    Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                    and affects the speed of inference since large values will perform large beam search in the next
                    step.

                maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the
                    expansions. The default (2.3) is selected from the paper. It performs a comparison
                    (max_log_prob - gamma <= log_prob[v]) where v is all vocabulary indices in the Vocab set and
                    max_log_prob is the "most" likely token to be predicted. Gamma therefore provides a margin of
                    additional tokens which can be potential candidates for expansion apart from the "most likely"
                    candidate. Lower values will reduce the number of expansions (by increasing pruning-by-value,
                    thereby improving speed but hurting accuracy). Higher values will increase the number of expansions
                    (by reducing pruning-by-value, thereby reducing speed but potentially improving accuracy). This is
                    a hyper parameter to be experimentally tuned on a validation set.

                softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        blank_id: The id of the RNNT blank token.
        supported_punctuation: Set of punctuation marks in the vocabulary
    """

    def __init__(self, decoding_cfg, decoder, joint, blank_id: int, supported_punctuation: Optional[Set] = None):
        super(AbstractRNNTDecoding, self).__init__()

        # Convert dataclass to config object
        if is_dataclass(decoding_cfg):
            decoding_cfg = OmegaConf.structured(decoding_cfg)

        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.supported_punctuation = supported_punctuation
        self.num_extra_outputs = joint.num_extra_outputs
        self.big_blank_durations = self.cfg.get("big_blank_durations", None)
        self.durations = self.cfg.get("durations", None)
        self.compute_hypothesis_token_set = self.cfg.get("compute_hypothesis_token_set", False)
        self.compute_langs = decoding_cfg.get('compute_langs', False)
        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.joint_fused_batch_size = self.cfg.get('fused_batch_size', None)
        self.compute_timestamps = self.cfg.get('compute_timestamps', None)
        self.tdt_include_token_duration = self.cfg.get('tdt_include_token_duration', False)
        self.word_seperator = self.cfg.get('word_seperator', ' ')
        self.segment_seperators = self.cfg.get('segment_seperators', ['.', '?', '!'])
        self.segment_gap_threshold = self.cfg.get('segment_gap_threshold', None)

        self._is_tdt = self.durations is not None and self.durations != []  # this means it's a TDT model.
        if self._is_tdt:
            if blank_id == 0:
                raise ValueError("blank_id must equal len(non_blank_vocabs) for TDT models")
            if self.big_blank_durations is not None and self.big_blank_durations != []:
                raise ValueError("duration and big_blank_durations can't both be not None")
            if self.cfg.strategy not in ['greedy', 'greedy_batch', 'beam', 'maes', "malsd_batch"]:
                raise ValueError(
                    "currently only greedy, greedy_batch, beam and maes inference is supported for TDT models"
                )

        if (
            self.big_blank_durations is not None and self.big_blank_durations != []
        ):  # this means it's a multi-blank model.
            if blank_id == 0:
                raise ValueError("blank_id must equal len(vocabs) for multi-blank RNN-T models")
            if self.cfg.strategy not in ['greedy', 'greedy_batch']:
                raise ValueError(
                    "currently only greedy and greedy_batch inference is supported for multi-blank models"
                )

        possible_strategies = ['greedy', 'greedy_batch', 'beam', 'tsd', 'alsd', 'maes', 'malsd_batch', "maes_batch"]
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)

            elif self.cfg.strategy in ['beam', 'tsd', 'alsd', 'maes']:
                self.preserve_alignments = self.cfg.beam.get('preserve_alignments', False)

        # Update compute timestamps
        if self.compute_timestamps is None:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.compute_timestamps = self.cfg.greedy.get('compute_timestamps', False)

            elif self.cfg.strategy in ['beam', 'tsd', 'alsd', 'maes']:
                self.compute_timestamps = self.cfg.beam.get('compute_timestamps', False)

        # Check if the model supports punctuation
        # and compile regex pattern to remove A space before supported punctuation marks if applicable
        # We remove only one space before punctuation marks as for some models punctuation marks are included in the vocabulary with a space.
        # The presence of multiple spaces before punctuation marks is a result of erroneous prediction of the ASR model, which should not be fixed during the decoding process.
        if self.supported_punctuation:
            punct_pattern = '|'.join([re.escape(p) for p in self.supported_punctuation])
            self.space_before_punct_pattern = re.compile(r'(\s)(' + punct_pattern + ')')

        # Test if alignments are being preserved for RNNT
        if not self._is_tdt and self.compute_timestamps is True and self.preserve_alignments is False:
            raise ValueError("If `compute_timesteps` flag is set, then `preserve_alignments` flag must also be set.")

        # initialize confidence-related fields
        self._init_confidence(self.cfg.get('confidence_cfg', None))

        if self._is_tdt:
            if self.preserve_frame_confidence is True and self.preserve_alignments is False:
                raise ValueError(
                    "If `preserve_frame_confidence` flag is set, then `preserve_alignments` flag must also be set."
                )
            self.tdt_include_token_duration = self.tdt_include_token_duration or self.compute_timestamps
            self._compute_offsets = self._compute_offsets_tdt
            self._refine_timestamps = self._refine_timestamps_tdt

        # Confidence estimation is not implemented for these strategies
        if (
            not self.preserve_frame_confidence
            and self.cfg.strategy in ['beam', 'tsd', 'alsd', 'maes']
            and self.cfg.beam.get('preserve_frame_confidence', False)
        ):
            raise NotImplementedError(f"Confidence calculation is not supported for strategy `{self.cfg.strategy}`")

        if self.cfg.strategy == 'greedy':
            if self.big_blank_durations is None or self.big_blank_durations == []:
                if not self._is_tdt:
                    self.decoding = rnnt_greedy_decoding.GreedyRNNTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        max_symbols_per_step=(
                            self.cfg.greedy.get('max_symbols', None)
                            or self.cfg.greedy.get('max_symbols_per_step', None)
                        ),
                        preserve_alignments=self.preserve_alignments,
                        preserve_frame_confidence=self.preserve_frame_confidence,
                        confidence_method_cfg=self.confidence_method_cfg,
                    )
                else:
                    self.decoding = rnnt_greedy_decoding.GreedyTDTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        durations=self.durations,
                        max_symbols_per_step=(
                            self.cfg.greedy.get('max_symbols', None)
                            or self.cfg.greedy.get('max_symbols_per_step', None)
                        ),
                        preserve_alignments=self.preserve_alignments,
                        preserve_frame_confidence=self.preserve_frame_confidence,
                        include_duration=self.tdt_include_token_duration,
                        include_duration_confidence=self.tdt_include_duration_confidence,
                        confidence_method_cfg=self.confidence_method_cfg,
                    )
            else:
                self.decoding = rnnt_greedy_decoding.GreedyMultiblankRNNTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    big_blank_durations=self.big_blank_durations,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None) or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    preserve_alignments=self.preserve_alignments,
                    preserve_frame_confidence=self.preserve_frame_confidence,
                    confidence_method_cfg=self.confidence_method_cfg,
                )

        elif self.cfg.strategy == 'greedy_batch':
            if self.big_blank_durations is None or self.big_blank_durations == []:
                if not self._is_tdt:
                    self.decoding = rnnt_greedy_decoding.GreedyBatchedRNNTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        max_symbols_per_step=(
                            self.cfg.greedy.get('max_symbols', None)
                            or self.cfg.greedy.get('max_symbols_per_step', None)
                        ),
                        preserve_alignments=self.preserve_alignments,
                        preserve_frame_confidence=self.preserve_frame_confidence,
                        confidence_method_cfg=self.confidence_method_cfg,
                        loop_labels=self.cfg.greedy.get('loop_labels', True),
                        use_cuda_graph_decoder=self.cfg.greedy.get('use_cuda_graph_decoder', True),
                        ngram_lm_model=self.cfg.greedy.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.greedy.get('ngram_lm_alpha', 0),
                    )
                else:
                    self.decoding = rnnt_greedy_decoding.GreedyBatchedTDTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        durations=self.durations,
                        max_symbols_per_step=(
                            self.cfg.greedy.get('max_symbols', None)
                            or self.cfg.greedy.get('max_symbols_per_step', None)
                        ),
                        preserve_alignments=self.preserve_alignments,
                        preserve_frame_confidence=self.preserve_frame_confidence,
                        include_duration=self.tdt_include_token_duration,
                        include_duration_confidence=self.tdt_include_duration_confidence,
                        confidence_method_cfg=self.confidence_method_cfg,
                        use_cuda_graph_decoder=self.cfg.greedy.get('use_cuda_graph_decoder', True),
                        ngram_lm_model=self.cfg.greedy.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.greedy.get('ngram_lm_alpha', 0),
                    )

            else:
                self.decoding = rnnt_greedy_decoding.GreedyBatchedMultiblankRNNTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    big_blank_durations=self.big_blank_durations,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None) or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    preserve_alignments=self.preserve_alignments,
                    preserve_frame_confidence=self.preserve_frame_confidence,
                    confidence_method_cfg=self.confidence_method_cfg,
                )

        elif self.cfg.strategy == 'beam':
            if self.big_blank_durations is None or self.big_blank_durations == []:
                if not self._is_tdt:
                    self.decoding = rnnt_beam_decoding.BeamRNNTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        beam_size=self.cfg.beam.beam_size,
                        return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                        search_type='default',
                        score_norm=self.cfg.beam.get('score_norm', True),
                        softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                        preserve_alignments=self.preserve_alignments,
                    )
                else:
                    self.decoding = tdt_beam_decoding.BeamTDTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        durations=self.durations,
                        beam_size=self.cfg.beam.beam_size,
                        return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                        search_type='default',
                        score_norm=self.cfg.beam.get('score_norm', True),
                        softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                        preserve_alignments=self.preserve_alignments,
                    )

        elif self.cfg.strategy == 'tsd':
            self.decoding = rnnt_beam_decoding.BeamRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                beam_size=self.cfg.beam.beam_size,
                return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                search_type='tsd',
                score_norm=self.cfg.beam.get('score_norm', True),
                tsd_max_sym_exp_per_step=self.cfg.beam.get('tsd_max_sym_exp', 10),
                softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                preserve_alignments=self.preserve_alignments,
            )

        elif self.cfg.strategy == 'alsd':
            self.decoding = rnnt_beam_decoding.BeamRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                beam_size=self.cfg.beam.beam_size,
                return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                search_type='alsd',
                score_norm=self.cfg.beam.get('score_norm', True),
                alsd_max_target_len=self.cfg.beam.get('alsd_max_target_len', 2),
                softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                preserve_alignments=self.preserve_alignments,
            )

        elif self.cfg.strategy == 'maes':
            if self.big_blank_durations is None or self.big_blank_durations == []:
                if not self._is_tdt:
                    self.decoding = rnnt_beam_decoding.BeamRNNTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        beam_size=self.cfg.beam.beam_size,
                        return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                        search_type='maes',
                        score_norm=self.cfg.beam.get('score_norm', True),
                        maes_num_steps=self.cfg.beam.get('maes_num_steps', 2),
                        maes_prefix_alpha=self.cfg.beam.get('maes_prefix_alpha', 1),
                        maes_expansion_gamma=self.cfg.beam.get('maes_expansion_gamma', 2.3),
                        maes_expansion_beta=self.cfg.beam.get('maes_expansion_beta', 2.0),
                        softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                        preserve_alignments=self.preserve_alignments,
                        ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.0),
                        hat_subtract_ilm=self.cfg.beam.get('hat_subtract_ilm', False),
                        hat_ilm_weight=self.cfg.beam.get('hat_ilm_weight', 0.0),
                    )
                else:
                    self.decoding = tdt_beam_decoding.BeamTDTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        durations=self.durations,
                        beam_size=self.cfg.beam.beam_size,
                        return_best_hypothesis=decoding_cfg.beam.get('return_best_hypothesis', True),
                        search_type='maes',
                        score_norm=self.cfg.beam.get('score_norm', True),
                        maes_num_steps=self.cfg.beam.get('maes_num_steps', 2),
                        maes_prefix_alpha=self.cfg.beam.get('maes_prefix_alpha', 1),
                        maes_expansion_gamma=self.cfg.beam.get('maes_expansion_gamma', 2.3),
                        maes_expansion_beta=self.cfg.beam.get('maes_expansion_beta', 2.0),
                        softmax_temperature=self.cfg.beam.get('softmax_temperature', 1.0),
                        preserve_alignments=self.preserve_alignments,
                        ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.3),
                    )
        elif self.cfg.strategy == 'malsd_batch':
            if self.big_blank_durations is None or self.big_blank_durations == []:
                if not self._is_tdt:
                    self.decoding = rnnt_beam_decoding.BeamBatchedRNNTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        beam_size=self.cfg.beam.beam_size,
                        search_type='malsd_batch',
                        max_symbols_per_step=self.cfg.beam.get("max_symbols", 10),
                        preserve_alignments=self.preserve_alignments,
                        ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.0),
                        blank_lm_score_mode=self.cfg.beam.get(
                            'blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL
                        ),
                        pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                        score_norm=self.cfg.beam.get('score_norm', True),
                        allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', True),
                        return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                    )
                else:
                    self.decoding = tdt_beam_decoding.BeamBatchedTDTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        durations=self.durations,
                        beam_size=self.cfg.beam.beam_size,
                        search_type='malsd_batch',
                        max_symbols_per_step=self.cfg.beam.get("max_symbols", 10),
                        preserve_alignments=self.preserve_alignments,
                        ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.0),
                        blank_lm_score_mode=self.cfg.beam.get(
                            'blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL
                        ),
                        pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                        score_norm=self.cfg.beam.get('score_norm', True),
                        allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', True),
                        return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                    )
        elif self.cfg.strategy == 'maes_batch':
            if self.big_blank_durations is None or self.big_blank_durations == []:
                if not self._is_tdt:
                    self.decoding = rnnt_beam_decoding.BeamBatchedRNNTInfer(
                        decoder_model=decoder,
                        joint_model=joint,
                        blank_index=self.blank_id,
                        beam_size=self.cfg.beam.beam_size,
                        search_type='maes_batch',
                        maes_num_steps=self.cfg.beam.get('maes_num_steps', 2),
                        maes_expansion_beta=self.cfg.beam.get('maes_expansion_beta', 2),
                        maes_expansion_gamma=self.cfg.beam.get('maes_expansion_gamma', 2.3),
                        preserve_alignments=self.preserve_alignments,
                        ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                        ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.0),
                        blank_lm_score_mode=self.cfg.beam.get(
                            'blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL
                        ),
                        pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                        score_norm=self.cfg.beam.get('score_norm', True),
                        allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', False),
                        return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                    )
        else:
            raise ValueError(
                f"Incorrect decoding strategy supplied. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )

        # Update the joint fused batch size or disable it entirely if needed.
        self.update_joint_fused_batch_size()

    def rnnt_decoder_predictions_tensor(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        return_hypotheses: bool = False,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> Union[List[Hypothesis], List[List[Hypothesis]]]:
        """
        Decode an encoder output by autoregressive decoding of the Decoder+Joint networks.

        Args:
            encoder_output: torch.Tensor of shape [B, D, T].
            encoded_lengths: torch.Tensor containing lengths of the padded encoder outputs. Shape [B].
            return_hypotheses: bool. If set to True it will return list of Hypothesis or NBestHypotheses

        Returns:
            If `return_all_hypothesis` is set:
                A list[list[Hypothesis]].
                    Look at rnnt_utils.Hypothesis for more information.

            If `return_all_hypothesis` is not set:
                A list[Hypothesis].
                List of best hypotheses
                    Look at rnnt_utils.Hypothesis for more information.
        """
        # Compute hypotheses
        with torch.inference_mode():
            hypotheses_list = self.decoding(
                encoder_output=encoder_output, encoded_lengths=encoded_lengths, partial_hypotheses=partial_hypotheses
            )  # type: [List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        prediction_list = hypotheses_list

        if isinstance(prediction_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []

            for nbest_hyp in prediction_list:  # type: NBestHypotheses
                n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(n_hyps)  # type: List[str]

                # If computing timestamps
                if self.compute_timestamps is True:
                    timestamp_type = self.cfg.get('rnnt_timestamp_type', 'all')
                    for hyp_idx in range(len(decoded_hyps)):
                        decoded_hyps[hyp_idx] = self.compute_rnnt_timestamps(decoded_hyps[hyp_idx], timestamp_type)

                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return all_hypotheses  # type: list[list[Hypothesis]]

            all_hyp = [[Hypothesis(h.score, h.y_sequence, h.text) for h in hh] for hh in all_hypotheses]
            return all_hyp

        else:
            hypotheses = self.decode_hypothesis(prediction_list)  # type: List[str]

            # If computing timestamps
            if self.compute_timestamps is True:
                timestamp_type = self.cfg.get('rnnt_timestamp_type', 'all')
                for hyp_idx in range(len(hypotheses)):
                    hypotheses[hyp_idx] = self.compute_rnnt_timestamps(hypotheses[hyp_idx], timestamp_type)

            if return_hypotheses:
                # greedy decoding, can get high-level confidence scores
                if self.preserve_frame_confidence and (
                    self.preserve_word_confidence or self.preserve_token_confidence
                ):
                    hypotheses = self.compute_confidence(hypotheses)
                return hypotheses

            return [Hypothesis(h.score, h.y_sequence, h.text) for h in hypotheses]

    def decode_hypothesis(self, hypotheses_list: List[Hypothesis]) -> List[Union[Hypothesis, NBestHypotheses]]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of strings.
        """
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            prediction = hypotheses_list[ind].y_sequence

            if type(prediction) != list:
                prediction = prediction.tolist()

            # RNN-T sample level is already preprocessed by implicit RNNT decoding
            # Simply remove any blank and possibly big blank tokens
            if self.big_blank_durations is not None and self.big_blank_durations != []:  # multi-blank RNNT
                num_extra_outputs = len(self.big_blank_durations)
                prediction = [p for p in prediction if p < self.blank_id - num_extra_outputs]
            elif self._is_tdt:  # TDT model.
                prediction = [p for p in prediction if p < self.blank_id]
            else:  # standard RNN-T
                prediction = [p for p in prediction if p != self.blank_id]

            # De-tokenize the integer tokens; if not computing timestamps
            if self.compute_timestamps is True and self._is_tdt:
                hypothesis = (prediction, None, None)
            elif self.compute_timestamps is True:
                # keep the original predictions, wrap with the number of repetitions per token and alignments
                # this is done so that `rnnt_decoder_predictions_tensor()` can process this hypothesis
                # in order to compute exact time stamps.
                alignments = copy.deepcopy(hypotheses_list[ind].alignments)
                token_repetitions = [1] * len(alignments)  # preserve number of repetitions per token
                hypothesis = (prediction, alignments, token_repetitions)
            else:
                hypothesis = self.decode_tokens_to_str_with_strip_punctuation(prediction)

                if self.compute_hypothesis_token_set:
                    hypotheses_list[ind].tokens = self.decode_ids_to_tokens(prediction)

            # De-tokenize the integer tokens
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
        if self._is_tdt:
            # if self.tdt_include_duration_confidence is True then frame_confidence elements consist of two numbers
            maybe_pre_aggregate = (
                (lambda x: self._aggregate_confidence(x)) if self.tdt_include_duration_confidence else (lambda x: x)
            )
            for hyp in hypotheses_list:
                token_confidence = []
                # trying to recover frame_confidence according to alignments
                subsequent_blank_confidence = []
                # going backwards since <blank> tokens are considered belonging to the last non-blank token.
                for fc, fa in zip(hyp.frame_confidence[::-1], hyp.alignments[::-1]):
                    # there is only one score per frame most of the time
                    if len(fa) > 1:
                        for i, a in reversed(list(enumerate(fa))):
                            if a[-1] == self.blank_id:
                                if not self.exclude_blank_from_confidence:
                                    subsequent_blank_confidence.append(maybe_pre_aggregate(fc[i]))
                            elif not subsequent_blank_confidence:
                                token_confidence.append(maybe_pre_aggregate(fc[i]))
                            else:
                                token_confidence.append(
                                    self._aggregate_confidence(
                                        [maybe_pre_aggregate(fc[i])] + subsequent_blank_confidence
                                    )
                                )
                                subsequent_blank_confidence = []
                    else:
                        i, a = 0, fa[0]
                        if a[-1] == self.blank_id:
                            if not self.exclude_blank_from_confidence:
                                subsequent_blank_confidence.append(maybe_pre_aggregate(fc[i]))
                        elif not subsequent_blank_confidence:
                            token_confidence.append(maybe_pre_aggregate(fc[i]))
                        else:
                            token_confidence.append(
                                self._aggregate_confidence([maybe_pre_aggregate(fc[i])] + subsequent_blank_confidence)
                            )
                            subsequent_blank_confidence = []
                token_confidence = token_confidence[::-1]
                hyp.token_confidence = token_confidence
        else:
            if self.exclude_blank_from_confidence:
                for hyp in hypotheses_list:
                    hyp.token_confidence = hyp.non_blank_frame_confidence
            else:
                for hyp in hypotheses_list:
                    timestep = hyp.timestamp.tolist() if isinstance(hyp.timestamp, torch.Tensor) else hyp.timestamp
                    offset = 0
                    token_confidence = []
                    if len(timestep) > 0:
                        for ts, te in zip(timestep, timestep[1:] + [len(hyp.frame_confidence)]):
                            if ts != te:
                                # <blank> tokens are considered to belong to the last non-blank token, if any.
                                token_confidence.append(
                                    self._aggregate_confidence(
                                        [hyp.frame_confidence[ts][offset]]
                                        + [fc[0] for fc in hyp.frame_confidence[ts + 1 : te]]
                                    )
                                )
                                offset = 0
                            else:
                                token_confidence.append(hyp.frame_confidence[ts][offset])
                                offset += 1
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

    @abstractmethod
    def decode_tokens_to_lang(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to
        compute the most likely language ID (LID) string given the tokens.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded LID string.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_ids_to_langs(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to
        decode a token id list into language ID (LID) list.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded LIDS.
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

    def update_joint_fused_batch_size(self):
        """ "
        Updates the fused batch size for the joint module if applicable.

        If `joint_fused_batch_size` is set, verifies that the joint module has
        the required `set_fused_batch_size` and `set_fuse_loss_wer` functions.
        If present, updates the batch size; otherwise, logs a warning.

        If `joint_fused_batch_size` is <= 0, disables fused batch processing.
        """
        if self.joint_fused_batch_size is None:
            # do nothing and let the Joint itself handle setting up of the fused batch
            return

        if not hasattr(self.decoding.joint, 'set_fused_batch_size'):
            logging.warning(
                "The joint module does not have `set_fused_batch_size(int)` as a setter function.\n"
                "Ignoring update of joint fused batch size."
            )
            return

        if not hasattr(self.decoding.joint, 'set_fuse_loss_wer'):
            logging.warning(
                "The joint module does not have `set_fuse_loss_wer(bool, RNNTLoss, RNNTWER)` "
                "as a setter function.\n"
                "Ignoring update of joint fused batch size."
            )
            return

        if self.joint_fused_batch_size > 0:
            self.decoding.joint.set_fused_batch_size(self.joint_fused_batch_size)
        else:
            logging.info("Joint fused batch size <= 0; Will temporarily disable fused batch step in the Joint.")
            self.decoding.joint.set_fuse_loss_wer(False)

    def compute_rnnt_timestamps(self, hypothesis: Hypothesis, timestamp_type: str = "all"):
        """
        Computes character, word, and segment timestamps for an RNN-T hypothesis.

        This function generates timestamps for characters, words, and segments within
        a hypothesis sequence. The type of timestamps computed depends on `timestamp_type`,
        which can be 'char', 'word', 'segment', or 'all'.

        Args:
            hypothesis (Hypothesis): Hypothesis.
            timestamp_type (str): Type of timestamps to compute. Options are 'char', 'word', 'segment', or 'all'.
                                Defaults to 'all'.

        Returns:
            Hypothesis: The updated hypothesis with computed timestamps for characters, words, and/or segments.
        """
        assert timestamp_type in ['char', 'word', 'segment', 'all']

        # Unpack the temporary storage
        decoded_prediction, alignments, token_repetitions = hypothesis.text

        # Retrieve offsets
        char_offsets = word_offsets = None
        char_offsets = self._compute_offsets(hypothesis, token_repetitions, self.blank_id)

        # finally, set the flattened decoded predictions to text field for later text decoding
        hypothesis.text = decoded_prediction

        # Assert number of offsets and hypothesis tokens are 1:1 match.
        num_flattened_tokens = 0
        for t in range(len(char_offsets)):
            # Count all tokens except for RNNT BLANK token emitted to designate "End of timestep"
            num_flattened_tokens += len([c for c in char_offsets[t]['char'] if c != self.blank_id])

        if num_flattened_tokens != len(hypothesis.text):
            raise ValueError(
                f"`char_offsets`: {char_offsets} and `processed_tokens`: {hypothesis.text}"
                " have to be of the same length, but are: "
                f"`len(offsets)`: {num_flattened_tokens} and `len(processed_tokens)`:"
                f" {len(hypothesis.text)}"
            )

        encoded_char_offsets = copy.deepcopy(char_offsets)

        # Correctly process the token ids to chars/subwords.
        for i, offsets in enumerate(char_offsets):
            decoded_chars = []
            for char in offsets['char']:
                if char != self.blank_id:  # ignore the RNNT Blank token
                    decoded_chars.append(self.decode_tokens_to_str([int(char)]))
            char_offsets[i]["char"] = decoded_chars

        encoded_char_offsets, char_offsets = self._refine_timestamps(
            encoded_char_offsets, char_offsets, self.supported_punctuation
        )

        # detect char vs subword models
        lens = []
        for v in char_offsets:
            tokens = v["char"]
            # each token may be either 1 unicode token or multiple unicode token
            # for character based models, only 1 token is used
            # for subword, more than one token can be used.
            # Computing max, then summing up total lens is a test to check for char vs subword
            # For char models, len(lens) == sum(lens)
            # but this is violated for subword models.
            max_len = max(len(c) for c in tokens)
            lens.append(max_len)

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
            segment_offsets = self._get_segment_offsets(
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

        # Convert the flattened token indices to text
        hypothesis.text = self.decode_tokens_to_str_with_strip_punctuation(hypothesis.text)

        if self.compute_hypothesis_token_set:
            hypothesis.tokens = self.decode_ids_to_tokens(decoded_prediction)

        return hypothesis

    @staticmethod
    def _compute_offsets(
        hypothesis: Hypothesis, token_repetitions: List[int], rnnt_token: int
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Utility method that calculates the indidual time indices where a token starts and ends.

        Args:
            hypothesis: A Hypothesis object that contains `text` field that holds the character / subword token
                emitted at every time step after rnnt collapse.
            token_repetitions: A list of ints representing the number of repetitions of each emitted token.
            rnnt_token: The integer of the rnnt blank token used during rnnt collapse.

        Returns:

        """
        start_index = 0
        # If the exact timestep information is available, utilize the 1st non-rnnt blank token timestep
        # as the start index.
        if hypothesis.timestamp is not None and len(hypothesis.timestamp) > 0:
            first_timestep = hypothesis.timestamp[0]
            first_timestep = first_timestep if isinstance(first_timestep, int) else first_timestep.item()
            start_index = max(0, first_timestep - 1)

        # Construct the start and end indices brackets
        end_indices = np.asarray(token_repetitions).cumsum()
        start_indices = np.concatenate(([start_index], end_indices[:-1]))

        # Process the TxU dangling alignment tensor, containing pairs of (logits, label)
        alignment_labels = [al_logits_labels for al_logits_labels in hypothesis.text[1]]
        for t in range(len(alignment_labels)):
            for u in range(len(alignment_labels[t])):
                alignment_labels[t][u] = alignment_labels[t][u][1]  # pick label from (logit, label) tuple

        # Merge the results per token into a list of dictionaries
        offsets = [
            {"char": a, "start_offset": s, "end_offset": e}
            for a, s, e in zip(alignment_labels, start_indices, end_indices)
        ]

        # Filter out RNNT token (blank at [t][0] position). This is because blank can only occur at end of a
        # time step for RNNT, so if 0th token is blank, then that timestep is skipped.
        offsets = list(filter(lambda offsets: offsets["char"][0] != rnnt_token, offsets))
        return offsets

    @staticmethod
    def _compute_offsets_tdt(hypothesis: Hypothesis, *args) -> List[Dict[str, Union[str, int]]]:
        """
        Utility method that calculates the indidual time indices where a token starts and ends.

        Args:
            hypothesis: A Hypothesis object that contains `text` field that holds the character / subword token
                emitted at a specific time step considering predicted durations of the previous tokens.

        Returns:

        """
        if isinstance(hypothesis.timestamp, torch.Tensor):
            hypothesis.token_duration = hypothesis.token_duration.cpu().tolist()

        if isinstance(hypothesis.timestamp, torch.Tensor):
            hypothesis.timestamp = hypothesis.timestamp.cpu().tolist()

        # Merge the results per token into a list of dictionaries
        offsets = [
            {"char": [t], "start_offset": s, "end_offset": s + d}
            for t, s, d in zip(hypothesis.text[0], hypothesis.timestamp, hypothesis.token_duration)
        ]
        return offsets

    @staticmethod
    def _refine_timestamps(
        encoded_char_offsets: List[Dict[str, Union[str, int]]],
        char_offsets: List[Dict[str, Union[str, int]]],
        supported_punctuation: Optional[Set] = None,
    ) -> List[Dict[str, Union[str, int]]]:

        # no refinement for rnnt

        return encoded_char_offsets, char_offsets

    @staticmethod
    def _refine_timestamps_tdt(
        encoded_char_offsets: List[Dict[str, Union[str, int]]],
        char_offsets: List[Dict[str, Union[str, int]]],
        supported_punctuation: Optional[Set] = None,
    ) -> List[Dict[str, Union[str, int]]]:

        if not supported_punctuation:
            return encoded_char_offsets, char_offsets

        for i, offset in enumerate(char_offsets):

            # Check if token is a punctuation mark
            # If so, set its start and end offset as start and end of the previous token
            # This is done because there was observed a behaviour, when punctuation marks are
            # predicted long after preceding token (i.e. after silence)
            if offset['char'][0] in supported_punctuation and i > 0:
                encoded_char_offsets[i]['start_offset'] = offset['start_offset'] = char_offsets[i - 1]['end_offset']
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
            segment_gap_threshold: Number of frames between 2 consecutive words necessary to form segments out of plain
            text.
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


class RNNTDecoding(AbstractRNNTDecoding):
    """
    Used for performing RNN-T auto-regressive decoding of the Decoder+Joint network given the encoder state.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.

            strategy:
                str value which represents the type of decoding that can occur.
                Possible values are :

                -   greedy, greedy_batch (for greedy decoding).

                -   beam, tsd, alsd (for beam search decoding).

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `alignments` is a List of List of
                Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

                The length of the list corresponds to the Acoustic Length (T).
                Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
                U is the number of target tokens for the current timestep Ti.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `rnnt_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of floats.

                    The length of the list corresponds to the Acoustic Length (T).
                    Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
                    U is the number of target tokens for the current timestep Ti.
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
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word
                    confidence.
                    Valid options are `mean`, `min`, `max`, `prod`.
                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
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

            The config may further contain the following sub-dictionaries:

                "greedy":
                    max_symbols: int, describing the maximum number of target tokens to decode per
                        timestep during greedy decoding. Setting to larger values allows longer sentences
                        to be decoded, at the cost of increased execution time.

                    preserve_frame_confidence: Same as above, overrides above value.

                    confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

                "beam":
                    beam_size: int, defining the beam size for beam search. Must be >= 1.
                        If beam_size == 1, will perform cached greedy search. This might be slightly different
                        results compared to the greedy search above.

                    score_norm: optional bool, whether to normalize the returned beam score in the hypotheses.
                        Set to True by default.

                    return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                        hypotheses after beam search has concluded. This flag is set by default.

                    tsd_max_sym_exp: optional int, determines number of symmetric expansions of the target symbols
                        per timestep of the acoustic model. Larger values will allow longer sentences to be decoded,
                        at increased cost to execution time.

                    alsd_max_target_len: optional int or float, determines the potential maximum target sequence
                        length. If an integer is provided, it can decode sequences of that particular maximum length.
                        If a float is provided, it can decode sequences of int(alsd_max_target_len * seq_len),
                        where seq_len is the length of the acoustic model output (T).

                        NOTE:
                            If a float is provided, it can be greater than 1!
                            By default, a float of 2.0 is used so that a target sequence can be at most twice
                            as long as the acoustic model output length T.

                    maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient,
                        and can be reduced to 1 to improve decoding speed while sacrificing some accuracy. int > 0.

                    maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to
                    keep this as 1 in order to reduce expensive beam search cost later. int >= 0.

                    maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                        Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                        and affects the speed of inference since large values will perform large beam search in the
                        next step.

                    maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the
                        expansions. The default (2.3) is selected from the paper. It performs a comparison
                        (max_log_prob - gamma <= log_prob[v]) where v is all vocabulary indices in the Vocab set and
                        max_log_prob is the "most" likely token to be predicted. Gamma therefore provides a margin of
                        additional tokens which can be potential candidates for expansion apart from the "most likely"
                        candidate. Lower values will reduce the number of expansions (by increasing pruning-by-value,
                        thereby improving speed but hurting accuracy). Higher values will increase the number of
                        expansions (by reducing pruning-by-value, thereby reducing speed but potentially improving
                        accuracy). This is a hyper parameter to be experimentally tuned on a validation set.

                    softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        vocabulary: The vocabulary (excluding the RNNT blank token) which will be used for decoding.
    """

    def __init__(
        self,
        decoding_cfg,
        decoder,
        joint,
        vocabulary,
    ):
        # we need to ensure blank is the last token in the vocab for the case of RNNT and Multi-blank RNNT.
        blank_id = len(vocabulary) + joint.num_extra_outputs
        supported_punctuation = {
            char for token in vocabulary for char in token if unicodedata.category(char).startswith('P')
        }

        if hasattr(decoding_cfg, 'model_type') and decoding_cfg.model_type == 'tdt':
            blank_id = len(vocabulary)

        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        super(RNNTDecoding, self).__init__(
            decoding_cfg=decoding_cfg,
            decoder=decoder,
            joint=joint,
            blank_id=blank_id,
            supported_punctuation=supported_punctuation,
        )

        if isinstance(self.decoding, rnnt_beam_decoding.BeamRNNTInfer) or isinstance(
            self.decoding, tdt_beam_decoding.BeamTDTInfer
        ):
            self.decoding.set_decoding_type('char')

    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """
        Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_chars(hypothesis.words, hypothesis.token_confidence)

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
        token_list = [self.labels_map[c] for c in tokens if c < self.blank_id - self.num_extra_outputs]
        return token_list

    def decode_tokens_to_lang(self, tokens: List[int]) -> str:
        """
        Compute the most likely language ID (LID) string given the tokens.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded LID string.
        """
        lang = self.tokenizer.ids_to_lang(tokens)
        return lang

    def decode_ids_to_langs(self, tokens: List[int]) -> List[str]:
        """
        Decode a token id list into language ID (LID) list.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded LIDS.
        """
        lang_list = self.tokenizer.ids_to_text_and_langs(tokens)
        return lang_list

    @staticmethod
    def get_words_offsets(
        char_offsets: List[
            Dict[
                str,
                Union[
                    str,
                    float,
                ],
            ]
        ],
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
        for offset in char_offsets:
            for char in offset['char']:
                state = "DELIMITER" if char == word_delimiter_char else "WORD"

                curr_punctuation = supported_punctuation and char.strip() in supported_punctuation

                # If current character is a punctuation,
                # we add it to the last formed word after removing uts last space (if it exists)
                # If there is already a word being formed, we add the punctuation to it by removing existent space at the end of the word.

                # This is for being consistent with the final hypothesis text,
                # For which we are removing a space before a punctuation.
                if curr_punctuation and state != "DELIMITER":
                    if word:
                        word = word[:-1] if word[-1] == ' ' else word
                        word += char
                    else:
                        last_built_word = word_offsets[-1]
                        last_built_word['end_offset'] = offset['end_offset']
                        if last_built_word['word'][-1] == ' ':
                            last_built_word['word'] = last_built_word['word'][:-1]
                        last_built_word['word'] += char

                    continue

                if state == last_state and state != "DELIMITER":
                    # If we are in the same state as before, we simply repeat what we've done before
                    end_offset = offset["end_offset"]
                    word += char
                else:
                    # Switching state
                    if state == "DELIMITER" and word:
                        # Finishing a word
                        word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                        word = ""
                    else:
                        # Starting a new word
                        start_offset = offset["start_offset"]
                        end_offset = offset["end_offset"]
                        word = char

                last_state = state

        if last_state == "WORD":
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        return word_offsets


class RNNTBPEDecoding(AbstractRNNTDecoding):
    """
    Used for performing RNN-T auto-regressive decoding of the Decoder+Joint network given the encoder state.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.

            strategy:
                str value which represents the type of decoding that can occur.
                Possible values are :

                -   greedy, greedy_batch (for greedy decoding).

                -   beam, tsd, alsd (for beam search decoding).

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `alignments` in it. Here, `alignments` is a List of List of
                Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

                The length of the list corresponds to the Acoustic Length (T).
                Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
                U is the number of target tokens for the current timestep Ti.

            compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrete intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

            compute_langs: a bool flag, which allows to compute language id (LID) information per token,
                word, and the entire sample (most likely language id). The LIDS will be available
                in the returned Hypothesis object as a dictionary

            rnnt_timestamp_type: A str value, which represents the types of timestamps that should be calculated.
                Can take the following values - "char" for character/subword time stamps, "word" for word level
                time stamps and "all" (default), for both character level and word level time stamps.

            word_seperator: Str token representing the seperator between words.

            segment_seperators: List containing tokens representing the seperator(s) between segments.

            segment_gap_threshold: The threshold (in frames) that caps the gap between two words necessary for forming
                the segments.

            preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of ints.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence
                scores. In order to obtain hypotheses with confidence scores, please utilize
                `rnnt_decoder_predictions_tensor` function with the `preserve_frame_confidence` flag set to True.

                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during decoding (sample / batched). When set to true, the Hypothesis will contain
                    the non-null value for `frame_confidence` in it. Here, `alignments` is a List of List of floats.

                    The length of the list corresponds to the Acoustic Length (T).
                    Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
                    U is the number of target tokens for the current timestep Ti.
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
                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word
                    confidence. Valid options are `mean`, `min`, `max`, `prod`.
                tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
                    attached to the regular frame confidence,
                    making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
                method_cfg: A dict-like object which contains the method name and settings to compute per-frame
                    confidence scores.

                    name:
                        The method name (str).
                        Supported values:

                            - 'max_prob' for using the maximum token probability as a confidence.

                            - 'entropy' for using a normalized entropy of a log-likelihood vector.

                    entropy_type: Which type of entropy to use (str).
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

                    alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                        When the alpha equals one, scaling is not applied to 'max_prob',
                        and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

                    entropy_norm: A mapping of the entropy value to the interval [0,1].
                        Supported values:

                            - 'lin' for using the linear mapping.

                            - 'exp' for using exponential mapping with linear shift.

            The config may further contain the following sub-dictionaries:

                "greedy":
                    max_symbols: int, describing the maximum number of target tokens to decode per
                        timestep during greedy decoding. Setting to larger values allows longer sentences
                        to be decoded, at the cost of increased execution time.

                    preserve_frame_confidence: Same as above, overrides above value.

                    confidence_method_cfg: Same as above, overrides confidence_cfg.method_cfg.

                "beam":
                    beam_size: int, defining the beam size for beam search. Must be >= 1.
                        If beam_size == 1, will perform cached greedy search. This might be slightly different
                        results compared to the greedy search above.

                    score_norm: optional bool, whether to normalize the returned beam score in the hypotheses.
                        Set to True by default.

                    return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                        hypotheses after beam search has concluded.

                    tsd_max_sym_exp: optional int, determines number of symmetric expansions of the target symbols
                        per timestep of the acoustic model. Larger values will allow longer sentences to be decoded,
                        at increased cost to execution time.

                    alsd_max_target_len: optional int or float, determines the potential maximum target sequence
                        length.If an integer is provided, it can decode sequences of that particular maximum length.
                        If a float is provided, it can decode sequences of int(alsd_max_target_len * seq_len),
                        where seq_len is the length of the acoustic model output (T).

                        NOTE:
                            If a float is provided, it can be greater than 1!
                            By default, a float of 2.0 is used so that a target sequence can be at most twice
                            as long as the acoustic model output length T.

                    maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient,
                        and can be reduced to 1 to improve decoding speed while sacrificing some accuracy. int > 0.

                    maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to
                        keep this as 1 in order to reduce expensive beam search cost later. int >= 0.

                    maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                        Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                        and affects the speed of inference since large values will perform large beam search in the
                        next step.

                    maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the
                        expansions. The default (2.3) is selected from the paper. It performs a comparison
                        (max_log_prob - gamma <= log_prob[v]) where v is all vocabulary indices in the Vocab set and
                        max_log_prob is the "most" likely token to be predicted. Gamma therefore provides a margin of
                        additional tokens which can be potential candidates for expansion apart from the "most likely"
                        candidate. Lower values will reduce the number of expansions (by increasing pruning-by-value,
                        thereby improving speed but hurting accuracy). Higher values will increase the number of
                        expansions (by reducing pruning-by-value, thereby reducing speed but potentially improving
                        accuracy). This is a hyper parameter to be experimentally tuned on a validation set.

                    softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        tokenizer: The tokenizer which will be used for decoding.
    """

    def __init__(self, decoding_cfg, decoder, joint, tokenizer: TokenizerSpec):
        blank_id = tokenizer.tokenizer.vocab_size  # RNNT or TDT models.

        if hasattr(tokenizer, 'supported_punctuation'):
            supported_punctuation = tokenizer.supported_punctuation
        else:
            supported_punctuation = {
                char for token in tokenizer.vocab for char in token if unicodedata.category(char).startswith('P')
            }

        # multi-blank RNNTs
        if hasattr(decoding_cfg, 'model_type') and decoding_cfg.model_type == 'multiblank':
            blank_id = tokenizer.tokenizer.vocab_size + joint.num_extra_outputs

        self.tokenizer = tokenizer
        self.tokenizer_type = self.define_tokenizer_type(tokenizer.vocab)

        super(RNNTBPEDecoding, self).__init__(
            decoding_cfg=decoding_cfg,
            decoder=decoder,
            joint=joint,
            blank_id=blank_id,
            supported_punctuation=supported_punctuation,
        )

        if isinstance(self.decoding, rnnt_beam_decoding.BeamRNNTInfer) or isinstance(
            self.decoding, tdt_beam_decoding.BeamTDTInfer
        ):
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
        Implemented by subclass in order to reduce token confidence to a word-level confidence.

        **Note**: Only supports Sentencepiece based tokenizers!

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_subwords_sentencepiece(
            hypothesis.words, hypothesis.token_confidence, hypothesis.y_sequence
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

    def decode_tokens_to_lang(self, tokens: List[int]) -> str:
        """
        Compute the most likely language ID (LID) string given the tokens.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded LID string.
        """
        lang = self.tokenizer.ids_to_lang(tokens)
        return lang

    def decode_ids_to_langs(self, tokens: List[int]) -> List[str]:
        """
        Decode a token id list into language ID (LID) list.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded LIDS.
        """
        lang_list = self.tokenizer.ids_to_text_and_langs(tokens)
        return lang_list

    def decode_hypothesis(self, hypotheses_list: List[Hypothesis]) -> List[Union[Hypothesis, NBestHypotheses]]:
        """
        Decode a list of hypotheses into a list of strings.
        Overrides the super() method optionally adding lang information

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of strings.
        """
        hypotheses = super().decode_hypothesis(hypotheses_list)
        if self.compute_langs:
            if isinstance(self.tokenizer, AggregateTokenizer):
                for ind in range(len(hypotheses_list)):
                    # Extract the integer encoded hypothesis
                    prediction = hypotheses_list[ind].y_sequence

                    if type(prediction) != list:
                        prediction = prediction.tolist()

                    # RNN-T sample level is already preprocessed by implicit RNNT decoding
                    # Simply remove any blank tokens
                    prediction = [p for p in prediction if p != self.blank_id]

                    hypotheses[ind].langs = self.decode_tokens_to_lang(prediction)
                    hypotheses[ind].langs_chars = self.decode_ids_to_langs(prediction)
            else:
                logging.warning(
                    "Ignoring request for lang output in hypotheses since the model does not use an aggregate \
                        tokenizer"
                )

        return hypotheses

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

            for char in offset['char']:
                if char == self.blank_id:
                    continue

                char = int(char)
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
class RNNTDecodingConfig:
    """
    RNNT Decoding config
    """

    model_type: str = "rnnt"  # one of "rnnt", "multiblank" or "tdt"
    strategy: str = "greedy_batch"

    compute_hypothesis_token_set: bool = False

    # preserve decoding alignments
    preserve_alignments: Optional[bool] = None

    # include token duration
    tdt_include_token_duration: Optional[bool] = None

    #  confidence config
    confidence_cfg: ConfidenceConfig = field(default_factory=lambda: ConfidenceConfig())

    # RNNT Joint fused batch size
    fused_batch_size: Optional[int] = None

    # compute RNNT time stamps
    compute_timestamps: Optional[bool] = None

    # compute language IDs
    compute_langs: bool = False

    # token representing word seperator
    word_seperator: str = " "

    # tokens representing segments seperators
    segment_seperators: Optional[List[str]] = field(default_factory=lambda: [".", "!", "?"])

    # threshold (in frames) that caps the gap between two words necessary for forming the segments
    segment_gap_threshold: Optional[int] = None

    # type of timestamps to calculate
    rnnt_timestamp_type: str = "all"  # can be char, word or all for both

    # greedy decoding config
    greedy: rnnt_greedy_decoding.GreedyBatchedRNNTInferConfig = field(
        default_factory=rnnt_greedy_decoding.GreedyBatchedRNNTInferConfig
    )

    # beam decoding config
    beam: rnnt_beam_decoding.BeamRNNTInferConfig = field(
        default_factory=lambda: rnnt_beam_decoding.BeamRNNTInferConfig(beam_size=4)
    )

    # can be used to change temperature for decoding
    temperature: float = 1.0

    # config for TDT decoding.
    durations: Optional[List[int]] = field(default_factory=list)

    # config for multiblank decoding.
    big_blank_durations: Optional[List[int]] = field(default_factory=list)


@dataclass
class RNNTBPEDecodingConfig(RNNTDecodingConfig):
    """
    RNNT BPE Decoding Config
    """

    pass
