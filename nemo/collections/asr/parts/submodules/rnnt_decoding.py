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
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, List, Optional, Set, Union

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig, GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding, rnnt_greedy_decoding, tdt_beam_decoding
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMixin
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BlankLMScoreMode, PruningMode
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.asr.parts.utils.timestamp_utils import get_segment_offsets, get_words_offsets
from nemo.collections.asr.parts.utils.tokenizer_utils import define_spe_tokenizer_type, extract_punctuation_from_vocab
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum

try:
    import kenlm

    KENLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KENLM_AVAILABLE = False


class TransducerModelType(PrettyStrEnum):
    RNNT = "rnnt"
    TDT = "tdt"
    MULTI_BLANK = "multi_blank"


class TransducerDecodingStrategyType(PrettyStrEnum):
    GREEDY = "greedy"
    GREEDY_BATCH = "greedy_batch"
    BEAM = "beam"
    TSD = "tsd"
    MAES = "maes"
    ALSD = "alsd"
    MALSD_BATCH = "malsd_batch"
    MAES_BATCH = "maes_batch"


TRANSDUCER_SUPPORTED_STRATEGIES: dict[TransducerModelType, set[TransducerDecodingStrategyType]] = {
    TransducerModelType.RNNT: {
        TransducerDecodingStrategyType.GREEDY,
        TransducerDecodingStrategyType.GREEDY_BATCH,
        TransducerDecodingStrategyType.BEAM,
        TransducerDecodingStrategyType.MAES,
        TransducerDecodingStrategyType.ALSD,
        TransducerDecodingStrategyType.TSD,
        TransducerDecodingStrategyType.MALSD_BATCH,
        TransducerDecodingStrategyType.MAES_BATCH,
    },
    TransducerModelType.TDT: {
        TransducerDecodingStrategyType.GREEDY,
        TransducerDecodingStrategyType.GREEDY_BATCH,
        TransducerDecodingStrategyType.BEAM,
        TransducerDecodingStrategyType.MAES,
        TransducerDecodingStrategyType.MALSD_BATCH,
    },
    TransducerModelType.MULTI_BLANK: {
        TransducerDecodingStrategyType.GREEDY,
        TransducerDecodingStrategyType.GREEDY_BATCH,
    },
}


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
        self._with_multiple_blanks = self.big_blank_durations is not None and len(self.big_blank_durations) > 0

        if self._is_tdt:
            if blank_id == 0:
                raise ValueError("blank_id must equal len(non_blank_vocabs) for TDT models")
            if self._with_multiple_blanks:
                raise ValueError("duration and big_blank_durations can't both be not None")

        if self._with_multiple_blanks and blank_id == 0:
            raise ValueError("blank_id must equal len(vocabs) for multi-blank RNN-T models")

        strategy = TransducerDecodingStrategyType(self.cfg.strategy)

        if self._is_tdt:
            model_type = TransducerModelType.TDT
        elif self._with_multiple_blanks:
            model_type = TransducerModelType.MULTI_BLANK
        else:
            model_type = TransducerModelType.RNNT

        self._model_type = model_type
        self._decoding_strategy_type = strategy

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

        # initialize confidence-related fields
        self._init_confidence(self.cfg.get('confidence_cfg', None))

        if model_type is TransducerModelType.TDT:
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

        if strategy in {TransducerDecodingStrategyType.GREEDY, TransducerDecodingStrategyType.GREEDY_BATCH}:
            ngram_lm_model = self.cfg.greedy.get('ngram_lm_model', None)
            ngram_lm_alpha = self.cfg.greedy.get('ngram_lm_alpha', 0)
            boosting_tree = self.cfg.greedy.get('boosting_tree', None)
            boosting_tree_alpha = self.cfg.greedy.get('boosting_tree_alpha', 0)
        else:
            ngram_lm_model = self.cfg.beam.get('ngram_lm_model', None)
            ngram_lm_alpha = self.cfg.beam.get('ngram_lm_alpha', 0)
            boosting_tree = self.cfg.beam.get('boosting_tree', None)
            boosting_tree_alpha = self.cfg.beam.get('boosting_tree_alpha', 0)

        # load fusion models from paths (ngram_lm_model and boosting_tree_model)
        fusion_models, fusion_models_alpha = [], []
        # load ngram_lm model from path
        if ngram_lm_model is not None:
            if strategy is TransducerDecodingStrategyType.MAES:
                fusion_models.append(self._load_kenlm_model(ngram_lm_model))
            else:
                fusion_models.append(NGramGPULanguageModel.from_file(lm_path=ngram_lm_model, vocab_size=self.blank_id))
            fusion_models_alpha.append(ngram_lm_alpha)
        # load boosting tree model from path
        if boosting_tree and not BoostingTreeModelConfig.is_empty(boosting_tree):
            if strategy is TransducerDecodingStrategyType.MAES:
                raise NotImplementedError(
                    f"Model {model_type} with strategy `{strategy}` does not support boosting tree."
                )
            fusion_models.append(
                GPUBoostingTreeModel.from_config(boosting_tree, tokenizer=getattr(self, 'tokenizer', None))
            )
            fusion_models_alpha.append(boosting_tree_alpha)
        if not fusion_models:
            fusion_models = None
            fusion_models_alpha = None

        match strategy, model_type:
            # greedy strategy
            case TransducerDecodingStrategyType.GREEDY, TransducerModelType.RNNT:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                        f"Recommended greedy strategy with LM is `greedy_batch`."
                    )
                self.decoding = rnnt_greedy_decoding.GreedyRNNTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None) or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    preserve_alignments=self.preserve_alignments,
                    preserve_frame_confidence=self.preserve_frame_confidence,
                    confidence_method_cfg=self.confidence_method_cfg,
                )
            case TransducerDecodingStrategyType.GREEDY, TransducerModelType.TDT:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree. "
                        f"Recommended greedy strategy with LM is `greedy_batch`."
                    )
                self.decoding = rnnt_greedy_decoding.GreedyTDTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    durations=self.durations,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None) or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    preserve_alignments=self.preserve_alignments,
                    preserve_frame_confidence=self.preserve_frame_confidence,
                    include_duration=self.tdt_include_token_duration,
                    include_duration_confidence=self.tdt_include_duration_confidence,
                    confidence_method_cfg=self.confidence_method_cfg,
                )
            case TransducerDecodingStrategyType.GREEDY, TransducerModelType.MULTI_BLANK:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                    )
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
            # greedy_batch strategy
            case TransducerDecodingStrategyType.GREEDY_BATCH, TransducerModelType.RNNT:
                self.decoding = rnnt_greedy_decoding.GreedyBatchedRNNTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None) or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    preserve_alignments=self.preserve_alignments,
                    preserve_frame_confidence=self.preserve_frame_confidence,
                    confidence_method_cfg=self.confidence_method_cfg,
                    loop_labels=self.cfg.greedy.get('loop_labels', True),
                    use_cuda_graph_decoder=self.cfg.greedy.get('use_cuda_graph_decoder', True),
                    fusion_models=fusion_models,
                    fusion_models_alpha=fusion_models_alpha,
                    enable_per_stream_biasing=self.cfg.greedy.get('enable_per_stream_biasing', False),
                )
            case TransducerDecodingStrategyType.GREEDY_BATCH, TransducerModelType.TDT:
                self.decoding = rnnt_greedy_decoding.GreedyBatchedTDTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    durations=self.durations,
                    max_symbols_per_step=(
                        self.cfg.greedy.get('max_symbols', None) or self.cfg.greedy.get('max_symbols_per_step', None)
                    ),
                    preserve_alignments=self.preserve_alignments,
                    preserve_frame_confidence=self.preserve_frame_confidence,
                    include_duration=self.tdt_include_token_duration,
                    include_duration_confidence=self.tdt_include_duration_confidence,
                    confidence_method_cfg=self.confidence_method_cfg,
                    use_cuda_graph_decoder=self.cfg.greedy.get('use_cuda_graph_decoder', True),
                    fusion_models=fusion_models,
                    fusion_models_alpha=fusion_models_alpha,
                    enable_per_stream_biasing=self.cfg.greedy.get('enable_per_stream_biasing', False),
                )
            case TransducerDecodingStrategyType.GREEDY_BATCH, TransducerModelType.MULTI_BLANK:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                    )
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
            # beam, maes, alsd, tsd strategies
            case TransducerDecodingStrategyType.BEAM, TransducerModelType.RNNT:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                        f"Recommended beam decoding strategy with LM is `malsd_batch`."
                    )
                logging.warning(
                    f"Decoding strategy `{strategy}` is experimental. "
                    "Recommended beam decoding strategy is `malsd_batch`."
                )
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
            case TransducerDecodingStrategyType.BEAM, TransducerModelType.TDT:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                        f"Recommended beam decoding strategy with LM is `malsd_batch`."
                    )
                logging.warning(
                    f"Decoding strategy `{strategy}` is experimental. "
                    "Recommended beam decoding strategy is `malsd_batch`."
                )
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
            case TransducerDecodingStrategyType.TSD, TransducerModelType.RNNT:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                        f"Recommended beam decoding strategy with LM is `malsd_batch`."
                    )
                logging.warning(
                    f"Decoding strategy `{strategy}` is experimental. "
                    "Recommended beam decoding strategy is `malsd_batch`."
                )
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
            case TransducerDecodingStrategyType.ALSD, TransducerModelType.RNNT:
                if fusion_models is not None:
                    raise NotImplementedError(
                        f"Model {model_type} with strategy `{strategy}` does not support n-gram LM models and boosting tree."
                        f"Recommended beam decoding strategy with LM is `malsd_batch`."
                    )
                logging.warning(
                    f"Decoding strategy `{strategy}` is experimental. "
                    "Recommended beam decoding strategy is `malsd_batch`."
                )
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
            case TransducerDecodingStrategyType.MAES, TransducerModelType.RNNT:
                logging.warning(
                    f"Decoding strategy `{strategy}` is experimental. "
                    "Recommended beam decoding strategy is `malsd_batch`."
                )
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
                    ngram_lm_model=fusion_models[0] if fusion_models is not None else None,
                    ngram_lm_alpha=fusion_models_alpha[0] if fusion_models_alpha is not None else 0.0,
                    hat_subtract_ilm=self.cfg.beam.get('hat_subtract_ilm', False),
                    hat_ilm_weight=self.cfg.beam.get('hat_ilm_weight', 0.0),
                )
            case TransducerDecodingStrategyType.MAES, TransducerModelType.TDT:
                logging.warning(
                    f"Decoding strategy `{strategy}` is experimental. "
                    "Recommended beam decoding strategy is `malsd_batch`."
                )
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
                    ngram_lm_model=fusion_models[0] if fusion_models is not None else None,
                    ngram_lm_alpha=fusion_models_alpha[0] if fusion_models_alpha is not None else 0.0,
                )
            # beam batch: malsd_batch and maes_batch strategies
            case TransducerDecodingStrategyType.MALSD_BATCH, TransducerModelType.RNNT:
                self.decoding = rnnt_beam_decoding.BeamBatchedRNNTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    beam_size=self.cfg.beam.beam_size,
                    search_type='malsd_batch',
                    max_symbols_per_step=self.cfg.beam.get("max_symbols", 10),
                    preserve_alignments=self.preserve_alignments,
                    fusion_models=fusion_models,
                    fusion_models_alpha=fusion_models_alpha,
                    blank_lm_score_mode=self.cfg.beam.get('blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL),
                    pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                    score_norm=self.cfg.beam.get('score_norm', True),
                    allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', True),
                    return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                )
            case TransducerDecodingStrategyType.MALSD_BATCH, TransducerModelType.TDT:
                self.decoding = tdt_beam_decoding.BeamBatchedTDTInfer(
                    decoder_model=decoder,
                    joint_model=joint,
                    blank_index=self.blank_id,
                    durations=self.durations,
                    beam_size=self.cfg.beam.beam_size,
                    search_type='malsd_batch',
                    max_symbols_per_step=self.cfg.beam.get("max_symbols", 10),
                    preserve_alignments=self.preserve_alignments,
                    fusion_models=fusion_models,
                    fusion_models_alpha=fusion_models_alpha,
                    blank_lm_score_mode=self.cfg.beam.get('blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL),
                    pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                    score_norm=self.cfg.beam.get('score_norm', True),
                    allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', True),
                    return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                )
            case TransducerDecodingStrategyType.MAES_BATCH, TransducerModelType.RNNT:
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
                    fusion_models=fusion_models,
                    fusion_models_alpha=fusion_models_alpha,
                    blank_lm_score_mode=self.cfg.beam.get('blank_lm_score_mode', BlankLMScoreMode.LM_WEIGHTED_FULL),
                    pruning_mode=self.cfg.beam.get('pruning_mode', PruningMode.LATE),
                    score_norm=self.cfg.beam.get('score_norm', True),
                    allow_cuda_graphs=self.cfg.beam.get('allow_cuda_graphs', False),
                    return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                )
            case _, _:
                raise NotImplementedError(
                    f"Transducer model of {model_type} type does not support {strategy} strategy. "
                    f"Supported strategies: {', '.join(map(str, TRANSDUCER_SUPPORTED_STRATEGIES[model_type]))}"
                )

        # Update the joint fused batch size or disable it entirely if needed.
        self.update_joint_fused_batch_size()

    @abstractproperty
    def tokenizer_type(self):
        """
        Implemented by subclass in order to get tokenizer type information for timestamps extraction.
        """
        raise NotImplementedError()

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
        for hyp in hypotheses_list:
            # Extract the integer encoded hypothesis
            prediction = hyp.y_sequence

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

            # De-tokenize the integer tokens;
            hyp.text = self.decode_tokens_to_str_with_strip_punctuation(prediction)

            if self.compute_hypothesis_token_set:
                hyp.tokens = self.decode_ids_to_tokens(prediction)

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

    def decode_ids_to_str(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens ids to a string.
        """
        if hasattr(self, 'tokenizer') and isinstance(self.tokenizer, AggregateTokenizer):
            return self.tokenizer.ids_to_text(tokens)
        else:
            return self.decode_tokens_to_str(self.decode_ids_to_tokens(tokens))

    def decode_tokens_to_str_with_strip_punctuation(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens to a string and removes a space before supported punctuation marks.
        """
        text = self.decode_ids_to_str(tokens)
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

        # Retrieve offsets
        char_offsets = word_offsets = None
        char_offsets = self._compute_offsets(hypothesis, self.blank_id)
        y_sequence_blank_removed = [t for t in hypothesis.y_sequence if t != self.blank_id]

        if len(char_offsets) != len(y_sequence_blank_removed):
            raise ValueError(
                f"`char_offsets`: {char_offsets} and `processed_tokens`: {y_sequence_blank_removed}"
                " have to be of the same length, but are: "
                f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                f" {len(y_sequence_blank_removed)}"
            )

        encoded_char_offsets = copy.deepcopy(char_offsets)

        # Correctly process the token ids to chars/subwords.
        for i, offsets in enumerate(char_offsets):
            chars_text = []
            chars_tokens = []
            for char in offsets['char']:
                # NB: if blank tokens are present, _refine_timestamps will not work properly
                # as offests and encoded_offsets will not be 1:1 match
                assert char != self.blank_id, "Offsets should not contain blank tokens"
                chars_tokens.append(self.decode_ids_to_tokens([int(char)])[0])
                chars_text.append(self.decode_ids_to_str([int(char)]))
            char_offsets[i]["char"] = chars_text
            encoded_char_offsets[i]["char"] = chars_tokens

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
            word_offsets = get_words_offsets(
                char_offsets=char_offsets,
                encoded_char_offsets=encoded_char_offsets,
                word_delimiter_char=self.word_seperator,
                supported_punctuation=self.supported_punctuation,
                tokenizer_type=self.tokenizer_type,
                decode_tokens_to_str=self.decode_tokens_to_str,
            )

        segment_offsets = None
        if timestamp_type in ['segment', 'all']:
            segment_offsets = get_segment_offsets(
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

        return hypothesis

    @staticmethod
    def _compute_offsets(hypothesis: Hypothesis, blank_id: int) -> List[Dict[str, Union[str, int]]]:
        """
        Utility method that calculates the indidual time indices where a token starts and ends.

        Args:
            hypothesis: A Hypothesis object that contains `text` field that holds the character / subword token
                emitted at every time step after rnnt collapse.

        Returns:
            List[Dict[str, Union[str, int]]]: A list of dictionaries, where each dictionary contains:
                - "char": List[str] - The character/subword token
                - "start_offset": int - The start time index of the token
                - "end_offset": int - The end time index of the token

        **Note**: Blank tokens are not included in the offsets.
        """
        if isinstance(hypothesis.timestamp, torch.Tensor):
            hypothesis.timestamp = hypothesis.timestamp.cpu().tolist()

        # Merge the results per token into a list of dictionaries
        offsets = [
            {"char": [t], "start_offset": s, "end_offset": s + 1}
            for t, s in zip(hypothesis.y_sequence, hypothesis.timestamp)
            if t != blank_id
        ]

        return offsets

    @staticmethod
    def _compute_offsets_tdt(hypothesis: Hypothesis, blank_id: int, *args) -> List[Dict[str, Union[str, int]]]:
        """
        Utility method that calculates the indidual time indices where a token starts and ends.

        Args:
            hypothesis: A Hypothesis object that contains `text` field that holds the character / subword token
                emitted at a specific time step considering predicted durations of the previous tokens.

        Returns:
            List[Dict[str, Union[str, int]]]: A list of dictionaries, where each dictionary contains:
                - "char": List[str] - The character/subword token
                - "start_offset": int - The start time index of the token
                - "end_offset": int - The end time index of the token

        **Note**: Blank tokens are not included in the offsets.
        """
        if isinstance(hypothesis.timestamp, torch.Tensor):
            hypothesis.token_duration = hypothesis.token_duration.cpu().tolist()

        if isinstance(hypothesis.timestamp, torch.Tensor):
            hypothesis.timestamp = hypothesis.timestamp.cpu().tolist()

        # Merge the results per token into a list of dictionaries
        offsets = [
            {"char": [t], "start_offset": s, "end_offset": s + d}
            for t, s, d in zip(hypothesis.y_sequence, hypothesis.timestamp, hypothesis.token_duration)
            if t != blank_id
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
    def _load_kenlm_model(ngram_lm_model: str):
        """
        Load a KenLM model from a file path.
        """
        if KENLM_AVAILABLE:
            return kenlm.Model(ngram_lm_model)
        else:
            raise ImportError(
                "KenLM package (https://github.com/kpu/kenlm) is not installed. " "Use ngram_lm_model=None."
            )


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
        supported_punctuation = extract_punctuation_from_vocab(vocabulary)

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

    @property
    def tokenizer_type(self):
        return "char"

    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """
        Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        return self._aggregate_token_confidence_chars(hypothesis.words, hypothesis.token_confidence)

    def decode_tokens_to_str(self, tokens: List[str]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = ''.join(tokens)
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
            supported_punctuation = extract_punctuation_from_vocab(tokenizer.vocab)

        # multi-blank RNNTs
        if hasattr(decoding_cfg, 'model_type') and decoding_cfg.model_type == 'multiblank':
            blank_id = tokenizer.tokenizer.vocab_size + joint.num_extra_outputs

        self.tokenizer = tokenizer

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

    @property
    def tokenizer_type(self):
        return define_spe_tokenizer_type(self.tokenizer.vocab)

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

    def decode_tokens_to_str(self, tokens: List[str]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of str representing the tokens.

        Returns:
            A decoded string.
        """
        hypothesis = self.tokenizer.tokens_to_text(tokens)
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
