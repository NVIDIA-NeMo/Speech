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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging


class ConfidenceMethodConstants:
    NAMES = ("max_prob", "entropy")
    ENTROPY_TYPES = ("gibbs", "tsallis", "renyi")
    ENTROPY_NORMS = ("lin", "exp")

    @classmethod
    def print(cls):
        return (
            cls.__name__
            + ": "
            + str({"NAMES": cls.NAMES, "ENTROPY_TYPES": cls.ENTROPY_TYPES, "ENTROPY_NORMS": cls.ENTROPY_NORMS})
        )


class ConfidenceConstants:
    AGGREGATIONS = ("mean", "min", "max", "prod")

    @classmethod
    def print(cls):
        return cls.__name__ + ": " + str({"AGGREGATIONS": cls.AGGREGATIONS})


@dataclass
class ConfidenceMethodConfig:
    """A Config which contains the method name and settings to compute per-frame confidence scores.

    Args:
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
    """

    name: str = "entropy"
    entropy_type: str = "tsallis"
    alpha: float = 0.33
    entropy_norm: str = "exp"
    temperature: str = "DEPRECATED"

    def __post_init__(self):
        if self.temperature != "DEPRECATED":
            # self.temperature has type str
            self.alpha = float(self.temperature)
            self.temperature = "DEPRECATED"
        if self.name not in ConfidenceMethodConstants.NAMES:
            raise ValueError(
                f"`name` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.NAMES) + '`'}. Provided: `{self.name}`"
            )
        if self.entropy_type not in ConfidenceMethodConstants.ENTROPY_TYPES:
            raise ValueError(
                f"`entropy_type` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_TYPES) + '`'}. Provided: `{self.entropy_type}`"
            )
        if self.alpha <= 0.0:
            raise ValueError(f"`alpha` must be > 0. Provided: {self.alpha}")
        if self.entropy_norm not in ConfidenceMethodConstants.ENTROPY_NORMS:
            raise ValueError(
                f"`entropy_norm` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_NORMS) + '`'}. Provided: `{self.entropy_norm}`"
            )


@dataclass
class ConfidenceConfig:
    """A config which contains the following key-value pairs related to confidence scores.

    Args:
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
        tdt_include_duration: Bool flag indicating that the duration confidence scores are to be calculated and
            attached to the regular frame confidence,
            making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
        method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
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
    """

    preserve_frame_confidence: bool = False
    preserve_token_confidence: bool = False
    preserve_word_confidence: bool = False
    exclude_blank: bool = True
    aggregation: str = "min"
    tdt_include_duration: bool = False
    method_cfg: ConfidenceMethodConfig = field(default_factory=lambda: ConfidenceMethodConfig())

    def __post_init__(self):
        # OmegaConf.structured ensures that post_init check is always executed
        self.method_cfg = OmegaConf.structured(
            self.method_cfg
            if isinstance(self.method_cfg, ConfidenceMethodConfig)
            else ConfidenceMethodConfig(**self.method_cfg)
        )
        if self.aggregation not in ConfidenceConstants.AGGREGATIONS:
            raise ValueError(
                f"`aggregation` has to be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceConstants.AGGREGATIONS) + '`'}. Provided: `{self.aggregation}`"
            )


def get_confidence_measure_bank():
    """Generate a dictionary with confidence measure functionals.

    Supported confidence measures:
        max_prob: normalized maximum probability
        entropy_gibbs_lin: Gibbs entropy with linear normalization
        entropy_gibbs_exp: Gibbs entropy with exponential normalization
        entropy_tsallis_lin: Tsallis entropy with linear normalization
        entropy_tsallis_exp: Tsallis entropy with exponential normalization
        entropy_renyi_lin: Rényi entropy with linear normalization
        entropy_renyi_exp: Rényi entropy with exponential normalization

    Returns:
        dictionary with lambda functions.
    """
    # helper functions
    # Gibbs entropy is implemented without alpha
    neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
    neg_entropy_alpha = lambda x, t: (x * t).exp().sum(-1)
    neg_entropy_alpha_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)
    # too big for a lambda
    def entropy_tsallis_exp(x, v, t):
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
        return (((1 - neg_entropy_alpha(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    def entropy_gibbs_exp(x, v, t):
        exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
        return ((neg_entropy_alpha_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    # use Gibbs entropies for Tsallis and Rényi with t == 1.0
    entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
    entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)
    # fill the measure bank
    confidence_measure_bank = {}
    # Maximum probability measure is implemented without alpha
    confidence_measure_bank["max_prob"] = (
        lambda x, v, t: (x.max(dim=-1)[0].exp() * v - 1) / (v - 1)
        if t == 1.0
        else ((x.max(dim=-1)[0] * t).exp() * math.pow(v, t) - 1) / (math.pow(v, t) - 1)
    )
    confidence_measure_bank["entropy_gibbs_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha_gibbs(x, t) / math.log(v) / math.pow(v, 1 - t)
    )
    confidence_measure_bank["entropy_gibbs_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_gibbs_exp(x, v, t)
    )
    confidence_measure_bank["entropy_tsallis_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + (1 - neg_entropy_alpha(x, t)) / (math.pow(v, 1 - t) - 1)
    )
    confidence_measure_bank["entropy_tsallis_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_tsallis_exp(x, v, t)
    )
    confidence_measure_bank["entropy_renyi_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha(x, t).log2() / (t - 1) / math.log(v, 2)
    )
    confidence_measure_bank["entropy_renyi_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
        if t == 1.0
        else (neg_entropy_alpha(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
    )
    return confidence_measure_bank


def get_confidence_aggregation_bank():
    """Generate a dictionary with confidence aggregation functions.

    Supported confidence aggregation functions:
        min: minimum
        max: maximum
        mean: arithmetic mean
        prod: product

    Returns:
        dictionary with functions.
    """
    confidence_aggregation_bank = {"mean": lambda x: sum(x) / len(x), "min": min, "max": max}
    # python 3.7 and earlier do not have math.prod
    if hasattr(math, "prod"):
        confidence_aggregation_bank["prod"] = math.prod
    else:
        import operator
        from functools import reduce

        confidence_aggregation_bank["prod"] = lambda x: reduce(operator.mul, x, 1)
    return confidence_aggregation_bank


class ConfidenceMethodMixin(ABC):
    """Confidence Method Mixin class.

    It initializes per-frame confidence method.
    """

    def _init_confidence_method(self, confidence_method_cfg: Optional[DictConfig] = None):
        """Initialize per-frame confidence method from config.
        """
        # OmegaConf.structured ensures that post_init check is always executed
        confidence_method_cfg = OmegaConf.structured(
            ConfidenceMethodConfig()
            if confidence_method_cfg is None
            else ConfidenceMethodConfig(**confidence_method_cfg)
        )

        # set confidence calculation method
        # we suppose that self.blank_id == len(vocabulary)
        self.num_tokens = (self.blank_id if hasattr(self, "blank_id") else self._blank_index) + 1
        self.alpha = confidence_method_cfg.alpha

        # init confidence measure bank
        self.confidence_measure_bank = get_confidence_measure_bank()

        measure = None
        # construct measure_name
        measure_name = ""
        if confidence_method_cfg.name == "max_prob":
            measure_name = "max_prob"
        elif confidence_method_cfg.name == "entropy":
            measure_name = '_'.join(
                [confidence_method_cfg.name, confidence_method_cfg.entropy_type, confidence_method_cfg.entropy_norm]
            )
        else:
            raise ValueError(f"Unsupported `confidence_method_cfg.name`: `{confidence_method_cfg.name}`")
        if measure_name not in self.confidence_measure_bank:
            raise ValueError(f"Unsupported measure setup: `{measure_name}`")
        measure = partial(self.confidence_measure_bank[measure_name], v=self.num_tokens, t=self.alpha)

        self._confidence_measure = measure

    def _get_confidence(self, x: torch.Tensor) -> list[float]:
        """Compute confidence, return list of confidence items for each item in batch"""
        return self._get_confidence_tensor(x).tolist()

    def _get_confidence_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute confidence, return tensor"""
        return self._confidence_measure(torch.nan_to_num(x))


class ConfidenceMixin(ABC):
    """Confidence Mixin class.

    It is responsible for confidence estimation method initialization and high-level confidence score calculation.
    """

    def _init_confidence(self, confidence_cfg: Optional[DictConfig] = None):
        """Initialize confidence-related fields and confidence aggregation function from config.
        """
        # OmegaConf.structured ensures that post_init check is always executed
        confidence_cfg = OmegaConf.structured(
            ConfidenceConfig() if confidence_cfg is None else ConfidenceConfig(**confidence_cfg)
        )
        self.confidence_method_cfg = confidence_cfg.method_cfg

        # extract the config
        self.preserve_word_confidence = confidence_cfg.get('preserve_word_confidence', False)
        # set preserve_frame_confidence and preserve_token_confidence to True
        # if preserve_word_confidence is True
        self.preserve_token_confidence = (
            confidence_cfg.get('preserve_token_confidence', False) | self.preserve_word_confidence
        )
        # set preserve_frame_confidence to True if preserve_token_confidence is True
        self.preserve_frame_confidence = (
            confidence_cfg.get('preserve_frame_confidence', False) | self.preserve_token_confidence
        )
        self.exclude_blank_from_confidence = confidence_cfg.get('exclude_blank', True)
        self.tdt_include_duration_confidence = confidence_cfg.get('tdt_include_duration', False)
        self.word_confidence_aggregation = confidence_cfg.get('aggregation', "min")

        # define aggregation functions
        self.confidence_aggregation_bank = get_confidence_aggregation_bank()
        self._aggregate_confidence = self.confidence_aggregation_bank[self.word_confidence_aggregation]

        # Update preserve frame confidence
        if self.cfg.strategy in ['greedy', 'greedy_batch']:
            if not self.preserve_frame_confidence:
                self.preserve_frame_confidence = self.cfg.greedy.get('preserve_frame_confidence', False)
                # OmegaConf.structured ensures that post_init check is always executed
                confidence_method_cfg = OmegaConf.structured(self.cfg.greedy).get('confidence_method_cfg', None)
                self.confidence_method_cfg = (
                    OmegaConf.structured(ConfidenceMethodConfig())
                    if confidence_method_cfg is None
                    else OmegaConf.structured(ConfidenceMethodConfig(**confidence_method_cfg))
                )
            if not self.tdt_include_duration_confidence:
                self.tdt_include_duration_confidence = self.cfg.greedy.get('tdt_include_duration_confidence', False)

    @abstractmethod
    def compute_confidence(self, hypotheses_list: List[Hypothesis]) -> List[Hypothesis]:
        """Computes high-level (per-token and/or per-word) confidence scores for a list of hypotheses.
        Assumes that `frame_confidence` is present in the hypotheses.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of hypotheses with high-level confidence scores.
        """
        raise NotImplementedError()

    @abstractmethod
    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        raise NotImplementedError()

    def _aggregate_token_confidence_chars(self, words: List[str], token_confidence: List[float]) -> List[float]:
        """Implementation of token confidence aggregation for character-based models.

        Args:
            words: List of words of a hypothesis.
            token_confidence: List of token-level confidence scores of a hypothesis.

        Returns:
            A list of word-level confidence scores.
        """
        word_confidence = []
        i = 0
        for word in words:
            word_len = len(word)
            word_confidence.append(self._aggregate_confidence(token_confidence[i : i + word_len]))
            # we assume that there is exactly one space token between words and exclude it from word confidence
            i += word_len + 1
        return word_confidence

    def _aggregate_token_confidence_subwords_sentencepiece(
        self, words: List[str], token_confidence: List[float], token_ids: List[int]
    ) -> List[float]:
        """Implementation of token confidence aggregation for subword-based models.

        **Note**: Only supports Sentencepiece based tokenizers !

        Args:
            words: List of words of a hypothesis.
            token_confidence: List of token-level confidence scores of a hypothesis.
            token_ids: List of token ids of a hypothesis.

        Returns:
            A list of word-level confidence scores.
        """
        word_confidence = []
        # run only if there are final words
        if len(words) > 0:
            j = 0
            prev_unk = False
            prev_underline = False
            for i, token_id in enumerate(token_ids):
                token = self.decode_ids_to_tokens([int(token_id)])[0]
                token_text = self.decode_tokens_to_str([int(token_id)])
                # treat `<unk>` as a separate word regardless of the next token
                # to match the result of `tokenizer.ids_to_text`
                if (token != token_text or prev_unk) and i > j:
                    # do not add confidence for `▁` if the current token starts with `▁`
                    # to match the result of `tokenizer.ids_to_text`
                    if not prev_underline:
                        word_confidence.append(self._aggregate_confidence(token_confidence[j:i]))
                    j = i
                prev_unk = token == '<unk>'
                prev_underline = token == '▁'
            if not prev_underline:
                word_confidence.append(self._aggregate_confidence(token_confidence[j : len(token_ids)]))
        if len(words) != len(word_confidence):
            raise RuntimeError(
                f"""Something went wrong with word-level confidence aggregation.\n
            Please check these values for debugging:\n
            len(words): {len(words)},\n
            len(word_confidence): {len(word_confidence)},\n
            recognized text: `{' '.join(words)}`"""
            )
        return word_confidence
