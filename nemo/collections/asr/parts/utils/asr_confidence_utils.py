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
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


@dataclass
class ConfidenceMethodConfig:
    name: str = "entropy"
    entropy_type: str = "tsallis"
    temperature: float = 0.33
    entropy_norm: str = "exp"

    def __post_init__(self):
        if self.name not in ("max_prob", "entropy"):
            raise ValueError(f"`name` has to be one of the following: `max_prob`, `entropy`. Provided: {self.name}")
        if self.entropy_type not in ("gibbs", "tsallis", "renui"):
            raise ValueError(
                f"`entropy_type` has to be one of the following: `gibbs`, `tsallis`, `renui`. Provided: {self.entropy_type}"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"`temperature` has to be > 0. Provided: {self.temperature}")
        if self.entropy_norm not in ("lin", "exp"):
            raise ValueError(
                f"`entropy_norm` has to be one of the following: `lin`, `exp`. Provided: {self.entropy_norm}"
            )


@dataclass
class ConfidenceConfig:
    preserve_frame_confidence: bool = False
    preserve_token_confidence: bool = False
    preserve_word_confidence: bool = False
    exclude_blank: bool = True
    aggregation: str = "min"
    method_cfg: ConfidenceMethodConfig = ConfidenceMethodConfig()

    def __post_init__(self):
        if self.aggregation not in ("mean", "min", "max", "prod"):
            raise ValueError(
                f"`aggregation` has to be one of the following: `mean`, `min`, `max`, `prod`. Provided: {self.aggregation}"
            )


def get_confidence_measure_bank():
    """Generate a dictionary with confidence measure functionals.

    Supported confidence measures:
        max_prob: normalized maximum probability
        entropy_gibbs_lin: Gibbs entropy with linear normalization
        entropy_gibbs_exp: Gibbs entropy with exponential normalization
        entropy_tsallis_lin: Tsallis entropy with linear normalization
        entropy_tsallis_exp: Tsallis entropy with exponential normalization
        entropy_renui_lin: Rényi entropy with linear normalization
        entropy_renui_exp: Rényi entropy with exponential normalization

    Returns:
        dictionary with lambda functions.
    """
    # helper functions
    # Gibbs entropy is implemented without temperature
    neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
    neg_entropy_temperature = lambda x, t: (x * t).exp().sum(-1)
    neg_entropy_temperature_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)
    # too big for a lambda
    def entropy_tsallis_exp(x, v, t):
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
        return (((1 - neg_entropy_temperature(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    def entropy_gibbs_exp(x, v, t):
        exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
        return ((neg_entropy_temperature_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    # use Gibbs entropies for Tsallis and Rényi with t == 1.0
    entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
    entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)
    # fill the measure bank
    confidence_measure_bank = {}
    # Maximum probability measure is implemented without temperature
    confidence_measure_bank["max_prob"] = (
        lambda x, v, t: (x.max(dim=-1)[0].exp() * v - 1) / (v - 1)
        if t == 1.0
        else ((x.max(dim=-1)[0] * t).exp() * math.pow(v, t) - 1) / (math.pow(v, t) - 1)
    )
    confidence_measure_bank["entropy_gibbs_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_temperature_gibbs(x, t) / math.log(v) / math.pow(v, 1 - t)
    )
    confidence_measure_bank["entropy_gibbs_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_gibbs_exp(x, v, t)
    )
    confidence_measure_bank["entropy_tsallis_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + (1 - neg_entropy_temperature(x, t)) / (math.pow(v, 1 - t) - 1)
    )
    confidence_measure_bank["entropy_tsallis_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_tsallis_exp(x, v, t)
    )
    confidence_measure_bank["entropy_renui_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_temperature(x, t).log2() / (t - 1) / math.log(v, 2)
    )
    confidence_measure_bank["entropy_renui_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
        if t == 1.0
        else (neg_entropy_temperature(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
    )
    return confidence_measure_bank


def get_confidence_aggregation_bank():
    """Generate a dictionary with confidence aggregation functions.

    Supported confidence measures:
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


class ConfidenceMeasureMixin(ABC):
    """Confidence Measure Mixin class.

    It initializes per-frame confidence measure.
    """

    def _init_confidence_measure(self, confidence_method_cfg: Optional[DictConfig] = None):
        """Initialize per-frame confidence measure from config.
        """
        if confidence_method_cfg is None:
            confidence_method_cfg = OmegaConf.structured(ConfidenceMethodConfig())

        # set confidence calculation method
        # we suppose that self.blank_id == len(vocabulary)
        self.num_tokens = (self.blank_id if hasattr(self, "blank_id") else self._blank_index) + 1
        self.temperature = confidence_method_cfg.temperature

        # init confidence measure bank
        self.confidence_measure_bank = get_confidence_measure_bank()

        method = None
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
        method = partial(self.confidence_measure_bank[measure_name], v=self.num_tokens, t=self.temperature)
        self._get_confidence = lambda x: method(x).tolist()


class ConfidenceMixin(ABC):
    """Confidence Mixin class.

    It initializes per-frame confidence measure.
    """

    def _init_confidence(self, confidence_cfg: Optional[DictConfig] = None):
        """Initialize confidence-related fields and confidence aggregation function from config.
        """
        if confidence_cfg is None:
            confidence_cfg = OmegaConf.structured(ConfidenceConfig())

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
        self.word_confidence_aggregation = confidence_cfg.get('aggregation', "min")
        self.confidence_method_cfg = confidence_cfg.get('method_cfg', None)

        # define aggregation functions
        self.confidence_aggregation_bank = get_confidence_aggregation_bank()
        self._aggregate_confidence = self.confidence_aggregation_bank[self.word_confidence_aggregation]

        # Update preserve frame confidence
        if self.preserve_frame_confidence is False:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.preserve_frame_confidence = self.cfg.greedy.get('preserve_frame_confidence', False)
                self.confidence_method_cfg = self.cfg.greedy.get('confidence_method_cfg', None)

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
