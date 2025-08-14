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

from abc import abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Union

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules.multitask_beam_decoding import (
    AEDBeamInfer,
    AEDBeamInferConfig,
    TransformerAEDBeamInfer,
)
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import (
    AEDGreedyInferConfig,
    TransformerAEDGreedyInfer,
)
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMixin
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging


class AbstractMultiTaskDecoding(ConfidenceMixin):
    """
    Used for performing AED auto-regressive decoding of the Multi task model given the encoder state.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy, greedy_batch (for greedy decoding).
                -   beam, tsd, alsd (for beam search decoding).

            compute_langs: a bool flag, which allows to compute language id (LID) information per token,
                word, and the entire sample (most likely language id). The LIDS will be available
                in the returned Hypothesis object as a dictionary

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `alignments` in it. Here, `alignments` is a List of List of
                Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence scores.
                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during greedy decoding. When set to true, the Hypothesis will contain the non-null value
                    for `frame_confidence` in it. Here, `frame_confidence` is a List of tensor floats.

                preserve_token_confidence: Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding. When set to true, the Hypothesis will contain the non-null value
                    for `token_confidence` in it. Here, `token_confidence` is a List of tensor floats.
                    The length of the list corresponds to the number of recognized tokens.

                preserve_word_confidence: Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding. When set to true, the Hypothesis will contain the non-null value
                    for `word_confidence` in it. Here, `word_confidence` is a List of tensor floats.
                    The length of the list corresponds to the number of recognized words.

                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word
                    confidence. Valid options are `mean`, `min`, `max`, `prod`.

                method_cfg: A dict-like object which contains settings to compute per-frame confidence scores.
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

            The config may further contain the following sub-dictionaries:
            "greedy":
                temperature: None (disabled) or float, specifying this enables temperature sampling instead of greedy decoding.
                max_generation_delta: int = -1  # -1 means up to the max length of the decoder
                preserve_alignments: bool = False (unsupported)

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                length_penalty: float, length penalty for beam search decoding. Must be >= 0.0.

                max_generation_delta: int,in case of encoder-decoder generation (e.g. NMT),
                    forbids generated sequences to be longer than the length of source sequences plus max_generation_delta

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.


        transformer_decoder: Transformer decoder module.
        log_softmax_module: Log Softmax projection module to the vocab size.
        tokenizer: Aggregate Tokenizer.
    """

    def __init__(
        self,
        decoding_cfg,
        transformer_decoder: torch.nn.Module,
        log_softmax_module: torch.nn.Module,
        tokenizer: TokenizerSpec,
    ):
        super().__init__()

        # Convert dataclass to config object
        if is_dataclass(decoding_cfg):
            decoding_cfg = OmegaConf.structured(decoding_cfg)

        self.cfg = decoding_cfg

        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.compute_langs = self.cfg.get('compute_langs', False)
        self.compute_hypothesis_token_set = self.cfg.get('compute_hypothesis_token_set', False)
        self.transformer_decoder = transformer_decoder
        self.log_softmax_module = log_softmax_module
        self.tokenizer = tokenizer

        # initialize confidence-related fields
        self._init_confidence(self.cfg.get('confidence_cfg', None))

        self.change_strategy(self.cfg.strategy)

    def change_strategy(self, strategy: str) -> "AbstractMultiTaskDecoding":
        possible_strategies = ['greedy', 'greedy_batch', 'beam']
        if strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}" f"but was provided {strategy}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if strategy in ['greedy', 'greedy_batch']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)

            elif strategy in ['beam']:
                self.preserve_alignments = self.cfg.beam.get('preserve_alignments', False)

        if strategy in ['greedy', 'greedy_batch']:
            if self.cfg.greedy.get('ngram_lm_model') is not None:
                raise ValueError(
                    "Greedy strategy cannot be used with ngram_lm_model. Use beam strategy with beam=1 instead."
                )
            self.decoding = TransformerAEDGreedyInfer(
                transformer_decoder=self.transformer_decoder,
                log_softmax_module=self.log_softmax_module,
                tokenizer=self.tokenizer,
                max_generation_delta=self.cfg.greedy.get('max_generation_delta', -1),
                preserve_alignments=self.preserve_alignments,
                preserve_token_confidence=self.preserve_token_confidence or self.preserve_frame_confidence,
                confidence_method_cfg=self.confidence_method_cfg,
                temperature=self.cfg.greedy.temperature,
                n_samples=self.cfg.greedy.n_samples,
            )

        elif strategy == 'beam':

            self.decoding = TransformerAEDBeamInfer(
                transformer_decoder=self.transformer_decoder,
                log_softmax_module=self.log_softmax_module,
                tokenizer=self.tokenizer,
                search_type=self.cfg.beam.get('search_type', 'default'),
                beam_size=self.cfg.beam.beam_size,
                length_penalty=self.cfg.beam.get('length_penalty', 0.0),
                max_generation_delta=self.cfg.beam.get('max_generation_delta', -1),
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
                ngram_lm_model=self.cfg.beam.get('ngram_lm_model', None),
                ngram_lm_alpha=self.cfg.beam.get('ngram_lm_alpha', 0.0),
                boosting_tree=self.cfg.beam.get('boosting_tree', None),
                boosting_tree_alpha=self.cfg.beam.get('boosting_tree_alpha', 0.0),
            )

        else:

            raise ValueError(
                f"Incorrect decoding strategy provided. Must be one of {possible_strategies}\n"
                f"but was provided {strategy}"
            )

    def decode_predictions_tensor(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
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
                encoder_hidden_states=encoder_hidden_states,
                encoder_input_mask=encoder_input_mask,
                decoder_input_ids=decoder_input_ids,
                partial_hypotheses=partial_hypotheses,
            )  # type: [List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        prediction_list = hypotheses_list

        if isinstance(prediction_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []

            for nbest_hyp in prediction_list:  # type: NBestHypotheses
                n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(n_hyps)

                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return all_hypotheses

            all_hyp = [[Hypothesis(h.score, h.y_sequence, h.text) for h in hh] for hh in all_hypotheses]
            return all_hyp

        else:
            hypotheses = self.decode_hypothesis(prediction_list)

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

            hypothesis = self.decode_ids_to_str(prediction)

            if self.compute_hypothesis_token_set:
                hypotheses_list[ind].tokens = self.decode_ids_to_tokens(prediction)

            # De-tokenize the integer tokens
            hypotheses_list[ind].text = hypothesis

        return hypotheses_list

    def compute_confidence(self, hypotheses_list: List[Hypothesis]) -> List[Hypothesis]:
        """
        Compute high-level (per-token and/or per-word) confidence scores for a list of hypotheses.
        Assumes that `token_confidence` is present in the hypotheses.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of hypotheses with high-level confidence scores.
        """
        if self.preserve_word_confidence:
            for hyp in hypotheses_list:
                hyp.word_confidence = self._aggregate_token_confidence(hyp)
        return hypotheses_list

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


class MultiTaskDecoding(AbstractMultiTaskDecoding):
    """
    Used for performing AED auto-regressive decoding of the Multi task model given the encoder state.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy, greedy_batch (for greedy decoding).
                -   beam, tsd, alsd (for beam search decoding).

            compute_langs: a bool flag, which allows to compute language id (LID) information per token,
                word, and the entire sample (most likely language id). The LIDS will be available
                in the returned Hypothesis object as a dictionary

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `alignments` in it. Here, `alignments` is a List of List of
                Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

            confidence_cfg: A dict-like object which contains the following key-value pairs related to confidence scores.
                preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
                    generated during greedy decoding. When set to true, the Hypothesis will contain the non-null value
                    for `frame_confidence` in it. Here, `frame_confidence` is a List of tensor floats.

                preserve_token_confidence: Bool flag which preserves the history of per-token confidence scores
                    generated during greedy decoding. When set to true, the Hypothesis will contain the non-null value
                    for `token_confidence` in it. Here, `token_confidence` is a List of tensor floats.
                    The length of the list corresponds to the number of recognized tokens.

                preserve_word_confidence: Bool flag which preserves the history of per-word confidence scores
                    generated during greedy decoding. When set to true, the Hypothesis will contain the non-null value
                    for `word_confidence` in it. Here, `word_confidence` is a List of tensor floats.
                    The length of the list corresponds to the number of recognized words.

                aggregation: Which aggregation type to use for collapsing per-token confidence into per-word
                    confidence. Valid options are `mean`, `min`, `max`, `prod`.

                method_cfg: A dict-like object which contains settings to compute per-frame confidence scores.
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

            The config may further contain the following sub-dictionaries:
            "greedy":
                temperature: None (disabled) or float, specifying this enables temperature sampling instead of greedy decoding.

                max_generation_delta: int = -1  # -1 means up to the max length of the decoder

                preserve_alignments: bool = False (unsupported)

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                length_penalty: float, length penalty for beam search decoding. Must be >= 0.0.

                max_generation_delta: int, maximum number of additional target tokens to generate

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.


        transformer_decoder: Transformer decoder module.
        log_softmax_module: Log Softmax projection module to the vocab size.
        tokenizer: TokenizerSpec.
    """

    def __init__(
        self,
        decoding_cfg,
        transformer_decoder: torch.nn.Module,
        log_softmax_module: torch.nn.Module,
        tokenizer: TokenizerSpec,
    ):
        self.tokenizer = tokenizer

        super().__init__(
            decoding_cfg=decoding_cfg,
            transformer_decoder=transformer_decoder,
            log_softmax_module=log_softmax_module,
            tokenizer=tokenizer,
        )

        if isinstance(self.decoding, AEDBeamInfer):
            self.decoding.set_decoding_type('subword')

    def decode_tokens_to_str(self, tokens: List[str], lang: Optional[str] = None) -> str:
        """
        Implemented by subclass in order to decoder a token str into a string.

        Args:
            tokens: List of str representing the tokens.

        Returns:
            A decoded string.
        """
        if lang is not None:
            hypothesis = self.tokenizer.tokens_to_text(tokens, lang)
        else:
            hypothesis = self.tokenizer.tokens_to_text(tokens)
        return hypothesis

    def decode_ids_to_str(self, tokens: List[int]) -> str:
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

                    hypotheses[ind].langs = self.decode_tokens_to_lang(prediction)
                    hypotheses[ind].langs_chars = self.decode_ids_to_langs(prediction)
            else:
                logging.warning(
                    "Ignoring request for lang output in hypotheses since the model does not use an aggregate tokenizer"
                )

        return hypotheses


@dataclass
class MultiTaskDecodingConfig:
    strategy: str = "beam"

    compute_hypothesis_token_set: bool = False

    # preserve decoding alignments
    preserve_alignments: Optional[bool] = None

    #  confidence config
    confidence_cfg: ConfidenceConfig = field(default_factory=lambda: ConfidenceConfig())

    # compute language IDs
    compute_langs: bool = False

    # greedy decoding config
    greedy: AEDGreedyInferConfig = field(default_factory=AEDGreedyInferConfig)

    # beam decoding config
    beam: AEDBeamInferConfig = field(default_factory=lambda: AEDBeamInferConfig(beam_size=1))

    # can be used to change temperature for decoding
    temperature: float = 1.0
