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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules.multitask_beam_decoding import (
    AEDBeamInfer,
    AEDBeamInferConfig,
    TransformerAEDBeamInfer,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging


class AbstractMultiTaskDecoding(ABC):
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

        possible_strategies = ['greedy', 'greedy_batch', 'beam']
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)

            elif self.cfg.strategy in ['beam']:
                self.preserve_alignments = self.cfg.beam.get('preserve_alignments', False)

        if self.cfg.strategy == 'greedy' or self.cfg.strategy == 'greedy_batch':

            # self.decoding = None
            raise NotImplementedError("Greedy decoding is not implemented yet.")

        elif self.cfg.strategy == 'beam':

            self.decoding = TransformerAEDBeamInfer(
                transformer_decoder=transformer_decoder,
                log_softmax_module=log_softmax_module,
                tokenizer=tokenizer,
                search_type=self.cfg.beam.get('search_type', 'default'),
                beam_size=self.cfg.beam.beam_size,
                length_penalty=self.cfg.beam.get('length_penalty', 0.0),
                max_generation_delta=self.cfg.beam.get('max_generation_delta', 50),
                return_best_hypothesis=self.cfg.beam.get('return_best_hypothesis', True),
                preserve_alignments=self.preserve_alignments,
            )

        else:

            raise ValueError(
                f"Incorrect decoding strategy provided. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )

    def decode_predictions_tensor(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        return_hypotheses: bool = False,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> Tuple[List[str], Optional[List[List[str]]], Optional[Union[Hypothesis, NBestHypotheses]]]:
        """
        Decode an encoder output by autoregressive decoding of the Decoder+Joint networks.

        Args:
            encoder_output: torch.Tensor of shape [B, D, T].
            encoded_lengths: torch.Tensor containing lengths of the padded encoder outputs. Shape [B].
            return_hypotheses: bool. If set to True it will return list of Hypothesis or NBestHypotheses

        Returns:
            If `return_best_hypothesis` is set:
                A tuple (hypotheses, None):
                hypotheses - list of Hypothesis (best hypothesis per sample).
                    Look at rnnt_utils.Hypothesis for more information.

            If `return_best_hypothesis` is not set:
                A tuple(hypotheses, all_hypotheses)
                hypotheses - list of Hypothesis (best hypothesis per sample).
                    Look at rnnt_utils.Hypothesis for more information.
                all_hypotheses - list of NBestHypotheses. Each NBestHypotheses further contains a sorted
                    list of all the hypotheses of the model per sample.
                    Look at rnnt_utils.NBestHypotheses for more information.
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
                return hypotheses, all_hypotheses

            best_hyp_text = [h.text for h in hypotheses]
            all_hyp_text = [h.text for hh in all_hypotheses for h in hh]
            return best_hyp_text, all_hyp_text

        else:
            hypotheses = self.decode_hypothesis(prediction_list)

            if return_hypotheses:
                return hypotheses, None

            best_hyp_text = [h.text for h in hypotheses]
            return best_hyp_text, None

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

            hypothesis = self.decode_tokens_to_str(prediction)

            if self.compute_hypothesis_token_set:
                hypotheses_list[ind].tokens = self.decode_ids_to_tokens(prediction)

            # De-tokenize the integer tokens
            hypotheses_list[ind].text = hypothesis

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

    def strip_special_tokens(self, text: str):
        """
        assuming all special tokens are of format <token>
        Note that if any label/pred is of format <token>, it will be stripped
        """
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        text = re.sub(r'<[^>]+>', '', text)
        # strip spaces at the beginning and end;
        # this is training data artifact, will be fixed in future (@kpuvvada)
        return text.strip()


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

    # compute language IDs
    compute_langs: bool = False

    # greedy decoding config
    # greedy: rnnt_greedy_decoding.GreedyBatchedRNNTInferConfig = field(
    #     default_factory=rnnt_greedy_decoding.GreedyBatchedRNNTInferConfig
    # )

    # beam decoding config
    beam: AEDBeamInferConfig = field(default_factory=lambda: AEDBeamInferConfig(beam_size=1))

    # can be used to change temperature for decoding
    temperature: float = 1.0
