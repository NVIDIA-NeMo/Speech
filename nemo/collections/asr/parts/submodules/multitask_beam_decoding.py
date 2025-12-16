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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch

from nemo.collections.asr.modules.transformer import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithFusionModels,
)
from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig, GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core import Typing, typecheck
from nemo.core.neural_types import ChannelType, HypothesisType, LabelsType, MaskType, NeuralType
from nemo.utils import logging


def pack_hypotheses(
    hypotheses: List[Hypothesis], beam_hypotheses: torch.Tensor, scores: List[Optional[float]]
) -> List[Hypothesis]:

    for idx, hyp in enumerate(hypotheses):  # type: Hypothesis
        if scores[idx] is not None:
            hyp.score = scores[idx]

        hypi = beam_hypotheses[idx]
        if torch.is_tensor(hypi):
            hyp.y_sequence = hypi.long()
        else:
            hyp.y_sequence = torch.tensor(hypi, dtype=torch.long)

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class AEDBeamInfer(ABC):
    def __init__(
        self,
        transformer_decoder: torch.nn.Module,
        log_softmax_module: torch.nn.Module,
        tokenizer: TokenizerSpec,
        search_type: str = 'default',
        return_best_hypothesis: bool = True,
        preserve_alignments: bool = False,
    ):
        super().__init__()

        self.transformer_decoder = transformer_decoder
        self.log_softmax_module = log_softmax_module
        self.tokenizer = tokenizer

        self.search_type = search_type
        self.return_best_hypothesis = return_best_hypothesis
        self.preserve_alignments = preserve_alignments

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ):
        raise NotImplementedError()

    def set_decoding_type(self, decoding_type: str):
        self.decoding_type = decoding_type


class TransformerAEDBeamInfer(AEDBeamInfer, Typing):
    """A beam decoder engine for AED Transformer models.

    Provides a common abstraction for batch level beam decoding.

    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        # Input can be of dimention -
        # ('B', 'T', 'D') [Log probs] or ('B', 'T') [Labels]

        return {
            "encoder_hidden_states": NeuralType(tuple(('B', 'T', 'D')), ChannelType()),
            "encoder_input_mask": NeuralType(tuple(('B', 'T')), MaskType()),
            "decoder_input_ids": NeuralType(('B', 'T'), LabelsType()),
            "partial_hypotheses": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        transformer_decoder: torch.nn.Module,
        log_softmax_module: torch.nn.Module,
        tokenizer: TokenizerSpec,
        search_type: str = 'default',
        beam_size: int = 1,
        length_penalty: float = 0.0,
        max_generation_delta: int = 50,
        return_best_hypothesis: bool = True,
        preserve_alignments: bool = False,
        ngram_lm_model: Path | str | None = None,
        ngram_lm_alpha: float = 0.0,
        boosting_tree: BoostingTreeModelConfig | None = None,
        boosting_tree_alpha: float = 0.0,
    ):
        super().__init__(
            transformer_decoder=transformer_decoder,
            log_softmax_module=log_softmax_module,
            tokenizer=tokenizer,
            search_type=search_type,
            return_best_hypothesis=return_best_hypothesis,
            preserve_alignments=preserve_alignments,
        )
        self.beam_size = beam_size
        self.bos = tokenizer.bos
        self.pad = tokenizer.pad
        self.eos = tokenizer.eos

        # load boosting tree model
        boosting_tree_model = None
        if boosting_tree is not None and (
            boosting_tree.model_path or boosting_tree.key_phrases_file or boosting_tree.key_phrases_list
        ):
            boosting_tree_model = GPUBoostingTreeModel.from_config(boosting_tree, tokenizer=tokenizer)

        # initialize fusion models (ngram LM, boosting tree)
        fusion_models, fusion_models_alpha = [], []
        if ngram_lm_model is not None:
            fusion_models.append(
                NGramGPULanguageModel.from_file(lm_path=ngram_lm_model, vocab_size=tokenizer.vocab_size)
            )
            fusion_models_alpha.append(ngram_lm_alpha)
        if boosting_tree_model is not None:
            fusion_models.append(boosting_tree_model)
            fusion_models_alpha.append(boosting_tree_alpha)

        if not fusion_models:
            self.beam_search = BeamSearchSequenceGenerator(
                embedding=transformer_decoder.embedding,
                decoder=transformer_decoder.decoder,
                log_softmax=log_softmax_module,
                max_sequence_length=transformer_decoder.max_sequence_length,
                beam_size=beam_size,
                bos=self.bos,
                pad=self.pad,
                eos=self.eos,
                len_pen=length_penalty,
                max_delta_length=max_generation_delta,
            )
        else:
            self.beam_search = BeamSearchSequenceGeneratorWithFusionModels(
                embedding=transformer_decoder.embedding,
                decoder=transformer_decoder.decoder,
                log_softmax=log_softmax_module,
                max_sequence_length=transformer_decoder.max_sequence_length,
                beam_size=beam_size,
                bos=self.bos,
                pad=self.pad,
                eos=self.eos,
                len_pen=length_penalty,
                max_delta_length=max_generation_delta,
                fusion_models=fusion_models,
                fusion_models_alpha=fusion_models_alpha,
            )

        self.preserve_alignments = preserve_alignments
        if self.preserve_alignments:
            logging.info(
                "Preservation of alignments was requested but {} does not implement it.".format(
                    self.__class__.__name__
                )
            )

    @typecheck()
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features) or (batch, timesteps) (each timestep is a label).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        with torch.inference_mode():
            self.transformer_decoder.eval()
            self.log_softmax_module.eval()

            topk_hypotheses, beam_scores, best_hypo = self.beam_search(
                encoder_hidden_states=encoder_hidden_states,
                encoder_input_mask=encoder_input_mask,
                decoder_input_ids=decoder_input_ids,
                return_beam_scores=True,
            )

            if not self.return_best_hypothesis:
                topk_hypotheses = [x.detach().cpu() for x in topk_hypotheses]  # each item is [beam, seq_len]
                beam_scores = [x.detach().cpu() for x in beam_scores]  # each item is [beam,]
                packed_result = []
                for i in range(len(topk_hypotheses)):
                    hypotheses = [Hypothesis(score=0.0, y_sequence=[], timestamp=[]) for _ in range(self.beam_size)]
                    # Pack results into Hypotheses
                    hypotheses = pack_hypotheses(hypotheses, topk_hypotheses[i], beam_scores[i])
                    packed_result.append(NBestHypotheses(hypotheses))
                self.format_hypotheses(packed_result, decoder_input_ids)
            else:
                beam_scores = [None for _ in range(len(best_hypo))]
                best_hypo = best_hypo.detach().cpu()
                hypotheses = [
                    Hypothesis(score=0.0, y_sequence=[], timestamp=[]) for _ in range(encoder_hidden_states.shape[0])
                ]
                # Pack results into Hypotheses
                packed_result = pack_hypotheses(hypotheses, best_hypo, beam_scores)
                self.format_hypotheses(packed_result, decoder_input_ids)

        self.transformer_decoder.train()
        self.log_softmax_module.train()

        return (packed_result,)

    def format_hypotheses(
        self, packed_result: List[Hypothesis | NBestHypotheses], decoder_input_ids: Union[torch.Tensor, None]
    ) -> None:
        """
        For each hypothesis in the mini-batch:
        * Remove the decoder input ids (prompt) from the predictions
        * Remove BOS, EOS, and PAD ids from the predictions.
        Modifies results in-place.
        """
        if decoder_input_ids is not None:
            assert (
                len(packed_result) == decoder_input_ids.shape[0]
            ), f"Mismatching number of examples {len(packed_result)=} {decoder_input_ids.shape[0]=}"
            decoder_input_ids = decoder_input_ids.detach().cpu()

            for h, prefix in zip(packed_result, decoder_input_ids):
                hypotheses = h.n_best_hypotheses if isinstance(h, NBestHypotheses) else [h]
                for hyp in hypotheses:
                    assert (hyp.y_sequence[: prefix.shape[0]] == prefix).all(), (
                        f"The decoder input IDs were not found at the beginning of prediction: "
                        f"{hyp.y_sequence=} {prefix=}"
                    )
                    hyp.y_sequence = hyp.y_sequence[prefix.shape[0] :]

        for h in packed_result:
            hyps = h.n_best_hypotheses if isinstance(h, NBestHypotheses) else [h]
            for hyp in hyps:
                ids = hyp.y_sequence
                ids_len = ids.shape[0]
                pos = -1
                while ids[pos] == self.pad or ids[pos] == self.eos:
                    pos -= 1
                    if ids_len + pos == -1:
                        break  # empty sequence
                if pos < -1:
                    hyp.y_sequence = ids[: pos + 1]


@dataclass
class AEDBeamInferConfig:
    beam_size: int = 1
    search_type: str = 'default'
    len_pen: float = 1.0
    max_generation_delta: int = -1  # -1 means up to the max length of the decoder
    return_best_hypothesis: bool = True
    preserve_alignments: bool = False
    # fusion models params
    ngram_lm_model: Optional[str] = None
    ngram_lm_alpha: float = 0.0
    boosting_tree: BoostingTreeModelConfig = field(default_factory=lambda: BoostingTreeModelConfig(depth_scaling=1.0))
    boosting_tree_alpha: float = 0.0
