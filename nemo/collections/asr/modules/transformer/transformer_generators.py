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

from contextlib import contextmanager
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.distributions import Categorical

from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.common.parts import NEG_INF, mask_padded_tokens

__all__ = [
    "GreedySequenceGenerator",
    "TopKSequenceGenerator",
    "BeamSearchSequenceGenerator",
    "BeamSearchSequenceGeneratorWithLanguageModel",
    "BeamSearchSequenceGeneratorWithNGramLM",
    "EnsembleBeamSearchSequenceGenerator",
]


class GreedySequenceGenerator(ConfidenceMethodMixin):
    """
    Greedy sequence generator based on the decoder followed by log_softmax.
    Optionally supports temperature sampling with ``n_samples`` and ``temperature`` options.

    Args:
        embedding: nn.Module, transforms input_ids into vector embeddings
        decoder: nn.Module, takes embeddings and produces hidden_states
        classifier: nn.Module, takes hidden_states and produces
            logits or log-probability distribution of tokens (ids)
        pad: index of padding token in the vocabulary
        bos: index of beginning of sequence token in the vocabulary
        eos: index of end of sequence token in the vocabulary
        max_sequence_length: maximum allowed length for generated sequences
        max_delta_length: in case of encoder-decoder generation (e.g. NMT),
            forbids generated sequences to be longer than the length of
            source sequences plus max_delta_length
        batch_size: size of the batch of generated sequences if neither
            source nor target starting sequences are provided
        n_samples: number of sequences to generate (requires ``temperature`` to be set)
        temperature: temperature for temperature sampling. Even with ``n_samples`` set to 1,
            enabling temperature will sample hypotheses instead of returning the best ones.

        preserve_step_confidence: Bool flag which preserves the history of per-step confidence scores generated
            during greedy decoding. When set to true, the results will contain additional List of tensor floats.
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-step
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

    def __init__(
        self,
        embedding,
        decoder,
        classifier: TokenClassifier,
        pad=0,
        bos=1,
        eos=2,
        max_sequence_length=512,
        max_delta_length=20,
        batch_size=1,
        n_samples=1,
        temperature=None,
        preserve_step_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.classifier = classifier
        self.pad, self.bos, self.eos = pad, bos, eos
        self.max_seq_length = max_sequence_length
        self.max_delta_len = max_delta_length
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.temperature = temperature
        self.preserve_step_confidence = preserve_step_confidence

        # set confidence calculation method
        self.num_tokens = getattr(self.classifier.mlp, f'layer{self.classifier.mlp.layers - 1}').out_features
        self._init_confidence_method(confidence_method_cfg)

    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
        return_scores: bool = True,
    ):
        """
        One step of autoregressive output generation.

        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding
        """

        decoder_hidden_states = self.embedding.forward(decoder_input_ids, start_pos=pos)
        decoder_input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()

        if encoder_hidden_states is not None:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states,
                decoder_input_mask,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                return_mems=True,
            )
        else:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states, decoder_input_mask, decoder_mems_list, return_mems=True
            )
        with self.classifier.with_log_softmax_enabled(return_scores) as clf:
            logits = clf.forward(hidden_states=decoder_mems_list[-1][:, -1:])
        return logits, decoder_mems_list

    def _prepare_for_search(self, decoder_input_ids=None, encoder_hidden_states=None):
        """
        Helper function which defines starting sequence to begin generating
        with and maximum allowed number of tokens to be generated.
        """

        decoder_parameter = next(self.decoder.parameters())
        batch_size = self.batch_size

        # for encoder-decoder generation, maximum length of generated sequence
        # is min(max_sequence_length, src_len + max_delta_length)
        if encoder_hidden_states is not None:
            batch_size, src_len, _ = encoder_hidden_states.size()
            if self.max_delta_len >= 0:
                max_seq_length = min(self.max_seq_length, src_len + self.max_delta_len)
            else:
                max_seq_length = self.max_seq_length
        else:
            max_seq_length = self.max_seq_length

        # if no input is provided, start with the batch of <bos> tokens
        if decoder_input_ids is not None:
            tgt = decoder_input_ids
            batch_size, tgt_len = decoder_input_ids.size()
        else:
            tgt = torch.zeros(batch_size, 1).long().fill_(self.bos).to(decoder_parameter.device)
            tgt_len = 1
        max_generation_length = max_seq_length - tgt_len

        return tgt, batch_size, max_generation_length

    def _forward(
        self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, return_beam_scores=False
    ):
        assert not return_beam_scores
        is_sampling = self.temperature is not None and self.n_samples > 1

        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)
        if is_sampling:
            tgt = torch.repeat_interleave(tgt, self.n_samples, dim=0)
            encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, self.n_samples, dim=0)
            encoder_input_mask = torch.repeat_interleave(encoder_input_mask, self.n_samples, dim=0)
            orig_batch_size = batch_size
            batch_size = batch_size * self.n_samples

        # pad profile tracks sequences ending with <eos> token to replace
        # everything after <eos> with <pad> token
        decoder_parameter = next(self.decoder.parameters())
        pad_profile = torch.zeros(batch_size).long().to(decoder_parameter.device)

        if self.preserve_step_confidence:
            if encoder_hidden_states is None:
                raise RuntimeError("`encoder_hidden_states` must be provided to compute confidence scores.")
            # start with prompt confidence which is always 1
            step_confidence = [torch.full_like(tgt, 1, dtype=encoder_hidden_states.dtype)]
        else:
            step_confidence = None

        decoder_mems_list = None
        for i in range(max_generation_length):

            if i == 0:
                input_ids = tgt
            else:
                input_ids = tgt[:, -1:]

            logits, decoder_mems_list = self._one_step_forward(
                input_ids,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                i,
                return_scores=return_beam_scores,
            )

            if self.temperature is None:  # Greedy decoding
                next_tokens = torch.argmax(logits[:, -1], dim=-1)
            else:  # Temperature sampling
                next_tokens = Categorical(logits=logits[:, -1] / self.temperature).sample()

            next_tokens = self.pad * pad_profile + next_tokens * (1 - pad_profile)
            pad_profile = torch.max(pad_profile, (next_tokens == self.eos).long())
            tgt = torch.cat((tgt, next_tokens.unsqueeze(1)), dim=-1)

            if self.preserve_step_confidence:
                step_confidence.append(
                    self._get_confidence_tensor(
                        torch.nn.functional.log_softmax(logits, dim=-1) if not return_beam_scores else logits
                    )
                )

            # abort generation if all sequences end with <eos>
            if pad_profile.sum() == batch_size:
                break

        step_confidence_tensor = (
            torch.cat(step_confidence, dim=1) if self.preserve_step_confidence and len(step_confidence) > 0 else None
        )

        samples = None
        if is_sampling:
            samples = list(tgt.view(orig_batch_size, self.n_samples, -1))
            tgt = tgt[:: self.n_samples]

        return tgt, samples, step_confidence_tensor

    def __call__(
        self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, return_beam_scores=False
    ):
        with torch.inference_mode():
            results = self._forward(
                decoder_input_ids, encoder_hidden_states, encoder_input_mask, return_beam_scores=return_beam_scores
            )
            if not return_beam_scores:
                return results
            else:
                prefixes, scores, tgt = results
                prefixes = prefixes.view(-1, self.beam_size, tgt.size(1)).split(1, dim=0)
                scores = scores.view(-1, self.beam_size).split(1, dim=0)
                prefixes = [x.squeeze(0) for x in prefixes]  # each item is [beam, seq_len]
                scores = [x.squeeze(0) for x in scores]  # each item is [beam,]
                return prefixes, scores, tgt

    def freeze(self) -> None:
        """Freeze weights of embedding, decoder, and classification layers to prevent memory leak."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.embedding.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()

    def unfreeze(self) -> None:
        """Unfreeze weights of embedding, decoder, and classification layers."""
        for param in self.embedding.parameters():
            param.requires_grad = True
        self.embedding.train()
        for param in self.decoder.parameters():
            param.requires_grad = True
        self.decoder.train()
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.classifier.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes embedding, decoder, and classifier modules,
        yields control and finally unfreezes the modules.
        """
        self.freeze()

        try:
            yield
        finally:
            self.unfreeze()


class TopKSequenceGenerator(GreedySequenceGenerator):
    """
    Top-k sequence generator based on the decoder followed by log_softmax.

    Args:
        *all args of GreedySequenceGenerator class
        beam_size: size of the beam (parameter k in top-k)
        temperature: temperature of top-k sampling, all logits are divided
            by temperature before rescaling. High temperature leads to
            uniform distribution, low leads to delta-like distribution.
    Kwargs:
        all remaining parameters of GreedySequenceGenerator class
    """

    def __init__(self, embedding, decoder, log_softmax, beam_size=1, temperature=1.0, **kwargs):
        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.beam_size = beam_size
        self.temp = temperature

    # @torch.no_grad()
    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
        return_scores: bool = True,
    ):
        log_probs, decoder_mems_list = super()._one_step_forward(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_input_mask,
            decoder_mems_list,
            pos,
            return_scores=return_scores,
        )

        batch_size, seq_len, vocab_size = log_probs.size()
        scores, indices = torch.topk(log_probs, self.beam_size, dim=-1)

        rescaled_logexp = torch.zeros_like(log_probs).scatter(-1, indices, scores.div(self.temp).exp())
        probs = rescaled_logexp / rescaled_logexp.norm(1, -1, keepdim=True)

        # We randomly sample next tokens from rescaled probability distribution
        # over top-k candidates and return a binary tensor which indicates
        # candidates that have been selected. We call this object
        # `pseudo_log_probs` as genuine log_probs should have -infs instead of
        # 0s and 0s instead of 1s.
        ids = torch.multinomial(probs.view(-1, vocab_size), 1).view(-1, seq_len, 1)
        pseudo_log_probs = torch.zeros_like(log_probs).scatter(-1, ids, 1.0)

        return pseudo_log_probs, decoder_mems_list


class BeamSearchSequenceGenerator(GreedySequenceGenerator):
    def __init__(self, embedding, decoder, log_softmax, beam_size=1, len_pen=0, **kwargs):
        """
        Beam Search sequence generator based on the decoder followed by
        log_softmax.

        Args:
            *all args of GreedySequenceGenerator class
            beam_size: size of the beam
            len_pen: length penalty parameter
        Kwargs:
            all remaining parameters of GreedySequenceGenerator class
        """

        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.beam_size = beam_size
        self.len_pen = len_pen

    @staticmethod
    def compute_len_penalty(lengths, alpha):
        """Returns length penalty according to https://arxiv.org/pdf/1609.08144.pdf"""
        return ((5 + lengths) / 6).pow(alpha)

    def _forward(
        self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, return_beam_scores=False
    ):
        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs, decoder_mems_list = self._one_step_forward(tgt, encoder_hidden_states, encoder_input_mask, None, 0)
        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, tgt.shape[1]), prefixes), dim=1)
        for j in range(len(decoder_mems_list)):
            decoder_mems_list[j] = decoder_mems_list[j].repeat(self.beam_size, 1, 1)

        # repeat source sequence beam_size times for beam search
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mems_list[0].size(2)

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long()

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        tgt_len = tgt.size(-1)
        for i in range(tgt_len, max_generation_length + tgt_len):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            log_probs, decoder_mems_list = self._one_step_forward(
                prefixes[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i
            )
            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size**2), self.beam_size, dim=1)
            scores = scores.view(-1, 1) * len_penalties

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size**2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
            for j in range(len(decoder_mems_list)):
                decoder_mems_list[j] = (
                    decoder_mems_list[j]
                    .view(-1, self.beam_size, p_len - 1, hidden_size)
                    .gather(1, mems_ids)
                    .view(-1, p_len - 1, hidden_size)
                )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

        # select best performing hypotheses in each element of the batch
        len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
        scores = scores / len_penalties
        best_guesses = (
            torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
        )
        tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses).squeeze(1)

        if return_beam_scores:
            return prefixes, scores * len_penalties, tgt
        else:
            return tgt


class BeamSearchSequenceGeneratorWithFusionModels(BeamSearchSequenceGenerator):
    def __init__(
        self, embedding, decoder, log_softmax, fusion_models, fusion_models_alpha, beam_size=1, len_pen=0, **kwargs
    ):
        """
        Beam Search sequence generator based on the decoder followed by
        log_softmax.

        Args:
            *all args of BeamSearchSequenceGenerator class
            ngram_lm_model: path to the n-gram language model; LM should use the same tokenizer as the current model
            ngram_lm_alpha: n-gram LM weight
        Kwargs:
            all remaining parameters of BeamSearchSequenceGenerator class
        """

        super().__init__(embedding, decoder, log_softmax, beam_size=beam_size, len_pen=len_pen, **kwargs)

        self.fusion_models = fusion_models
        self.fusion_models_alpha = fusion_models_alpha

    def _forward(
        self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, return_beam_scores=False
    ):
        device = encoder_hidden_states.device
        # force fusion models to use the same device as encoder_hidden_states, since current class is not nn.Module instance
        for fusion_model in self.fusion_models:
            fusion_model.to(device)

        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        batch_fusion_states_list = [
            fusion_model.get_init_states(batch_size=batch_size, bos=True) for fusion_model in self.fusion_models
        ]
        batch_fusion_states_candidates_list = []

        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs, decoder_mems_list = self._one_step_forward(tgt, encoder_hidden_states, encoder_input_mask, None, 0)
        # get fusion models scores
        for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
            fusion_scores, batch_fusion_states_candidates = fusion_model.advance(
                states=batch_fusion_states_list[fusion_model_idx], eos_id=self.eos
            )
            batch_fusion_states_candidates_list.append(batch_fusion_states_candidates)
            log_probs += self.fusion_models_alpha[fusion_model_idx] * fusion_scores[:, None, :]

        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)  # [Batch, Beam, 1]
        for fusion_model_idx, batch_fusion_states_candidates in enumerate(batch_fusion_states_candidates_list):
            batch_fusion_states_list[fusion_model_idx] = batch_fusion_states_candidates.gather(
                dim=1, index=prefixes.squeeze(-1)
            ).view(
                -1
            )  # [Batch, Beam] -> [Batch*Beam]

        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)  # [Batch*Beam, 1]

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, tgt.shape[1]), prefixes), dim=1)
        for j in range(len(decoder_mems_list)):
            decoder_mems_list[j] = decoder_mems_list[j].repeat(self.beam_size, 1, 1)

        # repeat source sequence beam_size times for beam search
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mems_list[0].size(2)

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long()

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        tgt_len = tgt.size(-1)
        for i in range(tgt_len, max_generation_length + tgt_len):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            log_probs, decoder_mems_list = self._one_step_forward(
                prefixes[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i
            )
            for fusion_model_idx, fusion_model in enumerate(self.fusion_models):
                fusion_scores, batch_fusion_states_candidates = fusion_model.advance(
                    states=batch_fusion_states_list[fusion_model_idx], eos_id=self.eos
                )
                log_probs += self.fusion_models_alpha[fusion_model_idx] * fusion_scores[:, None, :]
                batch_fusion_states_candidates_list[fusion_model_idx] = batch_fusion_states_candidates

            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)  # [Batch*Beam, Beam]

            for fusion_model_idx, batch_fusion_states_candidates in enumerate(batch_fusion_states_candidates_list):
                batch_fusion_states_list[fusion_model_idx] = batch_fusion_states_candidates.gather(
                    dim=1, index=prefixes_i
                )

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size**2), self.beam_size, dim=1)  # [Batch, Beam]

            for fusion_model_idx, batch_fusion_states in enumerate(batch_fusion_states_list):
                batch_fusion_states_list[fusion_model_idx] = (
                    batch_fusion_states.view(-1, self.beam_size**2).gather(dim=1, index=indices_i).view(-1)
                )

            scores = scores.view(-1, 1) * len_penalties  # [Batch*Beam, 1]

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size**2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
            for j in range(len(decoder_mems_list)):
                decoder_mems_list[j] = (
                    decoder_mems_list[j]
                    .view(-1, self.beam_size, p_len - 1, hidden_size)
                    .gather(1, mems_ids)
                    .view(-1, p_len - 1, hidden_size)
                )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

        # select best performing hypotheses in each element of the batch
        len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
        scores = scores / len_penalties
        best_guesses = (
            torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
        )
        tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses).squeeze(1)

        if return_beam_scores:
            return prefixes, scores * len_penalties, tgt
        else:
            return tgt


class EnsembleBeamSearchSequenceGenerator:
    def __init__(
        self,
        encoders,
        embeddings,
        decoders,
        log_softmaxes,
        beam_size=1,
        len_pen=0,
        pad=0,
        bos=1,
        eos=2,
        max_sequence_length=512,
        max_delta_length=20,
        batch_size=1,
        language_model=None,
        fusion_coef=None,
    ):
        """
        Ensemble Beam Search sequence generator based on the decoder followed by
        log_softmax. Averages the probabilities of different models.
        NOTE: All models must have been trained with the same BPE tokenizers.

        Args:
            encoders: A list of encoders
            embeddings: A list of decoder embedding layers
            decoders: A list of decoders
            log_softmaxes: A list of decoder output layers
            beam_size: Beam size
            len_pen: Length penalty to adjust logprob scores to favor longer sequences
            pad: pad id
            bos: beginning of sequence id
            eos: end of sequence id
            max_sequence_length: maximum sequence length
            max_delta_length: maximum length difference between input and output
            batch_size: batch size if not inferrable from input sequence
        """
        self.encoders = encoders
        self.embeddings = embeddings
        self.decoders = decoders
        self.log_softmaxes = log_softmaxes
        self.beam_size = beam_size
        self.len_pen = len_pen
        self.pad, self.bos, self.eos = pad, bos, eos
        self.max_seq_length = max_sequence_length
        self.max_delta_len = max_delta_length
        self.batch_size = batch_size
        assert len(embeddings) == len(decoders) == len(log_softmaxes) == len(encoders)
        self.num_models = len(encoders)
        self.language_model = language_model
        self.fusion_coef = fusion_coef

    @staticmethod
    def compute_len_penalty(lengths, alpha):
        """Returns length penalty according to https://arxiv.org/pdf/1609.08144.pdf"""
        return ((5 + lengths) / 6).pow(alpha)

    def _one_step_forward_lm(self, decoder_input_ids=None, lm_mems_list=None, pos=0):
        input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()
        lm_hidden_states = self.language_model.encoder.embedding.forward(decoder_input_ids, start_pos=pos)
        lm_mems_list = self.language_model.encoder.encoder.forward(
            lm_hidden_states,
            input_mask,
            lm_mems_list,
            return_mems=True,
        )
        lm_log_probs = self.language_model.log_softmax.forward(hidden_states=lm_mems_list[-1][:, -1:])
        return lm_log_probs, lm_mems_list

    def _one_step_forward(
        self,
        ensemble_index,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
    ):
        """
        One step of autoregressive output generation for one particular model.

        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding
        """

        decoder_hidden_states = self.embeddings[ensemble_index].forward(decoder_input_ids, start_pos=pos)
        decoder_input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()

        if encoder_hidden_states is not None:
            decoder_mems_list = self.decoders[ensemble_index].forward(
                decoder_hidden_states,
                decoder_input_mask,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                return_mems=True,
            )
        else:
            decoder_mems_list = self.decoders[ensemble_index].forward(
                decoder_hidden_states, decoder_input_mask, decoder_mems_list, return_mems=True
            )
        log_probs = self.log_softmaxes[ensemble_index].forward(hidden_states=decoder_mems_list[-1][:, -1:])
        return log_probs, decoder_mems_list

    def _prepare_for_search(self, decoder_input_ids=None, encoder_hidden_states=None):
        """
        Helper function which defines starting sequence to begin generating
        with and maximum allowed number of tokens to be generated.
        """

        decoder_parameter = next(self.decoders[0].parameters())
        batch_size = self.batch_size

        # for encoder-decoder generation, maximum length of generated sequence
        # is min(max_sequence_length, src_len + max_delta_length)
        if encoder_hidden_states is not None:
            batch_size, src_len, _ = encoder_hidden_states.size()
            if self.max_delta_len >= 0:
                max_seq_length = min(self.max_seq_length, src_len + self.max_delta_len)
            else:
                max_seq_length = self.max_seq_length
        else:
            max_seq_length = self.max_seq_length

        # if no input is provided, start with the batch of <bos> tokens
        if decoder_input_ids is not None:
            tgt = decoder_input_ids
            batch_size, tgt_len = decoder_input_ids.size()
        else:
            tgt = torch.zeros(batch_size, 1).long().fill_(self.bos).to(decoder_parameter.device)
            tgt_len = 1
        max_generation_length = max_seq_length - tgt_len

        return tgt, batch_size, max_generation_length

    def _get_encoder_hidden_states(self, src_ids, encoder_input_mask, ensemble_index):
        return self.encoders[ensemble_index](input_ids=src_ids, encoder_mask=encoder_input_mask)

    def _average_probs(self, probs_list):
        probs_list = torch.stack(probs_list)
        return torch.log(torch.exp(probs_list).mean(0))
        # probs = torch.stack(probs_list) # Ens x B x T x V
        # return torch.log(probs.sum(0) / probs.sum(-1).sum(0).unsqueeze(-1))

    def _forward(self, src_ids, encoder_input_mask, decoder_input_ids=None, return_beam_scores=False):
        encoder_hidden_states = [
            self._get_encoder_hidden_states(src_ids, encoder_input_mask, i) for i in range(self.num_models)
        ]
        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states[0])

        # generate initial buffer of beam_size prefixes-hypotheses
        outputs = [
            self._one_step_forward(i, tgt, encoder_hidden_states[i], encoder_input_mask, None, 0)
            for i in range(self.num_models)
        ]
        nmt_log_probs = self._average_probs([x[0] for x in outputs])
        decoder_mems_lists = [x[1] for x in outputs]

        if self.language_model is not None:
            lm_log_probs, lm_mems_list = self._one_step_forward_lm(tgt, None, 0)
            log_probs = nmt_log_probs + self.fusion_coef * lm_log_probs
        else:
            log_probs = nmt_log_probs
        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, 1), prefixes), dim=1)
        for i in range(self.num_models):
            for j in range(len(decoder_mems_lists[i])):
                decoder_mems_lists[i][j] = decoder_mems_lists[i][j].repeat(self.beam_size, 1, 1)

        if self.language_model is not None:
            for j in range(len(lm_mems_list)):
                lm_mems_list[j] = lm_mems_list[j].repeat(self.beam_size, 1, 1)
            lm_hidden_size = lm_mems_list[0].size(2)

        encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, encoder_input_mask.size(1))
        for i in range(self.num_models):
            _, src_length, hidden_size = encoder_hidden_states[i].size()
            encoder_hidden_states[i] = (
                encoder_hidden_states[i].repeat(1, self.beam_size, 1).view(-1, src_length, hidden_size)
            )

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long()

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        for i in range(max_generation_length):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            outputs = [
                self._one_step_forward(
                    model_num,
                    prefixes[:, -1:],
                    encoder_hidden_states[model_num],
                    encoder_input_mask,
                    decoder_mems_lists[model_num],
                    i + 1,
                )
                for model_num in range(self.num_models)
            ]
            nmt_log_probs = self._average_probs([x[0] for x in outputs])
            decoder_mems_lists = [x[1] for x in outputs]

            if self.language_model is not None:
                lm_log_probs, lm_mems_list = self._one_step_forward_lm(prefixes[:, -1:], lm_mems_list, i + 1)
                log_probs = nmt_log_probs + self.fusion_coef * lm_log_probs
            else:
                log_probs = nmt_log_probs
            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size**2), self.beam_size, dim=1)
            scores = scores.view(-1, 1) * len_penalties

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size**2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            for model_num in range(self.num_models):
                hidden_size = decoder_mems_lists[model_num][0].size(2)
                mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
                for j in range(len(decoder_mems_lists[model_num])):
                    decoder_mems_lists[model_num][j] = (
                        decoder_mems_lists[model_num][j]
                        .view(-1, self.beam_size, p_len - 1, hidden_size)
                        .gather(1, mems_ids)
                        .view(-1, p_len - 1, hidden_size)
                    )
            if self.language_model is not None:
                lm_mems_ids = (
                    indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, lm_hidden_size) // self.beam_size
                )
                for j in range(len(lm_mems_list)):
                    lm_mems_list[j] = (
                        lm_mems_list[j]
                        .view(-1, self.beam_size, p_len - 1, lm_hidden_size)
                        .gather(1, lm_mems_ids)
                        .view(-1, p_len - 1, lm_hidden_size)
                    )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

        # select best performing hypotheses in each element of the batch
        len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
        scores = scores / len_penalties
        best_guesses = (
            torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
        )
        tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses).squeeze(1)

        if return_beam_scores:
            return prefixes, scores * len_penalties, tgt
        else:
            return tgt

    def __call__(self, src_ids, encoder_input_mask, decoder_input_ids=None, return_beam_scores=False):
        with torch.inference_mode():
            return self._forward(src_ids, encoder_input_mask, decoder_input_ids, return_beam_scores)

    def freeze(self) -> None:
        """Freeze weights of embedding, decoder, and classification layers to prevent memory leak."""
        for model_num in range(self.num_models):
            for param in self.embeddings[model_num].parameters():
                param.requires_grad = False
            self.embeddings[model_num].eval()
            for param in self.decoders[model_num].parameters():
                param.requires_grad = False
            self.decoders[model_num].eval()
            for param in self.log_softmaxes[model_num].parameters():
                param.requires_grad = False
            self.log_softmaxes[model_num].eval()
            for param in self.encoders[model_num].parameters():
                param.requires_grad = False
            self.encoders[model_num].eval()

    def unfreeze(self) -> None:
        """Unfreeze weights of embedding, decoder, and classification layers."""
        for model_num in range(self.num_models):
            for param in self.embeddings[model_num].parameters():
                param.requires_grad = True
            self.embeddings[model_num].train()
            for param in self.decoders[model_num].parameters():
                param.requires_grad = True
            self.decoders[model_num].train()
            for param in self.log_softmaxes[model_num].parameters():
                param.requires_grad = True
            self.log_softmaxes[model_num].train()
            for param in self.encoders[model_num].parameters():
                param.requires_grad = True
            self.encoders[model_num].train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes embedding, decoder, and log_softmax modules,
        yields control and finally unfreezes the modules.
        """
        self.freeze()

        try:
            yield
        finally:
            self.unfreeze()


class BeamSearchSequenceGeneratorWithLanguageModel(GreedySequenceGenerator):
    def __init__(
        self, embedding, decoder, log_softmax, language_model, beam_size=1, len_pen=0, fusion_coef=0.0, **kwargs
    ):
        """
        Beam Search sequence generator based on the decoder followed by log_softmax
        with external language model fusion.
        Args:
            *all args of BeamSearchSequenceGenerator class
            language_model: nemo TransformerLMModel
            fusion_coef: coefficient before language model score, the resulting score is
                score = log P_NMT(y|x) + fusion_coef * log P_LM(y)
        Kwargs:
            all remaining parameters of GreedySequenceGenerator class
        """

        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.language_model = language_model
        self.beam_size = beam_size
        self.len_pen = len_pen
        self.fusion_coef = fusion_coef

    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        lm_mems_list=None,
        pos=0,
    ):

        nmt_log_probs, decoder_mems_list = super()._one_step_forward(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_input_mask,
            decoder_mems_list,
            pos,
        )
        input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()
        lm_hidden_states = self.language_model.encoder.embedding.forward(decoder_input_ids, start_pos=pos)

        lm_mems_list = self.language_model.encoder.encoder.forward(
            lm_hidden_states,
            input_mask,
            lm_mems_list,
            return_mems=True,
        )
        lm_log_probs = self.language_model.log_softmax.forward(hidden_states=lm_mems_list[-1][:, -1:])

        log_probs = nmt_log_probs + self.fusion_coef * lm_log_probs

        return log_probs, decoder_mems_list, lm_mems_list

    @staticmethod
    def compute_len_penalty(lengths, alpha):
        """Returns length penalty according to https://arxiv.org/pdf/1609.08144.pdf"""
        return ((5 + lengths) / 6).pow(alpha)

    def _forward(
        self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, return_beam_scores=False
    ):

        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs, decoder_mems_list, lm_mems_list = self._one_step_forward(
            tgt, encoder_hidden_states, encoder_input_mask, None, None, 0
        )
        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, 1), prefixes), dim=1)
        for j in range(len(decoder_mems_list)):
            decoder_mems_list[j] = decoder_mems_list[j].repeat(self.beam_size, 1, 1)
        for j in range(len(lm_mems_list)):
            lm_mems_list[j] = lm_mems_list[j].repeat(self.beam_size, 1, 1)

        # repeat source sequence beam_size times for beam search
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mems_list[0].size(2)
        lm_hidden_size = lm_mems_list[0].size(2)

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long()

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        for i in range(max_generation_length):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            log_probs, decoder_mems_list, lm_mems_list = self._one_step_forward(
                prefixes[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, lm_mems_list, i + 1
            )
            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size**2), self.beam_size, dim=1)
            scores = scores.view(-1, 1) * len_penalties

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size**2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
            for j in range(len(decoder_mems_list)):
                decoder_mems_list[j] = (
                    decoder_mems_list[j]
                    .view(-1, self.beam_size, p_len - 1, hidden_size)
                    .gather(1, mems_ids)
                    .view(-1, p_len - 1, hidden_size)
                )
            lm_mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, lm_hidden_size) // self.beam_size
            for j in range(len(lm_mems_list)):
                lm_mems_list[j] = (
                    lm_mems_list[j]
                    .view(-1, self.beam_size, p_len - 1, lm_hidden_size)
                    .gather(1, lm_mems_ids)
                    .view(-1, p_len - 1, lm_hidden_size)
                )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

        # select best performing hypotheses in each element of the batch
        len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
        scores = scores / len_penalties
        best_guesses = (
            torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
        )
        tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses).squeeze(1)

        if return_beam_scores:
            return prefixes, scores * len_penalties, tgt
        else:
            return tgt
