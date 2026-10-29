# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import functools
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM, TransformerEncoderNM
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithFusionModels,
    BeamSearchSequenceGeneratorWithLanguageModel,
    EnsembleBeamSearchSequenceGenerator,
    GreedySequenceGenerator,
)
from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, RMSNorm
from nemo.collections.asr.parts.context_biasing import GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules.multitask_beam_decoding import AEDBeamInferConfig, TransformerAEDBeamInfer
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import (
    AEDGreedyInferConfig,
    TransformerAEDGreedyInfer,
)
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.utils import logging


@pytest.fixture()
def deterministic_rng():
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)


@pytest.fixture()
def decoder_nm(deterministic_rng):
    return TransformerDecoderNM(
        vocab_size=8,
        hidden_size=2,
        num_layers=1,
        inner_size=4,
        num_attention_heads=1,
        max_sequence_length=32,
    ).eval()


@pytest.fixture()
def nnet(decoder_nm):
    ans = (
        decoder_nm.embedding,
        decoder_nm.decoder,
        TokenClassifier(hidden_size=2, num_classes=8),
    )
    ans = tuple(m.eval() for m in ans)
    return ans


@pytest.fixture()
def inputs():
    B, T, C = 1, 5, 2
    return (
        torch.tensor([[1]], dtype=torch.long),  # decoder_input_ids
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


@pytest.fixture()
def tokenizer():
    tok = Mock()
    tok.pad = 0
    tok.bos = 1
    tok.eos = 2
    return tok


@pytest.mark.parametrize('with_confidence', [False, True])
@pytest.mark.parametrize('return_xattn_scores', [False, True])
def test_greedy_decoding(inputs, nnet, deterministic_rng, with_confidence, return_xattn_scores):
    gen = GreedySequenceGenerator(
        *nnet, return_xattn_scores=return_xattn_scores, preserve_step_confidence=with_confidence
    )
    output = gen(*inputs)

    assert len(output) == 4
    best_path, hypotheses, confidence, xattn_list = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 25)
    if return_xattn_scores:
        assert len(xattn_list) == len(nnet[1].layers)
        assert xattn_list[0].shape == (1, 1, 24, 5)
    else:
        assert xattn_list is None

    assert hypotheses is None

    if with_confidence:
        assert confidence is not None
        assert torch.is_tensor(confidence)
        assert confidence.shape == best_path.shape
    else:
        assert confidence is None


@pytest.mark.parametrize('return_xattn_scores', [False, True])
def test_temperature_sampling_decoding(inputs, nnet, return_xattn_scores):
    gen = GreedySequenceGenerator(*nnet, return_xattn_scores=return_xattn_scores, temperature=10.0, n_samples=2)
    output = gen(*inputs)

    assert len(output) == 4
    best_path, hypotheses, _, xatt_list = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape[0] == 1

    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 1
    (seq0,) = hypotheses
    assert seq0.shape[0] == 2
    assert (seq0[0] != seq0[1]).any()

    if return_xattn_scores:
        assert len(xatt_list) == len(nnet[1].layers)
        assert xatt_list[0].shape == (2, 1, 24, 5)
    else:
        assert xatt_list is None


def test_beam_decoding_beam_scores_false(inputs, nnet):
    gen = BeamSearchSequenceGenerator(*nnet, beam_size=2)
    output = gen(*inputs, return_beam_scores=False)

    assert len(output) == 1
    (best_path,) = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (26,)


@pytest.mark.parametrize('return_xattn_scores', [False, True])
def test_beam_decoding_beam_scores_true(inputs, nnet, return_xattn_scores):
    gen = BeamSearchSequenceGenerator(*nnet, return_xattn_scores=return_xattn_scores, beam_size=2)
    output = gen(*inputs, return_beam_scores=True)

    assert len(output) == 4
    beam_paths, scores, best_path, xatt_scores_list = output

    assert beam_paths is not None
    assert isinstance(beam_paths, list)
    assert len(beam_paths) == 1
    (beam_paths_seq0,) = beam_paths
    assert torch.is_tensor(beam_paths_seq0)
    assert beam_paths_seq0.shape == (2, 26)

    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 1
    (scores_seq0,) = scores
    assert torch.is_tensor(scores_seq0)
    assert scores_seq0.shape == (2,)

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 26)

    if return_xattn_scores:
        assert xatt_scores_list is not None
        assert isinstance(xatt_scores_list, list)
        assert torch.is_tensor(xatt_scores_list[0])
        assert xatt_scores_list[0].shape == (1, 1, 25, 5)
    else:
        assert xatt_scores_list is None


def test_beam_decoding_beam_scores_true_with_fusion_models(inputs, nnet):
    """Test decoding with dummy unigram LM and boosting tree"""
    # load dummy ngpu-lm
    lm = NGramGPULanguageModel.dummy_unigram_lm(vocab_size=8)

    # load dummy boosting tree
    boosting_tree = GPUBoostingTreeModel.dummy_boosting_tree(vocab_size=8)

    fusion_models = [lm, boosting_tree]
    fusion_models_alpha = [0.2, 0.2]

    gen = BeamSearchSequenceGeneratorWithFusionModels(
        *nnet,
        return_xattn_scores=True,
        fusion_models=fusion_models,
        fusion_models_alpha=fusion_models_alpha,
        beam_size=2,
    )
    output = gen(*inputs, return_beam_scores=True)

    assert len(output) == 4
    beam_paths, scores, best_path, xatt_scores_list = output

    assert beam_paths is not None
    assert isinstance(beam_paths, list)
    assert len(beam_paths) == 1
    (beam_paths_seq0,) = beam_paths
    assert torch.is_tensor(beam_paths_seq0)
    assert beam_paths_seq0.shape == (2, 26)

    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 1
    (scores_seq0,) = scores
    assert torch.is_tensor(scores_seq0)
    assert scores_seq0.shape == (2,)

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 26)

    assert xatt_scores_list is not None
    assert isinstance(xatt_scores_list, list)
    assert torch.is_tensor(xatt_scores_list[0])
    assert xatt_scores_list[0].shape == (1, 1, 25, 5)


class TestBeamSearchBatching:
    """Beam search over a batch gives every source sequence the hypotheses and scores it gets alone."""

    PAD, BOS, EOS, VOCAB, HIDDEN = 0, 1, 2, 12, 16

    @classmethod
    def _modules(cls, seed=0):
        torch.manual_seed(seed)
        decoder = TransformerDecoderNM(
            vocab_size=cls.VOCAB,
            hidden_size=cls.HIDDEN,
            num_layers=2,
            inner_size=24,
            num_attention_heads=2,
            max_sequence_length=40,
        ).eval()
        classifier = TokenClassifier(hidden_size=cls.HIDDEN, num_classes=cls.VOCAB, num_layers=1, log_softmax=True)
        for parameter in list(decoder.parameters()) + list(classifier.parameters()):
            parameter.data.normal_(0, 0.5)
        return decoder, classifier.eval()

    @classmethod
    def _encoder(cls, mask_future=False):
        encoder = TransformerEncoderNM(
            vocab_size=cls.VOCAB,
            hidden_size=cls.HIDDEN,
            num_layers=1,
            inner_size=24,
            num_attention_heads=2,
            max_sequence_length=40,
            mask_future=mask_future,
        ).eval()
        for parameter in encoder.parameters():
            parameter.data.normal_(0, 0.5)
        return encoder

    @classmethod
    def _language_model(cls):
        torch.manual_seed(2)
        encoder = cls._encoder(mask_future=True)
        log_softmax = TokenClassifier(hidden_size=cls.HIDDEN, num_classes=cls.VOCAB, num_layers=1, log_softmax=True)
        for parameter in log_softmax.parameters():
            parameter.data.normal_(0, 0.5)
        return SimpleNamespace(encoder=encoder, log_softmax=log_softmax.eval())

    @classmethod
    def _inputs(cls, prompt_length):
        torch.manual_seed(1)
        batch, frames = 4, 7
        encoder_states = torch.randn(batch, frames, cls.HIDDEN)
        encoder_mask = (torch.arange(frames)[None, :] < torch.tensor([7, 5, 6, 4])[:, None]).float()
        prompt = torch.full((batch, prompt_length), cls.BOS, dtype=torch.long)
        if prompt_length > 1:
            prompt[:, 1:] = torch.randint(3, cls.VOCAB, (batch, prompt_length - 1))
        return prompt, encoder_states, encoder_mask

    @staticmethod
    def _assert_batched_matches_single(generator, prompt, encoder_states, encoder_mask):
        prefixes, scores, best, _ = generator(prompt, encoder_states, encoder_mask, return_beam_scores=True)
        for i in range(prompt.shape[0]):
            single_prefixes, single_scores, single_best, _ = generator(
                prompt[i : i + 1], encoder_states[i : i + 1], encoder_mask[i : i + 1], return_beam_scores=True
            )
            assert torch.equal(best[i], single_best[0]), i
            assert torch.equal(prefixes[i], single_prefixes[0]), i
            torch.testing.assert_close(scores[i], single_scores[0], atol=1e-5, rtol=1e-5)

    @staticmethod
    def _assert_flat_sequence_matches(i, beam_size, batched, single):
        """Compare sequence ``i`` of flat search outputs (hypotheses along one batch axis) with its single run."""
        prefixes, scores, best = batched
        single_prefixes, single_scores, single_best = single
        assert torch.equal(best[i], single_best[0]), i
        assert torch.equal(prefixes.view(-1, beam_size, prefixes.shape[1])[i], single_prefixes), i
        torch.testing.assert_close(scores.view(-1, beam_size)[i], single_scores.view(-1), atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize("prompt_length", [1, 3])
    def test_beam_search_batched_matches_single_sequence(self, prompt_length):
        decoder, classifier = self._modules()
        generator = BeamSearchSequenceGenerator(
            decoder.embedding,
            decoder.decoder,
            classifier,
            beam_size=3,
            len_pen=0.5,
            bos=self.BOS,
            pad=self.PAD,
            eos=self.EOS,
            max_delta_length=4,
        )
        self._assert_batched_matches_single(generator, *self._inputs(prompt_length))

    @pytest.mark.unit
    def test_beam_search_with_fusion_models_batched_matches_single_sequence(self):
        decoder, classifier = self._modules()
        generator = BeamSearchSequenceGeneratorWithFusionModels(
            decoder.embedding,
            decoder.decoder,
            classifier,
            fusion_models=[NGramGPULanguageModel.dummy_unigram_lm(vocab_size=self.VOCAB)],
            fusion_models_alpha=[0.3],
            beam_size=3,
            len_pen=0.5,
            bos=self.BOS,
            pad=self.PAD,
            eos=self.EOS,
            max_delta_length=4,
        )
        self._assert_batched_matches_single(generator, *self._inputs(3))

    @pytest.mark.unit
    def test_beam_search_batched_preserves_cross_attention_scores_per_sequence(self):
        decoder, classifier = self._modules()
        generator = BeamSearchSequenceGenerator(
            decoder.embedding,
            decoder.decoder,
            classifier,
            beam_size=3,
            len_pen=0.5,
            bos=self.BOS,
            pad=self.PAD,
            eos=self.EOS,
            max_delta_length=4,
            return_xattn_scores=True,
        )
        prompt, encoder_states, encoder_mask = self._inputs(3)
        _, _, best, xatt_scores = generator(prompt, encoder_states, encoder_mask, return_beam_scores=True)
        assert len(xatt_scores) == len(decoder.decoder.layers)
        for i in range(prompt.shape[0]):
            _, _, single_best, single_xatt_scores = generator(
                prompt[i : i + 1], encoder_states[i : i + 1], encoder_mask[i : i + 1], return_beam_scores=True
            )
            assert torch.equal(best[i], single_best[0]), i
            for layer_scores, single_layer_scores in zip(xatt_scores, single_xatt_scores):
                assert layer_scores.shape[1:] == single_layer_scores.shape[1:]
                torch.testing.assert_close(layer_scores[i], single_layer_scores[0], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_beam_search_with_language_model_batched_matches_single_sequence(self):
        decoder, classifier = self._modules()
        generator = BeamSearchSequenceGeneratorWithLanguageModel(
            decoder.embedding,
            decoder.decoder,
            classifier,
            self._language_model(),
            beam_size=3,
            len_pen=0.5,
            fusion_coef=0.3,
            bos=self.BOS,
            pad=self.PAD,
            eos=self.EOS,
            max_delta_length=4,
        )
        prompt, encoder_states, encoder_mask = self._inputs(1)
        # the search itself, since the inherited __call__ unpacks cross-attention scores this class does not return
        with torch.inference_mode():
            batched = generator._forward(prompt, encoder_states, encoder_mask, return_beam_scores=True)
            for i in range(prompt.shape[0]):
                single = generator._forward(
                    prompt[i : i + 1], encoder_states[i : i + 1], encoder_mask[i : i + 1], return_beam_scores=True
                )
                self._assert_flat_sequence_matches(i, generator.beam_size, batched, single)

    @pytest.mark.unit
    @pytest.mark.parametrize("with_language_model", [False, True])
    def test_ensemble_beam_search_batched_matches_single_sequence(self, with_language_model):
        models = [self._modules(seed=seed) for seed in (0, 3)]
        encoders = []
        for seed in (4, 5):
            torch.manual_seed(seed)
            encoders.append(self._encoder())
        generator = EnsembleBeamSearchSequenceGenerator(
            encoders,
            [decoder.embedding for decoder, _ in models],
            [decoder.decoder for decoder, _ in models],
            [classifier for _, classifier in models],
            beam_size=3,
            len_pen=0.5,
            bos=self.BOS,
            pad=self.PAD,
            eos=self.EOS,
            max_delta_length=4,
            language_model=self._language_model() if with_language_model else None,
            fusion_coef=0.3 if with_language_model else None,
        )
        prompt, _, encoder_mask = self._inputs(1)
        torch.manual_seed(6)
        src_ids = torch.randint(3, self.VOCAB, encoder_mask.shape)
        batched = generator(src_ids, encoder_mask, prompt, return_beam_scores=True)
        for i in range(prompt.shape[0]):
            single = generator(src_ids[i : i + 1], encoder_mask[i : i + 1], prompt[i : i + 1], return_beam_scores=True)
            self._assert_flat_sequence_matches(i, generator.beam_size, batched, single)


class TestSequenceGeneratorsKeyValueCache:
    """The generators give the same hypotheses over the decoder's key-value cache as over the cached layer inputs."""

    PAD, BOS, EOS, VOCAB, HIDDEN = 0, 1, 2, 12, 16
    TOLERANCE = dict(atol=1e-5, rtol=1e-5)

    @pytest.fixture()
    def cached_modules(self):
        torch.manual_seed(0)
        decoder = TransformerDecoderNM(
            vocab_size=self.VOCAB,
            hidden_size=self.HIDDEN,
            num_layers=2,
            inner_size=24,
            num_attention_heads=2,
            max_sequence_length=40,
            num_token_types=0,
            pre_ln=True,
            norm=functools.partial(RMSNorm, eps=1e-6),
            self_attn=functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0),
        ).eval()
        classifier = TokenClassifier(hidden_size=self.HIDDEN, num_classes=self.VOCAB, num_layers=1, log_softmax=True)
        for parameter in list(decoder.parameters()) + list(classifier.parameters()):
            parameter.data.normal_(0, 0.5)
        # Trained models never emit the pad token before the end of a hypothesis;
        # a random model would, and the cache masks the pad position as the full pass does,
        # while the memory path attends to it.
        classifier.mlp.layer0.bias.data[self.PAD] = -30.0
        return decoder, classifier.eval()

    @pytest.fixture()
    def cached_inputs(self):
        torch.manual_seed(1)
        batch, frames = 3, 7
        prompt = torch.tensor([[self.BOS, 4], [self.BOS, 7], [self.BOS, 3]])
        encoder_states = torch.randn(batch, frames, self.HIDDEN)
        encoder_mask = (torch.arange(frames)[None, :] < torch.tensor([7, 5, 6])[:, None]).float()
        return prompt, encoder_states, encoder_mask

    def _generators(self, cls, decoder, classifier, **kwargs):
        return [
            cls(
                decoder.embedding,
                decoder.decoder,
                classifier,
                bos=self.BOS,
                pad=self.PAD,
                eos=self.EOS,
                use_kv_cache=flag,
                **kwargs,
            )
            for flag in (False, True)
        ]

    @pytest.mark.unit
    def test_greedy(self, cached_modules, cached_inputs):
        default, cached = self._generators(
            GreedySequenceGenerator,
            *cached_modules,
            max_delta_length=5,
            return_xattn_scores=True,
            preserve_step_confidence=True,
        )
        best, _, confidence, xatt = default(*cached_inputs)
        cached_best, _, cached_confidence, cached_xatt = cached(*cached_inputs)
        assert torch.equal(best, cached_best)
        torch.testing.assert_close(confidence, cached_confidence, **self.TOLERANCE)
        for layer in range(len(xatt)):
            torch.testing.assert_close(xatt[layer], cached_xatt[layer], **self.TOLERANCE)

    @pytest.mark.unit
    def test_temperature_sampling(self, cached_modules, cached_inputs):
        default, cached = self._generators(
            GreedySequenceGenerator, *cached_modules, max_delta_length=5, temperature=2.0, n_samples=2
        )
        torch.manual_seed(3)
        best, samples, _, _ = default(*cached_inputs)
        torch.manual_seed(3)
        cached_best, cached_samples, _, _ = cached(*cached_inputs)
        assert torch.equal(best, cached_best)
        assert all(torch.equal(a, b) for a, b in zip(samples, cached_samples))

    @pytest.mark.unit
    def test_beam(self, cached_modules, cached_inputs):
        default, cached = self._generators(
            BeamSearchSequenceGenerator,
            *cached_modules,
            beam_size=3,
            len_pen=0.5,
            max_delta_length=5,
            return_xattn_scores=True,
        )
        prefixes, scores, best, xatt = default(*cached_inputs, return_beam_scores=True)
        cached_prefixes, cached_scores, cached_best, cached_xatt = cached(*cached_inputs, return_beam_scores=True)
        assert torch.equal(best, cached_best)
        for i in range(len(prefixes)):
            assert torch.equal(prefixes[i], cached_prefixes[i])
            torch.testing.assert_close(scores[i], cached_scores[i], **self.TOLERANCE)
        for layer in range(len(xatt)):
            torch.testing.assert_close(xatt[layer], cached_xatt[layer], **self.TOLERANCE)

    @pytest.mark.unit
    def test_beam_with_fusion_models(self, cached_modules, cached_inputs):
        default, cached = self._generators(
            BeamSearchSequenceGeneratorWithFusionModels,
            *cached_modules,
            fusion_models=[NGramGPULanguageModel.dummy_unigram_lm(vocab_size=self.VOCAB)],
            fusion_models_alpha=[0.3],
            beam_size=3,
            len_pen=0.5,
            max_delta_length=5,
        )
        prefixes, scores, best, _ = default(*cached_inputs, return_beam_scores=True)
        cached_prefixes, cached_scores, cached_best, _ = cached(*cached_inputs, return_beam_scores=True)
        assert torch.equal(best, cached_best)
        for i in range(len(prefixes)):
            assert torch.equal(prefixes[i], cached_prefixes[i])
            torch.testing.assert_close(scores[i], cached_scores[i], **self.TOLERANCE)

    @pytest.mark.unit
    def test_aed_infer_and_configuration(self, cached_modules, cached_inputs, tokenizer):
        decoder, classifier = cached_modules
        prompt, encoder_states, encoder_mask = cached_inputs
        results = {}
        for flag in (False, True):
            greedy = TransformerAEDGreedyInfer(
                decoder, classifier, tokenizer, max_generation_delta=5, use_kv_cache=flag
            )
            beam = TransformerAEDBeamInfer(
                decoder, classifier, tokenizer, beam_size=3, max_generation_delta=5, use_kv_cache=flag
            )
            assert greedy.greedy_search.use_kv_cache is flag and beam.beam_search.use_kv_cache is flag
            results[flag] = [
                [
                    hyp.y_sequence
                    for hyp in infer(
                        encoder_hidden_states=encoder_states, encoder_input_mask=encoder_mask, decoder_input_ids=prompt
                    )[0]
                ]
                for infer in (greedy, beam)
            ]
        for default, cached in zip(results[False], results[True]):
            assert all(torch.equal(a, b) for a, b in zip(default, cached))

        config = MultiTaskDecodingConfig(
            strategy="greedy",
            greedy=AEDGreedyInferConfig(use_kv_cache=True),
            beam=AEDBeamInferConfig(beam_size=2, use_kv_cache=True),
        )
        decoding = MultiTaskDecoding(config, decoder, classifier, tokenizer)
        assert decoding.decoding.greedy_search.use_kv_cache is True
        decoding.change_strategy("beam")
        assert decoding.decoding.beam_search.use_kv_cache is True
        default = MultiTaskDecoding(MultiTaskDecodingConfig(strategy="beam"), decoder, classifier, tokenizer)
        assert default.decoding.beam_search.use_kv_cache is False


@pytest.fixture()
def prompted_inputs():
    B, T, C = 1, 5, 2
    return (
        torch.tensor([[1, 0, 2, 3, 4]], dtype=torch.long),  # prompt
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


def test_transformer_aed_beam_infer_strips_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDBeamInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path = ans[0][0].y_sequence
    assert best_path is not None
    assert torch.is_tensor(best_path)

    # Now run the underlying beam search generator that doesn't trim anything.
    *_, (untrimmed,), _ = gen.beam_search(*prompted_inputs, return_beam_scores=True)
    assert untrimmed is not None
    assert torch.is_tensor(untrimmed)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed[decoder_input_ids.shape[1] :], best_path
    )  # stripped the prompt from the beggining


def test_transformer_aed_greedy_infer_strips_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    decoder_input_ids = torch.tensor([[1, 0, 2, 3, 4]], dtype=torch.long)  # prompt
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDGreedyInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path = ans[0][0].y_sequence
    assert best_path is not None
    assert torch.is_tensor(best_path)

    # Now run the underlying beam search generator that doesn't trim anything.
    (untrimmed,), _, _, _ = gen.greedy_search(*prompted_inputs)
    assert untrimmed is not None
    assert torch.is_tensor(untrimmed)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed[decoder_input_ids.shape[1] :], best_path
    )  # stripped the prompt from the beggining


def test_transformer_aed_beam_infer_trims_xatt_scores(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim eos and pads in xatt from the end.
    gen = TransformerAEDBeamInfer(decoder_nm, classifier, tokenizer, return_xattn_scores=True)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    hyp = ans[0][0]

    assert hyp.xatt_scores is not None
    seq_len = hyp.y_sequence.shape[0]
    decoder_input_ids_len = decoder_input_ids.shape[1]
    total_expected_len = seq_len + decoder_input_ids_len - 1

    # Check that the expected trimming has indeed been done.
    for layer_scores in hyp.xatt_scores:
        assert layer_scores.shape[1] == total_expected_len


def test_transformer_aed_greedy_infer_trims_xatt_scores(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim eos and pads in xatt from the end.
    gen = TransformerAEDGreedyInfer(decoder_nm, classifier, tokenizer, return_xattn_scores=True)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    hyp = ans[0][0]

    assert hyp.xatt_scores is not None
    seq_len = hyp.y_sequence.shape[0]
    decoder_input_ids_len = decoder_input_ids.shape[1]
    total_expected_len = seq_len + decoder_input_ids_len - 1

    # Check that the expected trimming has indeed been done.
    for layer_scores in hyp.xatt_scores:
        assert layer_scores.shape[1] == total_expected_len


@pytest.mark.parametrize("len_pen", [0.0, 0.7, 1.0])
def test_multitask_decoding_beam_len_pen_reaches_the_generator(decoder_nm, nnet, tokenizer, len_pen):
    # the beam strategy reads `beam.len_pen`, the field of AEDBeamInferConfig
    *_, classifier = nnet
    cfg = MultiTaskDecodingConfig(strategy="beam", beam=AEDBeamInferConfig(beam_size=2, len_pen=len_pen))
    decoding = MultiTaskDecoding(cfg, decoder_nm, classifier, tokenizer)
    assert decoding.decoding.beam_search.len_pen == len_pen


def test_multitask_decoding_beam_len_pen_from_dict_config(decoder_nm, nnet, tokenizer):
    *_, classifier = nnet
    cfg = OmegaConf.create({"strategy": "beam", "beam": {"beam_size": 2, "len_pen": 0.5}})
    decoding = MultiTaskDecoding(cfg, decoder_nm, classifier, tokenizer)
    assert decoding.decoding.beam_search.len_pen == 0.5


def test_multitask_decoding_beam_length_penalty_alias(decoder_nm, nnet, tokenizer):
    # configs that worked around the old lookup with `length_penalty` keep their penalty, with a removal notice
    *_, classifier = nnet
    cfg = OmegaConf.create({"strategy": "beam", "beam": {"beam_size": 2, "length_penalty": 0.3}})
    with patch.object(logging, "warning") as warning:
        decoding = MultiTaskDecoding(cfg, decoder_nm, classifier, tokenizer)
    assert decoding.decoding.beam_search.len_pen == 0.3
    assert "`beam.length_penalty` is deprecated and will be removed in the next release" in warning.call_args.args[0]


def test_multitask_decoding_beam_len_pen_default_is_zero(decoder_nm, nnet, tokenizer):
    # a config without the field decodes without a length penalty, as before the lookup read `len_pen`
    *_, classifier = nnet
    cfg = MultiTaskDecodingConfig(strategy="beam", beam=AEDBeamInferConfig(beam_size=2))
    decoding = MultiTaskDecoding(cfg, decoder_nm, classifier, tokenizer)
    assert decoding.decoding.beam_search.len_pen == 0.0
