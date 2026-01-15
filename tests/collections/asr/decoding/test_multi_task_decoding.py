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

from unittest.mock import Mock

import pytest
import torch

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithFusionModels,
    GreedySequenceGenerator,
)
from nemo.collections.asr.parts.context_biasing import GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules.multitask_beam_decoding import TransformerAEDBeamInfer
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import TransformerAEDGreedyInfer
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier


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
