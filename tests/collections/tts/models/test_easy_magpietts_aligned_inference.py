# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel, TrainingMode

pytestmark = pytest.mark.unit


class _Config(dict):
    __getattr__ = dict.__getitem__


class _InferenceHarness(nn.Module):
    """Exercise streaming changes without constructing a codec or language model."""

    embed_aligned_text_tokens = EasyMagpieTTSInferenceModel.embed_aligned_text_tokens
    _prepare_streaming_input = EasyMagpieTTSInferenceModel._prepare_streaming_input
    _process_predictions = EasyMagpieTTSInferenceModel._process_predictions
    _predict_aligned_text_tokens = EasyMagpieTTSInferenceModel._predict_aligned_text_tokens
    _finalize_aligned_text_predictions = EasyMagpieTTSInferenceModel._finalize_aligned_text_predictions
    _prepare_codes_for_decode = EasyMagpieTTSInferenceModel._prepare_codes_for_decode
    streaming_step = EasyMagpieTTSInferenceModel.streaming_step
    streaming_finalize = EasyMagpieTTSInferenceModel.streaming_finalize

    def __init__(self):
        super().__init__()
        self.cfg = _Config(embedding_dim=4, use_user_speaking_end_token=True)
        self.pad_id, self.bos_id, self.eos_id = 0, 5, 6
        self.use_aligned_text_head = True
        self.aligned_text_feedback = True
        self.aligned_text_feedback_scale = 1.0
        self.audio_embeddings = nn.ModuleList([nn.Embedding(32, 4)])
        self.text_embeddings = nn.Embedding(10, 4)
        self.text_embed_calls = []
        self.aligned_text_head = nn.Linear(4, 10)
        self.register_buffer('_aligned_text_output_mask', torch.arange(10) < self.bos_id)
        self._aligned_text_output_mask[self.eos_id] = True
        self._aligned_text_output_mask[self.pad_id] = True
        self.phoneme_tokenizer = None
        self.repeat_context = False
        self.num_audio_codebooks, self.frame_stacking_factor = 1, 2
        self.audio_bos_id, self.audio_eos_id = 13, 14
        self.audio_user_speaking_id, self.audio_user_speaking_end_id = 15, 16
        self.codec_decode_batch_size = 0
        self.codec_sil_codes = torch.tensor([0])
        self._codec_helper = SimpleNamespace(codes_to_audio=self._decode_audio)
        with torch.no_grad():
            self.text_embeddings.weight.copy_(torch.arange(40).reshape(10, 4))
            self.aligned_text_head.weight.zero_()
            self.aligned_text_head.bias.zero_()
            self.aligned_text_head.bias[2] = 1.0

    def embed_text_tokens(self, tokens, **kwargs):
        self.text_embed_calls.append(tokens.detach().clone())
        return self.text_embeddings(tokens)

    def embed_audio_tokens(self, tokens):
        return torch.zeros(tokens.size(0), tokens.size(-1), self.cfg.embedding_dim, device=tokens.device)

    def _predict_audio_codes(self, state):
        codes = torch.zeros(state.config.batch_size, 2, dtype=torch.long, device=state.config.device)
        return codes, codes

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.zeros_like(inputs_embeds), past_key_values=None)

    @staticmethod
    def _decode_audio(codes, lengths):
        return torch.zeros(codes.size(0), int(lengths.max()) * 10), lengths * 10, None


def _state(batch_size=2, use_cfg=False):
    zeros = torch.zeros(batch_size, dtype=torch.long)
    return SimpleNamespace(
        config=SimpleNamespace(
            batch_size=batch_size,
            device=torch.device('cpu'),
            training_mode=TrainingMode('streaming', 0, 0, 0),
            use_cfg=use_cfg,
            dummy_context_embedding_unconditional=torch.zeros(1, 1, 4),
        ),
        full_context_embedding=torch.zeros(batch_size, 1, 4),
        full_context_lens=zeros.clone(),
        context_position=zeros.clone(),
        text_finished=torch.ones(batch_size, dtype=torch.bool),
        text_tokens_seen=zeros.clone(),
        turn_text_tokens_seen=zeros.clone(),
        phoneme_steps=zeros.clone(),
        phoneme_stream_ended=torch.ones(batch_size, dtype=torch.bool),
        audio_steps=zeros.clone(),
        finished=torch.zeros(batch_size, dtype=torch.bool),
        gt_audio_embeddings=None,
        gt_audio_lens=None,
        last_audio_codes=None,
        last_aligned_text_tokens=None,
        last_hidden=torch.zeros(batch_size * (2 if use_cfg else 1), 1, 4),
        all_predictions=[],
        all_phoneme_predictions=[],
        all_aligned_text_predictions=[],
        audio_prediction_start_idx=torch.full((batch_size,), -1),
        audio_prediction_end_idx=torch.full((batch_size,), -1),
        cache_seq_len=0,
        past_key_values=None,
    )


def test_feedback_embeds_distinct_tokens_once_and_keeps_correct_gradients():
    model = _InferenceHarness()
    tokens = torch.tensor([[2, 2, 0], [4, 2, 0]])
    embedded = model.embed_aligned_text_tokens(tokens)
    assert model.text_embed_calls[0].tolist() == [[2, 4]]
    assert len(model.text_embed_calls) == 1
    assert torch.equal(embedded[tokens == 2], model.text_embeddings.weight[2].expand(3, -1))
    assert embedded[tokens == 0].eq(0).all()
    embedded.sum().backward()
    assert model.text_embeddings.weight.grad[2].eq(3).all()
    assert model.text_embeddings.weight.grad[4].eq(1).all()
    assert model.text_embeddings.weight.grad[0].eq(0).all()


@pytest.mark.parametrize('shape', [(2, 3), (2, 0), (0, 3)])
def test_empty_or_all_pad_feedback_never_calls_cas(shape):
    model = _InferenceHarness()
    embedded = model.embed_aligned_text_tokens(torch.zeros(shape, dtype=torch.long))
    assert embedded.shape == (*shape, 4)
    assert embedded.eq(0).all()
    assert model.text_embed_calls == []


@torch.no_grad()
def test_first_audio_step_uses_bos_feedback_in_both_cfg_branches():
    model, state = _InferenceHarness(), _state(use_cfg=True)
    inputs, _, _, needs_audio = model._prepare_streaming_input(state, None, False)
    expected = model.text_embeddings.weight[model.bos_id].expand(4, 1, -1)
    assert needs_audio.all()
    assert torch.equal(inputs, expected)


@torch.no_grad()
def test_feedback_zeroes_inactive_items_and_pad_tokens():
    model, state = _InferenceHarness(), _state(batch_size=3)
    state.audio_steps[:] = 1
    state.last_aligned_text_tokens = torch.tensor([3, 0, 4])
    state.finished[2] = True
    inputs, _, _, needs_audio = model._prepare_streaming_input(state, None, False)
    assert needs_audio.tolist() == [True, True, False]
    assert torch.equal(inputs[0, 0], model.text_embeddings.weight[3])
    assert inputs[1:].eq(0).all()


@torch.no_grad()
def test_feedback_can_be_disabled_without_disabling_output_head():
    model, state = _InferenceHarness(), _state()
    model.aligned_text_feedback = False
    inputs, _, _, _ = model._prepare_streaming_input(state, None, False)
    assert inputs.eq(0).all()
    assert model.text_embed_calls == []


@torch.no_grad()
def test_head_masks_non_bpe_controls_and_uses_only_conditional_hidden_states():
    model, state = _InferenceHarness(), _state(use_cfg=True)
    model.aligned_text_head = nn.Identity()
    logits = torch.zeros(4, 1, 10)
    logits[:, :, model.bos_id] = 100
    logits[:, :, 7:] = 99  # CFG, interruption and pronunciation-control IDs.
    logits[0, 0, model.eos_id] = 9
    logits[1, 0, model.pad_id] = 8
    logits[2:, :, 4] = 1000  # Unconditional predictions must not be used.
    state.last_hidden = logits
    assert model._predict_aligned_text_tokens(state).tolist() == [model.eos_id, model.pad_id]


@torch.no_grad()
def test_text_eos_does_not_finish_audio_and_inactive_predictions_are_pad():
    model, state = _InferenceHarness(), _state()
    model.aligned_text_head.bias[model.eos_id] = 10
    needs_audio = torch.tensor([True, False])
    model._process_predictions(state, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), needs_audio)
    assert not state.finished.any()
    assert state.last_aligned_text_tokens.tolist() == [model.eos_id, model.pad_id]
    assert state.all_aligned_text_predictions[0].tolist() == [model.eos_id, model.pad_id]


@torch.no_grad()
def test_user_prefill_appends_pad_and_resets_next_assistant_feedback_to_bos():
    model, state = _InferenceHarness(), _state()
    state.last_aligned_text_tokens = torch.tensor([3, 4])
    state, audio, phonemes = model.streaming_step(state, prefill_like_step=True, prefill_like_is_last_step=True)
    assert audio is None and phonemes is None
    assert state.all_aligned_text_predictions[0].eq(model.pad_id).all()
    assert state.last_aligned_text_tokens.eq(model.bos_id).all()
    assert model.text_embed_calls == []
    inputs, _, _, _ = model._prepare_streaming_input(state, None, False)
    assert torch.equal(inputs, model.text_embeddings.weight[model.bos_id].expand(2, 1, -1))


def test_finalize_trims_prefill_mixed_starts_and_partial_stacked_frames():
    model, state = _InferenceHarness(), _state(batch_size=3)
    state.all_predictions = [torch.zeros(3, 1, 4)] + [torch.zeros(3, 1, 2) for _ in range(3)]
    raw_tokens = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 4, 4, 1], [0, 0, 0, 0, 0]])
    state.all_aligned_text_predictions = list(raw_tokens.unbind(dim=1))
    state.audio_prediction_start_idx = torch.tensor([4, 6, -1])
    state.audio_prediction_end_idx = torch.tensor([9, -1, -1])
    tokens, lengths = model._finalize_aligned_text_predictions(state)
    assert lengths.tolist() == [3, 2, 0]
    assert tokens.tolist() == [[1, 2, 3], [4, 1, 0], [0, 0, 0]]


@pytest.mark.parametrize('branch', ['empty', 'zero_length', 'decode'])
@torch.no_grad()
def test_all_finalize_branches_expose_aligned_tokens_before_codec_padding(branch):
    model, state = _InferenceHarness(), _state(batch_size=1)
    if branch != 'empty':
        state.all_predictions = [torch.zeros(1, 1, 2, dtype=torch.long)]
        state.all_aligned_text_predictions = [torch.tensor([3])]
        state.audio_prediction_start_idx = torch.tensor([0])
        state.audio_prediction_end_idx = torch.tensor([0 if branch == 'zero_length' else 1])
    result = model.streaming_finalize(state)
    expected_length = 1 if branch == 'decode' else 0
    assert result.aligned_text_tokens.shape == (1, expected_length)
    assert result.aligned_text_tokens_lens.tolist() == [expected_length]
    if branch == 'decode':
        assert result.aligned_text_tokens.tolist() == [[3]]
        assert result.audio_codes_len.tolist() == [4]  # Existing codec minimum-length padding adds no text labels.


def test_disabled_feature_leaves_optional_outputs_empty():
    model, state = _InferenceHarness(), _state()
    model.use_aligned_text_head = False
    assert model._finalize_aligned_text_predictions(state) == (None, None)


def test_finalize_detects_missing_prefill_text_history():
    model, state = _InferenceHarness(), _state()
    state.all_predictions = [torch.zeros(2, 1, 4)]
    with pytest.raises(ValueError, match='one label per audio decoder step'):
        model._finalize_aligned_text_predictions(state)


@torch.no_grad()
def test_audio_eos_after_bulk_prefill_uses_frame_count_instead_of_chunk_count():
    model, state = _InferenceHarness(), _state(batch_size=1)
    state.all_predictions = [torch.zeros(1, 1, 6, dtype=torch.long)]  # Three prefill decoder steps in one chunk.
    state.all_aligned_text_predictions = list(torch.zeros(1, 3, dtype=torch.long).unbind(dim=1))
    state.audio_steps[:] = 3
    eos = torch.full((1, 2), model.audio_eos_id, dtype=torch.long)
    model._predict_audio_codes = lambda current_state: (eos, eos)
    model._process_predictions(state, torch.tensor([False]), torch.tensor([False]), torch.tensor([True]))
    assert state.audio_prediction_start_idx.tolist() == [6]
    assert state.audio_prediction_end_idx.tolist() == [6]
    tokens, lengths = model._finalize_aligned_text_predictions(state)
    assert tokens.shape == (1, 0)
    assert lengths.tolist() == [0]
