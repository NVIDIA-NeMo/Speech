# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.modules.aligner import init_aligned_text_head
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream, _model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('use_cfg', [False, True])
def test_output_only_head_preserves_audio_and_backbone_states_when_head_weights_change(use_cfg):
    baseline, observed = _model(), _head_model()
    states = [_init_stream(model, use_cfg=use_cfg, ragged=True) for model in (baseline, observed)]
    tokens = [torch.tensor([7, 10]), torch.tensor([8, 11]), torch.tensor([9, 12])]
    tokens += [torch.full((2,), observed.eos_id)] * 3 + [None] * 3
    with patch.object(observed.aligned_text_head, 'forward', wraps=observed.aligned_text_head.forward) as head:
        for token in tokens:
            with torch.no_grad():
                for parameter in observed.aligned_text_head.parameters():
                    parameter.normal_()
            results = [model.streaming_step(state, token) for model, state in zip((baseline, observed), states)]
            states = [result[0] for result in results]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            torch.testing.assert_close(states[0].finished, states[1].finished)
            assert states[0].cache_seq_len == states[1].cache_seq_len
            assert results[0][2] is results[1][2] is None
            if results[0][1] is not None:
                torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
                assert head.call_args.args[0].shape == (2, 16)
    assert states[0].all_aligned_text_predictions is None
    assert len(states[1].all_aligned_text_predictions) == len(states[1].all_predictions)
    output = observed.streaming_finalize(states[1])
    assert output.aligned_text_tokens.shape[0] == 2
    assert (output.aligned_text_tokens_lens > 0).all()


def test_inactive_items_get_pad_while_text_eos_does_not_end_audio():
    model = _head_model()
    state = _init_stream(model, ragged=True)
    predicted = torch.full((2,), model.eos_id)
    with patch.object(model, '_predict_aligned_text_tokens', return_value=predicted):
        for _ in range(8):
            state, codes, _ = model.streaming_step(state, torch.full((2,), 7))
            if codes is not None:
                break
    assert state.all_aligned_text_predictions[0].tolist() == [model.pad_id, model.eos_id]
    assert not state.finished.any()


def test_profile_prefill_and_text_warmup_do_not_invoke_the_head():
    model = _head_model()
    state = _init_stream(model)
    with patch.object(model.aligned_text_head, 'forward', side_effect=AssertionError('Prefill projected text')):
        state = model.streaming_prefill(state, torch.tensor([[7, 8], [9, 10]]))
        state, _, _ = model.streaming_step(state, torch.tensor([11, 12]), prefill_like_step=True)
    assert len(state.all_aligned_text_predictions) == 3
    assert torch.stack(state.all_aligned_text_predictions).eq(model.pad_id).all()


@pytest.mark.parametrize('eos_subframe', [0, 1])
def test_partial_final_audio_step_keeps_one_label_without_codec_padding_labels(eos_subframe):
    model = _head_model()
    state = _init_stream(model)

    def predict(current, needs_audio=None):
        codes = torch.zeros(2, model.num_audio_codebooks, 2, dtype=torch.long)
        codes[:, :, eos_subframe] = model.audio_eos_id
        return codes.reshape(2, -1), codes.reshape(2, -1)

    model._predict_audio_codes = predict
    with patch.object(model, '_predict_aligned_text_tokens', return_value=torch.tensor([7, 8])):
        for token in [model.bos_id, 7, model.eos_id, model.pad_id, model.pad_id]:
            state, _, _ = model.streaming_step(state, torch.full((2,), token))
        state, _, _ = model.streaming_step(state, None)
    assert state.finished.all()
    output = model.streaming_finalize(state)
    assert output.aligned_text_tokens_lens.tolist() == [eos_subframe, eos_subframe]
    assert output.aligned_text_tokens.shape == (2, eos_subframe)
    if eos_subframe:
        assert output.aligned_text_tokens.tolist() == [[7], [8]]
        assert output.audio_codes_len.tolist() == [4, 4]


def test_one_frame_prediction_masks_special_tokens_and_reads_only_conditional_states():
    model = _head_model()
    state = SimpleNamespace(config=SimpleNamespace(batch_size=2), last_hidden=torch.randn(4, 1, 16))
    logits = torch.zeros(2, model.text_vocab_size)
    logits[:, model.bos_id] = 100
    logits[:, model.cfg_unk_token_id] = 90
    logits[0, 7] = logits[1, model.eos_id] = 5
    with patch.object(model.aligned_text_head, 'forward', return_value=logits) as head:
        assert model._predict_aligned_text_tokens(state).tolist() == [7, model.eos_id]
    torch.testing.assert_close(head.call_args.args[0], state.last_hidden[:2, -1])


def test_no_audio_returns_an_empty_aligned_text_sequence():
    model = _head_model()
    output = model.streaming_finalize(_init_stream(model))
    assert output.aligned_text_tokens.shape == (2, 0)
    assert output.aligned_text_tokens_lens.tolist() == [0, 0]


def _head_model():
    model = _model()
    model.text_vocab_size = 64
    model._cfg.use_aligned_text_head = True
    model._cfg.aligned_text_head_dim = 4
    init_aligned_text_head(model, model.cfg)
    return model
