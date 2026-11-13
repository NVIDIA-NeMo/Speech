# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-owned phoneme EOS and decode preparation without host scalar reads."""
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.cfg_pairing import CFG_PAIR_METADATA_KEY
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration as Model
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner as Runner
from torch import nn
from torch.profiler import ProfilerActivity, profile


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('ended', [False, True])
def test_decode_preparation_does_not_read_device_scalars(device, ended):
    model = _model(device)
    info = _info(device)
    info['phoneme_ended'] = ended
    info.update(_decode(model, info))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _decode(model, info)
    keys = {event.key for event in prof.key_averages()}
    assert 'aten::item' not in keys
    assert 'aten::_local_scalar_dense' not in keys
    if device == 'cuda':
        assert 'cudaStreamSynchronize' not in keys


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_eos_consumed_once_then_stays_masked_across_reordering(device):
    model = _model(device)
    a, b = _info(device, phonemes=[2, 9]), _info(device, phonemes=[3, 4])
    a.update(_decode(model, a, row=0))
    b.update(_decode(model, b, row=1))
    assert model._dec_phoneme_valid.tolist()[:2] == [1, 1]
    assert isinstance(a['phoneme_ended'], torch.Tensor)
    assert a['phoneme_ended'].device.type == device
    assert a['phoneme_ended'].item() and not b['phoneme_ended'].item()
    previous = a['phoneme_ended'].clone()
    a['last_phoneme_token'] = torch.tensor([[5, 6]], device=device)
    # Each request carries its state to its new physical row.
    b.update(_decode(model, b, row=0))
    a.update(_decode(model, a, row=1))
    assert model._dec_phoneme_valid.tolist()[:2] == [1, 0]
    torch.testing.assert_close(a['phoneme_ended'], previous)
    # A fresh request reuses the ended request's row without inheriting its EOS.
    fresh = _info(device)
    fresh.update(_decode(model, fresh, row=1))
    assert model._dec_phoneme_valid[1].item() == 1
    assert not fresh['phoneme_ended'].item()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('ended', [False, True])
@pytest.mark.parametrize('offset', [0, 1, 2])
def test_phoneme_delay_bos_and_absent_feedback(device, ended, offset):
    model = _model(device)
    model.phonemes_delay = 1
    info = _info(device)
    info.update(decode_offset=offset, phoneme_ended=ended, last_phoneme_token=None)
    update = _decode(model, info)
    assert model._dec_phoneme_valid[0].item() == (offset == 1 and not ended)
    assert update['phoneme_ended'].item() == ended
    if offset == 1 and not ended:
        assert model._dec_phoneme_tokens[0].tolist() == [8, 8]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_eos_state_survives_missing_feedback_and_is_not_mutated(device):
    model = _model(device)
    info = _info(device)
    state = torch.tensor(False, device=device)
    info.update(phoneme_ended=state, last_phoneme_token=torch.tensor([[9, 2]], device=device))
    update = _decode(model, info)
    assert not state.item()  # An outstanding earlier update keeps its owned value.
    info.update(update, last_phoneme_token=None)
    update = _decode(model, info)
    assert update['phoneme_ended'].item()
    assert model._dec_phoneme_valid[0].item() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_preprocess_eos_state_uses_native_request_storage(device):
    model = _model(device)
    runner = object.__new__(Runner)
    runner.model = model
    runner.requests = {'a': SimpleNamespace()}
    runner.model_intermediate_buffer = {}
    info = _info(device, phonemes=[9, 2])
    update = _decode(model, info)
    runner._update_intermediate_buffer('a', update)
    stored = runner.model_intermediate_buffer['a']['phoneme_ended']
    assert stored.device.type == device
    update['phoneme_ended'].zero_()
    assert stored.item()  # Native storage must own the EOS tensor, not alias it.
    info.update(runner.model_intermediate_buffer['a'])
    _decode(model, info, row=2)
    assert model._dec_phoneme_valid[2].item() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('offset', [0, 1, 2, 4])
def test_text_and_audio_delay_and_fallback(device, offset):
    model = _model(device)
    model.has_phoneme = False
    model.speech_delay = 1
    info = _info(device)
    info.update(decode_offset=offset, text_tokens=[4, 5, 6], last_audio_codes=None)
    update = _decode(model, info)
    assert 'phoneme_ended' not in update
    assert model._dec_text_mask[0].item() == (offset < 3)
    if offset < 3:
        assert model._dec_text_tokens[0].item() == info['text_tokens'][offset]
    assert model._dec_audio_valid[0].item() == (offset >= 1)
    if offset >= 1:
        assert model._dec_audio_codes[0].tolist() == [7, 7]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_private_decode_masks_text_and_phonemes_with_device_eos(device):
    model = _model(device)
    model.arch.cfg_capability = {'version': 1}
    info = _info(device, phonemes=[9, 2])
    info.update(
        phoneme_ended=torch.tensor(False, device=device),
        use_cfg=True,
        _omni_is_prefill=False,
        **{CFG_PAIR_METADATA_KEY: {'branch': 'unconditional'}},
    )
    ids = torch.zeros(1, dtype=torch.long, device=device)
    update = model.preprocess(ids, None, **info)[2]
    assert model._dec_text_mask[0].item() == 0
    assert model._dec_phoneme_valid[0].item() == 0
    assert model._dec_audio_valid[0].item() == 1
    assert update['phoneme_ended'].item()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_user_audio_prefill_resets_owned_eos_for_next_turn(device):
    model = _model(device)
    model.arch.use_user_speaking_end_token = True
    model.arch.audio_user_speaking_end_id = 6
    model._maybe_set_code_sampling_params = lambda info: None
    model._build_user_audio_prefill_chunk = lambda **kwargs: torch.zeros(1, 4, device=device)
    runner = object.__new__(Runner)
    runner.model = model
    runner.requests = {'a': SimpleNamespace()}
    ended = torch.tensor(True, device=device)
    runner.model_intermediate_buffer = {'a': dict(phoneme_ended=ended)}
    info = _info(device, phonemes=[9, 9])
    info.update(
        runner.model_intermediate_buffer['a'],
        speaker_reference_prefilled=True,
        reference_conditioning_len=2,
        _omni_num_computed_tokens=4,
        _omni_prompt_len=5,
    )
    ids = torch.zeros(1, dtype=torch.long, device=device)
    update = model._preprocess_prefill(ids, None, 1, ids.device, info)[2]
    runner._update_intermediate_buffer('a', update)
    assert ended.item()  # Reset must not mutate the preceding turn's state.
    info.update(runner.model_intermediate_buffer['a'])
    update = _decode(model, info)
    assert not update['phoneme_ended'].item()
    assert model._dec_phoneme_valid[0].item() == 1
    assert model._dec_phoneme_tokens[0].tolist() == [8, 8]
    assert model._dec_audio_codes[0].tolist() == [6, 6]


def _model(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = object.__new__(Model)
    nn.Module.__init__(model)
    model.has_phoneme = True
    model.phonemes_delay = model.speech_delay = 0
    model.phoneme_eos_id, model.phoneme_bos_id, model.arch = (
        9,
        8,
        SimpleNamespace(phoneme_stacking_factor=2, audio_bos_id=7, use_user_speaking_end_token=False),
    )
    model.num_codebooks, model.embedding_dim = 2, 4
    model._combined_embeddings = torch.zeros(3, 4, device=device)
    for name, shape in [
        ('text_tokens', (3,)),
        ('text_mask', (3,)),
        ('phoneme_tokens', (3, 2)),
        ('phoneme_valid', (3,)),
        ('audio_codes', (3, 2)),
        ('audio_valid', (3,)),
    ]:
        setattr(model, '_dec_' + name, torch.zeros(shape, dtype=torch.long, device=device))
    return model


def _info(device, phonemes=(1, 2)):
    return dict(
        decode_offset=2,
        text_tokens=[3, 4, 5],
        last_phoneme_token=torch.tensor([phonemes], device=device),
        last_audio_codes=torch.tensor([[4, 5]], device=device),
    )


def _decode(model, info, row=0):
    ids = torch.zeros(1, dtype=torch.long, device=model._combined_embeddings.device)
    return model._preprocess_decode(ids, row, ids.device, info)[2]
