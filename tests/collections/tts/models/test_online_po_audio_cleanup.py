# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO

pytestmark = pytest.mark.unit


def test_disabled_audio_saving_does_not_create_a_directory(tmp_path):
    model = SimpleNamespace(cfg={'save_reward_audios': False}, logger=None, trainer=None, global_rank=0)
    model._get_audio_dir = lambda: pytest.fail('No audio directory should be requested')
    paths = EasyMagpieTTSModelOnlinePO._save_waveforms_to_paths(
        model, torch.zeros(2, 8), torch.tensor([4, 8]), 'generated', 24000
    )
    assert paths == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('save_audio', [False, True])
@pytest.mark.parametrize('failure', [None, 'asr', 'utmos'])
def test_reward_consumers_receive_trimmed_waveforms(tmp_path, save_audio, failure):
    audio = torch.randn(2, 2400, requires_grad=True)
    lens = torch.tensor([1200, 2400])
    model = SimpleNamespace(
        cfg={'save_reward_audios': save_audio},
        global_rank=0,
        global_step=0,
        logger=SimpleNamespace(log_dir=str(tmp_path)),
        trainer=None,
        device='cpu',
        rollout_cfg_mode='off',
        inference_cfg_scale=1.0,
        max_decoder_steps=10,
        audio_sampling_temperature=1.0,
        audio_sampling_topk=10,
        phoneme_sampling_temperature=1.0,
        phoneme_sampling_topk=10,
        output_sample_rate=24000,
        use_utmos=True,
        frame_stacking_factor=2,
        best_cer_threshold=1.0,
        worst_cer_threshold=1.0,
        scale_rewards=True,
    )
    for name in ('_get_audio_dir', '_save_waveforms_to_paths'):
        setattr(model, name, MethodType(getattr(EasyMagpieTTSModelOnlinePO, name), model))
    model.repeat_items_in_batch = lambda batch, count: batch
    codes = torch.zeros(2, 1, 12, dtype=torch.long)
    model.infer_batch = lambda **kwargs: SimpleNamespace(
        predicted_audio=audio,
        predicted_audio_lens=lens,
        predicted_codes=codes,
        predicted_codes_lens=torch.tensor([10, 11]),
    )
    model._unstack_rollout_phoneme_tokens = lambda output: (None, None)
    model._print_group_cer_wer_table = lambda **kwargs: None
    seen = []

    def consume(name, waveforms):
        assert [wav.numel() for wav in waveforms] == [1200, 2400]
        assert all(not wav.requires_grad for wav in waveforms)
        assert all(isinstance(wav, torch.Tensor) for wav in waveforms)
        seen.append(name)
        if failure == name:
            raise RuntimeError('scorer failed')

    def asr(waveforms, *args):
        consume('asr', waveforms)
        return ['hello', 'hello']

    def speaker(waveforms, *args):
        consume('speaker', waveforms)
        return torch.ones(2, 3), torch.ones(2, 3)

    def utmos(waveforms):
        consume('utmos', waveforms)
        return [3.0, 4.0]

    model._compute_pred_transcripts = asr
    model._compute_speaker_embeddings_parallel = speaker
    model._compute_utmos_scores_batched = utmos
    batch = {'raw_texts': ['hello', 'hello']}
    if failure:
        with pytest.raises(RuntimeError, match='scorer failed'):
            EasyMagpieTTSModelOnlinePO.generate_and_reward(model, batch, 1)
    else:
        result = EasyMagpieTTSModelOnlinePO.generate_and_reward(model, batch, 1)
        assert result['predicted_codes'] is codes
        assert [item['utmos'] for item in result['metrics']] == [3.0, 4.0]
        assert seen == ['asr', 'speaker', 'utmos']
    assert len(list(tmp_path.rglob('*.wav'))) == (2 if save_audio else 0)
    assert not hasattr(model, '_tmp_reward_audio_dir')


def test_reward_router_receives_waveforms_and_sample_rate():
    seen = []

    def transcribe(audio, sample_rate, languages):
        seen.extend(audio)
        assert sample_rate == 24000
        assert languages == ['en', 'hi']
        assert [wav.numel() for wav in audio] == [2400, 4800]
        return ['HELLO', 'WORLD']

    model = SimpleNamespace(
        output_sample_rate=24000,
        _reward_asr_router=SimpleNamespace(transcribe=transcribe),
        _normalize_reward_transcript=False,
        reward_asr_log_samples=0,
        trainer=None,
    )
    result = EasyMagpieTTSModelOnlinePO._compute_pred_transcripts(
        model, [torch.ones(2400), torch.ones(4800)], {'languages': ['en', 'hi']}
    )
    assert result == ['hello', 'world']
    assert len(seen) == 2


def test_reference_audio_returns_true_lengths_and_source_rate():
    saved = []
    model = SimpleNamespace(sample_rate=22050, _save_waveforms_to_paths=lambda **kwargs: saved.append(kwargs))
    batch = {'context_audio': torch.ones(2, 200), 'context_audio_lens': torch.tensor([100, 200])}
    audio, rate = EasyMagpieTTSModelOnlinePO._get_reference_audio(model, batch)
    assert rate == 22050
    assert [wav.numel() for wav in audio] == [100, 200]
    assert saved[0]['sample_rate'] == 22050


def test_decoded_reference_uses_codec_rate_and_target_fallback():
    def decode(codes, lengths):
        assert codes.shape[-1] == 6
        assert lengths.tolist() == [6]
        assert codes.eq(7).all()
        return torch.ones(1, 100), torch.tensor([60]), None

    model = SimpleNamespace(
        output_sample_rate=24000,
        _codec_converter=None,
        _save_waveforms_to_paths=lambda **kwargs: [],
        _prepare_codes_for_decode=lambda codes, lens: (codes, lens),
        _codec_helper=SimpleNamespace(codes_to_audio=decode),
    )
    batch = {
        'context_audio_codes': torch.ones(1, 2, 2, dtype=torch.long),
        'context_audio_codes_lens': torch.tensor([2]),
        'audio_codes': torch.full((1, 2, 6), 7, dtype=torch.long),
        'audio_codes_lens': torch.tensor([6]),
    }
    waveforms, rate = EasyMagpieTTSModelOnlinePO._get_reference_audio(model, batch)
    assert rate == 24000
    assert waveforms[0].numel() == 60


@pytest.mark.parametrize('configured_device,expected_device', [('cuda', 'cuda:1'), ('cpu', 'cpu')])
def test_utmos_receives_actual_sample_rate_and_selected_device(configured_device, expected_device):
    seen = []

    def score(waveforms, **kwargs):
        assert len(waveforms) == 3
        assert kwargs == {'sample_rate': 24000, 'batch_size': 2, 'num_workers': 0, 'device': expected_device}
        seen.append(True)
        return [1.0, 2.0, 3.0]

    model = SimpleNamespace(
        use_utmos=True,
        device='cuda:1',
        output_sample_rate=24000,
        utmos_device=configured_device,
        cfg={'utmos_device': configured_device, 'utmos_batch_size': 2},
        _utmos_calculator=SimpleNamespace(score_waveforms=score),
    )
    assert EasyMagpieTTSModelOnlinePO._compute_utmos_scores_batched(model, [torch.ones(100)] * 3) == [1.0, 2.0, 3.0]
    assert seen == [True]
