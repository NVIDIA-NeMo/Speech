# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO
from nemo.collections.tts.parts.utils import reward_audio

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def ram_audio_root(tmp_path, monkeypatch):
    # Lifecycle tests can run without Linux; storage placement is covered by
    # test_reward_audio.py, including the prohibition on a disk fallback.
    monkeypatch.setattr(reward_audio, '_RAM_AUDIO_ROOT', str(tmp_path))


def _make_model(tmp_path, save_reward_audios=False):
    model = SimpleNamespace(
        cfg={'save_reward_audios': save_reward_audios},
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
    )
    for name in ('_get_audio_dir', '_generate_and_reward'):
        if hasattr(EasyMagpieTTSModelOnlinePO, name):
            setattr(model, name, MethodType(getattr(EasyMagpieTTSModelOnlinePO, name), model))
    return model


@pytest.mark.parametrize('stage', ['write', 'asr', 'utmos'])
def test_generate_and_reward_removes_temporary_audio_on_failure(tmp_path, stage):
    model = _make_model(tmp_path)
    audio = MagicMock()
    audio.size.return_value = 1
    audio_lens = MagicMock()
    audio_lens.__getitem__.return_value.item.return_value = 2400
    model.repeat_items_in_batch = lambda batch, count: batch
    model.infer_batch = lambda **kwargs: SimpleNamespace(
        predicted_audio=audio,
        predicted_audio_lens=audio_lens,
        predicted_codes=object(),
        predicted_codes_lens=object(),
    )
    model._unstack_rollout_phoneme_tokens = lambda output: (None, None)
    artifacts = []

    def save(**kwargs):
        path = Path(model._get_audio_dir()) / 'generated.wav'
        path.write_bytes(b'iteration audio')
        artifacts.append(path)
        if stage == 'write':
            raise RuntimeError('reward failure')
        return [str(path)]

    def fail(*args):
        assert artifacts[0].exists()
        raise RuntimeError('reward failure')

    model._save_waveforms_to_paths = save
    model._compute_pred_transcripts = fail if stage == 'asr' else lambda *args: ['hello']
    model._compute_speaker_embeddings_parallel = lambda *args: (None, None)
    model._compute_utmos_scores_batched = fail

    with pytest.raises(RuntimeError, match='reward failure'):
        EasyMagpieTTSModelOnlinePO.generate_and_reward(model, {'raw_texts': ['hello']}, 1)

    assert artifacts
    assert not artifacts[0].exists()
    assert not artifacts[0].parent.exists()
    assert not hasattr(model, '_tmp_reward_audio_dir')


@pytest.mark.parametrize('save_reward_audios', [False, True])
def test_generate_and_reward_audio_lifetime_on_success(tmp_path, save_reward_audios):
    model = _make_model(tmp_path, save_reward_audios)
    artifacts = []
    result = {'predicted_codes': object(), 'advantages': object()}

    def reward(**kwargs):
        audio_dir = Path(model._get_audio_dir())
        for name in ('generated.wav', 'reference_context_audio.wav'):
            path = audio_dir / name
            path.write_bytes(b'iteration audio')
            artifacts.append(path)
        assert all(path.exists() for path in artifacts)
        return result

    model._generate_and_reward = reward
    actual = EasyMagpieTTSModelOnlinePO.generate_and_reward(model, {}, 1)

    assert actual is result
    assert all(path.exists() == save_reward_audios for path in artifacts)
    assert not hasattr(model, '_tmp_reward_audio_dir')


def test_temporary_audio_directories_are_unique_per_iteration(tmp_path):
    model = _make_model(tmp_path)
    directories = []

    def reward(**kwargs):
        directories.append(model._get_audio_dir())
        return {}

    model._generate_and_reward = reward
    for _ in range(2):
        EasyMagpieTTSModelOnlinePO.generate_and_reward(model, {}, 1)
    assert len(set(directories)) == 2
    assert all(not Path(path).exists() for path in directories)
