# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, replace
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.supervision import AlignmentItem
from omegaconf import OmegaConf

from nemo.collections.tts.data.silence_trimming import trim_tts_silence
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset
from nemo.collections.tts.parts.utils.silence_trimming import (
    SilenceTrimConfig,
    silence_trim_config,
    trim_long_silences,
)

pytestmark = pytest.mark.unit


def waveform(sr):
    speech = (0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    return np.concatenate([np.zeros(2 * sr), speech, np.zeros(3 * sr), speech, np.zeros(2 * sr)]).astype(np.float32)


def recording(audio, sr, name):
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format='WAV', subtype='FLOAT')
    return Recording.from_bytes(buffer.getvalue(), recording_id=name)


def cut_with_audio(sr=16000):
    target = recording(waveform(sr), sr, 'target')
    cut = target.to_cut()
    cut.supervisions = [
        SupervisionSegment(
            id='sup',
            recording_id=target.id,
            start=0,
            duration=cut.duration,
            text='hello world',
            speaker='| Language:en Dataset:Granary Speaker:1 |',
            language='en',
            alignment={'word': [AlignmentItem(symbol='world', start=6, duration=1)]},
        )
    ]
    cut.custom = {
        'target_audio': target,
        'context_audio': recording(waveform(24000), 24000, 'context'),
        'silence_trim': True,
        'task': 'tts',
    }
    return cut


@pytest.mark.parametrize('sr', [8000, 16000, 22050, 24000, 44100, 48000])
def test_preserves_speech_and_caps_pauses(sr):
    audio = waveform(sr)
    result = trim_long_silences(audio, sr)
    assert len(result.removed) == 3
    assert len(result.audio) / sr == pytest.approx(4.5, abs=0.04)
    # All original speech samples are retained bit-for-bit, in order.
    for start, end in [(2 * sr, 3 * sr), (6 * sr, 7 * sr)]:
        mapped = round(result.map_time(start / sr, sr) * sr)
        np.testing.assert_array_equal(result.audio[mapped : mapped + end - start], audio[start:end])
    for (start, end), (a, b) in zip(result.silence, result.removed):
        expected = 0.5 if start == 0 or end == len(audio) else 1.5
        assert ((end - start) - (b - a)) / sr == pytest.approx(expected, abs=1 / sr)
    assert result.audio.dtype == audio.dtype


@pytest.mark.parametrize('kind', ['empty', 'zero', 'constant', 'nan', 'noise', 'short_pause'])
def test_uncertain_or_short_signals_unchanged(kind):
    sr = 16000
    rng = np.random.default_rng(0)
    audio = {
        'empty': np.empty(0, dtype=np.float32),
        'zero': np.zeros(3 * sr, dtype=np.float32),
        'constant': np.full(3 * sr, 0.002, dtype=np.float32),
        'nan': np.full(sr, np.nan, dtype=np.float32),
        'noise': rng.normal(0, 0.01, 3 * sr).astype(np.float32),
        'short_pause': np.concatenate([np.ones(sr), np.zeros(sr), np.ones(sr)]).astype(np.float32),
    }[kind]
    result = trim_long_silences(audio, sr)
    assert result.audio is audio
    assert result.removed == []


def test_low_amplitude_speech_and_background():
    sr = 16000
    rng = np.random.default_rng(1)
    audio = waveform(sr) * 0.01 + rng.normal(0, 1e-6, 9 * sr).astype(np.float32)
    result = trim_long_silences(audio, sr)
    assert len(result.removed) == 3
    assert all(not (a < 3 * sr and b > 2 * sr or a < 7 * sr and b > 6 * sr) for a, b in result.removed)


def test_quiet_speech_restarts_near_end():
    sr = 16000
    # Quiet speech can restart during an apparently trailing pause. Its energy
    # is >30dB below preceding speech and below the initial -40dBFS ceiling.
    t = np.arange(sr) / sr
    loud = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    quiet = (0.01 * np.sin(2 * np.pi * 200 * t[: sr // 4])).astype(np.float32)
    audio = np.concatenate([loud, np.zeros(sr // 2, dtype=np.float32), quiet])
    result = trim_long_silences(audio, sr)
    assert result.removed == []
    assert result.audio is audio


@pytest.mark.parametrize('value', [None, False])
def test_disabled(value):
    assert silence_trim_config(value) is None
    cut = cut_with_audio()
    cut.custom['silence_trim'] = value
    assert next(iter(trim_tts_silence(CutSet.from_cuts([cut]), True))) is cut


@pytest.mark.parametrize(
    'settings',
    [
        {'max_internal_silence': 0.1},
        {'max_edge_silence': 0},
        {'speech_pad': 0.001},
        {'hop_length': 0},
        {'noise_percentile': 100},
        {'noise_margin_db': -1},
        {'max_threshold_db': float('nan')},
        {'max_internal_silence': True},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        SilenceTrimConfig(**settings)


def test_omegaconf_tag():
    config = silence_trim_config(OmegaConf.create({'max_internal_silence': 2.0}))
    assert config.max_internal_silence == 2.0
    assert silence_trim_config(asdict(config)) == config


def test_trim_target_and_timeline_preserving_context_and_original():
    cut = cut_with_audio()
    original = cut.to_dict()
    prepared = next(iter(trim_tts_silence(CutSet.from_cuts([cut]), False)))
    assert cut.to_dict() == original
    assert prepared.duration == pytest.approx(4.5, abs=0.04)
    assert prepared.supervisions[0].duration == prepared.duration
    assert prepared.target_audio.duration == prepared.duration
    assert prepared.context_audio is cut.context_audio
    assert prepared.context_audio.duration == 9.0
    assert prepared.context_audio.sampling_rate == 24000
    np.testing.assert_array_equal(prepared.context_audio.load_audio()[0], waveform(24000))
    result = trim_long_silences(waveform(16000), 16000)
    assert prepared.supervisions[0].alignment['word'][0].start == result.map_time(6, 16000)
    # Memory WAV wrapping must not quantize the training signal.
    np.testing.assert_array_equal(prepared.target_audio.load_audio()[0], result.audio)


@pytest.mark.parametrize('field', ['target_codes', 'source_codes'])
def test_reject_cached_codes(field):
    cut = cut_with_audio()
    cut.custom[field] = 'unused'
    with pytest.raises(ValueError, match='online codes'):
        trim_tts_silence(CutSet.from_cuts([cut]), True)
    prepared = next(iter(trim_tts_silence(CutSet.from_cuts([cut]), False)))
    assert not prepared.has_custom(field)


@pytest.mark.parametrize('load_cached_codes', [False, True])
def test_context_not_decoded_and_cached_context_codes_preserved(monkeypatch, load_cached_codes):
    cut = cut_with_audio()
    context_codes = object()
    cut.custom['context_codes'] = context_codes

    def fail_context_load(*args, **kwargs):
        raise AssertionError('Silence trimming must not load context audio')

    monkeypatch.setattr(cut.context_audio, 'load_audio', fail_context_load)
    prepared = next(iter(trim_tts_silence(CutSet.from_cuts([cut]), load_cached_codes)))
    assert prepared.duration == pytest.approx(4.5, abs=0.04)
    assert prepared.context_audio is cut.context_audio
    assert prepared.context_codes is context_codes


def test_reject_dialogue():
    cut = cut_with_audio()
    cut.custom['task'] = 'dialog'
    with pytest.raises(ValueError, match='single-supervision'):
        trim_tts_silence(CutSet.from_cuts([cut]), False)


class FakeTokenizer:
    tokens = list(range(100))
    pad = 0

    def encode(self, text, tokenizer_name):
        return [10, 11]


@pytest.mark.parametrize('dataset_class', [MagpieTTSLhotseDataset, MagpieTTSLhotseMultiturnDataset])
def test_both_datasets_trim_before_padding_and_online_codes(dataset_class):
    kwargs = dict(
        sample_rate=16000,
        codec_model_samples_per_frame=320,
        num_audio_codebooks=8,
        load_cached_codes_if_available=False,
        volume_norm=False,
        load_16khz_audio=True,
        context_duration_min=6.0,
        context_duration_max=6.0,
    )
    if dataset_class is MagpieTTSLhotseMultiturnDataset:
        kwargs.update(codec_model_input_sample_rate=16000, frame_stacking_factor=1)
    dataset = dataset_class(**kwargs)
    dataset.text_tokenizer = FakeTokenizer()
    dataset.bos_id, dataset.eos_id, dataset.pad_id = 100, 101, 0
    dataset.cfg_unk_token_id, dataset.interruption_token_id = 102, 103
    tagged = cut_with_audio()
    untagged = replace(cut_with_audio(), id='untagged', custom=dict(cut_with_audio().custom, silence_trim=False))
    batch = dataset[CutSet.from_cuts([tagged, untagged])]
    assert batch['audio_lens'][0].item() / 16000 == pytest.approx(4.5, abs=0.04)
    assert batch['audio_lens'][1].item() == 9 * 16000
    assert batch['audio_lens_16khz'].tolist() == [6 * 16000] * 2
    assert batch['context_audio_lens'].tolist() == [6 * 16000] * 2
    if dataset_class is MagpieTTSLhotseMultiturnDataset:
        assert batch['text_lens'][0].item() == batch['agent_mask_lens'][0].item()
        assert batch['source_audio_lens'].tolist() == batch['audio_lens'].tolist()
