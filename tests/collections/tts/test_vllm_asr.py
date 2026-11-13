# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models import easy_magpietts_preference_optimization as preference
from nemo.collections.tts.models.nemotron_tts.v2 import model as v2_model
from nemo.collections.tts.parts.utils import helpers

pytestmark = pytest.mark.unit


def _response(text, finish_reason='stop'):
    return nullcontext(
        io.BytesIO(json.dumps({'choices': [{'finish_reason': finish_reason, 'message': {'content': text}}]}).encode())
    )


def test_vllm_asr_concurrent_requests_preserve_order_and_reference_free_prompt(tmp_path, monkeypatch):
    paths = [tmp_path / 'reference_should_not_be_a_prompt_first.wav', tmp_path / 'second.wav']
    for path, data in zip(paths, [b'first', b'second']):
        path.write_bytes(data)
    second_completed = threading.Event()
    requests = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        assert request.full_url == 'http://127.0.0.1:8000/v1/chat/completions'
        assert timeout == 17
        assert payload['temperature'] == 0 and payload['top_p'] == 1 and payload['seed'] == 0
        assert payload['skip_special_tokens'] is True
        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
        assert len(payload['messages']) == 1 and payload['messages'][0]['role'] == 'user'
        content = payload['messages'][0]['content']
        prompt = content[0]['text']
        assert 'repeated words' in prompt and 'numbers' in prompt and 'translate' in prompt
        assert 'reference_should_not_be_a_prompt' not in prompt
        audio = base64.b64decode(content[1]['audio_url']['url'].split(',', 1)[1])
        if audio == b'first':
            assert second_completed.wait(5), 'The second request did not run concurrently.'
            return _response('FIRST FIRST')
        second_completed.set()
        return _response('second second')

    monkeypatch.setattr(helpers, 'urlopen', urlopen)
    transcripts = helpers.transcribe_with_vllm_from_filepaths(
        paths, {'base_url': 'http://127.0.0.1:8000/v1/', 'max_concurrency': 2, 'timeout_seconds': 17}
    )
    assert transcripts == ['FIRST FIRST', 'second second']
    assert len(requests) == 2


def test_vllm_asr_normalizes_only_numeric_outputs_in_each_spoken_language(tmp_path, monkeypatch):
    text = ['52.25%', '432GB', 'one one', '٣']
    expected = ['fifty two point two five percent', 'vierhundertzweiunddreißig gigabyte', 'one one', 'ثلاثة']
    paths = [tmp_path / f'{index}.wav' for index in range(len(text))]
    for index, path in enumerate(paths):
        path.write_bytes(str(index).encode())
    normalizer_languages = []

    def urlopen(request, **kwargs):
        content = json.loads(request.data)['messages'][0]['content']
        index = int(base64.b64decode(content[1]['audio_url']['url'].split(',', 1)[1]))
        return _response(text[index])

    def normalizer(language):
        normalizer_languages.append(language)
        return SimpleNamespace(normalize=lambda value: expected[text.index(value)])

    monkeypatch.setattr(helpers, 'urlopen', urlopen)
    monkeypatch.setattr(helpers, 'get_cached_text_normalizer', normalizer)
    assert (
        helpers.transcribe_with_vllm_from_filepaths(
            paths, {'base_url': 'http://localhost/v1'}, languages=['en', 'de-DE', 'fr', 'ar']
        )
        == expected
    )
    assert normalizer_languages == ['en', 'de', 'ar']


def test_vllm_asr_numeric_normalization_can_be_disabled(tmp_path, monkeypatch):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    monkeypatch.setattr(helpers, 'urlopen', lambda *args, **kwargs: _response('123'))
    monkeypatch.setattr(
        helpers, 'get_cached_text_normalizer', lambda language: pytest.fail('Normalization is disabled.')
    )
    assert helpers.transcribe_with_vllm_from_filepaths(
        [path], {'base_url': 'http://localhost/v1', 'normalize_transcripts': False}, languages=['en']
    ) == ['123']


@pytest.mark.parametrize('retry', ['vierhundertzweiunddreißig gigabyte', '432gb'])
def test_vllm_asr_retries_case_sensitive_units_only_when_digit_count_decreases(tmp_path, monkeypatch, retry):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    calls = []

    def normalize(text):
        calls.append(text)
        return '432GB' if text == '432GB' else retry

    monkeypatch.setattr(helpers, 'urlopen', lambda *args, **kwargs: _response('432GB'))
    monkeypatch.setattr(helpers, 'get_cached_text_normalizer', lambda language: SimpleNamespace(normalize=normalize))
    expected = retry if not any(char.isdecimal() for char in retry) else '432GB'
    assert helpers.transcribe_with_vllm_from_filepaths(
        [path], {'base_url': 'http://localhost/v1'}, languages=['de']
    ) == [expected]
    assert calls == ['432GB', '432gb']


def test_vllm_asr_keeps_recognized_cased_normalization_without_retrying(tmp_path, monkeypatch):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    calls = []

    def normalize(text):
        calls.append(text)
        return 'Four Hundred Thirty Two Gigabytes'

    monkeypatch.setattr(helpers, 'urlopen', lambda *args, **kwargs: _response('432GB'))
    monkeypatch.setattr(helpers, 'get_cached_text_normalizer', lambda language: SimpleNamespace(normalize=normalize))
    assert helpers.transcribe_with_vllm_from_filepaths(
        [path], {'base_url': 'http://localhost/v1'}, languages=['en']
    ) == ['Four Hundred Thirty Two Gigabytes']
    assert calls == ['432GB']


def test_vllm_asr_rejects_mismatched_language_metadata():
    with pytest.raises(ValueError, match='one language per audio file'):
        helpers.transcribe_with_vllm_from_filepaths(
            ['first.wav', 'second.wav'], {'base_url': 'http://localhost/v1'}, languages=['en']
        )


def test_vllm_asr_missing_or_unsupported_language_keeps_the_original_transcript(tmp_path, monkeypatch):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    seen = []

    def normalizer(language):
        seen.append(language)
        return None

    monkeypatch.setattr(helpers, 'urlopen', lambda *args, **kwargs: _response('123'))
    monkeypatch.setattr(helpers, 'get_cached_text_normalizer', normalizer)
    assert helpers.transcribe_with_vllm_from_filepaths(
        [path, path], {'base_url': 'http://localhost/v1'}, languages=[None, 'zz']
    ) == ['123', '123']
    assert seen == [None, 'zz']


def test_text_normalizers_and_unsupported_languages_are_cached(monkeypatch):
    created, warnings = [], []

    def create_normalizer(**options):
        created.append(options)
        if options['lang'] == 'zz':
            raise ValueError('unsupported language')
        return object()

    monkeypatch.setitem(
        sys.modules, 'nemo_text_processing.text_normalization.normalize', SimpleNamespace(Normalizer=create_normalizer)
    )
    monkeypatch.setattr(helpers.logging, 'warning', warnings.append)
    helpers.get_cached_text_normalizer.cache_clear()
    try:
        assert helpers.get_cached_text_normalizer('en') is helpers.get_cached_text_normalizer('en')
        for _ in range(2):
            assert helpers.get_cached_text_normalizer('zz') is None
            assert helpers.get_cached_text_normalizer(None) is None
        assert created == [{'input_case': 'cased', 'lang': 'en'}, {'input_case': 'cased', 'lang': 'zz'}]
        assert len(warnings) == 2
    finally:
        helpers.get_cached_text_normalizer.cache_clear()


def test_whisper_normalization_retains_its_default_language(monkeypatch):
    languages = []
    monkeypatch.setattr(preference, 'get_cached_text_normalizer', languages.append)
    policy = SimpleNamespace()
    preference.OnlinePreferenceOptimizationMixin._get_cached_normalizer(policy, None)
    preference.OnlinePreferenceOptimizationMixin._get_cached_normalizer(policy, 'de')
    assert languages == ['en', 'de']


@pytest.mark.parametrize('finish_reason', ['length', 'content_filter', None])
def test_vllm_asr_rejects_unfinished_or_blocked_transcripts(tmp_path, monkeypatch, finish_reason):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    monkeypatch.setattr(helpers, 'urlopen', lambda *args, **kwargs: _response('do not score this', finish_reason))
    with pytest.raises(RuntimeError, match='did not finish normally'):
        helpers.transcribe_with_vllm_from_filepaths([path], {'base_url': 'http://localhost/v1'})


def test_vllm_asr_http_failure_does_not_become_an_empty_success(tmp_path, monkeypatch):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')

    def unavailable(*args, **kwargs):
        raise HTTPError('http://localhost/v1/chat/completions', 503, 'unavailable', {}, None)

    monkeypatch.setattr(helpers, 'urlopen', unavailable)
    with pytest.raises(RuntimeError, match='503'):
        helpers.transcribe_with_vllm_from_filepaths([path], {'base_url': 'http://localhost/v1'})


@pytest.mark.parametrize('text', [None, '', '   '])
def test_vllm_asr_completed_empty_transcript_remains_empty(tmp_path, monkeypatch, text):
    path = tmp_path / 'silence.wav'
    path.write_bytes(b'audio')
    monkeypatch.setattr(helpers, 'urlopen', lambda *args, **kwargs: _response(text))
    transcripts = helpers.transcribe_with_vllm_from_filepaths([path], {'base_url': 'http://localhost/v1'})
    assert transcripts == ['']
    assert preference.word_error_rate(transcripts, ['spoken words'], use_cer=True) == 1.0


@pytest.mark.parametrize(
    'message',
    [
        None,
        {},
        {'content': []},
        {'content': 0},
        {'content': None, 'refusal': 'blocked'},
        {'content': None, 'reasoning': 'The spoken words were routed here.'},
        {'content': None, 'reasoning_content': 'The spoken words were routed here.'},
        {'content': None, 'tool_calls': [{'function': {'name': 'transcribe'}}]},
        {'content': None, 'function_call': {'name': 'transcribe'}},
    ],
)
def test_vllm_asr_malformed_or_refused_message_is_not_an_empty_transcript(tmp_path, monkeypatch, message):
    path = tmp_path / 'audio.wav'
    path.write_bytes(b'audio')
    response = {'choices': [{'finish_reason': 'stop', 'message': message}]}
    monkeypatch.setattr(
        helpers, 'urlopen', lambda *args, **kwargs: nullcontext(io.BytesIO(json.dumps(response).encode()))
    )
    with pytest.raises(RuntimeError, match='returned no transcript'):
        helpers.transcribe_with_vllm_from_filepaths([path], {'base_url': 'http://localhost/v1'})


@pytest.mark.parametrize('options', [{'max_concurrency': 0}, {'timeout_seconds': 0}, {'max_tokens': 0}])
def test_vllm_asr_rejects_invalid_request_limits(options):
    with pytest.raises(ValueError, match='must be positive'):
        helpers.transcribe_with_vllm_from_filepaths(['unused.wav'], {'base_url': 'http://localhost/v1', **options})


@pytest.mark.parametrize('selection', [{'val_asr_backend': 'vllm'}, {'reward_asr_model': 'vllm'}])
def test_v2_remote_validation_does_not_load_local_asr_weights(monkeypatch, selection):
    def initialize_base(self, cfg, trainer=None):
        nn.Module.__init__(self)
        self._cfg = cfg

    def forbidden(*args, **kwargs):
        raise AssertionError('Local ASR weights must not load for the vLLM backend.')

    monkeypatch.setattr(v2_model.NemotronTTSBase, '__init__', initialize_base)
    monkeypatch.setattr(v2_model, 'init_training_alignment', lambda *args: None)
    monkeypatch.setattr(v2_model.WhisperProcessor, 'from_pretrained', forbidden)
    monkeypatch.setattr(v2_model.WhisperForConditionalGeneration, 'from_pretrained', forbidden)
    monkeypatch.setattr(v2_model.nemo_asr.models.EncDecRNNTBPEModel, 'from_pretrained', forbidden)
    monkeypatch.setattr(
        v2_model.nemo_asr.models.EncDecSpeakerLabelModel,
        'from_pretrained',
        lambda **kwargs: SimpleNamespace(freeze=lambda: None),
    )
    model = v2_model.NemotronTTSModel(
        OmegaConf.create({'run_val_inference': True, 'use_multilingual_asr': True, **selection})
    )
    assert model.val_asr_backend == 'vllm'
    assert model.whisper_model is None and model.whisper_processor is None and model._eval_asr_model is None


def test_remote_reward_initialization_and_multilingual_transcription_share_one_backend(monkeypatch):
    class Parent:
        def __init__(self, cfg, trainer):
            self.cfg = cfg
            self.trainer = trainer

    class Policy(preference.OnlinePreferenceOptimizationMixin, Parent):
        pass

    monkeypatch.setattr(
        preference.nemo_asr.models.EncDecRNNTBPEModel,
        'from_pretrained',
        lambda *args, **kwargs: pytest.fail('Parakeet must not load.'),
    )
    monkeypatch.setattr(
        v2_model.WhisperProcessor, 'from_pretrained', lambda *args, **kwargs: pytest.fail('Whisper must not load.')
    )
    monkeypatch.setattr(
        v2_model.WhisperForConditionalGeneration,
        'from_pretrained',
        lambda *args, **kwargs: pytest.fail('Whisper must not load.'),
    )
    monkeypatch.setattr(
        preference.nemo_asr.models.EncDecSpeakerLabelModel,
        'from_pretrained',
        lambda **kwargs: SimpleNamespace(freeze=lambda: None),
    )
    cfg = OmegaConf.create({'reference_free': True, 'reward_asr_model': 'vllm', 'vllm_asr': {'base_url': 'same/v1'}})
    policy = Policy(cfg)
    assert policy.val_asr_backend == 'vllm'
    assert policy.whisper_model is None and policy._eval_asr_model is None
    paths = ['english.wav', 'french.wav', 'hindi.wav']

    def transcribe(audio_paths, config, languages=None):
        assert audio_paths == paths and config.base_url == 'same/v1'
        assert languages == ['en', 'fr', 'hi']
        return ['ONE ONE', 'un un', 'एक एक']

    monkeypatch.setattr(preference, 'transcribe_with_vllm_from_filepaths', transcribe)
    assert policy._compute_pred_transcripts(paths, {'languages': ['en', 'fr', 'hi']}, 'vllm') == [
        'one one',
        'un un',
        'एक एक',
    ]
