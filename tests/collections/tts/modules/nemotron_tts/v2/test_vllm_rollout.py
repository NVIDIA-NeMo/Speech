# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action and policy-version recovery for the optional vLLM rollout backend."""

import os
import signal
from io import StringIO
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBase
from nemo.collections.tts.modules.nemotron_tts.v2 import vllm_rollout
from nemo.collections.tts.modules.nemotron_tts.v2.vllm_rollout import VLLMRolloutClient, unpack_rollout_rows

pytestmark = pytest.mark.unit


def _client(work_dir):
    client = VLLMRolloutClient.__new__(VLLMRolloutClient)
    client.rank_dir = work_dir.resolve()
    client.version = 0
    client.startup_seconds = 0.0
    client.config = {}
    client.process = None
    client.log_file = None
    return client


def _model(stacking_factor=2, with_phonemes=True):
    model = SimpleNamespace(
        device=torch.device('cpu'),
        num_audio_codebooks=2,
        frame_stacking_factor=stacking_factor,
        phoneme_stacking_factor=2,
        phoneme_tokenizer=SimpleNamespace(bos_token_id=5, eos_token_id=6) if with_phonemes else None,
        audio_eos_id=33,
        mask_token_id=36,
    )
    model.unstack_codes = MethodType(NemotronTTSBase.unstack_codes, model)
    return model


def _row(audio, phones=(), audio_valid=True, phone_valid=False, audio_logps=None, phone_logps=None):
    audio_logps = audio_logps or [-0.5] * len(audio)
    phone_logps = phone_logps or [-0.25] * len(phones)
    return torch.tensor(
        [*audio, *phones, int(audio_valid), int(phone_valid), *audio_logps, *phone_logps], dtype=torch.float32
    )


def test_unpack_preserves_terminal_groups_and_aligns_behavior_probabilities():
    model = _model()
    first = torch.stack(
        [
            _row([36] * 4, [1, 2], audio_valid=False, phone_valid=True),
            _row([3, 4, 5, 6], [3, 4], phone_valid=True, audio_logps=[-0.1, -0.2, -0.3, -0.4]),
            _row([7, 33, 9, 10], [6, 2], phone_valid=True, audio_logps=[-0.6, -0.7, -0.8, -0.9]),
            _row([36] * 4, [4, 4], audio_valid=False, phone_valid=True),
        ]
    )
    capped = _row([11, 12, 13, 14], [2, 3], phone_valid=True).unsqueeze(0)

    rollout, decoded, lengths = unpack_rollout_rows(model, [first, capped])

    assert rollout['_po_audio_codes_lens'].tolist() == [2, 1]
    torch.testing.assert_close(rollout['_po_audio_codes_stacked'][0], first[1:3, :4].T.long())
    assert rollout['_po_audio_codes_stacked'][0, :, -1].tolist() == [7, 33, 9, 10]
    assert (rollout['_po_audio_codes_stacked'][1, :, -1] == model.mask_token_id).all()
    torch.testing.assert_close(rollout['_po_audio_behavior_logps'][0], first[1:3, 8:12])
    assert rollout['_po_audio_behavior_logps'].dtype == torch.float32
    assert rollout['_po_audio_behavior_logps'][1, 1].count_nonzero() == 0
    assert rollout['_po_phoneme_tokens'][0].tolist() == [5, 1, 2, 3, 4, 6, 2]
    assert rollout['_po_phoneme_tokens_lens'].tolist() == [7, 3]
    torch.testing.assert_close(rollout['_po_phoneme_behavior_logps'][0], first[:3, 12:14])
    assert lengths.tolist() == [3, 2]
    torch.testing.assert_close(decoded[0], torch.tensor([[3, 4, 7], [5, 6, 9]]))
    torch.testing.assert_close(decoded[1, :, :2], torch.tensor([[11, 12], [13, 14]]))


def test_first_frame_eos_retains_actions_without_decodable_audio():
    model = _model(stacking_factor=1, with_phonemes=False)

    rollout, decoded, lengths = unpack_rollout_rows(model, [_row([33, 7]).unsqueeze(0)])

    assert rollout['_po_audio_codes_lens'].tolist() == [1]
    assert rollout['_po_audio_codes_stacked'][0, :, 0].tolist() == [33, 7]
    assert '_po_phoneme_tokens' not in rollout
    assert lengths.tolist() == [0] and decoded.shape == (1, 2, 0)


@pytest.mark.parametrize('invalid_layout', [False, True])
def test_unpack_rejects_missing_acoustic_actions_or_wrong_layout(invalid_layout):
    packed = torch.zeros(1, 3 if invalid_layout else 14)

    with pytest.raises(ValueError, match='layout|no acoustic actions'):
        unpack_rollout_rows(_model(), [packed])


@pytest.mark.parametrize('mismatch', ['version', 'request_ids'])
def test_client_rejects_stale_or_reordered_responses_and_closes(tmp_path, monkeypatch, mismatch):
    client = _client(tmp_path)
    model = SimpleNamespace(
        device=torch.device('cpu'),
        global_step=0,
        global_rank=0,
        cfg={},
        tokenizer=SimpleNamespace(pad=0),
        default_inference_mode='test',
        mode_name_to_mode={'test': object()},
        prepare_context_tensors=lambda **kwargs: (torch.zeros(1, 1, 4), torch.tensor([1]), None, None),
        embed_text_tokens=lambda *args, **kwargs: torch.zeros(1, 2, 4),
    )
    batch = {
        'text': torch.tensor([[1, 2]]),
        'text_lens': torch.tensor([2]),
        'context_text_tokens': torch.zeros(1, 1, dtype=torch.long),
        'context_text_tokens_lens': torch.tensor([1]),
        'context_audio_codes': torch.zeros(1, 2, 1, dtype=torch.long),
        'context_audio_codes_lens': torch.tensor([1]),
    }
    calls = []

    def request(route, payload=None):
        calls.append(route)
        if route == '/rollout':
            sent = torch.load(payload['path'], weights_only=True)
            ids = [prompt['request_id'] for prompt in sent['prompts']]
            response = {
                'version': 1 if mismatch == 'version' else 0,
                'request_ids': [] if mismatch == 'request_ids' else ids,
            }
            path = tmp_path / 'response.pt'
            torch.save(response, path)
            return {'path': str(path)}
        return {}

    client._request = request
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)

    with pytest.raises(RuntimeError, match='policy version or request order'):
        client.generate(model, batch, max_audio_steps=3, temperature=0.7, top_k=80, phoneme_sampling_method='sample')

    assert calls == ['/wake', '/rollout']
    assert not hasattr(model, '_policy_rollout')


@pytest.mark.parametrize('phase', ['refit', 'context'])
def test_early_failure_closes_without_queuing_cleanup_rpc(tmp_path, monkeypatch, phase):
    client = _client(tmp_path)
    client.version = -1 if phase == 'refit' else 0
    client.converter = SimpleNamespace(select_weights=lambda *args: {'decoder.weight': torch.ones(1)})

    def prepare_context(**kwargs):
        raise ValueError('context failed')

    model = SimpleNamespace(
        global_step=0,
        cfg=SimpleNamespace(hidden_dim=4),
        default_inference_mode='test',
        mode_name_to_mode={'test': object()},
        prepare_context_tensors=prepare_context,
        state_dict=lambda: {},
    )
    batch = {
        'context_audio_codes': torch.zeros(1, 2, 1, dtype=torch.long),
        'context_audio_codes_lens': torch.tensor([1]),
        'context_text_tokens': torch.zeros(1, 1, dtype=torch.long),
        'context_text_tokens_lens': torch.tensor([1]),
    }
    calls, closed = [], []

    def request(route, payload=None):
        calls.append(route)
        if route == '/refit':
            raise ValueError('refit failed')
        if route == '/sleep':
            raise AssertionError('A failed worker must be closed without another queued RPC.')
        return {}

    client._request = request
    client.close = lambda: closed.append(True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)

    with pytest.raises(ValueError, match=f'{phase} failed'):
        client.generate(model, batch)

    assert calls[0] == '/wake' and '/sleep' not in calls
    assert closed


def test_health_and_successful_cleanup_requests_have_bounded_timeouts(tmp_path, monkeypatch):
    client = _client(tmp_path)
    client.url, client.timeout = 'http://127.0.0.1:12345', 900
    timeouts = []

    def open_request(request, timeout):
        timeouts.append(timeout)
        return StringIO('{}')

    monkeypatch.setattr(vllm_rollout, 'urlopen', open_request)
    for route in ('/health', '/sleep', '/refit'):
        client._request(route)
    assert timeouts == [5, 30, 900]


def test_request_detects_dead_serving_leader_before_opening_a_connection(tmp_path, monkeypatch):
    client = _client(tmp_path)
    client.process = SimpleNamespace(poll=lambda: -6)

    def unexpected_connection(*args, **kwargs):
        raise AssertionError('No HTTP request should be made after the serving leader exits.')

    monkeypatch.setattr(vllm_rollout, 'urlopen', unexpected_connection)
    with pytest.raises(RuntimeError, match='worker exited'):
        client._request('/refit', {})


def test_worker_runtime_environment_preserves_the_trainer_configuration(tmp_path, monkeypatch):
    client = _client(tmp_path)
    client.timeout = 5
    client.converter = SimpleNamespace(__file__=str(tmp_path / 'convert_to_vllm.py'))
    client._request = lambda *args, **kwargs: {'error': None}
    captured = {}

    def start_worker(command, **kwargs):
        captured.update(kwargs['env'])
        ready = Path(command[command.index('--ready-file') + 1])
        ready.write_text('{"url": "http://127.0.0.1:12345"}')
        return SimpleNamespace(poll=lambda: None, pid=12345)

    monkeypatch.setattr(vllm_rollout.subprocess, 'Popen', start_worker)
    monkeypatch.setattr(vllm_rollout.atexit, 'register', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    for name in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        monkeypatch.setenv(name, 'expandable_segments:True')
    trainer_tmp = str(tmp_path / ('long-training-extraction-path-' * 8))
    monkeypatch.setenv('TMPDIR', trainer_tmp)

    client._start(SimpleNamespace(), tmp_path)

    assert captured['TMPDIR'] == '/tmp' and os.environ['TMPDIR'] == trainer_tmp
    for name in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        assert captured[name] == 'expandable_segments:False'
        assert os.environ[name] == 'expandable_segments:True'
    client.process = None
    client.log_file.close()


@pytest.mark.skipif(os.name != 'posix', reason='Process-group cleanup applies to POSIX serving workers')
def test_close_terminates_workers_after_the_serving_leader_exits(tmp_path, monkeypatch):
    client = _client(tmp_path)
    signals, waits = [], []
    client.process = SimpleNamespace(pid=12345, poll=lambda: 0, wait=lambda timeout=None: waits.append(timeout))
    monkeypatch.setattr(vllm_rollout.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))

    client.close()

    assert client.process is None
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert waits == [15, None]
