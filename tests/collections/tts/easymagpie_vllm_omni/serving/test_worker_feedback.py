# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker feedback regressions against the pinned Omni hooks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easymagpie_vllm_omni.cfg_pairing import CFGPair
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


def make_runner(postprocess=None):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(
        strict_worker_feedback=True,
        has_postprocess=True,
        has_phoneme=False,
        gpu_resident_buffer_keys={'last_audio_codes', 'last_phoneme_token'},
        postprocess=postprocess or (lambda h, **kw: {'last_audio_codes': h[-1:]}),
    )
    runner.requests = {rid: SimpleNamespace() for rid in ('a', 'b')}
    runner.input_batch = SimpleNamespace(req_ids=['b', 'a'])
    runner.query_start_loc = SimpleNamespace(cpu=np.array([0, 2, 3]))
    runner.model_intermediate_buffer = {}
    return runner


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_flat_feedback_owns_declared_tensor(device):
    runner = make_runner()
    source = torch.tensor([[1, 2]], device=device)
    runner._update_intermediate_buffer('a', {'last_audio_codes': source, 'offset': 3})
    stored = runner.model_intermediate_buffer['a']['last_audio_codes']
    source.fill_(9)
    assert stored.device == source.device
    assert stored.dtype == source.dtype and stored.shape == source.shape
    assert stored.tolist() == [[1, 2]]
    assert runner.requests['a'].additional_information_cpu['offset'] == 3


def test_feedback_exception_escapes_and_latches_before_successor(monkeypatch):
    def fail(h, **kwargs):
        raise ValueError('broken feedback')

    runner = make_runner(fail)
    with pytest.raises(ValueError, match='broken feedback'):
        runner._process_additional_information_updates(torch.ones(3, 2), {}, np.array([2, 1]), SimpleNamespace())
    calls = []
    monkeypatch.setattr(GPUARModelRunner, 'execute_model', lambda *a, **k: calls.append('execute'))
    monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', lambda *a, **k: calls.append('sample'))
    for entry in (runner.execute_model, runner.sample_tokens):
        with pytest.raises(RuntimeError, match='unusable'):
            entry(None)
    assert not calls


def test_feedback_span_tail_and_payload_suppression(monkeypatch):
    runner = make_runner()
    monkeypatch.setattr(runner, '_request_needs_downstream_stage_payload', lambda rid: False)
    runner._process_additional_information_updates(
        torch.tensor([[1, 2], [3, 4], [5, 6], [99, 99]]),
        {},
        np.array([2, 1]),
        SimpleNamespace(),
        req_ids_filter=set(),
    )
    assert runner.model_intermediate_buffer['b']['last_audio_codes'].tolist() == [[3, 4]]
    assert runner.model_intermediate_buffer['a']['last_audio_codes'].tolist() == [[5, 6]]


def test_missing_feedback_is_failure():
    runner = make_runner(lambda *a, **k: {})
    with pytest.raises(RuntimeError, match='last_audio_codes'):
        runner._process_additional_information_updates(torch.ones(3, 2), {}, np.array([2, 1]), SimpleNamespace())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_control_aligned_before_bookkeeping(monkeypatch, device):
    runner = make_runner()
    pair = CFGPair.create('a')
    runner.input_batch.req_ids = [pair.unconditional_id, 'b', 'a']
    runner.requests[pair.unconditional_id] = SimpleNamespace(
        additional_information_cpu=pair.metadata(pair.unconditional_id)
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array([False, False, False]))
    sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[9], [8], [7]], device=device))
    seen = []
    monkeypatch.setattr(
        GPUARModelRunner,
        '_bookkeeping_sync',
        lambda self, step, out, *args: seen.append(out.sampled_token_ids.clone()),
    )
    runner._bookkeeping_sync(SimpleNamespace(), sampled, None, torch.ones(5, 2), 5)
    assert seen[0].tolist() == [[7], [8], [7]]


@pytest.mark.parametrize('failed_key', ['last_audio_codes', 'last_phoneme_token'])
def test_storage_failure_after_partial_commit_latches(monkeypatch, failed_key):
    runner = make_runner(lambda h, **kw: {'last_audio_codes': h[-1:], 'last_phoneme_token': h[-1:, :1]})
    runner.model.has_phoneme = True
    original = runner._store_value
    calls = []

    def store(dest, key, value, keys):
        calls.append(key)
        if len(calls) > 2 and key == failed_key:
            raise OSError('storage failure')
        return original(dest, key, value, keys)

    monkeypatch.setattr(runner, '_store_value', store)
    with pytest.raises(OSError, match='storage failure'):
        runner._process_additional_information_updates(torch.ones(3, 2), {}, np.array([2, 1]), SimpleNamespace())
    assert 'last_audio_codes' in runner.model_intermediate_buffer['b']
    # A new request identity cannot make this worker healthy again.
    runner.input_batch.req_ids = ['fresh']
    with pytest.raises(RuntimeError, match='unusable'):
        runner.execute_model(None)
    runner.model_intermediate_buffer.clear()  # Cleanup remains possible.
    assert make_runner()._check_feedback_health() is None


def test_inline_builder_commits_once_and_passes_result_marker(monkeypatch):
    calls = []
    runner = make_runner(lambda h, **kw: calls.append(h.clone()) or {'last_audio_codes': h[-1:]})
    markers = []
    monkeypatch.setattr(
        GPUARModelRunner,
        '_build_omni_model_runner_output_from_snapshot',
        lambda self, **kw: markers.append(kw['postprocess_already_applied']) or 'output',
    )
    kwargs = dict(
        hidden_states=torch.ones(3, 2),
        multimodal_outputs={},
        num_scheduled_tokens_np=np.array([2, 1]),
        scheduler_output=SimpleNamespace(),
        req_ids_output_copy=['b', 'a'],
        query_start_loc_cpu=np.array([0, 2, 3]),
    )
    assert runner._build_omni_model_runner_output_from_snapshot(**kwargs) == 'output'
    assert len(calls) == 2 and markers == [True]
    runner._build_omni_model_runner_output_from_snapshot(**kwargs, postprocess_already_applied=True)
    assert len(calls) == 2
    assert 'postprocess_already_applied' not in kwargs


def test_empty_and_private_rows_do_not_commit():
    calls = []
    runner = make_runner(lambda *a, **k: calls.append(True))
    private = CFGPair.create('a').unconditional_id
    runner._process_additional_information_updates(
        torch.ones(2, 2),
        {},
        np.array([0, 2]),
        SimpleNamespace(),
        req_ids=['a', private],
        query_start_loc_cpu=np.array([0, 0, 2]),
    )
    assert not calls and not runner.model_intermediate_buffer


def test_nested_and_ordinary_storage_keep_their_contract():
    runner = make_runner()
    runner.model.gpu_resident_buffer_keys.add(('nested', 'gpu'))
    source = torch.tensor([3])
    runner._update_intermediate_buffer('a', {'nested': {'gpu': source, 'ordinary': source}, 'metadata': 'ok'})
    source.fill_(7)
    result = runner.model_intermediate_buffer['a']
    assert result['nested']['gpu'].tolist() == [3]
    assert result['nested']['ordinary'].tolist() == [7]  # Base CPU storage retains contiguous tensors.
    assert result['metadata'] == 'ok'


@pytest.mark.parametrize('mask', [[True, False, True], [True, False, False]])
def test_discarded_control_rows(monkeypatch, mask):
    runner = make_runner()
    pair = CFGPair.create('a')
    runner.input_batch.req_ids = [pair.unconditional_id, 'b', 'a']
    runner.requests[pair.unconditional_id] = SimpleNamespace(
        additional_information_cpu=pair.metadata(pair.unconditional_id)
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array(mask))
    sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[9], [8], [7]]))
    monkeypatch.setattr(GPUARModelRunner, '_bookkeeping_sync', lambda *a, **k: None)
    if mask[0] != mask[2]:
        with pytest.raises(RuntimeError, match='discard masks disagree'):
            runner._bookkeeping_sync(None, sampled)
    else:
        runner._bookkeeping_sync(None, sampled)
    assert sampled.sampled_token_ids.tolist() == [[9], [8], [7]]


@pytest.mark.parametrize('profile', ['easymagpie', 'easymagpie_lm'])
def test_noncfg_profiles_keep_inline_builder_and_strict_feedback(profile):
    import yaml
    from conftest import EASYMAGPIE_ROOT
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    config = yaml.safe_load((EASYMAGPIE_ROOT / 'deploy' / f'{profile}.yaml').read_text())
    stage = next(s for s in config['stages'] if s['stage_id'] == 0)
    assert stage['async_scheduling'] is True
    runner = make_runner()
    runner.use_async_scheduling = True
    runner.omni_prefix_cache = None
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=True)
    assert EasyMagpieTTSForConditionalGeneration.strict_worker_feedback
    assert not runner._should_use_async_omni_output()
    runner._process_additional_information_updates(torch.ones(3, 2), {}, np.array([2, 1]), SimpleNamespace())
    assert set(runner.model_intermediate_buffer) == {'a', 'b'}


def test_real_model_postprocess_preserves_prefill_tail_and_phoneme():
    from types import MethodType
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    runner = make_runner()
    runner.model.has_phoneme = True
    runner.model._dec_phoneme_tokens = torch.tensor([[10], [11], [12], [99]])
    runner.model.postprocess = MethodType(EasyMagpieTTSForConditionalGeneration.postprocess, runner.model)
    runner._process_additional_information_updates(
        torch.ones(4, 8),
        {'audio_codes': torch.tensor([[1, 2], [3, 4], [5, 6], [99, 99]])},
        np.array([2, 1]),
        SimpleNamespace(),
    )
    assert runner.model_intermediate_buffer['b']['last_audio_codes'].tolist() == [[3, 4]]
    assert runner.model_intermediate_buffer['b']['last_phoneme_token'].tolist() == [[11]]
    assert runner.model_intermediate_buffer['a']['last_phoneme_token'].tolist() == [[12]]


def test_cuda_feedback_copy_and_dependent_read_share_stream():
    runner = make_runner()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.full((1, 4), 5, dtype=torch.long, device='cuda')
        runner._update_intermediate_buffer('a', {'last_audio_codes': source})
        source.fill_(8)
        dependent = runner.model_intermediate_buffer['a']['last_audio_codes'] + 1
    stream.synchronize()
    assert dependent.tolist() == [[6, 6, 6, 6]]


def test_multiple_reordered_pairs_use_sampled_request_rows(monkeypatch):
    runner = make_runner()
    pairs = [CFGPair.create('a'), CFGPair.create('b')]
    ids = [pairs[1].unconditional_id, 'a', 'ordinary', pairs[0].unconditional_id, 'b']
    runner.input_batch.req_ids = ids
    for pair in pairs:
        runner.requests[pair.unconditional_id] = SimpleNamespace(
            additional_information_cpu=pair.metadata(pair.unconditional_id)
        )
    runner.discard_request_mask = SimpleNamespace(np=np.zeros(len(ids), dtype=bool))
    sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[90], [10], [30], [80], [20]]))
    observed = []
    monkeypatch.setattr(
        GPUARModelRunner,
        '_bookkeeping_sync',
        lambda self, step, output, *a: observed.append(output.sampled_token_ids.tolist()),
    )
    runner._bookkeeping_sync(SimpleNamespace(num_scheduled_tokens=dict(zip(ids, [3, 2, 1, 2, 3]))), sampled)
    assert observed == [[[20], [10], [30], [10], [20]]]


def test_missing_request_state_cannot_mark_feedback_applied():
    runner = make_runner()
    del runner.requests['a']
    with pytest.raises(RuntimeError, match='request state is missing'):
        runner._process_additional_information_updates(torch.ones(3, 2), {}, np.array([2, 1]), SimpleNamespace())
    assert runner._feedback_failure is not None


def test_local_dispatch_rejects_successor_after_failed_commit(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    runner = make_runner(lambda *a, **k: (_ for _ in ()).throw(ValueError('dispatch failure')))
    mutations = []
    monkeypatch.setattr(GPUARModelRunner, 'execute_model', lambda *a, **k: mutations.append('execute'))
    with ThreadPoolExecutor(max_workers=1) as dispatch:
        first = dispatch.submit(
            runner._process_additional_information_updates,
            torch.ones(3, 2),
            {},
            np.array([2, 1]),
            SimpleNamespace(),
        )
        second = dispatch.submit(runner.execute_model, SimpleNamespace(request_id='fresh'))
        with pytest.raises(ValueError, match='dispatch failure'):
            first.result(timeout=5)
        with pytest.raises(RuntimeError, match='unusable'):
            second.result(timeout=5)
    assert not mutations
