# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape CFG metadata and captured predictor replay contracts."""
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.cfg_graph import CFGGraphBuffers
from easymagpie_vllm_omni.cfg_pairing import CFGPairingLayout
from test_cfg_synthesis import _predictor


def test_graph_mapping_updates_in_place_and_clears_reused_and_padded_rows():
    mapping = CFGGraphBuffers(6, device='cpu')
    addresses = [value.data_ptr() for value in (mapping.owners, mapping.partners, mapping.active)]
    mapping.update(CFGPairingLayout(('a',), ((3,),), ((0,),)), 5, 6)
    assert mapping.owners.tolist() == [3, 1, 2, 3, 4, 5]
    assert mapping.partners.tolist() == [0, 1, 2, 0, 4, 5]
    assert mapping.active.tolist() == [True] * 5 + [False]
    mapping.update(CFGPairingLayout(('b',), ((0,),), ((1,),)), 2, 6)
    assert mapping.owners.tolist() == [0, 0, 2, 3, 4, 5]
    assert mapping.partners.tolist() == [1, 1, 2, 3, 4, 5]
    mapping.update(None, 1, 6)
    assert mapping.owners.tolist() == list(range(6))
    assert mapping.partners.tolist() == list(range(6))
    assert mapping.active.tolist() == [True] + [False] * 5
    assert addresses == [value.data_ptr() for value in (mapping.owners, mapping.partners, mapping.active)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_fixed_shape_guidance_matches_eager_and_masks_padding(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    logits = torch.tensor(
        [
            [[0.0, 3.0, 5.0, -10.0]],
            [[0.0, 1.0, 6.0, -10.0]],
            [[0.0, 7.0, 1.0, -10.0]],
            [[3.0, 4.0, 0.0, -10.0]],
            [[0.0, 1.0, 0.0, -10.0]],
            [[0.0, 0.0, 0.0, 100.0]],
        ],
        device=device,
        dtype=dtype,
    )
    predictor, _, embedder = _predictor([logits, logits], device)
    layout = CFGPairingLayout(('a', 'b'), ((3,), (2,)), ((0,), (4,)))
    mapping = CFGGraphBuffers(6, device=device)
    mapping.update(layout, 5, 6)
    kwargs = dict(
        positions=torch.ones(6, dtype=torch.long, device=device),
        predict=torch.tensor([True] * 5 + [False], device=device),
        cfg_scale=torch.tensor([2.5, 1.0, 3.0, 2.5, 3.0, 1.0], device=device),
        temperature=0.0,
        top_k=4,
    )
    hidden = torch.zeros(6, 2, device=device)
    expected = predictor.generate_codes(hidden, pairing=layout, **kwargs)
    result = predictor.generate_codes(hidden, graph_pairing=mapping, **kwargs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result[-1].tolist() == [8, 8]
    assert result.tolist()[:5] == [[0, 0], [2, 2], [1, 1], [0, 0], [1, 1]]
    assert embedder.inputs[-1][-1].tolist() == [8]


def test_fixed_shape_predictor_uses_fresh_noise_and_runtime_settings():
    logits = torch.zeros(4, 1, 4)
    predictor, _, _ = _predictor([logits])
    mapping = CFGGraphBuffers(4, device='cpu')
    mapping.update(CFGPairingLayout(('r',), ((2,),), ((0,),)), 3, 4)
    temperature = torch.ones(4)
    settings = dict(
        positions=torch.zeros(4, dtype=torch.long),
        graph_pairing=mapping,
        cfg_scale=torch.ones(4),
        temperature=temperature,
        top_k=torch.ones(4, dtype=torch.long),
    )
    results = [predictor.generate_codes(torch.zeros(4, 2), **settings).clone() for _ in range(24)]
    assert len({tuple(value[2].tolist()) for value in results}) > 1
    assert all(torch.equal(value[0], value[2]) and value[3].item() == 8 for value in results)
    temperature.zero_()
    assert predictor.generate_codes(torch.zeros(4, 2), **settings).flatten().tolist() == [0, 0, 0, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_graph_replay_updates_pairing_and_settings_without_recapture():
    logits = torch.zeros(4, 1, 4, device='cuda')
    logits[:, 0, 0] = 4
    logits[1, 0, 1] = 8
    predictor, _, _ = _predictor([logits], 'cuda')
    mapping = CFGGraphBuffers(4, device='cuda')
    hidden = torch.zeros(4, 2, device='cuda')
    positions = torch.zeros(4, dtype=torch.long, device='cuda')
    temperature = torch.zeros(4, device='cuda')
    scale = torch.ones(4, device='cuda')
    top_k = torch.full((4,), 4, dtype=torch.long, device='cuda')

    def run():
        return predictor.generate_codes(
            hidden, positions=positions, graph_pairing=mapping, cfg_scale=scale, temperature=temperature, top_k=top_k
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    # Capture without live pair metadata, as the real runner does during startup.
    with torch.cuda.graph(graph):
        result = run()
    for layout, active in [
        (CFGPairingLayout(('a',), ((1,),), ((0,),)), 3),
        (CFGPairingLayout(('b',), ((0,),), ((1,),)), 2),
        (None, 1),
    ]:
        mapping.update(layout, active, 4)
        expected = run().clone()
        graph.replay()
        torch.testing.assert_close(result, expected)
    mapping.update(CFGPairingLayout(('c',), ((0,),), ((1,),)), 2, 4)
    logits.zero_()
    temperature.fill_(1.0)
    draws = []
    for _ in range(24):
        graph.replay()
        value = result.clone()
        assert torch.equal(value[0], value[1])
        draws.append(value[0].item())
    assert len(set(draws)) > 1


@pytest.mark.parametrize('compiled_prefill', [False, True])
def test_graph_guard_is_opt_in_and_keeps_diagnostic_and_async_restrictions(compiled_prefill):
    from easymagpie_vllm_omni.cfg_scheduler import validate_cfg_pairing_config
    from test_cfg_scheduler import _config
    from vllm.config import CompilationMode, CUDAGraphMode

    config = _config()
    config.model_config.enforce_eager = False
    config.model_config.hf_config.cfg_capability = {"version": 1}
    config.compilation_config = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE if compiled_prefill else CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE if compiled_prefill else CUDAGraphMode.FULL_DECODE_ONLY,
    )
    with pytest.raises(ValueError, match='CFG pairing probe'):
        validate_cfg_pairing_config(config)
    validate_cfg_pairing_config(config, allow_decode_graphs=True)
    for mode in [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE]:
        config.compilation_config.cudagraph_mode = mode
        with pytest.raises(ValueError, match='CFG pairing probe'):
            validate_cfg_pairing_config(config, allow_decode_graphs=True)
    config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE if compiled_prefill else CUDAGraphMode.FULL_DECODE_ONLY
    )
    config.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match='CFG pairing probe'):
        validate_cfg_pairing_config(config, allow_decode_graphs=True)


@pytest.mark.parametrize('compiled_prefill', [False, True])
def test_graph_guard_rejects_mismatched_compilation_policy(compiled_prefill):
    from easymagpie_vllm_omni.cfg_graph import uses_cfg_decode_graphs
    from easymagpie_vllm_omni.cfg_scheduler import validate_cfg_pairing_config
    from test_cfg_scheduler import _config
    from vllm.config import CompilationMode, CUDAGraphMode

    config = _config()
    config.model_config.enforce_eager = False
    config.model_config.hf_config.cfg_capability = {'version': 1}
    config.scheduler_config.scheduler_cls = 'easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler'
    config.compilation_config = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE if compiled_prefill else CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY if compiled_prefill else CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    with pytest.raises(ValueError, match='CFG pairing probe'):
        validate_cfg_pairing_config(config, allow_decode_graphs=True)
    assert not uses_cfg_decode_graphs(config)


def test_runner_stages_unwrapped_model_after_reordering_and_clears_padding(monkeypatch):
    from easymagpie_vllm_omni.cfg_pairing import CFGPair
    from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    pair = CFGPair.create('r')
    model = SimpleNamespace(
        cfg_graph_buffers=CFGGraphBuffers(4, device='cpu'),
        _dec_audio_valid=torch.ones(4, dtype=torch.long),
        _dec_phoneme_valid=torch.ones(4, dtype=torch.long),
        has_phoneme=True,
    )
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    # Writes to this wrapper would not reach the underlying captured model.
    runner.model = SimpleNamespace()
    monkeypatch.setattr(EasyMagpieGPUARModelRunner, 'get_model', lambda self: model)
    runner.requests = {
        rid: SimpleNamespace(additional_information_cpu={**pair.metadata(rid), 'use_cfg': True})
        for rid in pair.physical_ids
    }
    ids = list(pair.physical_ids)
    runner.input_batch = SimpleNamespace(req_ids=ids, num_computed_tokens_cpu=[5, 5])
    step = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1))

    def prepare(self, *args):
        ids.reverse()
        return 'prepared'

    monkeypatch.setattr(GPUARModelRunner, '_preprocess', prepare)
    assert runner._preprocess(step, 4) == 'prepared'
    assert model.cfg_graph_buffers.owners.tolist() == [1, 1, 2, 3]
    assert model._dec_audio_valid.tolist() == [1, 1, 0, 0]
    assert model._dec_phoneme_valid.tolist() == [1, 1, 0, 0]
    # Same capture size, reused for plain conditional traffic.
    ids[:] = ['plain']
    step.num_scheduled_tokens = {'plain': 1}
    runner._preprocess(step, 4)
    assert model.cfg_graph_buffers.owners.tolist() == [0, 1, 2, 3]
    assert model.cfg_graph_buffers.active.tolist() == [True, False, False, False]


def test_graph_phonemes_do_not_update_private_or_padding_rows():
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
    from test_cfg_synthesis import _input_model

    model = _input_model()
    model.arch = SimpleNamespace(phoneme_stacking_factor=1, phoneme_vocab_size=3)
    model.phoneme_final_proj = lambda hidden: torch.tensor([[0.0, 10.0, 0.0]]).expand(hidden.shape[0], -1)
    model.phoneme_confidence_unk_threshold = 0.0
    model._dec_phoneme_tokens = torch.full((4, 1), 2, dtype=torch.long)
    model._dec_phoneme_valid = torch.ones(4, dtype=torch.long)
    model.cfg_guidance_layout = None  # Dummy capture sees no Python pair.
    model.cfg_graph_buffers = CFGGraphBuffers(4, device='cpu')
    model.cfg_graph_buffers.update(CFGPairingLayout(('r',), ((1,),), ((0,),)), 3, 4)
    EasyMagpieTTSForConditionalGeneration._predict_phonemes(model, torch.zeros(4, 2), slice(0, 4))
    assert model._dec_phoneme_tokens.flatten().tolist() == [2, 1, 1, 2]
    assert model._dec_phoneme_valid.tolist() == [0, 1, 1, 0]


def test_graph_forward_copies_public_feedback_and_suppresses_padding_eos():
    from test_cfg_synthesis import _input_model
    from torch import nn

    model = _input_model()
    model.cfg_guidance_layout = None
    model.cfg_graph_buffers = CFGGraphBuffers(4, device='cpu')
    model.cfg_graph_buffers.update(CFGPairingLayout(('r',), ((1,),), ((0,),)), 3, 4)
    model.has_phoneme = False
    model.audio_eos_id = 7
    model._dec_audio_codes = torch.tensor([[99], [5], [4], [7]])
    model._dec_audio_valid = torch.tensor([0, 1, 1, 1])
    model._dec_text_tokens = torch.zeros(4, dtype=torch.long)
    model._dec_text_mask = torch.zeros(4, dtype=torch.long)
    model._out_codes = torch.zeros(4, 1, dtype=torch.long)
    model._token_stop = torch.zeros(4, dtype=torch.bool)
    model._sample_stop = torch.zeros(4, dtype=torch.bool)
    model._acoustic_temperature = torch.zeros(4)
    model._acoustic_top_k = torch.full((4,), 4, dtype=torch.long)
    model._cfg_scale = torch.ones(4)
    model.text_embedding = nn.Embedding(4, 2)
    model._get_query_dispatch = lambda: (None, 0, None)
    logits = torch.zeros(4, 1, 4)
    logits[1, 0, 3] = 10  # Public EOS must be copied to the private history.
    model.code_predictor, _, _ = _predictor([logits])
    model.code_predictor.embed_audio_frame = lambda codes: codes.float().expand(-1, 2)

    class Backbone(nn.Module):
        def forward(self, **kwargs):
            self.inputs = kwargs['inputs_embeds'].clone()
            return kwargs['inputs_embeds']

    model.backbone = Backbone()
    with torch.inference_mode():
        model.forward(torch.zeros(4, dtype=torch.long), positions=torch.ones(4, dtype=torch.long))
    assert model._dec_audio_codes.flatten().tolist() == [5, 5, 4, 7]
    assert model.backbone.inputs[:, 0].tolist() == [5.0, 5.0, 4.0, 0.0]
    assert model._out_codes.flatten().tolist() == [7, 7, 0, 8]
    assert model._sample_stop.tolist() == [True, True, False, False]


def test_padded_paired_prefill_preserves_distinct_embeddings_and_only_advances_caches(monkeypatch):
    from easymagpie_vllm_omni import easymagpie
    from easymagpie_vllm_omni.config import NO_AUDIO_FRAME
    from test_cfg_synthesis import _input_model
    from torch import nn

    model = _input_model()
    model.has_phoneme = False
    model.cfg_guidance_layout = CFGPairingLayout(('r',), ((0, 1, 2),), ((3, 4, 5),))
    model.cfg_graph_buffers = CFGGraphBuffers(8, device='cpu')
    model.cfg_graph_buffers.update(model.cfg_guidance_layout, 6, 8)
    model._combined_embeddings = torch.zeros(8, 2)
    model._dec_audio_codes = torch.arange(8).reshape(8, 1)
    model._dec_audio_valid = torch.zeros(8, dtype=torch.long)
    model._out_codes = torch.ones(8, 1, dtype=torch.long)
    model._token_stop = torch.ones(8, dtype=torch.bool)
    model._sample_stop = torch.ones(8, dtype=torch.bool)
    model.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=easymagpie.CUDAGraphMode.FULL_AND_PIECEWISE, cudagraph_capture_sizes=[1, 2, 4, 8]
        )
    )
    metadata = SimpleNamespace(max_query_len=3, query_start_loc=torch.tensor([0, 3, 6]))
    monkeypatch.setattr(easymagpie, 'get_forward_context', lambda: SimpleNamespace(attn_metadata={'attn': metadata}))

    class Backbone(nn.Module):
        def forward(self, **kwargs):
            return kwargs['inputs_embeds']

    model.backbone = Backbone()
    advanced = []
    # Any acoustic sampling here would fail: this object exposes only cache advancement.
    model.code_predictor.advance = lambda hidden, positions: advanced.append((hidden.clone(), positions.clone()))
    embeddings = torch.tensor([[1.0, 2.0]] * 3 + [[9.0, 10.0]] * 3 + [[0.0, 0.0]] * 2)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0])
    with torch.inference_mode():
        result = model.forward(torch.zeros(8, dtype=torch.long), positions=positions, inputs_embeds=embeddings)
    torch.testing.assert_close(result, embeddings)
    assert len(advanced) == 1
    torch.testing.assert_close(advanced[0][0], embeddings)
    torch.testing.assert_close(advanced[0][1], positions)
    assert model._out_codes.eq(NO_AUDIO_FRAME).all()
    assert not model._sample_stop.any()


def test_graph_profile_parses_decode_only_configuration_and_keeps_codec_eager():
    from pathlib import Path

    import yaml
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    root = Path(__file__).resolve().parents[5] / 'tools/easymagpie_vllm_omni'
    graph = yaml.safe_load((root / 'deploy/easymagpie_cfg_graph.yaml').read_text())
    eager = yaml.safe_load((root / 'deploy/easymagpie_cfg.yaml').read_text())
    stage = graph['stages'][0]
    config = CompilationConfig(**stage['compilation_config'])
    assert config.mode == CompilationMode.NONE
    assert config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert config.cudagraph_capture_sizes == [1, 2]
    assert stage['enforce_eager'] is False
    assert stage['async_scheduling'] is False
    assert graph['dtype'] == eager['dtype'] == 'float16'
    assert graph['stages'][1] == eager['stages'][1]


@pytest.mark.parametrize('compiled_prefill', [False, True])
def test_fixed_shape_activation_is_scoped_to_paired_synthesis(compiled_prefill):
    from easymagpie_vllm_omni.cfg_graph import uses_cfg_decode_graphs
    from vllm.config import CompilationMode, CUDAGraphMode

    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        scheduler_config=SimpleNamespace(scheduler_cls='easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler'),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE if compiled_prefill else CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE if compiled_prefill else CUDAGraphMode.FULL_DECODE_ONLY,
        ),
    )
    assert uses_cfg_decode_graphs(config)
    for scheduler in [
        None,
        'easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler',
        'easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGPairingScheduler',
    ]:
        config.scheduler_config.scheduler_cls = scheduler
        assert not uses_cfg_decode_graphs(config)


def test_compiled_profile_dispatches_prefill_piecewise_and_decode_full():
    import yaml
    from conftest import EASYMAGPIE_ROOT
    from easymagpie_vllm_omni.cfg_graph import uses_cfg_decode_graphs
    from easymagpie_vllm_omni.cfg_scheduler import validate_cfg_pairing_config
    from test_cfg_scheduler import _config
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    profile = yaml.safe_load((EASYMAGPIE_ROOT / 'deploy/easymagpie_cfg_compile.yaml').read_text())
    fallback = yaml.safe_load((EASYMAGPIE_ROOT / 'deploy/easymagpie_cfg_graph.yaml').read_text())
    stage = profile['stages'][0]
    assert profile['stages'][1] == fallback['stages'][1]
    assert {k: v for k, v in stage.items() if k != 'compilation_config'} == {
        k: v for k, v in fallback['stages'][0].items() if k != 'compilation_config'
    }
    compilation = CompilationConfig(**stage['compilation_config'])
    assert compilation.mode == CompilationMode.VLLM_COMPILE
    assert compilation.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    compilation.set_splitting_ops_for_v1(all2all_backend='allgather_reducescatter')
    assert compilation.is_attention_compiled_piecewise()

    config = _config()
    config.compilation_config = compilation
    config.model_config.enforce_eager = False
    config.model_config.hf_config.cfg_capability = {'version': 1}
    config.scheduler_config.scheduler_cls = 'easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler'
    config.scheduler_config.max_num_seqs = stage['max_num_seqs']
    config.num_speculative_tokens = 0
    config.lora_config = None
    validate_cfg_pairing_config(config, allow_decode_graphs=True)
    assert uses_cfg_decode_graphs(config)
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(compilation.cudagraph_mode)
    mode, batch = dispatcher.dispatch(10, uniform_decode=False)
    assert mode == CUDAGraphMode.PIECEWISE
    assert batch.num_tokens == 16
    mode, batch = dispatcher.dispatch(2, uniform_decode=True)
    assert mode == CUDAGraphMode.FULL
    assert batch.num_tokens == 2
    # Long prefill remains supported outside the capture range, without graph replay.
    mode, _ = dispatcher.dispatch(compilation.max_cudagraph_capture_size + 1, uniform_decode=False)
    assert mode == CUDAGraphMode.NONE
