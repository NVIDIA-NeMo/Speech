# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance contracts: actual mixing, isolation and shared feedback."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from easymagpie_vllm_omni.cfg import CFGOptions, build_cfg_pairs, guided_logits, unconditional_prefill


@pytest.mark.parametrize('scale', [0.0, 1.0, 2.5])
def test_guidance_matches_native_formula(scale):
    conditional = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 1.0, 2.0]])
    unconditional = torch.tensor([[-2.0, 3.0, 1.0], [1.0, 0.0, -4.0]])
    result = guided_logits(conditional, unconditional, torch.full((2, 1), scale))
    torch.testing.assert_close(result, scale * conditional + (1 - scale) * unconditional)


def test_guidance_changes_the_distribution_and_can_change_selected_code():
    conditional = torch.tensor([[3.0, 2.0]])
    unconditional = torch.tensor([[4.0, -2.0]])
    assert conditional.argmax(-1).item() == 0
    assert guided_logits(conditional, unconditional, torch.tensor([[2.5]])).argmax(-1).item() == 1


def test_request_scales_do_not_leak_between_rows():
    conditional = torch.tensor([[3.0, 2.0], [3.0, 2.0]])
    unconditional = torch.tensor([[4.0, -2.0], [4.0, -2.0]])
    result = guided_logits(conditional, unconditional, torch.tensor([[1.0], [2.5]]))
    assert result.argmax(-1).tolist() == [0, 1]


@pytest.mark.parametrize('scale', [float('nan'), float('inf'), -1.0, True])
def test_reject_invalid_guidance_scale(scale):
    with pytest.raises(ValueError, match='cfg_scale'):
        CFGOptions.from_hf_config(SimpleNamespace(easymagpie_cfg=True)).resolve_scale(scale)


def test_cfg_is_opt_in_for_existing_checkpoints():
    assert not CFGOptions.from_hf_config(SimpleNamespace()).enabled


def test_unconditional_prefix_masks_conditioning_and_text_warmup():
    unknown = torch.tensor([2.0, -1.0, 4.0])
    result = unconditional_prefill(unknown, offset=1, length=5, conditioning_len=3)
    torch.testing.assert_close(result[:2], unknown.expand(2, -1))
    assert torch.count_nonzero(result[2:]).item() == 0


def test_pair_mapping_uses_ids_instead_of_batch_order():
    records = [
        ('a', 'unconditional', 8, 1, 2.5),
        ('b', 'conditional', 2, 1, 1.5),
        ('a', 'conditional', 5, 1, 2.5),
        ('b', 'unconditional', 12, 1, 1.5),
    ]
    pairs = build_cfg_pairs(records)
    assert pairs == [('a', 5, 8, 2.5), ('b', 2, 12, 1.5)]


def test_unpaired_or_misaligned_branch_fails_loudly():
    with pytest.raises(ValueError, match='pair'):
        build_cfg_pairs([('a', 'conditional', 0, 1, 2.5)])
    with pytest.raises(ValueError, match='length'):
        build_cfg_pairs([('a', 'conditional', 0, 2, 2.5), ('a', 'unconditional', 2, 1, 2.5)])


def test_cfg_scale_one_matches_existing_sampler():
    from conftest import build_vllm_config
    from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor

    torch.manual_seed(123)
    predictor = EasyMagpieCodePredictor(vllm_config=build_vllm_config()).eval()
    predictor.init_forbidden_mask()
    predictor.temperature = 0.0
    conditional, unconditional = torch.randn(2, 64), torch.randn(2, 64)
    torch.manual_seed(456)
    expected = predictor.generate_codes(conditional)
    torch.manual_seed(456)
    actual, _ = predictor.generate_codes_cfg(conditional, unconditional, torch.ones(2, 1))
    assert torch.equal(actual, expected)
    assert not any('cfg_code_loop' in name for name, _ in predictor.named_parameters())


def test_local_transformer_batches_branches_and_shares_sampled_history():
    from conftest import build_vllm_config
    from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor

    predictor = EasyMagpieCodePredictor(vllm_config=build_vllm_config()).eval()
    predictor.init_forbidden_mask()
    observed = []
    handle = predictor.local_transformer.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    try:
        codes, _ = predictor.generate_codes_cfg(torch.randn(2, 64), torch.randn(2, 64), torch.full((2, 1), 2.5))
    finally:
        handle.remove()
    assert codes.shape == (2, predictor.num_codebooks)
    assert len(observed) == predictor.num_codebooks  # one batched call, not two serial calls
    for step, inputs in enumerate(observed):
        assert inputs.shape[0] == 4
        if step:
            torch.testing.assert_close(inputs[:2, 1 : step + 1], inputs[2:, 1 : step + 1])


def test_scheduler_companion_has_independent_history_and_is_hidden(monkeypatch):
    from easymagpie_vllm_omni.cfg import CFG_COMPANION_PREFIX
    from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
    from vllm_omni.engine.serialization import deserialize_additional_information

    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler._cfg_enabled = True
    scheduler._cfg_options = CFGOptions(enabled=True)
    scheduler._cfg_companions = {}
    scheduler.requests = {}
    scheduler.chunk_transfer_adapter = None
    admitted = []

    def admit(self, request):
        self.requests[request.request_id] = request
        admitted.append(request)

    monkeypatch.setattr(OmniARAsyncScheduler, 'add_request', admit)
    request = SimpleNamespace(
        request_id='a',
        num_prompt_tokens=8,
        resumable=False,
        mm_features=None,
        additional_information={'speaker_id': 'voice', 'text_tokens': [1, 2]},
        _all_token_ids=[0] * 8,
    )
    scheduler.add_request(request)
    assert [r.request_id for r in admitted] == ['a', CFG_COMPANION_PREFIX + 'a']
    admitted[1]._all_token_ids.append(99)
    assert request._all_token_ids == [0] * 8
    assert deserialize_additional_information(admitted[1].additional_information)['_cfg_role'] == 'unconditional'
    assert deserialize_additional_information(request.additional_information)['cfg_scale'] == 2.5

    outputs = {0: [SimpleNamespace(request_id='a'), SimpleNamespace(request_id=CFG_COMPANION_PREFIX + 'a')]}
    monkeypatch.setattr(OmniARAsyncScheduler, 'update_from_output', lambda *args: outputs)
    result = scheduler.update_from_output(None, None)
    assert [r.request_id for r in result[0]] == ['a']


def test_scheduler_rejects_split_guidance_pair(monkeypatch):
    from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler

    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler._cfg_enabled = True
    scheduler._cfg_companions = {'a': 'unconditional-a'}
    scheduler.requests = {}
    monkeypatch.setattr(OmniARAsyncScheduler, 'schedule', lambda *args: SimpleNamespace(num_scheduled_tokens={'a': 1}))
    with pytest.raises(RuntimeError, match='atomically'):
        scheduler.schedule()


@pytest.mark.parametrize('enabled', [False, True])
def test_cfg_dispatch_cannot_be_skipped_by_outer_graph_replay(monkeypatch, enabled):
    from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    model = SimpleNamespace(_cfg_enabled=enabled)
    wrapper = object.__new__(CUDAGraphWrapper)
    wrapper.runnable = model
    monkeypatch.setattr(GPUARModelRunner, 'load_model', lambda self: setattr(self, 'model', wrapper))
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.load_model()
    assert runner.model is (model if enabled else wrapper)


def test_padded_cfg_batch_keeps_real_rows_and_hides_padding():
    from conftest import build_vllm_config
    from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor

    predictor = EasyMagpieCodePredictor(vllm_config=build_vllm_config()).eval()
    predictor.init_forbidden_mask()
    predictor.temperature = 0.0
    conditional, unconditional = torch.randn(3, 64), torch.randn(3, 64)
    expected = predictor.generate_codes(conditional)
    codes, diagnostics = predictor.generate_codes_cfg(conditional, unconditional, torch.ones(3, 1), padded_batch=4)
    assert codes.shape == expected.shape
    assert diagnostics.shape == (3, 3)
    assert torch.equal(codes, expected)


@pytest.mark.parametrize('scale', [-1.0, float('nan'), True])
def test_serving_rejects_invalid_cfg_scale_before_engine_submission(scale):
    from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls

    config = SimpleNamespace(easymagpie_cfg=True)
    ctx = SimpleNamespace(engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    adapter = _build_adapter_cls()(ctx)
    request = SimpleNamespace(input='hello', voice='eng', ref_audio=None, extra_params={'cfg_scale': scale})
    assert 'cfg_scale' in adapter.validate(request)


def test_cfg_streaming_is_rejected_before_loading_voice_or_tokenizer():
    from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls

    config = SimpleNamespace(easymagpie_cfg=True)
    ctx = SimpleNamespace(engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    adapter = _build_adapter_cls()(ctx)
    with pytest.raises(ValueError, match='whole-text'):
        adapter.build_streaming_spec(SimpleNamespace())


def test_cfg_off_accepts_existing_exports_with_only_backbone_vocab_size():
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.arch = SimpleNamespace(cfg_unk_id=None, resolved_text_eos_id=lambda size: size - 3)
    model.text_embedding = torch.nn.Embedding(32, 4)
    model.backbone = torch.nn.Identity()
    model.code_predictor = SimpleNamespace(_cfg_code_loop=SimpleNamespace(trace=False))
    model._dec_audio_valid = torch.zeros(8)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=32)), compilation_config=SimpleNamespace()
    )
    model._init_cfg(config)
    assert not model._cfg_enabled
    assert model._cfg_unknown_id == 30
