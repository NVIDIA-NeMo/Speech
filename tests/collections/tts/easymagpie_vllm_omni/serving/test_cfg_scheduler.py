# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the actual pinned scheduler and hybrid cache allocator on CPU.

These tests allocate real engine cache block identities, not GPU storage. They
do not establish numerical model/cache parity or replace a GPU engine run.
"""
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.cfg_pairing import CFG_PAIRING_PROBE_KEY, CFGPairingError, CFGPairingTransferFilter
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGPairingScheduler, validate_cfg_pairing_config
from vllm.config import SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.request import OmniRequest


def test_allocates_separate_backbone_mamba_and_predictor_cache_histories():
    scheduler = _scheduler()
    scheduler.add_request(_request())
    pair = scheduler.cfg_pair
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 3)
    physical = {req.req_id: req for req in first.scheduled_new_reqs}
    for gid in range(2):
        cond_blocks = physical[pair.conditional_id].block_ids[gid]
        uncond_blocks = physical[pair.unconditional_id].block_ids[gid]
        assert cond_blocks and uncond_blocks
        assert set(cond_blocks).isdisjoint(uncond_blocks)
    for rid in pair.physical_ids:
        info = deserialize_additional_information(physical[rid].additional_information)
        assert info == {CFG_PAIRING_PROBE_KEY: True, "speaker_id": "eng", **pair.metadata(rid)}
    # Request-owned mutable token histories are independent.
    conditional, unconditional = (scheduler.requests[rid] for rid in pair.physical_ids)
    assert conditional._all_token_ids is not unconditional._all_token_ids
    assert conditional._output_token_ids is not unconditional._output_token_ids

    outputs = scheduler.update_from_output(first, _output(pair, conditional_token=0, unconditional_token=1))
    assert [out.request_id for out in outputs[0].outputs] == [pair.conditional_id]
    assert [list(scheduler.requests[rid].output_token_ids) for rid in pair.physical_ids] == [[0], [0]]
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1)
    assert second.scheduled_cached_reqs.num_computed_tokens == [3, 3]
    scheduler.update_from_output(second, _output(pair, conditional_token=1, unconditional_token=0))
    assert scheduler.cfg_pair is None
    assert scheduler.requests == {}
    # Both physical identities still reach the worker cleanup notification.
    assert scheduler.schedule().finished_req_ids == set(pair.physical_ids)


def test_cfg_synthesis_requires_reconverted_artifact_and_complete_text():
    scheduler = _scheduler()
    config = scheduler.vllm_config.model_config.hf_config
    config.use_codec_latent_audio_embedding = True
    config.codec_num_levels_per_group = [8, 8, 16]
    info = {"use_cfg": True, "speaker_id": "eng", "text_tokens": [2, 3]}
    with pytest.raises(ValueError, match="reconvert"):
        scheduler.add_request(_request(info=info))
    assert scheduler.requests == {} and scheduler.cfg_pair is None
    config.cfg_capability = {"version": 1, "context_cas_disabled": False}
    config.cfg_unk_token_id = 12
    with pytest.raises(ValueError, match="complete target text"):
        scheduler.add_request(_request(info={"use_cfg": True, "speaker_id": "eng"}))
    scheduler.add_request(_request(info=info))
    pair = scheduler.cfg_pair
    assert scheduler.schedule().num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 3)


def test_codec_connector_never_receives_hidden_payload_or_completion():
    scheduler = _scheduler()
    transfers = []
    # Add a transfer spy only after scheduling so native block allocation and
    # request-data construction still execute without a transport connection.
    scheduler.add_request(_request())
    pair = scheduler.cfg_pair
    step = scheduler.schedule()
    scheduler.chunk_transfer_adapter = CFGPairingTransferFilter(
        SimpleNamespace(
            save_async=lambda payload, request, stopped: transfers.append((request.request_id, payload, stopped)),
        )
    )
    output = _output(pair, conditional_token=1, unconditional_token=0)
    output.inter_stage_outputs = [{"codes": "private"}, {"codes": "public"}]
    result = scheduler.update_from_output(step, output)
    assert transfers == [(pair.conditional_id, {"codes": "public"}, True)]
    assert [out.request_id for out in result[0].outputs] == [pair.conditional_id]
    assert result[0].finished_requests == {pair.conditional_id}


@pytest.mark.parametrize("branch", ["conditional_id", "unconditional_id"])
def test_abort_from_either_branch_frees_both_and_allows_slot_reuse(branch):
    scheduler = _scheduler()
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request())
    pair = scheduler.cfg_pair
    step = scheduler.schedule()
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free_before
    assert scheduler.finish_requests(getattr(pair, branch), RequestStatus.FINISHED_ABORTED) == [("speech", 0)]
    assert scheduler.cfg_pair is None
    assert scheduler.requests == {}
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
    # A late output from the cancelled step must not resurrect either history.
    scheduler.update_from_output(step, _output(pair))
    scheduler.add_request(_request("replacement"))
    assert scheduler.schedule().total_num_scheduled_tokens == 6


def test_partial_admission_aborts_pair_before_computed_counters_advance():
    scheduler = _scheduler(num_blocks=4)  # null block + one hybrid branch, not two
    request = _request()
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(request)
    with pytest.raises(CFGPairingError, match="aligned"):
        scheduler.schedule()
    assert request.num_computed_tokens == 0
    assert scheduler.requests == {}
    assert scheduler.running == []
    assert not scheduler.waiting
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before


def test_paired_scheduler_rejects_a_second_inflight_step():
    scheduler = _scheduler()
    scheduler.add_request(_request())
    scheduler.schedule()
    with pytest.raises(CFGPairingError, match="synchronous"):
        scheduler.schedule()


def test_running_cache_pressure_rejects_preemption_without_replaying_either_branch(monkeypatch):
    scheduler = _scheduler()
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request())
    pair = scheduler.cfg_pair
    first = scheduler.schedule()
    scheduler.update_from_output(first, _output(pair))
    conditional = scheduler.requests[pair.conditional_id]
    allocate = scheduler.kv_cache_manager.allocate_slots

    def no_shadow_capacity(request, *args, **kwargs):
        return None if request.request_id == pair.unconditional_id else allocate(request, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", no_shadow_capacity)
    with pytest.raises(CFGPairingError, match="preemption"):
        scheduler.schedule()
    assert conditional.num_computed_tokens == 3
    assert not scheduler.requests
    assert not scheduler.running
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before


def test_failure_admitting_second_branch_cleans_first_and_restores_public_metadata(monkeypatch):
    scheduler = _scheduler()
    request = _request()
    info = request.additional_information
    add_request = OmniARScheduler.add_request

    def fail_shadow(self, incoming):
        if incoming.request_id != request.request_id:
            raise RuntimeError("admission failed")
        add_request(self, incoming)

    monkeypatch.setattr(OmniARScheduler, "add_request", fail_shadow)
    with pytest.raises(RuntimeError, match="admission failed"):
        scheduler.add_request(request)
    assert not scheduler.requests
    assert not scheduler.waiting
    assert scheduler.cfg_pair is None
    assert request.additional_information is info


def test_runner_failure_cleans_both_histories(monkeypatch):
    scheduler = _scheduler()
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request())
    pair = scheduler.cfg_pair
    step = scheduler.schedule()

    def fail_output(self, *args):
        raise RuntimeError("output failed")

    monkeypatch.setattr(OmniARScheduler, "update_from_output", fail_output)
    with pytest.raises(RuntimeError, match="output failed"):
        scheduler.update_from_output(step, _output(pair))
    assert not scheduler.requests
    assert scheduler.cfg_pair is None
    assert scheduler._cfg_step_pending is False
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before


def test_pair_rejects_resumable_and_unmarked_or_full_cfg_requests_before_admission():
    scheduler = _scheduler()
    requests = [
        _request(info={"speaker_id": "eng"}),
        _request(info={"speaker_id": "eng", CFG_PAIRING_PROBE_KEY: True, "use_cfg": True}),
        _request(resumable=True),
    ]
    for request in requests:
        with pytest.raises(ValueError):
            scheduler.add_request(request)
        assert scheduler.cfg_pair is None
        assert scheduler.requests == {}


def test_pair_rejects_long_prompts_and_another_logical_request():
    scheduler = _scheduler()
    with pytest.raises(ValueError, match="token budget"):
        scheduler.add_request(_request(prompt=[0] * 65))
    scheduler.add_request(_request())
    with pytest.raises(CFGPairingError, match="one logical"):
        scheduler.add_request(_request("another"))
    assert len(scheduler.requests) == 2


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model_config", "enforce_eager", False),
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "enable_chunked_prefill", True),
        ("cache_config", "enable_prefix_caching", True),
        ("scheduler_config", "max_num_seqs", 1),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
    ],
)
def test_probe_validates_unsupported_execution_modes(section, key, value):
    config = _config()
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError, match="CFG pairing probe"):
        validate_cfg_pairing_config(config)


def _config(num_blocks=64):
    return SimpleNamespace(
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            is_encoder_decoder=False,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            enable_chunked_prefill=False,
            async_scheduling=False,
        ),
        model_config=SimpleNamespace(
            enforce_eager=True,
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=64,
            enable_return_routed_experts=False,
            stage_id=0,
            hf_config=SimpleNamespace(acoustic_predictor_type="acoustic_codes_predictor"),
        ),
        cache_config=SimpleNamespace(num_gpu_blocks=num_blocks, enable_prefix_caching=False, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_index=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        observability_config=SimpleNamespace(kv_cache_metrics=False, enable_mfu_metrics=False),
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        use_v2_model_runner=False,
    )


def _scheduler(num_blocks=64):
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    mamba = MambaSpec(block_size=4, shapes=((8,),), dtypes=(torch.float32,), mamba_cache_mode="none")
    caches = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["backbone.attn", "code_predictor.cache_layers.0", "code_predictor.cache_layers.1"], attention
            ),
            KVCacheGroupSpec(["backbone.mamba"], mamba),
        ],
    )
    return EasyMagpieCFGPairingScheduler(
        vllm_config=_config(num_blocks),
        kv_cache_config=caches,
        structured_output_manager=SimpleNamespace(should_advance=lambda request: False),
        block_size=4,
        mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda model: False),
    )


def _request(request_id="speech", *, info=None, prompt=None, resumable=False):
    return OmniRequest(
        request_id=request_id,
        external_req_id=request_id,
        prompt_token_ids=[0, 0, 0] if prompt is None else prompt,
        sampling_params=SamplingParams(max_tokens=4, stop_token_ids=[1], temperature=0, detokenize=False),
        pooling_params=None,
        resumable=resumable,
        additional_information=serialize_additional_information(
            {CFG_PAIRING_PROBE_KEY: True, "speaker_id": "eng"} if info is None else info
        ),
    )


def _output(pair, conditional_token=0, unconditional_token=0):
    # Deliberately reverse physical rows to catch positional pairing assumptions.
    ids = list(reversed(pair.physical_ids))
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: index for index, rid in enumerate(ids)},
        sampled_token_ids=[[unconditional_token], [conditional_token]],
    )
