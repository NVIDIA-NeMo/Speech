# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched CFG companions with independent native vLLM attention/Mamba slots."""
from __future__ import annotations

import copy
import math

from easymagpie_vllm_omni.cfg import CFG_COMPANION_PREFIX, CFGOptions
from vllm.logger import init_logger
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information

logger = init_logger(__name__)


class CFGPairSchedulerMixin:
    """Run one backbone batch for both branches; hide companions from clients/codec.

    CFG is an opt-in whole-text, known-speaker deployment. The bounded prefill
    contract and full KV-capacity reservation prevent a pair from being split
    by token budgets, admission limits or cache preemption. No scheduler loop
    is copied, and all CFG-off requests keep the existing scheduler behavior.
    """

    _cfg_enabled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg_options = CFGOptions.from_hf_config(self.vllm_config.model_config.hf_config)
        self._cfg_enabled = self._cfg_options.enabled
        self._cfg_companions: dict[str, str] = {}
        if not self._cfg_enabled:
            return
        config = self.scheduler_config
        if config.max_num_seqs % 2:
            raise ValueError('CFG requires an even max_num_seqs (two physical slots per speech request)')
        if config.enable_chunked_prefill:
            raise ValueError('CFG requires enable_chunked_prefill=false to keep both branches synchronized')
        if self.vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('CFG requires enable_prefix_caching=false for isolated branch state')
        required_tokens = config.max_num_seqs * self._cfg_options.max_prefill_tokens
        if self.max_num_scheduled_tokens < required_tokens:
            raise ValueError(
                f'CFG requires max_num_batched_tokens >= {required_tokens}; reduce max_num_seqs otherwise'
            )
        # Reserve enough native pages for every admitted branch at max_model_len.
        # This deliberately conservative bound also covers state-cache groups.
        cache = self.kv_cache_manager.kv_cache_config
        required_blocks = sum(
            config.max_num_seqs * math.ceil(self.max_model_len / group.kv_cache_spec.block_size)
            for group in cache.kv_cache_groups
        )
        if cache.num_blocks <= required_blocks + self.kv_cache_manager.watermark_blocks:
            raise ValueError('CFG KV capacity cannot retain every paired branch; reduce max_model_len/max_num_seqs')
        adapter = self.chunk_transfer_adapter
        if adapter is not None:
            adapter.save_async = _conditional_only_save(adapter.save_async)
        logger.info(
            'EasyMagpie CFG enabled: shared weights, paired batch, %d logical requests, default scale=%g',
            config.max_num_seqs // 2,
            self._cfg_options.scale,
        )

    def add_request(self, request):
        if not self._cfg_enabled:
            return super().add_request(request)
        if request.request_id.startswith(CFG_COMPANION_PREFIX):
            raise ValueError('Request ID uses the reserved CFG companion prefix')
        if getattr(request, 'resumable', False) or getattr(request, 'mm_features', None):
            raise ValueError(
                'CFG currently accepts whole-text known-speaker synthesis; streaming/user audio is unsupported'
            )
        info = deserialize_additional_information(getattr(request, 'additional_information', None))
        if not info.get('speaker_id'):
            raise ValueError('CFG requires a known speaker_id')
        if request.num_prompt_tokens > self._cfg_options.max_prefill_tokens:
            raise ValueError('CFG conditioning prefix exceeds easymagpie_cfg_max_prefill_tokens')
        scale = self._cfg_options.resolve_scale(info.get('cfg_scale'))
        companion_id = CFG_COMPANION_PREFIX + request.request_id
        # Deep-copy mutable token and streaming state; only the loaded model weights
        # are shared. The two requests receive independent native KV/Mamba slots.
        companion = copy.deepcopy(request)
        companion.request_id = companion_id
        companion.external_req_id = companion_id
        conditional = {**info, '_cfg_pair_id': request.request_id, '_cfg_role': 'conditional', 'cfg_scale': scale}
        negative = {**conditional, '_cfg_role': 'unconditional', 'omni_final_stage_id': 0}
        request.additional_information = serialize_additional_information(conditional)
        companion.additional_information = serialize_additional_information(negative)
        self._cfg_companions[request.request_id] = companion_id
        super().add_request(request)
        super().add_request(companion)

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        if self._cfg_enabled:
            scheduled = output.num_scheduled_tokens
            for parent, companion in self._cfg_companions.items():
                if (parent in scheduled or companion in scheduled) and scheduled.get(parent) != scheduled.get(
                    companion
                ):
                    states = {
                        key: {
                            name: getattr(self.requests.get(key), name, None)
                            for name in (
                                'status',
                                'num_computed_tokens',
                                'num_output_placeholders',
                                'num_prompt_tokens',
                                'num_tokens',
                                'max_tokens',
                                'is_prefill_chunk',
                            )
                        }
                        for key in (parent, companion)
                    }
                    raise RuntimeError(
                        f'CFG branches were not scheduled atomically: {parent}; '
                        f'scheduled={scheduled}; states={states}'
                    )
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if self._cfg_enabled:
            for client_outputs in outputs.values():
                if hasattr(client_outputs, 'outputs'):
                    client_outputs.outputs[:] = [
                        o for o in client_outputs.outputs if not o.request_id.startswith(CFG_COMPANION_PREFIX)
                    ]
                elif isinstance(client_outputs, list):
                    client_outputs[:] = [
                        o for o in client_outputs if not o.request_id.startswith(CFG_COMPANION_PREFIX)
                    ]
            for parent, companion in list(self._cfg_companions.items()):
                if parent not in self.requests and companion not in self.requests:
                    del self._cfg_companions[parent]
        return outputs

    def finish_requests(self, request_ids, finished_status):
        if self._cfg_enabled and request_ids is not None:
            ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
            request_ids = list(
                dict.fromkeys(ids + [self._cfg_companions[i] for i in ids if i in self._cfg_companions])
            )
        return super().finish_requests(request_ids, finished_status)


def _conditional_only_save(original):
    def save(multimodal_output, request, *args, **kwargs):
        if request.request_id.startswith(CFG_COMPANION_PREFIX):
            return None
        return original(multimodal_output, request, *args, **kwargs)

    return save
