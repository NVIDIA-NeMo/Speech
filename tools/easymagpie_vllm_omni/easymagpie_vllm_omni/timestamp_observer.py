# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, read-only observation of one V2 main-backbone attention head.

Only the selected head's K history is retained. Q/K copies are graph-safe;
request dispatch and timestamp transport stay outside CUDA graphs. Ordinary
attention and all sampling execute unchanged. The native path decoder runs in
the consuming frontend, keeping Python path decoding off the GPU worker loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from easymagpie_vllm_omni.timestamps import pack_alignment_packet


class V2TimestampObserver:
    """Capture current selected-head Q/K and publish causal text-key posteriors."""

    def __init__(self, model, vllm_config):
        if model.arch.acoustic_predictor_type != 'acoustic_codes_predictor':
            raise ValueError('Attention timestamps currently require the V2 acoustic predictor architecture')
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError('V2 attention timestamps currently require tensor/pipeline parallel size one')
        if getattr(model, '_cfg_enabled', False):
            raise ValueError('V2 attention timestamps currently require CFG off')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('V2 attention timestamps require --no-enable-prefix-caching')
        self.model = model
        config = json.loads((Path(model.model_path) / 'config.json').read_text())
        if not config.get('timestamp_runtime_file'):
            raise ValueError('Reconvert the checkpoint to bundle the native timestamp runtime')
        settings = config.get('timestamp_config') or {}
        blocks = [
            (i, layer.mixer)
            for i, layer in enumerate(model.backbone.layers)
            if getattr(layer, 'mixer', None).__class__.__name__ == 'NemotronHAttention'
        ]
        selection = settings.get('layer', 'middle')
        ordinal = len(blocks) // 2 if selection == 'middle' else int(selection)
        if not 0 <= ordinal < len(blocks):
            raise ValueError('Timestamp layer is outside the main-backbone full-attention layers')
        self.block_index, attention = blocks[ordinal]
        self.head = int(settings.get('head', 0))
        if not 0 <= self.head < attention.num_heads:
            raise ValueError('Timestamp head is outside the main-backbone query heads')
        self.kv_head = self.head // (attention.num_heads // attention.num_kv_heads)
        self.head_dim = attention.head_dim
        self.scaling = attention.scaling
        self.max_len = int(vllm_config.model_config.max_model_len)
        self.text_bos_id = int(config.get('text_bos_id', model.text_eos_id - 1))
        self.text_pad_id = int(config.get('text_pad_id', model.text_eos_id + 1))
        from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig

        codec = EasyMagpieCodecConfig(**json.loads((Path(model.model_path) / 'codec_native/config.json').read_text()))
        self.codec_frame_seconds = codec.samples_per_codec_frame / codec.output_sample_rate
        self.stacking = model.arch.frame_stacking_factor
        self._queries = []
        self._states: dict[str, _RequestState] = {}
        self.output_packets = []
        kwargs = dict(device=model._combined_embeddings.device, dtype=model._combined_embeddings.dtype)
        size = vllm_config.scheduler_config.max_num_batched_tokens
        self._query = torch.empty(size, self.head_dim, **kwargs)
        self._key = torch.empty(size, self.head_dim, **kwargs)

        def capture(_module, args):
            query, key = args[:2]
            count = query.shape[0]
            self._query[:count].copy_(query.reshape(count, attention.num_heads, self.head_dim)[:, self.head])
            self._key[:count].copy_(key.reshape(count, attention.num_kv_heads, self.head_dim)[:, self.kv_head])

        self._hook = attention.attn.register_forward_pre_hook(capture)

    def register_query(self, input_ids, info, update, batch_start):
        """Record CPU request identity and the exact positions of consumed source tokens."""
        request_id = info.get('request_id')
        if request_id is None:
            return  # Synthetic memory profiling/capture has no logical requests.
        enabled = bool(info.get('return_timestamps', False))
        length = int(input_ids.shape[0])
        offset = int(info['_omni_num_computed_tokens'])
        prefill = bool(info['_omni_is_prefill'])
        decode_offset = int(info.get('decode_offset', 0) or 0)
        state = self._states.get(request_id)
        # Async scheduling can already have queued another step when EOS is
        # sampled. Retain the closed state until finished_req_ids arrives;
        # neither recreate it mid-sequence nor publish audio beyond EOS.
        enabled = enabled and not (state is not None and state.closed)
        self._queries.append((request_id, batch_start, length, offset, prefill, decode_offset, enabled))
        if not enabled:
            return
        if not info.get('speaker_id'):
            raise ValueError('V2 attention timestamps currently require a known speaker_id')
        if state is None:
            if offset != 0:
                raise ValueError('Timestamp recording must start with the first request prefill')
            state = _RequestState(
                keys=torch.empty(self.max_len, self.head_dim, device=self._key.device, dtype=self._key.dtype)
            )
            self._states[request_id] = state
        if offset != state.key_length:
            raise ValueError('Timestamp K history is incomplete; restart a preempted request with a new request ID')
        tokens = update.get('text_tokens', info.get('text_tokens')) or []
        if prefill:
            prefix_count = int(info.get('text_prefill_num', 0) or 0)
            first_text_position = int(info['_omni_prompt_len']) - prefix_count
            prefix = info.get('prefill_text_tokens') or tokens[:prefix_count]
            for position in range(max(offset, first_text_position), offset + length):
                index = position - first_text_position
                if index < len(prefix):
                    self._register_token(state, index, int(prefix[index]), position)
        elif decode_offset < len(tokens):
            self._register_token(state, decode_offset, int(tokens[decode_offset]), offset)

    def finish_forward(self):
        """Read current attention after code prediction; emit one compact packet per request."""
        queries, self._queries = self._queries, []
        self.output_packets = [torch.empty(0, dtype=torch.uint8) for _ in queries]
        active = [(i, q) for i, q in enumerate(queries) if q[-1]]
        for _, (request_id, start, length, offset, *_rest) in active:
            state = self._states[request_id]
            if offset + length > self.max_len:
                raise ValueError('Timestamp key history exceeded max_model_len')
            state.keys[offset : offset + length].copy_(self._key[start : start + length])
            state.key_length += length
        audio_queries = [(i, q) for i, q in active if not q[4] and q[5] >= self.model.speech_delay]
        if not audio_queries:
            return
        indices = [q[1] + q[2] - 1 for _, q in audio_queries]
        codes = self.model._out_codes[indices].detach().cpu().tolist()
        for (packet_index, query), code_row in zip(audio_queries, codes):
            request_id, start, length, *_ = query
            state = self._states[request_id]
            eos = [
                s
                for s in range(self.stacking)
                if any(
                    code_row[c * self.stacking + s] == self.model.audio_eos_id
                    for c in range(self.model.arch.num_audio_codebooks)
                )
            ]
            valid = min(eos) if eos else self.stacking
            q = self._query[start + length - 1].float().reshape(1, -1)
            k = state.keys[: state.key_length].float()
            logits = (q @ k.transpose(0, 1)).reshape(-1) * self.scaling
            if state.positions_gpu is None:
                state.positions_gpu = torch.tensor(state.positions, device=k.device, dtype=torch.long)
            # Keep the denominator over ALL causal keys, exactly as in NeMo.
            probabilities = torch.softmax(logits, dim=-1).nan_to_num(0.0).index_select(0, state.positions_gpu)
            packet = {
                'type': 'attention.observation',
                'sequence': state.sequence,
                'attention_block': self.block_index,
                'attention_head': self.head,
                'codec_frame_seconds': self.codec_frame_seconds,
                'codec_frame_start': state.codec_frames,
                'valid_codec_frames': valid,
                'registrations': state.pending_registrations,
                'token_indices': list(range(len(state.positions))),
                'probabilities': probabilities.cpu().tolist(),
                'audio_eos': bool(eos),
            }
            self.output_packets[packet_index] = pack_alignment_packet(packet)
            state.pending_registrations = []
            state.sequence += 1
            state.codec_frames += valid
            if eos:
                state.closed = True

    def discard_requests(self, request_ids):
        """Release K history on completion, cancellation and generation-cap termination."""
        for request_id in request_ids:
            self._states.pop(request_id, None)

    def _register_token(self, state, input_index, token_id, key_position):
        if not 0 <= token_id < self.text_bos_id or token_id == self.text_pad_id:
            return  # Match NeMo's native content-key eligibility, including control/IPA exclusions.
        registration = {
            'token_index': len(state.positions),
            'input_token_index': input_index,
            'token_id': token_id,
            'text_key_position': key_position,
        }
        state.positions.append(key_position)
        state.pending_registrations.append(registration)
        state.positions_gpu = None


@dataclass
class _RequestState:
    keys: torch.Tensor
    key_length: int = 0
    positions: list[int] = field(default_factory=list)
    pending_registrations: list[dict] = field(default_factory=list)
    positions_gpu: torch.Tensor | None = None
    codec_frames: int = 0
    sequence: int = 0
    closed: bool = False


def patch_alignment_transport() -> None:
    """Expose and drain the alignment tensor in vLLM-Omni 0.24 without accumulating old deltas."""
    from vllm_omni.engine import output_modality
    from vllm_omni.utils import mm_outputs

    mm_outputs._CLIENT_MM_ROOT_KEYS = mm_outputs._CLIENT_MM_ROOT_KEYS | {'alignment'}
    output_modality.DRAINABLE_MODALITIES.add('alignment')
