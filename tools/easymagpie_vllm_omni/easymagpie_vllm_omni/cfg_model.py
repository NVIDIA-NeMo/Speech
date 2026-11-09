# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Efficient paired-branch CFG execution for the EasyMagpie model."""
from __future__ import annotations

import json
import os

import torch
from easymagpie_vllm_omni.cfg import CFGOptions, build_cfg_pairs, unconditional_prefill
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


class CFGModelMixin:
    """Use one native vLLM backbone call for both independently cached branches."""

    _cfg_enabled = False

    def _init_cfg(self, vllm_config):
        self._cfg_options = CFGOptions.from_hf_config(vllm_config.model_config.hf_config)
        self._cfg_enabled = self._cfg_options.enabled
        self._cfg_records = []
        self._cfg_prefill_keys = set()
        self._cfg_trace_path = os.environ.get('EASYMAGPIE_CFG_TRACE_PATH')
        self._cfg_traced = set()
        self._cfg_logged_metadata = False
        logger.warning(
            'EasyMagpie worker CFG enabled=%s default_scale=%g tracing=%s',
            self._cfg_enabled,
            self._cfg_options.scale,
            bool(self._cfg_trace_path),
        )
        self.code_predictor._cfg_code_loop.trace = bool(self._cfg_trace_path)
        self._cfg_conditional_mask = torch.ones_like(self._dec_audio_valid)
        self._cfg_backbone_forward = (self.backbone,)
        mode = getattr(vllm_config.compilation_config, 'cudagraph_mode', None)
        if self._cfg_enabled and mode is not None and mode.has_full_cudagraphs():
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            from vllm.config import CUDAGraphMode

            # Capture only tensor computation. Pair dispatch stays outside this
            # graph, while the expensive backbone retains full replay speed.
            self._cfg_backbone_forward = (
                CUDAGraphWrapper(self.backbone, vllm_config, runtime_mode=CUDAGraphMode.FULL),
            )
        vocab = self.text_embedding.num_embeddings
        self._cfg_unknown_id = getattr(self.arch, 'cfg_unk_id', None)
        if self._cfg_unknown_id is None:
            self._cfg_unknown_id = self.arch.resolved_text_eos_id(vocab) + 1
        if self._cfg_enabled and not 0 <= self._cfg_unknown_id < vocab:
            raise ValueError('CFG_UNK is outside the exported text-embedding table; reconvert the checkpoint')

    def _register_cfg_query(self, input_ids, info, start):
        if self._cfg_enabled and not getattr(self, '_cfg_logged_metadata', False) and info.get('speaker_id'):
            logger.warning(
                'EasyMagpie first real query: CFG enabled=%s pair_metadata=%s role=%s',
                self._cfg_enabled,
                '_cfg_pair_id' in info,
                info.get('_cfg_role'),
            )
            self._cfg_logged_metadata = True
        if '_cfg_pair_id' in info and not self._cfg_enabled:
            raise RuntimeError('Scheduler enabled CFG but the GPU model did not enable CFG')
        if self._cfg_enabled and info.get('speaker_id') and '_cfg_pair_id' not in info:
            raise RuntimeError('CFG model received an unpaired speech request')
        if not self._cfg_enabled or '_cfg_pair_id' not in info:
            return  # Dummy profiling/capture calls have no real request metadata.
        key, role = info['_cfg_pair_id'], info['_cfg_role']
        start = self._batch_slot_offset(input_ids, start)
        length = int(input_ids.shape[0])
        scale = self._cfg_options.resolve_scale(info.get('cfg_scale'))
        self._cfg_records.append((key, role, start, length, scale))
        if info['_omni_is_prefill']:
            self._cfg_prefill_keys.add(key)
        self._cfg_conditional_mask[start : start + length] = int(role == 'conditional')

    def _mask_cfg_prefill(self, take, conditioning_len, offset, info):
        if not self._cfg_enabled or info.get('_cfg_role') != 'unconditional':
            return take
        unknown = self.text_embedding(torch.tensor(self._cfg_unknown_id, device=take.device)).to(take.dtype)
        return unconditional_prefill(unknown, offset=offset, length=take.shape[0], conditioning_len=conditioning_len)

    def _forward_cfg(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        records, self._cfg_records = self._cfg_records, []
        prefills, self._cfg_prefill_keys = self._cfg_prefill_keys, set()
        pairs = build_cfg_pairs(records)
        num_tokens = input_ids.shape[0]
        profile = not records
        if profile:
            # Warm both compiled subgraphs, including the actual guided sampler,
            # at each physical capture size. No client metadata exists here.
            logical = max(1, num_tokens // 2)
            pairs = [
                (f'profile-{i}', i, min(i + logical, num_tokens - 1), self._cfg_options.scale) for i in range(logical)
            ]
            self._cfg_conditional_mask[:num_tokens].zero_()
            self._cfg_conditional_mask[:logical] = 1
        decode = [p for p in pairs if p[0] not in prefills]
        combined = self._combined_embeddings[:num_tokens]
        if inputs_embeds is None:
            combined.zero_()
        else:
            combined.copy_(inputs_embeds)
        self._out_codes[:num_tokens].zero_()
        self._token_stop[:num_tokens].zero_()
        if decode:
            indices = torch.tensor([[p[1], p[2]] for p in decode], dtype=torch.long, device=input_ids.device)
            conditional, unconditional = indices[:, 0], indices[:, 1]
            self._assemble_decode_embeddings(combined, indices.flatten())
        # Both requests own distinct native attention/Mamba state. There is only
        # ONE backbone module, one weight copy and one batched forward call.
        hidden = self._cfg_backbone_forward[0](
            input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors, inputs_embeds=combined
        )
        if decode:
            scales = torch.tensor([p[3] for p in decode], dtype=torch.float32, device=input_ids.device).unsqueeze(-1)
            context = get_forward_context()
            original_descriptor = context.batch_descriptor
            padded_batch = len(decode)
            from vllm.config import CUDAGraphMode

            if context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                physical = next((size for size in sorted(sizes) if size >= 2 * padded_batch and size % 2 == 0), None)
                if physical is None:
                    raise ValueError('CFG capture sizes must include an even size large enough for every branch pair')
                padded_batch = physical // 2
            context.batch_descriptor = BatchDescriptor(num_tokens=2 * padded_batch)
            try:
                codes, diagnostics = self.code_predictor.generate_codes_cfg(
                    hidden[conditional], hidden[unconditional], scales, padded_batch=padded_batch
                )
            finally:
                context.batch_descriptor = original_descriptor
            self._out_codes[conditional] = codes
            self._out_codes[unconditional] = codes
            self._flag_audio_eos(codes, conditional)
            self._flag_audio_eos(codes, unconditional)
            if self.has_phoneme:
                self._predict_phonemes(hidden, conditional)
            if self._cfg_trace_path and not profile:
                self._trace_cfg(decode, conditional, diagnostics)
        if self.has_phoneme:
            last = [p[1] for p in pairs if p[0] in prefills]
            if last:
                self._predict_phonemes(hidden, torch.tensor(last, device=input_ids.device, dtype=torch.long))
        logits_index = kwargs.get('logits_index')
        if logits_index is not None:
            self._sample_stop[: logits_index.shape[0]] = self._token_stop[logits_index]
        else:
            self._sample_stop[:num_tokens].copy_(self._token_stop[:num_tokens])
        return hidden

    def _trace_cfg(self, pairs, conditional, diagnostics):
        if all(p[0] in self._cfg_traced for p in pairs):
            return
        # Opt-in diagnostics synchronize only the first real acoustic frame per
        # request. The production/throughput path never reads GPU scalars here.
        valid = self._dec_audio_valid[conditional].detach().cpu().tolist()
        values = diagnostics.detach().float().cpu().tolist()
        with open(self._cfg_trace_path, 'a', encoding='utf-8') as stream:
            for pair, active, value in zip(pairs, valid, values):
                key, cond_row, uncond_row, scale = pair
                if not active or key in self._cfg_traced:
                    continue
                self._cfg_traced.add(key)
                stream.write(
                    json.dumps(
                        {
                            'request_id': key,
                            'cfg_active': True,
                            'cfg_scale': scale,
                            'conditional_row': cond_row,
                            'unconditional_row': uncond_row,
                            'backbone_branches_batched': True,
                            'conditional_unconditional_logits_max_abs': value[0],
                            'guided_conditional_logits_max_abs': value[1],
                            'first_codebook_argmax_changed': bool(value[2]),
                        }
                    )
                    + '\n'
                )
