# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local vLLM-Omni LM rollouts with versioned file-based policy refits."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class FileRequest(BaseModel):
    path: str
    version: int = Field(ge=0)


class RolloutService:
    """Serialize policy changes while generating one cohort concurrently."""

    def __init__(
        self, engine, arch, work_dir, settings, extract_delta=None, sampling_params_factory=None, stop_token_id=1
    ):
        self.engine = engine
        self.arch = arch
        self.work_dir = Path(work_dir).resolve()
        self.settings = settings
        self.version = -1
        self.sleeping = False
        self.error = None
        self.lock = asyncio.Lock()
        self.stop_token_id = stop_token_id
        self.packed_width = (
            2 * (arch.num_audio_codebooks * arch.frame_stacking_factor + arch.phoneme_stacking_factor) + 2
        )
        if extract_delta is None:
            from benchmark_model import _extract_step_audio_codes

            extract_delta = _extract_step_audio_codes
        self.extract_delta = extract_delta
        self.sampling_params_factory = sampling_params_factory or _sampling_params

    async def refit(self, request: FileRequest):
        async with self.lock:
            self._check_healthy()
            if request.version < self.version:
                raise HTTPException(409, 'Cannot restore an older policy version.')
            path = _confined_path(request.path, self.work_dir)
            started = time.perf_counter()
            self.version = -1
            try:
                await self._wake()
                acknowledgments = await asyncio.wait_for(
                    self.engine.collective_rpc(
                        method='reload_grpo_weights', args=(str(path), request.version), stage_ids=[0]
                    ),
                    timeout=min(120.0, float(self.settings.get('timeout_seconds', 600))),
                )
                workers = _validate_refit_ack(acknowledgments, request.version)
            except Exception as exc:
                self.error = f'Policy refit failed: {exc}'
                raise HTTPException(503, self.error) from exc
            self.version = request.version
            return {
                'version': self.version,
                'acknowledgments': workers,
                'timings': {'refit_seconds': time.perf_counter() - started},
            }

    async def rollout(self, request: FileRequest):
        async with self.lock:
            self._check_healthy()
            if request.version != self.version:
                raise HTTPException(409, f'Policy version {self.version} does not match {request.version}.')
            path = _confined_path(request.path, self.work_dir)
            started = time.perf_counter()
            payload = await asyncio.to_thread(torch.load, path, map_location='cpu', weights_only=True)
            if not isinstance(payload, dict) or payload.get('version') != request.version:
                raise HTTPException(400, 'Rollout file has a missing or mismatched policy version.')
            prompts = payload.get('prompts')
            if not isinstance(prompts, list) or not prompts:
                raise HTTPException(400, 'Rollout file must contain a nonempty prompts list.')
            inputs = [self._prepare_prompt(prompt) for prompt in prompts]
            signatures = {(info['temperature'], info['top_k'], info['phoneme_sampling_method']) for info in inputs}
            if len(signatures) != 1:
                raise HTTPException(400, 'A cohort must use one temperature, top-k, and phoneme sampling method.')
            loaded_at = time.perf_counter()
            cohort_id = uuid.uuid4().hex
            request_ids = [
                info.get('request_id', f'grpo-v{self.version}-{cohort_id}-{index}')
                for index, info in enumerate(inputs)
            ]
            if any(not isinstance(request_id, str) or not request_id for request_id in request_ids):
                raise HTTPException(400, 'Request IDs must be nonempty strings.')
            if len(set(request_ids)) != len(request_ids):
                raise HTTPException(400, 'Request IDs must be unique within a cohort.')
            tasks = []
            try:
                await self._wake()
                tasks = [
                    asyncio.create_task(self._generate(info, request_id))
                    for info, request_id in zip(inputs, request_ids)
                ]
                rows = await asyncio.gather(*tasks)
            except BaseException as exc:
                self.error = f'Rollout cohort failed: {exc}'
                for task in tasks:
                    task.cancel()
                drained = await asyncio.gather(
                    *(self.engine.abort(request_id) for request_id in request_ids), return_exceptions=True
                )
                await asyncio.gather(*tasks, return_exceptions=True)
                if any(isinstance(result, BaseException) for result in drained):
                    self.error += '; engine request draining failed'
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise HTTPException(503, self.error) from exc
            timings = {
                'input_seconds': loaded_at - started,
                'generation_seconds': time.perf_counter() - loaded_at,
            }
            output_path = self.work_dir / f'rollout_v{self.version}_{cohort_id}.pt'
            await asyncio.to_thread(
                _atomic_torch_save,
                {'version': self.version, 'rows': rows, 'request_ids': request_ids, 'timings': timings},
                output_path,
            )
            return {'path': str(output_path), 'version': self.version}

    async def sleep(self):
        async with self.lock:
            if self.settings.get('sleep', True) and not self.sleeping:
                try:
                    await self.engine.sleep(stage_ids=[0], level=1)
                except Exception as exc:
                    self.error = f'Engine sleep failed: {exc}'
                    raise HTTPException(503, self.error) from exc
                self.sleeping = True
            return {'version': self.version, 'sleeping': self.sleeping}

    async def wake(self):
        async with self.lock:
            self._check_healthy()
            await self._wake()
            return {'version': self.version, 'sleeping': self.sleeping}

    def _check_healthy(self):
        if self.error is not None:
            raise HTTPException(503, self.error)

    async def _wake(self):
        if self.sleeping:
            try:
                await self.engine.wake_up(stage_ids=[0])
            except Exception as exc:
                self.error = f'Engine wake failed: {exc}'
                raise HTTPException(503, self.error) from exc
            self.sleeping = False

    def _prepare_prompt(self, prompt):
        if not isinstance(prompt, dict):
            raise HTTPException(400, 'Each prompt must be a dictionary.')
        info = dict(prompt, grpo_rollout=True)
        for name in ('grpo_context_embeddings', 'grpo_text_embeddings'):
            value = info.get(name)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.size(1) != self.arch.embedding_dim:
                raise HTTPException(400, f'{name} must have shape [rows, {self.arch.embedding_dim}].')
            info[name] = value.to(device="cpu", dtype=torch.float32).contiguous()
        tokens = info.get('grpo_text_token_ids')
        if not isinstance(tokens, list) or not tokens or len(tokens) != info['grpo_text_embeddings'].size(0):
            raise HTTPException(400, 'Text token IDs must match the text embedding rows.')
        if info.get('phoneme_sampling_method') not in ('sample', 'argmax'):
            raise HTTPException(400, 'Unsupported phoneme sampling method.')
        if not isinstance(info.get('max_audio_steps'), int) or info['max_audio_steps'] < 1:
            raise HTTPException(400, 'max_audio_steps must be a positive integer.')
        if not 0 < float(info.get('temperature', 0)) < float('inf') or int(info.get('top_k', 0)) < 1:
            raise HTTPException(400, 'A positive temperature and top_k are required.')
        return info

    async def _generate(self, info, request_id):
        parameters = self.sampling_params_factory(
            max_tokens=info['max_audio_steps'] + self.arch.streaming_speech_delay - self.arch.streaming_phonemes_delay,
            stop_token_id=self.stop_token_id,
        )
        context_rows = info['grpo_context_embeddings'].size(0)
        prompt = {
            'prompt_token_ids': [0] * (context_rows + self.arch.text_prefill_num),
            'additional_information': info,
        }
        generator = self.engine.generate(prompt, sampling_params_list=[parameters], request_id=request_id)
        chunks = []
        try:
            async for output in generator:
                delta = self.extract_delta(output)
                if delta is not None and delta.numel():
                    if delta.ndim != 2 or delta.size(1) != self.packed_width:
                        raise RuntimeError(f'Expected packed rollout width {self.packed_width}, got {delta.shape}.')
                    chunks.append(delta.detach().to(device='cpu', dtype=torch.float32).clone())
        finally:
            await generator.aclose()
        if not chunks:
            raise RuntimeError(f'Request {request_id} returned no packed rollout rows.')
        rows = torch.cat(chunks, dim=0)
        if not torch.isfinite(rows).all():
            raise RuntimeError(f'Request {request_id} returned non-finite rollout rows.')
        audio_valid_column = self.arch.num_audio_codebooks * self.arch.frame_stacking_factor
        audio_valid_column += self.arch.phoneme_stacking_factor
        if not (rows[:, audio_valid_column] > 0.5).any():
            raise RuntimeError(f'Request {request_id} returned no valid audio rows.')
        return rows


def create_app(service):
    app = FastAPI()

    @app.get('/health')
    async def health():
        state = {
            'version': service.version,
            'sleeping': service.sleeping,
            'busy': service.lock.locked(),
            'error': service.error,
        }
        return JSONResponse(state, status_code=503 if service.error else 200)

    app.post('/refit')(service.refit)
    app.post('/rollout')(service.rollout)
    app.post('/sleep')(service.sleep)
    app.post('/wake')(service.wake)
    return app


def _confined_path(path, work_dir):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = work_dir / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(400, f'IPC file does not exist: {path}') from exc
    if not candidate.is_relative_to(work_dir) or not candidate.is_file():
        raise HTTPException(400, 'IPC files must be regular files inside the rank work directory.')
    return candidate


def _validate_refit_ack(payload, version):
    if isinstance(payload, Mapping):
        if 'todo' in payload or 'error' in payload:
            raise RuntimeError(f'Worker rejected refit: {payload}')
        if 'version' in payload or 'loaded_weights' in payload:
            loaded = payload.get('loaded_weights')
            if payload.get('version') != version or type(loaded) is not int or loaded <= 0:
                raise RuntimeError(f'Invalid refit acknowledgment: {payload}')
            return 1
        count = sum(_validate_refit_ack(value, version) for value in payload.values())
    elif isinstance(payload, (list, tuple)):
        count = sum(_validate_refit_ack(value, version) for value in payload)
    else:
        raise RuntimeError(f'Unexpected refit acknowledgment: {payload!r}')
    if count == 0:
        raise RuntimeError('No worker acknowledged the policy refit.')
    return count


def _sampling_params(*, max_tokens, stop_token_id):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=[stop_token_id],
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )


def _atomic_torch_save(payload, path):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


async def _serve(args):
    import vllm_plugin_easymagpie_omni

    vllm_plugin_easymagpie_omni.register()

    from easymagpie_vllm_omni.config import EasyMagpieOmniArch
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
    from vllm_omni import AsyncOmni

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    model_config = json.loads((args.model / 'config.json').read_text())
    if not model_config.get('grpo_rollout', False):
        raise ValueError('The converted model must enable grpo_rollout.')
    hf_config = SimpleNamespace(**model_config)
    arch = EasyMagpieOmniArch.from_hf_config(hf_config)
    if args.config.lstrip().startswith(('{', 'null')):
        settings = json.loads(args.config)
    else:
        settings = json.loads(Path(args.config).read_text())
    settings = settings or {}
    deploy = yaml.safe_load((Path(__file__).resolve().parents[1] / 'deploy/easymagpie_lm.yaml').read_text())
    deploy['dtype'] = 'bfloat16'
    deploy['stages'][0].update(
        max_num_seqs=int(settings.get('max_num_seqs', 16)),
        gpu_memory_utilization=float(settings.get('gpu_memory_utilization', 0.15)),
        enforce_eager=bool(settings.get('enforce_eager', False)),
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        devices='0',
    )
    for name in ('max_model_len', 'max_num_batched_tokens'):
        if name in settings:
            deploy['stages'][0][name] = int(settings[name])
    deploy_path = work_dir / 'deploy.yaml'
    deploy_path.write_text(yaml.safe_dump(deploy))
    engine = AsyncOmni(
        model=str(args.model),
        deploy_config=str(deploy_path),
        log_stats=False,
        stage_init_timeout=float(settings.get('timeout_seconds', 600)),
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        service = RolloutService(
            engine,
            arch,
            work_dir,
            settings,
            stop_token_id=EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(hf_config),
        )
        sock.bind(('127.0.0.1', 0))
        sock.listen(128)
        server = uvicorn.Server(uvicorn.Config(create_app(service), log_level='warning', access_log=False))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done():
                await task
                raise RuntimeError('The rollout HTTP server did not start.')
            await asyncio.sleep(0.01)
        host, port = sock.getsockname()
        ready = {'url': f'http://{host}:{port}', 'pid': os.getpid(), 'host': host, 'port': port}
        temporary = args.ready_file.with_name(f'.{args.ready_file.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(ready))
        os.replace(temporary, args.ready_file)
        await task
    finally:
        args.ready_file.unlink(missing_ok=True)
        sock.close()
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--config', default='{}', help='JSON settings or a path to a JSON settings file.')
    args = parser.parse_args()
    os.environ.setdefault('VLLM_PLUGINS', 'easymagpie_omni')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    asyncio.run(_serve(args))


if __name__ == '__main__':
    main()
