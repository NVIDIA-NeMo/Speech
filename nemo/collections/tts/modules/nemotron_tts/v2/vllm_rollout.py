# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local vLLM-Omni rollout worker for native V2 online policy training."""

import atexit
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from nemo.collections.tts.models.nemotron_tts.v2.base import InferBatchOutput


def unpack_rollout_rows(model, rows):
    """Recover exact sampled actions and waveform-ready codes from packed engine rows."""
    device = model.device
    channels = model.num_audio_codebooks * model.frame_stacking_factor
    phones = model.phoneme_stacking_factor if model.phoneme_tokenizer is not None else 0
    audio_actions, phone_actions, audio_logps, phone_logps, decoded = [], [], [], [], []
    for packed in rows:
        if packed.ndim != 2 or packed.size(1) != 2 * (channels + phones) + 2:
            raise ValueError('The vLLM rollout action layout does not match the V2 model.')
        audio_mask, phone_mask = packed[:, channels + phones : channels + phones + 2].bool().unbind(-1)
        actions = packed[audio_mask, :channels].long().T.contiguous()
        if actions.size(1) == 0:
            raise ValueError('The vLLM rollout produced no acoustic actions.')
        eos_steps = (actions == model.audio_eos_id).any(0).nonzero(as_tuple=True)[0]
        if eos_steps.numel():
            actions = actions[:, : int(eos_steps[0]) + 1]
        audio_actions.append(actions)
        audio_logps.append(packed[audio_mask, channels + phones + 2 : 2 * channels + phones + 2][: actions.size(1)])
        unstacked = model.unstack_codes(
            actions.unsqueeze(0), torch.tensor([actions.size(1)]), model.frame_stacking_factor
        )[0][0]
        eos = (unstacked == model.audio_eos_id).any(0).nonzero(as_tuple=True)[0]
        decoded.append(unstacked[:, : int(eos[0])] if eos.numel() else unstacked)
        ph = packed[phone_mask, channels : channels + phones].long()
        pl = packed[phone_mask, 2 * channels + phones + 2 :]
        if phones:
            eos = (ph == model.phoneme_tokenizer.eos_token_id).any(-1).nonzero(as_tuple=True)[0]
            if eos.numel():
                ph, pl = ph[: int(eos[0]) + 1], pl[: int(eos[0]) + 1]
        phone_actions.append(ph)
        phone_logps.append(pl)

    lengths = torch.tensor([actions.size(1) for actions in audio_actions], device=device)
    codes = torch.full((len(rows), channels, int(lengths.max())), model.mask_token_id, dtype=torch.long, device=device)
    behavior = torch.zeros(len(rows), codes.size(-1), channels, device=device, dtype=torch.float32)
    for item, actions in enumerate(audio_actions):
        codes[item, :, : actions.size(1)] = actions.to(device)
        behavior[item, : actions.size(1)] = audio_logps[item].to(device)
    rollout = {
        '_po_audio_codes_stacked': codes,
        '_po_audio_codes_lens': lengths,
        '_po_audio_behavior_logps': behavior,  # [batch, decoder audio steps, stacked codebooks].
    }
    if phones:
        ph_lengths = torch.tensor([ph.size(0) * phones + 1 for ph in phone_actions], device=device)
        tokens = torch.full(
            (len(rows), int(ph_lengths.max())), model.phoneme_tokenizer.eos_token_id, dtype=torch.long, device=device
        )
        tokens[:, 0] = model.phoneme_tokenizer.bos_token_id
        max_phone_steps = max(ph.size(0) for ph in phone_actions)
        ph_behavior = torch.zeros(len(rows), max_phone_steps, phones, device=device, dtype=torch.float32)
        for item, ph in enumerate(phone_actions):
            tokens[item, 1 : 1 + ph.numel()] = ph.flatten().to(device)
            ph_behavior[item, : ph.size(0)] = phone_logps[item].to(device)
        rollout.update(
            _po_phoneme_tokens=tokens,
            _po_phoneme_tokens_lens=ph_lengths,
            _po_phoneme_behavior_logps=ph_behavior,  # [batch, phoneme steps, stacked phoneme channels].
        )
    decoded_lengths = torch.tensor([codes.size(1) for codes in decoded], device=device)
    decoded_codes = torch.zeros(
        len(rows), model.num_audio_codebooks, int(decoded_lengths.max()), device=device, dtype=torch.long
    )
    for item, codes in enumerate(decoded):
        decoded_codes[item, :, : codes.size(1)] = codes.to(device)
    return rollout, decoded_codes, decoded_lengths


class VLLMRolloutClient:
    """Own one isolated serving subprocess and synchronize its actor before generation."""

    def __init__(self, model):
        self.config = dict(model.cfg.get('vllm_rollout') or {})
        self.timeout = float(self.config.get('timeout_seconds', 900))
        self.work_dir = Path(self.config.get('work_dir', './vllm_rollouts')).resolve()
        self.rank_dir = self.work_dir / f'rank_{model.global_rank}'
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.log_file = None
        self.url = None
        self.version = None
        self.converter = importlib.import_module('tools.easymagpie_vllm_omni.scripts.convert_to_vllm')
        started = time.perf_counter()
        model_path = Path(self.config.get('model_path') or self.work_dir / 'model').resolve()
        error = [None]
        if model.global_rank == 0:
            try:
                if not (model_path / 'config.json').is_file():
                    args = self.converter.parse_args(
                        [
                            '--nemo_file',
                            'live-policy',
                            '--codec_model_path',
                            model.cfg.codecmodel_path,
                            '--outdir',
                            str(model_path),
                            '--dtype',
                            'bfloat16',
                            '--grpo-rollout',
                        ]
                    )
                    with torch.no_grad():
                        self.converter.convert(args, model=model)
                if not json.loads((model_path / 'config.json').read_text()).get('grpo_rollout', False):
                    raise ValueError('The vLLM model must be converted with --grpo-rollout.')
            except Exception as exc:
                error[0] = str(exc)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f'Unable to prepare the vLLM rollout model: {error[0]}')
        try:
            torch.cuda.empty_cache()
            self._start(model, model_path)
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - started

    def _start(self, model, model_path):
        ready = self.rank_dir / 'ready.json'
        ready.unlink(missing_ok=True)
        config_file = self.rank_dir / 'server_config.json'
        server_config = {
            'gpu_memory_utilization': 0.15,
            'max_num_seqs': 16,
            'enforce_eager': False,
            'sleep': True,
            **self.config,
        }
        config_file.write_text(json.dumps(server_config))
        server = Path(self.converter.__file__).with_name('grpo_server.py')
        environment = dict(os.environ)
        environment['TMPDIR'] = '/tmp'
        current_device = torch.cuda.current_device()
        visible = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
        selected = visible[current_device] if visible[0] and current_device < len(visible) else str(current_device)
        for name in list(environment):
            if name.startswith(('SLURM_', 'OMPI_', 'PMI_', 'PMIX_', 'NCCL_', 'TORCHELASTIC_')) or name in {
                'RANK',
                'WORLD_SIZE',
                'LOCAL_RANK',
                'LOCAL_WORLD_SIZE',
                'GROUP_RANK',
                'ROLE_RANK',
                'MASTER_ADDR',
                'MASTER_PORT',
                'NODE_RANK',
                'LIGHTNING_NODE_RANK',
            }:
                del environment[name]
        environment['CUDA_VISIBLE_DEVICES'] = str(self.config.get('cuda_visible_devices', selected))
        for name in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
            environment[name] = 'expandable_segments:False'
        self.log_file = (self.rank_dir / 'server.log').open('ab')
        command = [
            self.config.get('python_executable', sys.executable),
            str(server),
            '--model',
            str(model_path),
            '--work-dir',
            str(self.rank_dir),
            '--ready-file',
            str(ready),
            '--config',
            str(config_file),
        ]
        self.process = subprocess.Popen(
            command, env=environment, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
        atexit.register(self.close)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'vLLM rollout worker exited; see {self.rank_dir / "server.log"}')
            if ready.is_file():
                self.url = json.loads(ready.read_text())['url']
                try:
                    health = self._request('/health')
                    if not health.get('error'):
                        return
                except OSError:
                    pass
            time.sleep(0.25)
        self.close()
        raise TimeoutError(f'vLLM rollout worker did not become ready; see {self.rank_dir / "server.log"}')

    def _request(self, route, payload=None):
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f'vLLM rollout worker exited; see {self.rank_dir / "server.log"}')
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + route, data=body, headers={'Content-Type': 'application/json'})
        timeout = min(self.timeout, 5 if route == '/health' else 30 if route == '/sleep' else self.timeout)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            raise RuntimeError(f'vLLM rollout {route} failed: {detail}') from exc

    def _save(self, name, value):
        path = self.rank_dir / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(value, temporary)
        os.replace(temporary, path)
        return str(path)

    def generate(self, model, batch, **options):
        timings = {'startup': self.startup_seconds, 'weight_sync': 0.0}
        self.startup_seconds = 0.0
        version = int(model.global_step)
        torch.cuda.empty_cache()
        failed = True
        try:
            self._request('/wake', {})
            started = time.perf_counter()
            if self.version != version:
                weights = self.converter.select_weights(model.state_dict(), int(model.cfg.hidden_dim), torch.bfloat16)
                weights = {name: value.detach().cpu() for name, value in weights.items()}
                response = self._request('/refit', {'path': self._save('weights.pt', weights), 'version': version})
                if response.get('version') != version:
                    raise RuntimeError('The vLLM worker acknowledged a different policy version.')
                self.version = version
                timings['weight_sync'] = time.perf_counter() - started

            if 'context_audio_codes' in batch:
                context_codes, context_lens = batch['context_audio_codes'], batch['context_audio_codes_lens']
            else:
                context_codes, context_lens = model._codec_helper.audio_to_codes(
                    batch['context_audio'], batch['context_audio_lens']
                )
            context, context_lengths, _, _ = model.prepare_context_tensors(
                context_text_tokens=batch['context_text_tokens'],
                context_text_tokens_lens=batch['context_text_tokens_lens'],
                context_audio_codes=context_codes,
                context_audio_codes_lens=context_lens,
                training_mode=model.mode_name_to_mode[model.default_inference_mode],
                dropout_conditional_input=False,
            )
            text = model.embed_text_tokens(
                batch['text'], text_lens=batch['text_lens'], is_multiturn=model.cfg.get('use_multiturn_dataset', False)
            )
            text = text.masked_fill((batch['text'] == model.tokenizer.pad).unsqueeze(-1), 0.0)
            prompts = [
                {
                    'grpo_rollout': True,
                    'grpo_context_embeddings': context[item, : int(context_lengths[item])].cpu(),
                    'grpo_text_embeddings': text[item, : int(length)].cpu(),
                    'grpo_text_token_ids': batch['text'][item, : int(length)].tolist(),
                    **options,
                }
                for item, length in enumerate(batch['text_lens'])
            ]
            cohort = time.time_ns()
            request_ids = [f'{version}-{model.global_rank}-{cohort}-{item}' for item in range(len(prompts))]
            for prompt, request_id in zip(prompts, request_ids):
                prompt['request_id'] = request_id
            path = self._save('request.pt', {'version': version, 'prompts': prompts})
            started = time.perf_counter()
            response = self._request('/rollout', {'path': path, 'version': version})
            result_path = Path(response['path']).resolve()
            if result_path.parent != self.rank_dir:
                raise RuntimeError('The vLLM worker returned a file outside its rollout directory.')
            result = torch.load(result_path, map_location='cpu', weights_only=True)
            if not self.config.get('keep_rollout_files', False):
                result_path.unlink()
            if result.get('version') != version or result.get('request_ids') != request_ids:
                raise RuntimeError('The vLLM response policy version or request order is incorrect.')
            failed = False
        finally:
            if failed:
                self.close()
            else:
                try:
                    self._request('/sleep', {})
                except BaseException:
                    self.close()
                    raise
        timings['generation'] = time.perf_counter() - started
        rollout, codes, lengths = unpack_rollout_rows(model, result['rows'])
        model._policy_rollout = rollout
        started = time.perf_counter()
        codes, lengths = model._prepare_codes_for_decode(codes, lengths)
        audio, audio_lengths, _ = model._codec_helper.codes_to_audio(codes, lengths)
        timings['decode'] = time.perf_counter() - started
        if getattr(model, '_trainer', None) is not None:
            for name, seconds in timings.items():
                model.log(f'train_vllm_{name}_time_sec', seconds, sync_dist=True, prog_bar=False)
        return InferBatchOutput(audio, audio_lengths, codes, lengths, {'vllm_timings': timings})

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                elif process.poll() is None:
                    process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name != 'posix':
                    process.kill()
                    process.wait()
            finally:
                if os.name == 'posix':
                    # The serving leader may have exited before its GPU workers.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if self.log_file is not None:
            self.log_file.close()
