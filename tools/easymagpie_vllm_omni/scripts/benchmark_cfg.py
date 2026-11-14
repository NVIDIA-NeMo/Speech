# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare real single-turn CFG/off speech and steady-state HTTP throughput.

Input is JSONL with ``id`` and ``text``. Each output retains the exact text,
PCM waveform, settings and timings. Starts one server at a time on the same GPU.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voice', required=True)
    parser.add_argument('--scale', type=float, default=2.5)
    parser.add_argument('--sample-temperature', type=float, default=0.0)
    parser.add_argument('--vllm', default='vllm')
    parser.add_argument('--sample-rate', type=int, default=22050)
    parser.add_argument('--max-new-tokens', type=int, default=1600)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--request-timeout', type=int, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.sample_temperature) or args.sample_temperature < 0:
        parser.error('sample-temperature must be finite and non-negative')
    if args.scale < 0 or args.concurrency < 1 or args.repeats < 1:
        parser.error('scale must be non-negative; concurrency/repeats must be positive')
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input IDs must be unique and the manifest must be nonempty')
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / 'inputs.json', rows)
    summary = []
    for enabled in (False, True):
        name = 'cfg_on' if enabled else 'cfg_off'
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'deploy/easymagpie_cfg.yaml').read_text())
        config['stages'][0]['hf_overrides']['easymagpie_cfg'] = enabled
        config['stages'][0]['hf_overrides']['easymagpie_cfg_scale'] = args.scale
        deploy = args.output / f'{name}.yaml'
        deploy.write_text(yaml.safe_dump(config, sort_keys=False))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        env = dict(os.environ)
        # Numerical tracing is checked separately: exclude trace synchronization
        # and per-frame diagnostic reductions from all throughput measurements.
        env.pop('EASYMAGPIE_CFG_TRACE_PATH', None)
        env['EASYMAGPIE_CFG_ENABLED'] = '1' if enabled else '0'
        env['CFG_SCALE'] = str(args.scale)
        env['VLLM_PLUGINS'] = 'easymagpie_omni'
        command = [
            args.vllm,
            'serve',
            args.model,
            '--deploy-config',
            str(deploy),
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--trust-remote-code',
            '--disable-log-stats',
            '--disable-uvicorn-access-log',
            '--uvicorn-log-level',
            'warning',
            '--omni',
        ]
        _write(args.output / f'{name}_command.json', command)
        with (args.output / f'{name}_server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                _wait_ready(server, url, args.startup_timeout)
                # Compile/warm the exact paired batch sizes before collecting timing.
                for workers in (1, args.concurrency):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                        list(
                            pool.map(
                                lambda i: _speech(
                                    args,
                                    url,
                                    rows[i % len(rows)],
                                    enabled,
                                    args.scale,
                                    0.7,
                                    name + f'_warm{workers}_{i}',
                                    False,
                                ),
                                range(workers),
                            )
                        )
                conditions = [('scale_1', 1.0), ('scale_2p5', args.scale)] if enabled else [('off', None)]
                for condition, scale in conditions:
                    for row in rows:
                        result = _speech(
                            args,
                            url,
                            row,
                            enabled,
                            scale,
                            args.sample_temperature,
                            f'{name}_{condition}_{row["id"]}',
                            True,
                        )
                        print(json.dumps({'sample': result}), flush=True)
                for workers in (1, args.concurrency):
                    jobs = rows * args.repeats
                    began = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [
                            pool.submit(
                                _speech,
                                args,
                                url,
                                row,
                                enabled,
                                args.scale if enabled else None,
                                0.7,
                                f'{name}_perf{workers}_{i}',
                                False,
                            )
                            for i, row in enumerate(jobs)
                        ]
                        results = [f.result() for f in futures]
                    elapsed = time.perf_counter() - began
                    audio_seconds = sum(r['audio_seconds'] for r in results)
                    record = {
                        'condition': name,
                        'cfg_scale': args.scale if enabled else None,
                        'concurrency': workers,
                        'requests': len(results),
                        'wall_s': elapsed,
                        'audio_seconds': audio_seconds,
                        'audio_seconds_per_second': audio_seconds / elapsed,
                        'median_ttfa_s': statistics.median(r['ttfa_s'] for r in results),
                        'median_request_rtf': statistics.median(r['elapsed_s'] / r['audio_seconds'] for r in results),
                        'tracing': False,
                        'warmup_excluded': True,
                        'requests_detail': results,
                    }
                    summary.append(record)
                    _write(args.output / 'performance.json', summary)
                    print(
                        json.dumps({'performance': {k: v for k, v in record.items() if k != 'requests_detail'}}),
                        flush=True,
                    )
            finally:
                _stop(server)
    _write(args.output / 'complete.json', {'complete': True, 'inputs': len(rows), 'settings': vars(args)})


def _speech(args, url, row, enabled, scale, temperature, name, save):
    suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)[:180] + '_' + suffix
    extra = {'temperature': temperature, 'top_k': 80}
    if enabled:
        extra['cfg_scale'] = scale
    payload = {
        'input': row['text'],
        'voice': args.voice,
        'response_format': 'pcm',
        'stream': True,
        'stream_format': 'audio',
        'max_new_tokens': args.max_new_tokens,
        'extra_params': extra,
    }
    request = urllib.request.Request(
        url + '/v1/audio/speech', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}
    )
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        first = response.read(2)
        ttfa = time.perf_counter() - began
        audio = first + response.read()
    elapsed = time.perf_counter() - began
    if not audio or len(audio) % 2:
        raise ValueError(f'Invalid PCM output for {name}: {len(audio)} bytes')
    # Retain failed outputs for diagnosis. This loose 20 words/s lower bound
    # catches severe early stops without rejecting legitimate short utterances.
    suspiciously_short = len(audio) / 2 / args.sample_rate < len(row['text'].split()) / 20
    result = {
        'name': name,
        'id': row['id'],
        'text': row['text'],
        'cfg_enabled': enabled,
        'cfg_scale': scale if enabled else None,
        'temperature': temperature,
        'top_k': 80,
        'elapsed_s': elapsed,
        'ttfa_s': ttfa,
        'audio_seconds': len(audio) / 2 / args.sample_rate,
        'pcm_sha256': hashlib.sha256(audio).hexdigest(),
        'suspiciously_short': suspiciously_short,
    }
    if save or suspiciously_short:
        path = args.output / ('diagnostics' if suspiciously_short else 'samples') / f'{name}.wav'
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(args.sample_rate)
            output.writeframes(audio)
        result['audio'] = str(path)
        _write(path.with_suffix('.json'), result)
        path.with_suffix('.txt').write_text(row['text'] + '\n')
    if suspiciously_short:
        raise ValueError(f'Suspected early stop for {name}; exact text and audio saved to {path}')
    return result


def _wait_ready(server, url, timeout):
    began = time.monotonic()
    while time.monotonic() - began < timeout:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited {server.returncode}; inspect the saved server log')
        try:
            with urllib.request.urlopen(url + '/health', timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError('Server initialization timed out')


def _stop(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
