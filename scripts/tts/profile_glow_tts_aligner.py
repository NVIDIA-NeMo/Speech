# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile the tiny Glow-TTS aligner and compact MAS without model/data assets.

Example:
    python scripts/tts/profile_glow_tts_aligner.py --device cuda --output profile.json

Wall-clock measurements synchronize CUDA at their boundaries, so the CPU MAS
baseline includes GPU-to-CPU score transfer, Numba search, and return transfer.
Compilation and Numba initialization are excluded by separate warmup iterations.
The additional turn-batching comparison keeps total frame and token counts equal
while measuring the smaller search lattices produced by independent turns.
"""

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner
from nemo.collections.tts.parts.utils.monotonic_alignment import TRITON_AVAILABLE, maximum_path_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda', help='Torch device, such as cuda, cuda:0, or cpu')
    parser.add_argument('--output', type=Path, help='Optional JSON report; also prints JSON to stdout')
    parser.add_argument(
        '--varying-shapes-only', action='store_true', help='Only measure first-use and varying-shape Triton MAS calls'
    )
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument(
        '--long-iterations', type=int, default=5, help='Iteration cap for cases with at least 2048 frames'
    )
    parser.add_argument(
        '--shapes', default='32,256,82;16,768,246;1,5250,1680', help='Semicolon-separated B,T,N shapes'
    )
    parser.add_argument('--audio-channels', type=int, default=80, help='Default: codec dimension 40, stacked by two')
    parser.add_argument('--text-channels', type=int, default=1536)
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--num-flows', type=int, default=4)
    parser.add_argument('--num-text-layers', type=int, default=2)
    parser.add_argument('--dequantization-noise', type=float, default=0.01)
    parser.add_argument('--cpu-threads', type=int, help='Optional PyTorch and Numba CPU thread count')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.long_iterations < 1:
        parser.error('Warmup and iteration counts must be positive')
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error('--cpu-threads must be positive')
        import numba

        torch.set_num_threads(args.cpu_threads)
        numba.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    torch.manual_seed(args.seed)
    varying_shapes = _profile_varying_shapes(device, args.iterations)
    if args.varying_shapes_only:
        _write_report(
            {
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': _git_commit(),
                'environment': _environment(device),
                'varying_shapes_mas': varying_shapes,
            },
            args.output,
        )
        return
    shapes = _parse_shapes(args.shapes, parser)
    shapes.extend([('separate_turns', (8, 256, 82)), ('whole_conversation', (1, 2048, 656))])
    model = TinyGlowTTSAligner(
        audio_channels=args.audio_channels,
        text_channels=args.text_channels,
        hidden_channels=args.hidden_channels,
        num_flows=args.num_flows,
        num_text_layers=args.num_text_layers,
        dequantization_noise=args.dequantization_noise,
        mas_backend='auto',
    ).to(device)
    model.train()
    cases = []
    for name, shape in shapes:
        iterations = min(args.iterations, args.long_iterations) if shape[1] >= 2048 else args.iterations
        print(f'Profiling {name}: {shape}, {iterations} measured iterations', file=sys.stderr, flush=True)
        cases.append(_profile_case(name, shape, model, device, args.warmup, iterations))
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'environment': _environment(device),
        'config': {
            'warmup': args.warmup,
            'iterations': args.iterations,
            'long_iterations': args.long_iterations,
            'audio_channels': args.audio_channels,
            'text_channels': args.text_channels,
            'hidden_channels': args.hidden_channels,
            'num_flows': args.num_flows,
            'num_text_layers': args.num_text_layers,
            'dequantization_noise': args.dequantization_noise,
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameters': sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            'seed': args.seed,
        },
        'cases': cases,
        'varying_shapes_mas': varying_shapes,
        'turn_batching': _turn_comparison(cases),
        'notes': [
            'Synthetic fixed audio and detached text features; no codec, tokenizer, or TTS decoder is loaded.',
            'Flow forward includes text prior, acoustic flow, score matrix, MAS, and selected-path likelihood.',
            'Forward and backward wall times synchronize CUDA; compilation and warmup are excluded.',
            'Model gradients are cleared between iterations; no optimizer step is included.',
            'CPU MAS timings include score and path transfers. CUDA memory excludes the CPU baseline scratch memory.',
            'The separate-turn and whole-conversation cases preserve total frames/tokens but have different lattices.',
        ],
    }
    _write_report(report, args.output)


def _write_report(report, output):
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.tmp')
        temporary.write_text(serialized + '\n', encoding='utf-8')
        temporary.replace(output)
    print(serialized)


def _profile_varying_shapes(device, iterations):
    if device.type != 'cuda' or not TRITON_AVAILABLE:
        return {'supported': False, 'reason': 'CUDA and Triton are required'}
    # All exact shapes and layouts share BLOCK_TEXT=128. Time the initial bucket
    # invocation separately, then first-use calls for new runtime dimensions.
    # A fresh TRITON_CACHE_DIR can be supplied externally to measure cold compile.
    warm_shape = (2, 203, 65)
    warm_scores = torch.randn(warm_shape, device=device)
    warm_text_lens = torch.full((2,), 65, device=device, dtype=torch.long)
    warm_audio_lens = torch.full((2,), 203, device=device, dtype=torch.long)
    _synchronize(device)
    started = time.perf_counter()
    warm_path = maximum_path_indices(warm_scores, warm_text_lens, warm_audio_lens, backend='triton')
    _synchronize(device)
    first_bucket_ms = 1000.0 * (time.perf_counter() - started)
    del warm_scores, warm_text_lens, warm_audio_lens, warm_path
    measured = []
    for index, num_text in enumerate([67, 73, 82, 89, 97, 103, 113, 127]):
        num_audio = round(3.125 * num_text) + index
        batch_size = 1 + index % 4
        transposed = index % 2 == 1
        shape = (batch_size, num_text, num_audio) if transposed else (batch_size, num_audio, num_text)
        scores = torch.randn(shape, device=device)
        if transposed:
            scores = scores.transpose(1, 2)
        text_lens = torch.full((batch_size,), num_text, device=device, dtype=torch.long)
        audio_lens = torch.full((batch_size,), num_audio, device=device, dtype=torch.long)
        _synchronize(device)
        started = time.perf_counter()
        path = maximum_path_indices(scores, text_lens, audio_lens, backend='triton')
        _synchronize(device)
        first_shape_ms = 1000.0 * (time.perf_counter() - started)
        del path

        def run_mas():
            return maximum_path_indices(scores, text_lens, audio_lens, backend='triton')

        steady = _measure_call(run_mas, device, warmup=1, iterations=iterations)
        measured.append(
            {
                'shape': [batch_size, num_audio, num_text],
                'strides': list(scores.stride()),
                'first_call_wall_ms': first_shape_ms,
                'steady': steady,
            }
        )
        del scores, text_lens, audio_lens
    _cleanup(device)
    return {
        'supported': True,
        'text_width_bucket': 128,
        'first_bucket_shape': list(warm_shape),
        'first_bucket_call_wall_ms': first_bucket_ms,
        'fresh_disk_cache_guaranteed': False,
        'iterations_per_shape': iterations,
        'cases': measured,
        'fresh_shape_first_call': _timing_summary([case['first_call_wall_ms'] for case in measured]),
        'notes': [
            'Initial bucket call can include compilation or disk-cache loading.',
            'Exact lengths and contiguous/transposed strides vary within the warmed bucket.',
            'Supply a fresh TRITON_CACHE_DIR before starting Python to measure cold bucket compilation.',
            'No private Triton cache internals or hard latency thresholds are used.',
        ],
    }


def _parse_shapes(value, parser):
    shapes = []
    try:
        for index, raw in enumerate(value.split(';')):
            shape = tuple(int(part) for part in raw.split(','))
            if len(shape) != 3 or min(shape) < 1 or shape[1] < shape[2]:
                raise ValueError
            shapes.append((f'case_{index}', shape))
    except ValueError:
        parser.error('--shapes requires positive B,T,N triples with T >= N')
    return shapes


def _profile_case(name, shape, model, device, warmup, iterations):
    batch_size, audio_steps, text_steps = shape
    text_lens = torch.full((batch_size,), text_steps, device=device, dtype=torch.long)
    audio_lens = torch.full((batch_size,), audio_steps, device=device, dtype=torch.long)
    scores = torch.randn(shape, device=device, dtype=torch.float32)
    backends = ['cpu']
    if device.type == 'cuda' and TRITON_AVAILABLE and text_steps <= 4096:
        backends.append('triton')
    expected = maximum_path_indices(scores, text_lens, audio_lens, backend='cpu')
    parity = {}
    for backend in backends:
        actual = maximum_path_indices(scores, text_lens, audio_lens, backend=backend)
        parity[backend] = torch.equal(actual.cpu(), expected.cpu())
        if not parity[backend]:
            raise AssertionError(f'MAS correctness parity failed for {name}, {backend}')
    del expected, actual
    mas_measurements = {}
    for backend in backends:

        def run_mas():
            return maximum_path_indices(scores, text_lens, audio_lens, backend=backend)

        mas_measurements[backend] = _measure_call(run_mas, device, warmup, iterations)
    del scores
    _cleanup(device)
    audio = torch.randn(batch_size, audio_steps, model.audio_channels, device=device).detach()
    text = torch.randn(batch_size, text_steps, model.text_channels, device=device).detach()
    flow_measurements = _measure_flow(model, audio, text, audio_lens, text_lens, device, warmup, iterations)
    del audio, text, audio_lens, text_lens
    model.zero_grad(set_to_none=True)
    _cleanup(device)
    return {
        'name': name,
        'shape': list(shape),
        'audio_frames_per_text_token': audio_steps / text_steps,
        'total_audio_frames': batch_size * audio_steps,
        'total_text_tokens': batch_size * text_steps,
        'score_cells': batch_size * audio_steps * text_steps,
        'iterations': iterations,
        'mas_correctness_matches_cpu': parity,
        'mas': mas_measurements,
        'flow': flow_measurements,
    }


def _measure_call(call, device, warmup, iterations):
    for _ in range(warmup):
        result = call()
    del result
    _synchronize(device)
    gc.collect()
    memory_before = _reset_memory_peak(device)
    samples = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter()
        result = call()
        _synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - started))
        del result
    return {**_timing_summary(samples), **_memory_peak(device, memory_before)}


def _measure_flow(model, audio, text, audio_lens, text_lens, device, warmup, iterations):
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        output = model(audio, text, audio_lens, text_lens)
        output.loss.backward()
    del output
    model.zero_grad(set_to_none=True)
    _synchronize(device)
    gc.collect()
    memory_before = _reset_memory_peak(device)
    forward, backward, total = [], [], []
    loss = None
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        output = model(audio, text, audio_lens, text_lens)
        _synchronize(device)
        forward_finished = time.perf_counter()
        output.loss.backward()
        _synchronize(device)
        backward_finished = time.perf_counter()
        forward.append(1000.0 * (forward_finished - started))
        backward.append(1000.0 * (backward_finished - forward_finished))
        total.append(1000.0 * (backward_finished - started))
        loss = float(output.loss.detach().cpu())
        del output
    if loss is None or not math.isfinite(loss):
        raise RuntimeError('Aligner returned a nonfinite likelihood while profiling')
    return {
        'mas_backend': 'triton' if device.type == 'cuda' and TRITON_AVAILABLE and text.size(1) <= 4096 else 'cpu',
        'nll_forward': _timing_summary(forward),
        'nll_backward': _timing_summary(backward),
        'total': _timing_summary(total),
        'last_nll': loss,
        **_memory_peak(device, memory_before),
    }


def _timing_summary(samples):
    ordered = sorted(samples)
    return {
        'median_ms': statistics.median(samples),
        'p95_ms': ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        'min_ms': ordered[0],
        'max_ms': ordered[-1],
    }


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _cleanup(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _reset_memory_peak(device):
    if device.type != 'cuda':
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def _memory_peak(device, before):
    if device.type != 'cuda':
        return {'peak_allocated_bytes': None, 'peak_extra_bytes': None}
    peak = torch.cuda.max_memory_allocated(device)
    return {'peak_allocated_bytes': peak, 'peak_extra_bytes': peak - before}


def _turn_comparison(cases):
    separate = next(case for case in cases if case['name'] == 'separate_turns')
    whole = next(case for case in cases if case['name'] == 'whole_conversation')
    common_backends = separate['mas'].keys() & whole['mas'].keys()
    return {
        'turn_count': separate['shape'][0],
        'total_audio_frames_equal': separate['total_audio_frames'] == whole['total_audio_frames'],
        'total_text_tokens_equal': separate['total_text_tokens'] == whole['total_text_tokens'],
        'whole_over_separate_score_cells': whole['score_cells'] / separate['score_cells'],
        'whole_over_separate_mas_wall_time': {
            backend: whole['mas'][backend]['median_ms'] / separate['mas'][backend]['median_ms']
            for backend in sorted(common_backends)
        },
        'whole_over_separate_flow_total_wall_time': (
            whole['flow']['total']['median_ms'] / separate['flow']['total']['median_ms']
        ),
    }


def _environment(device):
    import numba

    environment = {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'device': str(device),
        'torch_cpu_threads': torch.get_num_threads(),
        'numba_cpu_threads': numba.get_num_threads(),
        'triton_available': TRITON_AVAILABLE,
    }
    if TRITON_AVAILABLE:
        import triton

        environment['triton_version'] = triton.__version__
    if device.type == 'cuda':
        environment['gpu_name'] = torch.cuda.get_device_name(device)
        environment['gpu_compute_capability'] = list(torch.cuda.get_device_capability(device))
        environment['allow_tf32_matmul'] = torch.backends.cuda.matmul.allow_tf32
    return environment


def _git_commit():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == '__main__':
    main()
