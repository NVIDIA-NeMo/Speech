# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Benchmark script comparing standard joint + loss vs fused Triton joint + loss.

Usage:
    python benchmark_rnnt_joint.py --joint standard --loss rnnt_triton --dtype float32
    python benchmark_rnnt_joint.py --joint triton --loss rnnt_triton --dtype float32
    python benchmark_rnnt_joint.py --joint triton_vocab --loss rnnt_triton --dtype bfloat16
    python benchmark_rnnt_joint.py --joint standard --loss warprnnt_numba --dtype bfloat16
    python benchmark_rnnt_joint.py --joint triton --dtype float32 -fo  # forward only
"""

import argparse
import sys
from dataclasses import dataclass

import torch

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.parts.rnnt_triton.rnnt_joint_triton import rnnt_joint_logprobs_triton
from nemo.collections.asr.parts.rnnt_triton.rnnt_joint_vocab_logprobs_triton import rnnt_joint_vocab_logprobs_triton
from nemo.collections.asr.parts.rnnt_triton.rnnt_loss_triton import rnnt_loss_from_logprobs_triton


@dataclass
class BenchmarkResults:
    joint: str
    loss: str
    dtype: str
    batch_size: int
    max_time: int
    max_target_plus_1: int
    hidden_dim: int
    vocab_size: int
    forward_memory_gb: float
    backward_peak_memory_gb: float
    max_peak_memory_gb: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if dtype_str.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str.lower()]


def benchmark_standard_joint(
    loss_name: str,
    dtype: torch.dtype,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    batch_size: int = 64,
    max_time: int = 150,
    hidden_dim: int = 640,
    num_classes: int = 1024,
    max_targets: int = 36,
    forward_only: bool = False,
) -> BenchmarkResults:
    """Benchmark standard joint (materialize logits) + loss."""
    device = torch.device('cuda')
    vocab_size = num_classes + 1
    blank_id = num_classes
    max_target_plus_1 = max_targets + 1

    enc_proj = torch.randn(batch_size, max_time, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    pred_proj = torch.randn(batch_size, max_target_plus_1, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    linear = torch.nn.Linear(hidden_dim, vocab_size, device=device, dtype=dtype)
    linear.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim, device=device, dtype=dtype))
    targets = torch.randint(0, blank_id, (batch_size, max_targets), device=device, dtype=torch.long)
    src_lengths = torch.full([batch_size], max_time, device=device, dtype=torch.long)
    tgt_lengths = torch.full([batch_size], max_targets, device=device, dtype=torch.long)

    loss_module = RNNTLoss(
        # RNNTLoss expects `num_classes` to be blank index (vocab_size - 1), not vocab size.
        num_classes=num_classes,
        loss_name=loss_name,
        loss_kwargs={'fastemit_lambda': 0.0},
        reduction='sum',
    )

    def run_fwd():
        hidden = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
        hidden.relu_()
        logits = linear(hidden)
        loss = loss_module(log_probs=logits, targets=targets, input_lengths=src_lengths, target_lengths=tgt_lengths)
        return loss

    def clear_grads():
        if enc_proj.grad is not None:
            enc_proj.grad = None
        if pred_proj.grad is not None:
            pred_proj.grad = None
        linear.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(warmup_iters):
        clear_grads()
        loss = run_fwd()
        if not forward_only:
            loss.backward()
        del loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Memory: forward only
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    clear_grads()
    loss = run_fwd()
    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated() - baseline_memory

    if forward_only:
        del loss
        backward_peak_memory = 0
        max_peak_memory = forward_memory
    else:
        # Memory: backward (after forward)
        torch.cuda.reset_peak_memory_stats()
        backward_baseline_memory = torch.cuda.memory_allocated()
        loss.backward()
        torch.cuda.synchronize()
        backward_peak_memory = torch.cuda.max_memory_allocated() - backward_baseline_memory
        del loss

        # Max peak memory (combined fwd+bwd)
        clear_grads()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        loss = run_fwd()
        loss.backward()
        torch.cuda.synchronize()
        max_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
        del loss

    # Timing: forward
    forward_times = []
    for _ in range(bench_iters):
        clear_grads()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = run_fwd()
        end.record()
        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))
        if not forward_only:
            loss.backward()
        del loss

    # Timing: backward
    backward_times = []
    if not forward_only:
        for _ in range(bench_iters):
            clear_grads()
            loss = run_fwd()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            backward_times.append(start.elapsed_time(end))
            del loss

    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times) if backward_times else 0.0
    dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'
    return BenchmarkResults(
        joint='standard',
        loss=loss_name,
        dtype=dtype_str,
        batch_size=batch_size,
        max_time=max_time,
        max_target_plus_1=max_target_plus_1,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        forward_memory_gb=forward_memory / (1024**3),
        backward_peak_memory_gb=backward_peak_memory / (1024**3),
        max_peak_memory_gb=max_peak_memory / (1024**3),
        forward_time_ms=avg_fwd,
        backward_time_ms=avg_bwd,
        total_time_ms=avg_fwd + avg_bwd,
    )


def benchmark_triton_joint(
    dtype: torch.dtype,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    batch_size: int = 64,
    max_time: int = 150,
    hidden_dim: int = 640,
    num_classes: int = 1024,
    max_targets: int = 36,
    forward_only: bool = False,
) -> BenchmarkResults:
    """Benchmark fused Triton joint + Triton loss."""
    device = torch.device('cuda')
    vocab_size = num_classes + 1
    blank_id = num_classes
    max_target_plus_1 = max_targets + 1

    enc_proj = torch.randn(batch_size, max_time, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    pred_proj = torch.randn(batch_size, max_target_plus_1, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    linear = torch.nn.Linear(hidden_dim, vocab_size, device=device, dtype=dtype)
    linear.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim, device=device, dtype=dtype))
    targets = torch.randint(0, blank_id, (batch_size, max_targets), device=device, dtype=torch.long)
    src_lengths = torch.full([batch_size], max_time, device=device, dtype=torch.long)
    tgt_lengths = torch.full([batch_size], max_targets, device=device, dtype=torch.long)

    def run_fwd():
        target_logprobs, blank_logprobs = rnnt_joint_logprobs_triton(
            encoder_output_projected=enc_proj,
            predictor_output_projected=pred_proj,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=linear.weight,
            bias=linear.bias,
            blank_id=blank_id,
        )
        loss_batch = rnnt_loss_from_logprobs_triton(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        return loss_batch.sum()

    def clear_grads():
        if enc_proj.grad is not None:
            enc_proj.grad = None
        if pred_proj.grad is not None:
            pred_proj.grad = None
        linear.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(warmup_iters):
        clear_grads()
        loss = run_fwd()
        if not forward_only:
            loss.backward()
        del loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Memory: forward only
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    clear_grads()
    loss = run_fwd()
    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated() - baseline_memory

    if forward_only:
        del loss
        backward_peak_memory = 0
        max_peak_memory = forward_memory
    else:
        # Memory: backward (after forward)
        torch.cuda.reset_peak_memory_stats()
        backward_baseline_memory = torch.cuda.memory_allocated()
        loss.backward()
        torch.cuda.synchronize()
        backward_peak_memory = torch.cuda.max_memory_allocated() - backward_baseline_memory
        del loss

        # Max peak memory (combined fwd+bwd)
        clear_grads()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        loss = run_fwd()
        loss.backward()
        torch.cuda.synchronize()
        max_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
        del loss

    # Timing: forward
    forward_times = []
    for _ in range(bench_iters):
        clear_grads()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = run_fwd()
        end.record()
        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))
        if not forward_only:
            loss.backward()
        del loss

    # Timing: backward
    backward_times = []
    if not forward_only:
        for _ in range(bench_iters):
            clear_grads()
            loss = run_fwd()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            backward_times.append(start.elapsed_time(end))
            del loss

    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times) if backward_times else 0.0
    dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'
    return BenchmarkResults(
        joint='triton',
        loss='rnnt_triton',
        dtype=dtype_str,
        batch_size=batch_size,
        max_time=max_time,
        max_target_plus_1=max_target_plus_1,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        forward_memory_gb=forward_memory / (1024**3),
        backward_peak_memory_gb=backward_peak_memory / (1024**3),
        max_peak_memory_gb=max_peak_memory / (1024**3),
        forward_time_ms=avg_fwd,
        backward_time_ms=avg_bwd,
        total_time_ms=avg_fwd + avg_bwd,
    )


def benchmark_triton_vocab_joint(
    loss_name: str,
    dtype: torch.dtype,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    batch_size: int = 64,
    max_time: int = 150,
    hidden_dim: int = 640,
    num_classes: int = 1024,
    max_targets: int = 36,
    forward_only: bool = False,
) -> BenchmarkResults:
    if loss_name != 'rnnt_triton':
        raise ValueError("Joint implementation `triton_vocab` supports only `rnnt_triton` loss.")

    device = torch.device('cuda')
    vocab_size = num_classes + 1
    blank_id = num_classes
    max_target_plus_1 = max_targets + 1

    enc_proj = torch.randn(batch_size, max_time, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    pred_proj = torch.randn(batch_size, max_target_plus_1, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    linear = torch.nn.Linear(hidden_dim, vocab_size, device=device, dtype=dtype)
    linear.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim, device=device, dtype=dtype))
    targets = torch.randint(0, blank_id, (batch_size, max_targets), device=device, dtype=torch.long)
    src_lengths = torch.full([batch_size], max_time, device=device, dtype=torch.long)
    tgt_lengths = torch.full([batch_size], max_targets, device=device, dtype=torch.long)

    def run_fwd():
        joint_hidden = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
        joint_hidden.relu_()
        target_logprobs, blank_logprobs = rnnt_joint_vocab_logprobs_triton(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=linear.weight,
            bias=linear.bias,
            blank_id=blank_id,
        )
        loss_batch = rnnt_loss_from_logprobs_triton(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        return loss_batch.sum()

    def clear_grads():
        if enc_proj.grad is not None:
            enc_proj.grad = None
        if pred_proj.grad is not None:
            pred_proj.grad = None
        linear.zero_grad(set_to_none=True)

    for _ in range(warmup_iters):
        clear_grads()
        loss = run_fwd()
        if not forward_only:
            loss.backward()
        del loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    clear_grads()
    loss = run_fwd()
    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated() - baseline_memory

    if forward_only:
        del loss
        backward_peak_memory = 0
        max_peak_memory = forward_memory
    else:
        torch.cuda.reset_peak_memory_stats()
        backward_baseline_memory = torch.cuda.memory_allocated()
        loss.backward()
        torch.cuda.synchronize()
        backward_peak_memory = torch.cuda.max_memory_allocated() - backward_baseline_memory
        del loss

        clear_grads()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        loss = run_fwd()
        loss.backward()
        torch.cuda.synchronize()
        max_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
        del loss

    forward_times = []
    for _ in range(bench_iters):
        clear_grads()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = run_fwd()
        end.record()
        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))
        if not forward_only:
            loss.backward()
        del loss

    backward_times = []
    if not forward_only:
        for _ in range(bench_iters):
            clear_grads()
            loss = run_fwd()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            backward_times.append(start.elapsed_time(end))
            del loss

    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times) if backward_times else 0.0
    dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'
    return BenchmarkResults(
        joint='triton_vocab',
        loss='rnnt_triton',
        dtype=dtype_str,
        batch_size=batch_size,
        max_time=max_time,
        max_target_plus_1=max_target_plus_1,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        forward_memory_gb=forward_memory / (1024**3),
        backward_peak_memory_gb=backward_peak_memory / (1024**3),
        max_peak_memory_gb=max_peak_memory / (1024**3),
        forward_time_ms=avg_fwd,
        backward_time_ms=avg_bwd,
        total_time_ms=avg_fwd + avg_bwd,
    )


def print_results(results: BenchmarkResults):
    print(
        f"\nGPU benchmark shape: enc=[{results.batch_size}, {results.max_time}, {results.hidden_dim}], "
        f"pred=[{results.batch_size}, {results.max_target_plus_1}, {results.hidden_dim}], "
        f"vocab_size={results.vocab_size}"
    )

    headers = [
        "Joint",
        "Loss",
        "Precision",
        "Fwd Memory",
        "Bwd Memory",
        "Max Memory",
        "Fwd Time",
        "Bwd Time",
        "Total Time",
    ]
    row = [
        results.joint,
        results.loss,
        results.dtype,
        f"{results.forward_memory_gb:.3f} GB",
        f"{results.backward_peak_memory_gb:.3f} GB",
        f"{results.max_peak_memory_gb:.3f} GB",
        f"{results.forward_time_ms:.3f} ms",
        f"{results.backward_time_ms:.3f} ms",
        f"{results.total_time_ms:.3f} ms",
    ]

    column_widths = [max(len(header), len(value)) for header, value in zip(headers, row)]
    separator = "-+-".join("-" * width for width in column_widths)
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, column_widths))
    row_line = " | ".join(value.ljust(width) for value, width in zip(row, column_widths))

    print(separator)
    print(header_line)
    print(separator)
    print(row_line)
    print(separator)


def main():
    parser = argparse.ArgumentParser(description='Benchmark RNN-T Joint implementations')
    parser.add_argument(
        '--joint',
        type=str,
        required=True,
        choices=['standard', 'triton', 'triton_vocab'],
        help='Joint implementation to benchmark',
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='rnnt_triton',
        choices=['warprnnt_numba', 'rnnt_triton'],
        help='Loss implementation',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'fp32', 'bfloat16', 'bf16'],
        help='Data type (default: float32)',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-time', type=int, default=150)
    parser.add_argument('--max-targets', type=int, default=36, help='Maximum target sequence length (default: 36)')
    parser.add_argument('--hidden-dim', type=int, default=640)
    parser.add_argument('--num-classes', type=int, default=1024)
    parser.add_argument('--warmup-iterations', type=int, default=10)
    parser.add_argument('--benchmark-iterations', type=int, default=100)
    parser.add_argument(
        '-fo', '--forward-only', action='store_true', help='Benchmark forward pass only (skip backward)'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    dtype = get_dtype(args.dtype)

    torch.manual_seed(42)

    if args.joint == 'standard':
        results = benchmark_standard_joint(
            loss_name=args.loss,
            dtype=dtype,
            warmup_iters=args.warmup_iterations,
            bench_iters=args.benchmark_iterations,
            batch_size=args.batch_size,
            max_time=args.max_time,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            max_targets=args.max_targets,
            forward_only=args.forward_only,
        )
    elif args.joint == 'triton':
        if args.loss != 'rnnt_triton':
            raise ValueError("Joint implementation `triton` supports only `rnnt_triton` loss.")
        results = benchmark_triton_joint(
            dtype=dtype,
            warmup_iters=args.warmup_iterations,
            bench_iters=args.benchmark_iterations,
            batch_size=args.batch_size,
            max_time=args.max_time,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            max_targets=args.max_targets,
            forward_only=args.forward_only,
        )
    else:
        results = benchmark_triton_vocab_joint(
            loss_name=args.loss,
            dtype=dtype,
            warmup_iters=args.warmup_iterations,
            bench_iters=args.benchmark_iterations,
            batch_size=args.batch_size,
            max_time=args.max_time,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            max_targets=args.max_targets,
            forward_only=args.forward_only,
        )

    print_results(results)


if __name__ == '__main__':
    main()
