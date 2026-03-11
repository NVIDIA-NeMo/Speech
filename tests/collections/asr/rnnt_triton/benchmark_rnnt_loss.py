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
Benchmark script to compare Triton vs Numba implementations of RNN-T loss.

Usage:
    python benchmark_rnnt_loss.py --loss-name rnnt_triton --dtype float32 --vocab-size 1024
    python benchmark_rnnt_loss.py --loss-name warprnnt_numba --dtype bfloat16 --vocab-size 1025
"""

import argparse
import sys
from dataclasses import dataclass

import torch

from nemo.collections.asr.losses.rnnt import RNNTLoss


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    loss_name: str
    dtype: str
    vocab_size: int
    batch_size: int
    max_time: int
    max_target_plus_1: int
    input_memory_gb: float
    forward_memory_gb: float
    backward_peak_memory_gb: float
    max_peak_memory_gb: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if dtype_str.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str.lower()]


def benchmark_rnnt_loss(
    loss_name: str,
    dtype: torch.dtype,
    vocab_size: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    batch_size: int = 64,
    max_time: int = 129,
    max_target_plus_1: int = 65,
) -> BenchmarkResults:
    """
    Benchmark RNN-T loss implementation.

    Args:
        loss_name: Loss implementation name ('warprnnt_numba' or 'rnnt_triton')
        dtype: Data type for tensors
        vocab_size: Vocabulary size (V)
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
        batch_size: Batch size
        max_time: Maximum time dimension (T)
        max_target_plus_1: Maximum target dimension + 1 (U+1)

    Returns:
        BenchmarkResults with timing and memory measurements
    """
    device = torch.device('cuda')
    blank_id = vocab_size - 1

    # Create the loss module via NeMo entry point
    loss_module = RNNTLoss(
        # RNNTLoss expects `num_classes` to be blank index (vocab_size - 1), not vocab size.
        num_classes=blank_id,
        loss_name=loss_name,
        loss_kwargs={'fastemit_lambda': 0.0},
        reduction='sum',
    )

    # Create input tensors
    torch.manual_seed(42)
    U = max_target_plus_1 - 1
    logits = torch.randn(
        batch_size,
        max_time,
        max_target_plus_1,
        vocab_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    targets = torch.randint(0, blank_id, (batch_size, U), device=device, dtype=torch.long)
    src_lengths = torch.full([batch_size], max_time, device=device, dtype=torch.long)
    tgt_lengths = torch.full([batch_size], U, device=device, dtype=torch.long)

    # Measure input memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    input_memory = torch.cuda.memory_allocated()

    # Warmup iterations
    for _ in range(warmup_iters):
        logits_warmup = logits.detach().clone().requires_grad_(True)
        loss = loss_module(
            log_probs=logits_warmup,
            targets=targets,
            input_lengths=src_lengths,
            target_lengths=tgt_lengths,
        )
        loss.backward()
        del logits_warmup, loss

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    # Benchmark forward pass timing
    forward_times = []
    for _ in range(bench_iters):
        if logits.grad is not None:
            logits.grad = None

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss = loss_module(
            log_probs=logits,
            targets=targets,
            input_lengths=src_lengths,
            target_lengths=tgt_lengths,
        )
        end.record()

        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))

        loss.backward()
        del loss

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Measure memory for forward pass only (single iteration)
    torch.cuda.reset_peak_memory_stats()
    if logits.grad is not None:
        logits.grad = None
    loss = loss_module(
        log_probs=logits,
        targets=targets,
        input_lengths=src_lengths,
        target_lengths=tgt_lengths,
    )
    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated() - baseline_memory

    # Measure memory for backward pass
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    backward_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    del loss

    # Measure max peak memory across combined forward + backward (no reset in between)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if logits.grad is not None:
        logits.grad = None
    loss = loss_module(
        log_probs=logits,
        targets=targets,
        input_lengths=src_lengths,
        target_lengths=tgt_lengths,
    )
    loss.backward()
    torch.cuda.synchronize()
    max_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    del loss

    # Benchmark backward pass timing
    backward_times = []
    for _ in range(bench_iters):
        if logits.grad is not None:
            logits.grad = None

        loss = loss_module(
            log_probs=logits,
            targets=targets,
            input_lengths=src_lengths,
            target_lengths=tgt_lengths,
        )

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        backward_times.append(start.elapsed_time(end))
        del loss

    # Calculate statistics
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    # Memory calculations
    input_memory_gb = input_memory / (1024**3)
    forward_memory_gb = forward_memory / (1024**3)
    backward_peak_memory_gb = backward_peak_memory / (1024**3)
    max_peak_memory_gb = max_peak_memory / (1024**3)

    dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'

    return BenchmarkResults(
        loss_name=loss_name,
        dtype=dtype_str,
        vocab_size=vocab_size,
        batch_size=batch_size,
        max_time=max_time,
        max_target_plus_1=max_target_plus_1,
        input_memory_gb=input_memory_gb,
        forward_memory_gb=forward_memory_gb,
        backward_peak_memory_gb=backward_peak_memory_gb,
        max_peak_memory_gb=max_peak_memory_gb,
        forward_time_ms=avg_forward_time,
        backward_time_ms=avg_backward_time,
        total_time_ms=avg_forward_time + avg_backward_time,
    )


def print_results(results: BenchmarkResults):
    """Print benchmark results in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark Results: {results.loss_name}")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  - Loss: {results.loss_name}")
    print(f"  - Dtype: {results.dtype}")
    print(f"  - Vocab size: {results.vocab_size}")
    print(
        f"  - Logits shape: [{results.batch_size}, {results.max_time}, {results.max_target_plus_1}, {results.vocab_size}]"
    )
    print(f"\nMemory Usage (above input tensors):")
    print(f"  - Input Memory: {results.input_memory_gb:.3f} GB")
    print(f"  - Forward Peak: {results.forward_memory_gb:.3f} GB")
    print(f"  - Backward Peak: {results.backward_peak_memory_gb:.3f} GB")
    print(f"  - Max Peak (fwd+bwd): {results.max_peak_memory_gb:.3f} GB")
    print(f"\nTiming (averaged):")
    print(f"  - Forward: {results.forward_time_ms:.3f} ms")
    print(f"  - Backward: {results.backward_time_ms:.3f} ms")
    print(f"  - Total: {results.total_time_ms:.3f} ms")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark RNN-T loss implementations (Triton vs Numba)')
    parser.add_argument(
        '--loss-name',
        type=str,
        required=True,
        choices=['warprnnt_numba', 'rnnt_triton'],
        help='Loss implementation to benchmark',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'fp32', 'bfloat16', 'bf16'],
        help='Data type for tensors (default: float32)',
    )
    parser.add_argument('--vocab-size', type=int, default=1024, help='Vocabulary size V (default: 1024)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--max-time', type=int, default=129, help='Maximum time dimension T (default: 129)')
    parser.add_argument(
        '--max-target-plus-1', type=int, default=65, help='Maximum target dimension + 1 (U+1) (default: 65)'
    )
    parser.add_argument('--warmup-iterations', type=int, default=10, help='Number of warmup iterations (default: 10)')
    parser.add_argument(
        '--benchmark-iterations', type=int, default=100, help='Number of benchmark iterations (default: 100)'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Loss: {args.loss_name}")
    print(f"Logits shape: [{args.batch_size}, {args.max_time}, {args.max_target_plus_1}, {args.vocab_size}]")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")

    dtype = get_dtype(args.dtype)

    results = benchmark_rnnt_loss(
        loss_name=args.loss_name,
        dtype=dtype,
        vocab_size=args.vocab_size,
        warmup_iters=args.warmup_iterations,
        bench_iters=args.benchmark_iterations,
        batch_size=args.batch_size,
        max_time=args.max_time,
        max_target_plus_1=args.max_target_plus_1,
    )

    print_results(results)


if __name__ == '__main__':
    main()
