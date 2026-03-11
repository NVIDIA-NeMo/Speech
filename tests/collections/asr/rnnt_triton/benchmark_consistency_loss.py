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
Benchmark script to compare Triton vs PyTorch implementations of KL divergence loss
for RNN-T consistency training.

Usage:
    python benchmark_consistency_loss.py --dtype float32 --symmetrical --use-triton
    python benchmark_consistency_loss.py --dtype bfloat16 --no-symmetrical --no-use-triton
"""

import argparse
import sys
from dataclasses import dataclass

import torch

from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency import ConsistencyFullRNNTLoss


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    dtype: str
    symmetrical: bool
    use_triton: bool
    input_memory_gb: float
    forward_memory_gb: float
    backward_peak_memory_gb: float
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


def benchmark_consistency_loss(
    dtype: torch.dtype,
    symmetrical: bool,
    use_triton: bool,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    batch_size: int = 16,
    max_time: int = 129,
    max_target_plus_1: int = 65,
    vocab_size: int = 2048,
) -> BenchmarkResults:
    """
    Benchmark consistency loss implementation.

    Args:
        dtype: Data type for tensors
        symmetrical: Whether to use symmetric KL divergence
        use_triton: Whether to use Triton implementation
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
        batch_size: Batch size
        max_time: Maximum time dimension (T)
        max_target_plus_1: Maximum target dimension + 1 (U+1)
        vocab_size: Vocabulary size (V)

    Returns:
        BenchmarkResults with timing and memory measurements
    """
    device = torch.device('cuda')

    # Create the loss module
    module = ConsistencyFullRNNTLoss(
        symmetrical=symmetrical,
        use_triton=use_triton,
        reduction='mean_volume',
    )

    # Create input tensors
    torch.manual_seed(42)
    teacher_logits = torch.randn(
        batch_size, max_time, max_target_plus_1, vocab_size,
        device=device, dtype=dtype
    )
    student_logits = torch.randn(
        batch_size, max_time, max_target_plus_1, vocab_size,
        device=device, dtype=dtype, requires_grad=True
    )

    # Measure input memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    input_memory = torch.cuda.memory_allocated()

    # Warmup iterations (use clone for warmup only)
    for _ in range(warmup_iters):
        student_logits_warmup = student_logits.detach().clone().requires_grad_(True)
        loss = module(
            teacher_logits=teacher_logits,
            student_logits=student_logits_warmup,
        )
        loss.backward()
        del student_logits_warmup, loss

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Reset memory stats after warmup - baseline is the input tensors
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    # Benchmark forward pass timing
    forward_times = []
    for _ in range(bench_iters):
        # Zero grad instead of clone to avoid memory inflation
        if student_logits.grad is not None:
            student_logits.grad = None

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss = module(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
        )
        end.record()

        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))

        # Clean up for next iteration
        loss.backward()
        del loss

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Measure memory for forward pass only (single iteration)
    torch.cuda.reset_peak_memory_stats()
    if student_logits.grad is not None:
        student_logits.grad = None
    loss = module(teacher_logits=teacher_logits, student_logits=student_logits)
    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated() - baseline_memory

    # Measure memory for backward pass
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    backward_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    del loss

    # Benchmark backward pass timing
    backward_times = []
    for _ in range(bench_iters):
        # Zero grad instead of clone
        if student_logits.grad is not None:
            student_logits.grad = None

        loss = module(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
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
    input_memory_gb = input_memory / (1024 ** 3)
    forward_memory_gb = forward_memory / (1024 ** 3)
    backward_peak_memory_gb = backward_peak_memory / (1024 ** 3)

    dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'

    return BenchmarkResults(
        dtype=dtype_str,
        symmetrical=symmetrical,
        use_triton=use_triton,
        input_memory_gb=input_memory_gb,
        forward_memory_gb=forward_memory_gb,
        backward_peak_memory_gb=backward_peak_memory_gb,
        forward_time_ms=avg_forward_time,
        backward_time_ms=avg_backward_time,
        total_time_ms=avg_forward_time + avg_backward_time,
    )


def print_results(results: BenchmarkResults):
    """Print benchmark results in a formatted way."""
    impl = "Triton" if results.use_triton else "PyTorch"
    sym = "Yes" if results.symmetrical else "No"

    print(f"\n{'=' * 60}")
    print(f"Benchmark Results: {impl} Implementation")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  - Implementation: {impl}")
    print(f"  - Dtype: {results.dtype}")
    print(f"  - Symmetrical: {sym}")
    print(f"\nMemory Usage (above input tensors):")
    print(f"  - Input Memory: {results.input_memory_gb:.3f} GB")
    print(f"  - Forward Peak: {results.forward_memory_gb:.3f} GB")
    print(f"  - Backward Peak: {results.backward_peak_memory_gb:.3f} GB")
    print(f"\nTiming (averaged):")
    print(f"  - Forward: {results.forward_time_ms:.3f} ms")
    print(f"  - Backward: {results.backward_time_ms:.3f} ms")
    print(f"  - Total: {results.total_time_ms:.3f} ms")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Triton vs PyTorch consistency loss implementations'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'fp32', 'bfloat16', 'bf16'],
        help='Data type for tensors (default: float32)'
    )
    parser.add_argument(
        '--symmetrical',
        action='store_true',
        dest='symmetrical',
        help='Use symmetric KL divergence'
    )
    parser.add_argument(
        '--no-symmetrical',
        action='store_false',
        dest='symmetrical',
        help='Use asymmetric KL divergence'
    )
    parser.add_argument(
        '--use-triton',
        action='store_true',
        dest='use_triton',
        help='Use Triton implementation'
    )
    parser.add_argument(
        '--no-use-triton',
        action='store_false',
        dest='use_triton',
        help='Use PyTorch implementation'
    )
    parser.add_argument(
        '--warmup-iterations',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    parser.add_argument(
        '--benchmark-iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--max-time',
        type=int,
        default=129,
        help='Maximum time dimension T (default: 129)'
    )
    parser.add_argument(
        '--max-target-plus-1',
        type=int,
        default=65,
        help='Maximum target dimension + 1 (U+1) (default: 65)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=2048,
        help='Vocabulary size V (default: 2048)'
    )

    parser.set_defaults(symmetrical=False, use_triton=False)
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Logits shape: [{args.batch_size}, {args.max_time}, {args.max_target_plus_1}, {args.vocab_size}]")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Benchmark iterations: {args.benchmark_iterations}")

    dtype = get_dtype(args.dtype)

    results = benchmark_consistency_loss(
        dtype=dtype,
        symmetrical=args.symmetrical,
        use_triton=args.use_triton,
        warmup_iters=args.warmup_iterations,
        bench_iters=args.benchmark_iterations,
        batch_size=args.batch_size,
        max_time=args.max_time,
        max_target_plus_1=args.max_target_plus_1,
        vocab_size=args.vocab_size,
    )

    print_results(results)


if __name__ == '__main__':
    main()
