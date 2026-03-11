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
Memory profiling script to investigate memory usage in Triton consistency loss.

This script profiles memory usage at each stage:
1. After allocating inputs (teacher, student, mask)
2. After forward pass
3. After backward pass

This helps understand:
- Why symmetric and non-symmetric show identical memory usage
- What contributes to the ~3 GB additional memory in benchmarks
"""

import torch

from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import FusedKLDivTriton


def profile_memory(symmetric: bool, dtype: torch.dtype = torch.float32):
    """
    Profile memory usage for KL divergence loss.

    Args:
        symmetric: Whether to use symmetric KL divergence
        dtype: Data type for tensors
    """
    # Clear GPU state
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    B, T, U, V = 16, 129, 65, 2048
    dtype_str = "float32" if dtype == torch.float32 else "bfloat16"
    bytes_per_element = 4 if dtype == torch.float32 else 2
    tensor_size_gb = (B * T * U * V * bytes_per_element) / (1024 ** 3)

    print(f"\n{'='*70}")
    print(f"Memory Profile: symmetric={symmetric}, dtype={dtype_str}")
    print(f"{'='*70}")
    print(f"Tensor shape: [{B}, {T}, {U}, {V}]")
    print(f"Single tensor size: {tensor_size_gb:.3f} GB")
    print(f"{'='*70}")

    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial memory: {initial_memory / 1e9:.3f} GB")

    # Allocate inputs
    teacher = torch.randn(B, T, U, V, device='cuda', dtype=dtype)
    student = torch.randn(B, T, U, V, device='cuda', dtype=dtype, requires_grad=True)
    mask = torch.ones(B, T, U, device='cuda', dtype=torch.bool)

    torch.cuda.synchronize()
    after_inputs = torch.cuda.memory_allocated()
    print(f"\nAfter inputs:")
    print(f"  - Current: {after_inputs / 1e9:.3f} GB")
    print(f"  - Delta: +{(after_inputs - initial_memory) / 1e9:.3f} GB")
    print(f"  - Expected: ~{2 * tensor_size_gb:.3f} GB (teacher + student)")

    # Set baseline before forward
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    loss = FusedKLDivTriton.apply(teacher, student, mask, symmetric)

    torch.cuda.synchronize()
    fwd_current = torch.cuda.memory_allocated()
    fwd_peak = torch.cuda.max_memory_allocated()

    print(f"\nAfter forward:")
    print(f"  - Current: {fwd_current / 1e9:.3f} GB (+{(fwd_current - baseline) / 1e9:.3f} GB from baseline)")
    print(f"  - Peak:    {fwd_peak / 1e9:.3f} GB (+{(fwd_peak - baseline) / 1e9:.3f} GB from baseline)")
    print(f"  - Loss shape: {loss.shape}")

    # Reset for backward measurement
    torch.cuda.reset_peak_memory_stats()
    pre_backward = torch.cuda.memory_allocated()

    # Backward pass
    loss.sum().backward()

    torch.cuda.synchronize()
    bwd_current = torch.cuda.memory_allocated()
    bwd_peak = torch.cuda.max_memory_allocated()

    print(f"\nAfter backward:")
    print(f"  - Current: {bwd_current / 1e9:.3f} GB (+{(bwd_current - baseline) / 1e9:.3f} GB from baseline)")
    print(f"  - Peak:    {bwd_peak / 1e9:.3f} GB (+{(bwd_peak - baseline) / 1e9:.3f} GB from baseline)")
    print(f"  - student.grad shape: {student.grad.shape if student.grad is not None else None}")

    # Analyze components
    print(f"\nMemory breakdown analysis:")
    print(f"  - Expected student_grad_logits: ~{tensor_size_gb:.3f} GB")
    print(f"  - Expected student.grad: ~{tensor_size_gb:.3f} GB")
    if symmetric:
        print(f"  - Expected teacher_grad_logits: ~{tensor_size_gb:.3f} GB")
        print(f"  - Note: teacher_grad is returned but teacher has no requires_grad,")
        print(f"    so PyTorch should immediately discard it")

    # Check if teacher has gradient
    print(f"\nGradient status:")
    print(f"  - teacher.requires_grad: {teacher.requires_grad}")
    print(f"  - teacher.grad: {teacher.grad}")
    print(f"  - student.requires_grad: {student.requires_grad}")
    print(f"  - student.grad is not None: {student.grad is not None}")

    # Clean up for next run
    del teacher, student, mask, loss
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        'symmetric': symmetric,
        'dtype': dtype_str,
        'baseline_gb': baseline / 1e9,
        'fwd_current_delta_gb': (fwd_current - baseline) / 1e9,
        'fwd_peak_delta_gb': (fwd_peak - baseline) / 1e9,
        'bwd_current_delta_gb': (bwd_current - baseline) / 1e9,
        'bwd_peak_delta_gb': (bwd_peak - baseline) / 1e9,
    }


def profile_with_teacher_grad(symmetric: bool, dtype: torch.dtype = torch.float32):
    """
    Profile memory when teacher also requires gradient.

    This tests whether symmetric mode shows higher memory when teacher
    actually needs gradients.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    B, T, U, V = 16, 129, 65, 2048
    dtype_str = "float32" if dtype == torch.float32 else "bfloat16"
    bytes_per_element = 4 if dtype == torch.float32 else 2
    tensor_size_gb = (B * T * U * V * bytes_per_element) / (1024 ** 3)

    print(f"\n{'='*70}")
    print(f"Memory Profile (TEACHER REQUIRES GRAD): symmetric={symmetric}, dtype={dtype_str}")
    print(f"{'='*70}")

    # Allocate inputs - BOTH require grad
    teacher = torch.randn(B, T, U, V, device='cuda', dtype=dtype, requires_grad=True)
    student = torch.randn(B, T, U, V, device='cuda', dtype=dtype, requires_grad=True)
    mask = torch.ones(B, T, U, device='cuda', dtype=torch.bool)

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # Forward
    loss = FusedKLDivTriton.apply(teacher, student, mask, symmetric)
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated()

    # Backward
    torch.cuda.reset_peak_memory_stats()
    loss.sum().backward()
    torch.cuda.synchronize()
    bwd_current = torch.cuda.memory_allocated()
    bwd_peak = torch.cuda.max_memory_allocated()

    print(f"Baseline: {baseline / 1e9:.3f} GB")
    print(f"After backward:")
    print(f"  - Current: {bwd_current / 1e9:.3f} GB (+{(bwd_current - baseline) / 1e9:.3f} GB)")
    print(f"  - Peak:    {bwd_peak / 1e9:.3f} GB (+{(bwd_peak - baseline) / 1e9:.3f} GB)")
    print(f"  - teacher.grad is not None: {teacher.grad is not None}")
    print(f"  - student.grad is not None: {student.grad is not None}")

    if symmetric:
        print(f"\nExpected peak for symmetric with both grads:")
        print(f"  - teacher_grad_logits: ~{tensor_size_gb:.3f} GB")
        print(f"  - student_grad_logits: ~{tensor_size_gb:.3f} GB")
        print(f"  - teacher.grad: ~{tensor_size_gb:.3f} GB")
        print(f"  - student.grad: ~{tensor_size_gb:.3f} GB")
        print(f"  - Total: ~{4 * tensor_size_gb:.3f} GB")

    del teacher, student, mask, loss
    torch.cuda.empty_cache()


def profile_benchmark_clone_impact():
    """
    Profile the impact of clone operation used in benchmark loop.

    The benchmark does: student_logits_bench = student_logits.detach().clone().requires_grad_(True)
    This creates an extra copy that inflates memory measurements.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    B, T, U, V = 16, 129, 65, 2048
    dtype = torch.float32
    bytes_per_element = 4
    tensor_size_gb = (B * T * U * V * bytes_per_element) / (1024 ** 3)

    print(f"\n{'='*70}")
    print(f"Clone Impact Analysis")
    print(f"{'='*70}")
    print(f"Single tensor size: {tensor_size_gb:.3f} GB")

    # Create inputs
    teacher = torch.randn(B, T, U, V, device='cuda', dtype=dtype)
    student = torch.randn(B, T, U, V, device='cuda', dtype=dtype, requires_grad=True)
    mask = torch.ones(B, T, U, device='cuda', dtype=torch.bool)

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    print(f"\nBaseline (after inputs): {baseline / 1e9:.3f} GB")

    # Simulate benchmark clone
    torch.cuda.reset_peak_memory_stats()
    student_clone = student.detach().clone().requires_grad_(True)

    torch.cuda.synchronize()
    after_clone = torch.cuda.memory_allocated()
    print(f"After clone: {after_clone / 1e9:.3f} GB (+{(after_clone - baseline) / 1e9:.3f} GB)")
    print(f"  - Clone adds: ~{tensor_size_gb:.3f} GB (one full tensor)")

    # Forward with clone
    torch.cuda.reset_peak_memory_stats()
    loss = FusedKLDivTriton.apply(teacher, student_clone, mask, False)
    loss.sum().backward()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    current = torch.cuda.memory_allocated()

    print(f"\nAfter forward+backward (with clone in scope):")
    print(f"  - Current: {current / 1e9:.3f} GB (+{(current - baseline) / 1e9:.3f} GB from baseline)")
    print(f"  - Peak:    {peak / 1e9:.3f} GB (+{(peak - baseline) / 1e9:.3f} GB from baseline)")

    print(f"\nExpected memory components:")
    print(f"  - Clone tensor: ~{tensor_size_gb:.3f} GB")
    print(f"  - student_grad_logits: ~{tensor_size_gb:.3f} GB")
    print(f"  - student_clone.grad: ~{tensor_size_gb:.3f} GB")
    print(f"  - Total: ~{3 * tensor_size_gb:.3f} GB")

    del teacher, student, student_clone, mask, loss
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("="*70)
    print("MEMORY PROFILING: Triton Consistency Loss")
    print("="*70)

    # Profile both modes without clone (direct usage)
    results_nonsym = profile_memory(symmetric=False)
    results_sym = profile_memory(symmetric=True)

    # Profile with teacher also requiring grad
    profile_with_teacher_grad(symmetric=False)
    profile_with_teacher_grad(symmetric=True)

    # Profile clone impact (simulating benchmark)
    profile_benchmark_clone_impact()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nDirect usage (no clone):")
    print(f"  Non-symmetric backward peak delta: {results_nonsym['bwd_peak_delta_gb']:.3f} GB")
    print(f"  Symmetric backward peak delta:     {results_sym['bwd_peak_delta_gb']:.3f} GB")
    print(f"  Difference: {results_sym['bwd_peak_delta_gb'] - results_nonsym['bwd_peak_delta_gb']:.3f} GB")

    print(f"\nKey insight: If symmetric shows similar memory to non-symmetric,")
    print(f"it's because teacher_grad_logits is allocated but immediately freed")
    print(f"since teacher.requires_grad=False.")
