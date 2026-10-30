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
Benchmark script for the RNN-T Joint + loss + consistency loss.

Compares the legacy path (full logits consumed independently by the RNN-T loss and by the consistency
loss, one full-size gradient tensor per loss) with the fused path (`Joint.forward_for_loss`, a single
gradient buffer for the Joint output).

NB: `batch_size` here is the *double* batch: the first half is the teacher pass, the second half is the
student pass of the same utterances.

Usage:
    python benchmark_rnnt_joint_consistency.py --loss turbo_rnnt --consistency fused
    python benchmark_rnnt_joint_consistency.py --loss turbo_rnnt --consistency separate
    python benchmark_rnnt_joint_consistency.py --loss warprnnt_numba --consistency separate
    python benchmark_rnnt_joint_consistency.py --sweep                     # full matrix, several vocab sizes
"""

import argparse
import sys
from dataclasses import dataclass

import torch

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import ConsistencyRNNTLoss

VALID_LOSSES = ("warprnnt_numba", "turbo_rnnt")
VALID_CONSISTENCY_MODES = ("none", "separate", "fused")
SWEEP_VOCAB_SIZES = (1024, 8192)
SWEEP_CONFIGURATIONS = (
    ("warprnnt_numba", "none"),
    ("turbo_rnnt", "none"),
    ("warprnnt_numba", "separate"),
    ("turbo_rnnt", "separate"),
    ("turbo_rnnt", "fused"),
)


@dataclass
class BenchmarkResults:
    loss: str
    consistency: str
    dtype: str
    batch_size: int
    max_time: int
    max_target_plus_1: int
    hidden_dim: int
    vocab_size: int
    input_memory_gb: float
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


def _create_joint(
    hidden_dim: int,
    vocab_size: int,
    dropout_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> RNNTJoint:
    """Create NeMo's standard Joint over already-projected benchmark inputs."""
    joint_module = RNNTJoint(
        jointnet={
            'encoder_hidden': hidden_dim,
            'pred_hidden': hidden_dim,
            'joint_hidden': hidden_dim,
            'activation': 'relu',
            'dropout': dropout_p,
        },
        num_classes=vocab_size - 1,
        log_softmax=False,
    )
    joint_module.enc = torch.nn.Identity()
    joint_module.pred = torch.nn.Identity()
    joint_module = joint_module.to(device=device, dtype=dtype)
    vocab_projection = joint_module.joint_net[-1]
    vocab_projection.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim, device=device, dtype=dtype))
    return joint_module


def run_benchmark(
    loss: str,
    consistency: str,
    dtype: torch.dtype,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    batch_size: int = 64,
    max_time: int = 150,
    hidden_dim: int = 640,
    vocab_size: int = 1024,
    max_targets: int = 72,
    dropout_p: float = 0.2,
    consistency_weight: float = 0.2,
    symmetrical: bool = True,
) -> BenchmarkResults:
    """Run one (loss, consistency mode, vocab) benchmark and return results."""
    if loss not in VALID_LOSSES:
        raise ValueError(f"Invalid loss={loss!r}. Expected one of {VALID_LOSSES}.")
    if consistency not in VALID_CONSISTENCY_MODES:
        raise ValueError(f"Invalid consistency={consistency!r}. Expected one of {VALID_CONSISTENCY_MODES}.")
    if consistency == "fused" and loss != "turbo_rnnt":
        raise ValueError("The fused consistency loss requires loss='turbo_rnnt'.")
    if batch_size % 2 != 0:
        raise ValueError(f"The double batch size must be even, got {batch_size}.")

    device = torch.device('cuda')
    blank_id = vocab_size - 1
    max_target_plus_1 = max_targets + 1
    half_batch_size = batch_size // 2

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_before_inputs = torch.cuda.memory_allocated()

    encoder_output_projected = torch.randn(
        batch_size, max_time, hidden_dim, device=device, dtype=dtype, requires_grad=True
    )
    predictor_output_projected = torch.randn(
        batch_size, max_target_plus_1, hidden_dim, device=device, dtype=dtype, requires_grad=True
    )
    joint_module = _create_joint(
        hidden_dim=hidden_dim, vocab_size=vocab_size, dropout_p=dropout_p, device=device, dtype=dtype
    )
    targets = torch.randint(0, blank_id, (half_batch_size, max_targets), device=device, dtype=torch.long).repeat(2, 1)
    src_lengths = torch.full([batch_size], max_time, device=device, dtype=torch.long)
    tgt_lengths = torch.full([batch_size], max_targets, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    input_memory = torch.cuda.memory_allocated() - baseline_before_inputs

    loss_module = RNNTLoss(
        # RNNTLoss expects `num_classes` to be the blank index (vocab_size - 1), not the vocab size
        num_classes=vocab_size - 1,
        loss_name=loss,
        loss_kwargs={'fastemit_lambda': 0.0},
        reduction='mean_volume',
    )
    consistency_loss_module = ConsistencyRNNTLoss(symmetrical=symmetrical, reduction='mean_volume', backend='triton')

    def run_fwd_separate() -> torch.Tensor:
        logits = joint_module(
            encoder_outputs=encoder_output_projected.transpose(1, 2),
            decoder_outputs=predictor_output_projected.transpose(1, 2),
        )
        loss_value = loss_module(
            log_probs=logits, targets=targets, input_lengths=src_lengths, target_lengths=tgt_lengths
        )
        if consistency == "none":
            return loss_value
        consistency_loss_value = consistency_loss_module(
            teacher_logits=logits[:half_batch_size],
            student_logits=logits[half_batch_size:],
            src_lengths=src_lengths[:half_batch_size],
            tgt_lengths=tgt_lengths[:half_batch_size],
        )
        return loss_value + consistency_weight * consistency_loss_value

    def run_fwd_fused() -> torch.Tensor:
        loss_inputs = joint_module.forward_for_loss(
            encoder_outputs=encoder_output_projected.transpose(1, 2),
            decoder_outputs=predictor_output_projected.transpose(1, 2),
            encoder_lengths=src_lengths,
            transcripts=targets,
            transcript_lengths=tgt_lengths,
            compute_consistency=consistency != "none",
            consistency_symmetrical=symmetrical,
        )
        loss_value = loss_module.forward_from_logprobs(
            target_logprobs=loss_inputs.target_logprobs,
            blank_logprobs=loss_inputs.blank_logprobs,
            input_lengths=src_lengths,
            target_lengths=tgt_lengths,
        )
        if consistency == "none":
            return loss_value
        consistency_loss_value = consistency_loss_module.forward_from_kl(
            consistency_kl=loss_inputs.consistency_kl,
            src_lengths=src_lengths[:half_batch_size],
            tgt_lengths=tgt_lengths[:half_batch_size],
        )
        return loss_value + consistency_weight * consistency_loss_value

    run_fwd = run_fwd_fused if consistency == "fused" else run_fwd_separate

    def clear_grads():
        encoder_output_projected.grad = None
        predictor_output_projected.grad = None
        joint_module.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(warmup_iters):
        clear_grads()
        loss_val = run_fwd()
        loss_val.backward()
        del loss_val
    clear_grads()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Memory: forward only
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    loss_val = run_fwd()
    torch.cuda.synchronize()
    forward_memory = torch.cuda.max_memory_allocated() - baseline_memory

    # Memory: backward (after forward)
    torch.cuda.reset_peak_memory_stats()
    backward_baseline_memory = torch.cuda.memory_allocated()
    loss_val.backward()
    torch.cuda.synchronize()
    backward_peak_memory = torch.cuda.max_memory_allocated() - backward_baseline_memory
    del loss_val

    # Max peak memory (combined fwd+bwd)
    clear_grads()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    loss_val = run_fwd()
    loss_val.backward()
    torch.cuda.synchronize()
    max_peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    del loss_val

    # Timing: forward
    forward_times = []
    for _ in range(bench_iters):
        clear_grads()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss_val = run_fwd()
        end.record()
        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))
        loss_val.backward()
        del loss_val

    # Timing: backward
    backward_times = []
    for _ in range(bench_iters):
        clear_grads()
        loss_val = run_fwd()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss_val.backward()
        end.record()
        torch.cuda.synchronize()
        backward_times.append(start.elapsed_time(end))
        del loss_val

    average_forward_time = sum(forward_times) / len(forward_times)
    average_backward_time = sum(backward_times) / len(backward_times)
    return BenchmarkResults(
        loss=loss,
        consistency=consistency,
        dtype='float32' if dtype == torch.float32 else 'bfloat16',
        batch_size=batch_size,
        max_time=max_time,
        max_target_plus_1=max_target_plus_1,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        input_memory_gb=input_memory / (1024**3),
        forward_memory_gb=forward_memory / (1024**3),
        backward_peak_memory_gb=backward_peak_memory / (1024**3),
        max_peak_memory_gb=max_peak_memory / (1024**3),
        forward_time_ms=average_forward_time,
        backward_time_ms=average_backward_time,
        total_time_ms=average_forward_time + average_backward_time,
    )


def _fmt_mem(value: float) -> str:
    return f"{value:.3f} GB"


def _fmt_time(value: float) -> str:
    return f"{value:.3f} ms"


TABLE_COLUMNS: tuple[tuple[str, str, callable], ...] = (
    ("Vocab", "vocab_size", str),
    ("Loss", "loss", str),
    ("Consistency", "consistency", str),
    ("Input Memory", "input_memory_gb", _fmt_mem),
    ("Fwd Memory", "forward_memory_gb", _fmt_mem),
    ("Bwd Memory", "backward_peak_memory_gb", _fmt_mem),
    ("Fwd+Bwd Memory", "max_peak_memory_gb", _fmt_mem),
    ("Fwd Time", "forward_time_ms", _fmt_time),
    ("Bwd Time", "backward_time_ms", _fmt_time),
    ("Fwd+Bwd Time", "total_time_ms", _fmt_time),
)


def print_results_table(results: list[BenchmarkResults]) -> None:
    """Pretty-print a table of benchmark results."""
    if not results:
        print("(no benchmark results)")
        return

    headers = [column[0] for column in TABLE_COLUMNS]
    rows = [[formatter(getattr(result, field)) for _, field, formatter in TABLE_COLUMNS] for result in results]
    column_widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    separator = "+-" + "-+-".join("-" * width for width in column_widths) + "-+"
    separator_mid = "| " + " | ".join("-" * width for width in column_widths) + " |"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, column_widths)) + " |"

    print(separator)
    print(header_line)
    print(separator_mid)
    for row in rows:
        print("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, column_widths)) + " |")
    print(separator)


def main():
    parser = argparse.ArgumentParser(description='Benchmark RNN-T Joint + loss + consistency loss')
    parser.add_argument('--loss', type=str, default='turbo_rnnt', choices=list(VALID_LOSSES))
    parser.add_argument('--consistency', type=str, default='fused', choices=list(VALID_CONSISTENCY_MODES))
    parser.add_argument('--sweep', action='store_true', help='Run the full matrix over several vocabulary sizes')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'fp32', 'bfloat16', 'bf16'])
    parser.add_argument('--batch-size', type=int, default=64, help='Double batch size (teacher + student halves)')
    parser.add_argument('--max-time', type=int, default=150)
    parser.add_argument('--max-targets', type=int, default=72)
    parser.add_argument('--hidden-dim', type=int, default=640)
    parser.add_argument('--vocab-size', type=int, default=1024)
    parser.add_argument('--no-symmetrical', action='store_true', help='Use the non-symmetric consistency loss')
    parser.add_argument('-wi', '--warmup-iterations', type=int, default=5)
    parser.add_argument('-bi', '--benchmark-iterations', type=int, default=20)
    parser.add_argument('--no-dropout', action='store_true', help='Disable dropout (enabled by default, p=0.2)')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    dtype = get_dtype(args.dtype)
    dropout_p = 0.0 if args.no_dropout else 0.2
    torch.manual_seed(42)

    if args.sweep:
        configurations = [
            (vocab_size, loss, consistency)
            for vocab_size in SWEEP_VOCAB_SIZES
            for loss, consistency in SWEEP_CONFIGURATIONS
        ]
    else:
        configurations = [(args.vocab_size, args.loss, args.consistency)]

    results = []
    for vocab_size, loss, consistency in configurations:
        print(f"Running: vocab_size={vocab_size}, loss={loss}, consistency={consistency}")
        results.append(
            run_benchmark(
                loss=loss,
                consistency=consistency,
                dtype=dtype,
                warmup_iters=args.warmup_iterations,
                bench_iters=args.benchmark_iterations,
                batch_size=args.batch_size,
                max_time=args.max_time,
                hidden_dim=args.hidden_dim,
                vocab_size=vocab_size,
                max_targets=args.max_targets,
                dropout_p=dropout_p,
                symmetrical=not args.no_symmetrical,
            )
        )

    first_result = results[0]
    print(
        f"\nShape: encoder=[{first_result.batch_size}, {first_result.max_time}, {first_result.hidden_dim}], "
        f"predictor=[{first_result.batch_size}, {first_result.max_target_plus_1}, {first_result.hidden_dim}], "
        f"dtype={first_result.dtype} (batch size is the double batch)"
    )
    print_results_table(results)


if __name__ == '__main__':
    main()
