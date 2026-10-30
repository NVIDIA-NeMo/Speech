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
    python benchmark_rnnt_joint.py --joint standard --loss turbo_rnnt --dtype bfloat16
    python benchmark_rnnt_joint.py --joint fused --loss turbo_rnnt --dtype bfloat16
    python benchmark_rnnt_joint.py --joint fused_vocab --loss turbo_rnnt --dtype bfloat16
    python benchmark_rnnt_joint.py --joint fused_vocab --dtype bf16-mixed
    python benchmark_rnnt_joint.py --joint fused_vocab --dtype bf16-mixed --mm-full-precision
    python benchmark_rnnt_joint.py --joint standard --loss warprnnt_numba --dtype bfloat16
    python benchmark_rnnt_joint.py --joint fused --dtype bfloat16 -fo  # forward only

For a multi-config sweep (matrix defined at top of the file) see
`benchmark_rnnt_joint_multi.py`.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_fused_joint import rnnt_fused_joint
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import rnnt_joint_fused_vocab
from nemo.collections.asr.parts.turbo_transducer.rnnt_loss_triton import rnnt_loss_from_logprobs_triton

STANDARD_BACKEND_LABEL = "-"

VALID_JOINTS = ("standard", "fused_vocab", "fused")
VALID_JOINT_BACKENDS = ("triton", "cutile")
VALID_LOSSES = ("warprnnt_numba", "turbo_rnnt")
VALID_DTYPES = ("float32", "fp32", "bfloat16", "bf16", "bfloat16-mixed", "bf16-mixed")


@dataclass
class BenchmarkResults:
    joint: str
    joint_backend: str
    loss: str
    fused_batch_size: int
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
    mm_full_precision: bool = False
    use_high_precision_mm: bool = False


def get_dtype(dtype_str: str) -> torch.dtype | str:
    """Resolve aliases; mixed precision keeps its name to distinguish it from pure BF16."""
    dtype_map = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'bfloat16-mixed': 'bfloat16-mixed',
        'bf16-mixed': 'bfloat16-mixed',
    }
    if dtype_str.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str.lower()]


def _normalize_joint_backend(joint: str, joint_backend: str | None) -> str:
    if joint == "standard":
        # Standard joint has no backend dispatch; always render "-".
        return STANDARD_BACKEND_LABEL
    if joint_backend is None or joint_backend == STANDARD_BACKEND_LABEL:
        return "triton"
    if joint_backend not in VALID_JOINT_BACKENDS:
        raise ValueError(
            f"Invalid joint_backend={joint_backend!r} for joint={joint!r}. " f"Expected one of {VALID_JOINT_BACKENDS}."
        )
    return joint_backend


def _create_standard_joint(
    hidden_dim: int,
    vocab_size: int,
    dropout_p: float,
    fused_batch_size: int,
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
        fuse_loss_wer=fused_batch_size > 0,
        fused_batch_size=fused_batch_size if fused_batch_size > 0 else None,
    )
    joint_module.enc = torch.nn.Identity()
    joint_module.pred = torch.nn.Identity()
    joint_module = joint_module.to(device=device, dtype=dtype)
    vocab_projection = joint_module.joint_net[-1]
    vocab_projection.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim, device=device, dtype=dtype))
    return joint_module


def run_benchmark(
    joint: str,
    loss: str,
    dtype: torch.dtype | str,
    joint_backend: str | None = None,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    batch_size: int = 64,
    max_time: int = 150,
    hidden_dim: int = 640,
    vocab_size: int = 1024,
    max_targets: int = 72,
    forward_only: bool = False,
    dropout_p: float = 0.2,
    fused_batch_size: int = 0,
    mm_full_precision: bool = False,
    use_high_precision_mm: bool = False,
) -> BenchmarkResults:
    """Benchmark Joint+loss. Mixed precision uses BF16 projected inputs, FP32 parameters and forward autocast.

    Backward runs outside autocast, as in training. ``mm_full_precision`` selects FP32 operands for optimized
    Joint matmuls; ``use_high_precision_mm`` additionally requests IEEE math in Triton. Standard Joint follows
    PyTorch's precision settings and does not accept these kernel-specific overrides.
    """
    dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype
    mixed_precision = dtype == "bfloat16-mixed"
    dtype_str = "bfloat16-mixed" if mixed_precision else str(dtype).removeprefix("torch.")
    if dtype not in (torch.float32, torch.bfloat16, "bfloat16-mixed"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    activation_dtype = torch.bfloat16 if mixed_precision else dtype
    parameter_dtype = torch.float32 if mixed_precision else dtype
    if joint not in VALID_JOINTS:
        raise ValueError(f"Invalid joint={joint!r}. Expected one of {VALID_JOINTS}.")
    if loss not in VALID_LOSSES:
        raise ValueError(f"Invalid loss={loss!r}. Expected one of {VALID_LOSSES}.")
    if joint != "standard" and loss != "turbo_rnnt":
        raise ValueError(f"Joint={joint!r} requires loss='turbo_rnnt', got loss={loss!r}.")
    if fused_batch_size < 0:
        raise ValueError(f"fused_batch_size must be non-negative, got {fused_batch_size}.")
    if fused_batch_size > batch_size:
        raise ValueError(f"fused_batch_size must not exceed batch_size ({batch_size}), got {fused_batch_size}.")
    if joint != "standard" and fused_batch_size > 0:
        raise ValueError("fused_batch_size is supported only for joint='standard'.")
    if joint == "standard" and (mm_full_precision or use_high_precision_mm):
        raise ValueError("Matmul precision overrides are supported only for optimized Joint implementations.")

    joint_backend = _normalize_joint_backend(joint, joint_backend)

    device = torch.device('cuda')
    blank_id = vocab_size - 1
    max_target_plus_1 = max_targets + 1

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_before_inputs = torch.cuda.memory_allocated()

    enc_proj = torch.randn(batch_size, max_time, hidden_dim, device=device, dtype=activation_dtype, requires_grad=True)
    pred_proj = torch.randn(
        batch_size, max_target_plus_1, hidden_dim, device=device, dtype=activation_dtype, requires_grad=True
    )
    standard_joint = None
    if joint == "standard":
        standard_joint = _create_standard_joint(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            dropout_p=dropout_p,
            fused_batch_size=fused_batch_size,
            device=device,
            dtype=parameter_dtype,
        )
        linear = standard_joint.joint_net[-1]
    else:
        linear = torch.nn.Linear(hidden_dim, vocab_size, device=device, dtype=parameter_dtype)
        linear.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim, device=device, dtype=parameter_dtype))
    targets = torch.randint(0, blank_id, (batch_size, max_targets), device=device, dtype=torch.long)
    src_lengths = torch.full([batch_size], max_time, device=device, dtype=torch.long)
    tgt_lengths = torch.full([batch_size], max_targets, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    input_memory = torch.cuda.memory_allocated() - baseline_before_inputs

    loss_module = None
    if joint == "standard":
        loss_module = RNNTLoss(
            # RNNTLoss expects `num_classes` to be blank index (vocab_size - 1), not vocab size.
            num_classes=vocab_size - 1,
            loss_name=loss,
            loss_kwargs={'fastemit_lambda': 0.0},
            reduction='sum',
        )
        if fused_batch_size > 0:
            standard_joint.set_loss(loss_module)
            # Required by RNNTJoint's fused path, but unused when compute_wer=False.
            standard_joint.set_wer(object())

    def run_fwd_standard() -> torch.Tensor:
        if fused_batch_size > 0:
            loss_value, _, _, _ = standard_joint(
                encoder_outputs=enc_proj.transpose(1, 2),
                decoder_outputs=pred_proj.transpose(1, 2),
                encoder_lengths=src_lengths,
                transcripts=targets,
                transcript_lengths=tgt_lengths,
            )
            return loss_value

        logits = standard_joint(encoder_outputs=enc_proj.transpose(1, 2), decoder_outputs=pred_proj.transpose(1, 2))
        return loss_module(log_probs=logits, targets=targets, input_lengths=src_lengths, target_lengths=tgt_lengths)

    def run_fwd_fused_vocab() -> torch.Tensor:
        joint_hidden = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
        joint_hidden.relu_()
        if dropout_p > 0.0:
            joint_hidden = F.dropout(joint_hidden, p=dropout_p, training=True)
        target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=linear.weight,
            bias=linear.bias,
            blank_id=blank_id,
            backend=joint_backend,
            mm_full_precision=mm_full_precision,
            use_high_precision_mm=use_high_precision_mm,
        )
        loss_batch = rnnt_loss_from_logprobs_triton(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        return loss_batch.sum()

    def run_fwd_fused() -> torch.Tensor:
        target_logprobs, blank_logprobs = rnnt_fused_joint(
            encoder_output_projected=enc_proj,
            predictor_output_projected=pred_proj,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=linear.weight,
            bias=linear.bias,
            blank_id=blank_id,
            dropout_p=dropout_p,
            backend=joint_backend,
            mm_full_precision=mm_full_precision,
            use_high_precision_mm=use_high_precision_mm,
        )
        loss_batch = rnnt_loss_from_logprobs_triton(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        return loss_batch.sum()

    run_fwd_impl = {
        "standard": run_fwd_standard,
        "fused_vocab": run_fwd_fused_vocab,
        "fused": run_fwd_fused,
    }[joint]

    def run_fwd():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            return run_fwd_impl()

    def clear_grads():
        if enc_proj.grad is not None:
            enc_proj.grad = None
        if pred_proj.grad is not None:
            pred_proj.grad = None
        linear.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(warmup_iters):
        clear_grads()
        loss_val = run_fwd()
        if not forward_only:
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

    if forward_only:
        del loss_val
        backward_peak_memory = 0
        max_peak_memory = forward_memory
    else:
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
        if not forward_only:
            loss_val.backward()
        del loss_val

    # Timing: backward
    backward_times = []
    if not forward_only:
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

    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times) if backward_times else 0.0
    return BenchmarkResults(
        joint=joint,
        joint_backend=joint_backend,
        loss=loss,
        fused_batch_size=fused_batch_size,
        dtype=dtype_str,
        batch_size=batch_size,
        max_time=max_time,
        max_target_plus_1=max_target_plus_1,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        input_memory_gb=input_memory / (1024**3),
        forward_memory_gb=forward_memory / (1024**3),
        backward_peak_memory_gb=backward_peak_memory / (1024**3),
        max_peak_memory_gb=max_peak_memory / (1024**3),
        forward_time_ms=avg_fwd,
        backward_time_ms=avg_bwd,
        total_time_ms=avg_fwd + avg_bwd,
        mm_full_precision=mm_full_precision,
        use_high_precision_mm=use_high_precision_mm,
    )


# Column spec drives both the printed table and the CSV. Each entry:
#   (header_label, results_field, formatter)
# `formatter` returns the display string for the table; the CSV emits the
# raw numeric value (for float fields) or the string (otherwise).
def _fmt_mem(value: float) -> str:
    return f"{value:.3f} GB"


def _fmt_time(value: float) -> str:
    return f"{value:.3f} ms"


def _fmt_int(value: int) -> str:
    return str(value)


def _fmt_str(value: str) -> str:
    return value


TABLE_COLUMNS: tuple[tuple[str, str, callable], ...] = (
    ("Vocab", "vocab_size", _fmt_int),
    ("Joint", "joint", _fmt_str),
    ("Joint Backend", "joint_backend", _fmt_str),
    ("Loss", "loss", _fmt_str),
    ("Fused Batch Size", "fused_batch_size", _fmt_int),
    ("Dtype", "dtype", _fmt_str),
    ("MM Full Precision", "mm_full_precision", str),
    ("High Precision MM", "use_high_precision_mm", str),
    ("Input Memory", "input_memory_gb", _fmt_mem),
    ("Fwd Memory", "forward_memory_gb", _fmt_mem),
    ("Bwd Memory", "backward_peak_memory_gb", _fmt_mem),
    ("Fwd+Bwd Memory", "max_peak_memory_gb", _fmt_mem),
    ("Fwd Time", "forward_time_ms", _fmt_time),
    ("Bwd Time", "backward_time_ms", _fmt_time),
    ("Fwd+Bwd Time", "total_time_ms", _fmt_time),
)


def print_results_table(results: list[BenchmarkResults]) -> None:
    """Pretty-print a table of benchmark results using the spec column order."""
    if not results:
        print("(no benchmark results)")
        return

    headers = [col[0] for col in TABLE_COLUMNS]
    rows = []
    for r in results:
        rows.append([formatter(getattr(r, field)) for _, field, formatter in TABLE_COLUMNS])

    column_widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    separator = "+-" + "-+-".join("-" * w for w in column_widths) + "-+"
    separator_mid = "| " + " | ".join("-" * w for w in column_widths) + " |"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, column_widths)) + " |"

    print(separator)
    print(header_line)
    print(separator_mid)
    for row in rows:
        print("| " + " | ".join(cell.ljust(w) for cell, w in zip(row, column_widths)) + " |")
    print(separator)


_FLOAT_FIELDS: frozenset[str] = frozenset(
    {
        "input_memory_gb",
        "forward_memory_gb",
        "backward_peak_memory_gb",
        "max_peak_memory_gb",
        "forward_time_ms",
        "backward_time_ms",
        "total_time_ms",
    }
)


def write_csv(results: list[BenchmarkResults], path: str | Path) -> None:
    """Write benchmark results to CSV. Numeric cells written as raw floats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [col[0] for col in TABLE_COLUMNS]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for r in results:
            row = []
            for _, field_name, _ in TABLE_COLUMNS:
                value = getattr(r, field_name)
                if field_name in _FLOAT_FIELDS:
                    row.append(f"{value:.6f}")
                else:
                    row.append(value)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Benchmark RNN-T Joint implementations (single config)')
    parser.add_argument(
        '--joint',
        type=str,
        required=True,
        choices=list(VALID_JOINTS),
        help='Joint implementation to benchmark',
    )
    parser.add_argument(
        '--joint-backend',
        type=str,
        default=None,
        choices=[*VALID_JOINT_BACKENDS, STANDARD_BACKEND_LABEL],
        help=(
            'Joint backend (only meaningful for fused/fused_vocab). '
            f'Defaults to {STANDARD_BACKEND_LABEL!r} for standard joint and "triton" otherwise.'
        ),
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='turbo_rnnt',
        choices=list(VALID_LOSSES),
        help='Loss implementation',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=VALID_DTYPES,
        help='Mixed precision uses FP32 parameters and BF16 forward autocast (default: bfloat16)',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--fused-batch-size',
        type=int,
        default=0,
        help="Sub-batch size for standard Joint+loss computation; 0 disables fused batching (default: 0)",
    )
    parser.add_argument('--max-time', type=int, default=150)
    parser.add_argument('--max-targets', type=int, default=72, help='Maximum target sequence length (default: 72)')
    parser.add_argument('--hidden-dim', type=int, default=640)
    parser.add_argument(
        '--vocab-size', type=int, default=1024, help='Vocab size including blank (blank_id=vocab_size-1)'
    )
    parser.add_argument('-wi', '--warmup-iterations', type=int, default=10)
    parser.add_argument('-bi', '--benchmark-iterations', type=int, default=100)
    parser.add_argument(
        '-fo', '--forward-only', action='store_true', help='Benchmark forward pass only (skip backward)'
    )
    parser.add_argument('--no-dropout', action='store_true', help='Disable dropout (enabled by default, p=0.2)')
    parser.add_argument('--csv-output', type=str, default=None, help='Optional path to write CSV')
    parser.add_argument('--mm-full-precision', action='store_true', help='Use FP32 matmul operands in optimized Joint')
    parser.add_argument(
        '--use-high-precision-mm', action='store_true', help='Use FP32 operands and IEEE matmuls in Triton Joint'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    dtype = get_dtype(args.dtype)

    torch.manual_seed(42)
    dropout_p = 0.0 if args.no_dropout else 0.2

    result = run_benchmark(
        joint=args.joint,
        loss=args.loss,
        dtype=dtype,
        joint_backend=args.joint_backend,
        warmup_iters=args.warmup_iterations,
        bench_iters=args.benchmark_iterations,
        batch_size=args.batch_size,
        max_time=args.max_time,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        max_targets=args.max_targets,
        forward_only=args.forward_only,
        dropout_p=dropout_p,
        fused_batch_size=args.fused_batch_size,
        mm_full_precision=args.mm_full_precision,
        use_high_precision_mm=args.use_high_precision_mm,
    )

    print(
        f"\nShape: enc=[{result.batch_size}, {result.max_time}, {result.hidden_dim}], "
        f"pred=[{result.batch_size}, {result.max_target_plus_1}, {result.hidden_dim}], "
        f"vocab_size={result.vocab_size}, dtype={result.dtype}, fused_batch_size={result.fused_batch_size}"
    )
    print_results_table([result])

    if args.csv_output:
        write_csv([result], args.csv_output)
        print(f"\nCSV written: {args.csv_output}")


if __name__ == '__main__':
    main()
