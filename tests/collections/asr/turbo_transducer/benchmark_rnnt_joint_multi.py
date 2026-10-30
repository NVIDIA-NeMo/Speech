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
Multi-config RNN-T Joint benchmark.

The experiment matrix is a module-level constant — edit the constants below
to change what gets benchmarked. The script then sweeps `VOCAB_SIZES` x
`EXPERIMENTS`, prints one aggregated table (ordered by vocab), and
optionally writes a CSV.

Usage:
    python benchmark_rnnt_joint_multi.py --csv-output _checks/benchmark.csv
    python benchmark_rnnt_joint_multi.py --dtype bf16-mixed
    python benchmark_rnnt_joint_multi.py --dtype bfloat16-mixed --mm-full-precision

Mixed precision keeps the projection parameters in FP32 and runs forward under BF16 autocast.
Matmul precision overrides apply to optimized Joint implementations; standard Joint follows PyTorch settings.
"""

import argparse
import sys

import torch

from tests.collections.asr.turbo_transducer.benchmark_rnnt_joint import (
    VALID_DTYPES,
    BenchmarkResults,
    get_dtype,
    print_results_table,
    run_benchmark,
    write_csv,
)

# Vocab sizes include the blank symbol. `num_classes = vocab_size - 1` is what
# `run_benchmark` expects (blank_id == num_classes).
VOCAB_SIZES: list[int] = [
    # 1024,
    1025,
    # 8192,
    8193,
    # 13088,
    # 16385
]

# Each entry is (joint, joint_backend, loss). Ordering preserved within each
# vocab group in the printed/CSV output.
EXPERIMENTS: list[tuple[str, str, str]] = [
    # ("standard", "-", "warprnnt_numba"),
    ("standard", "-", "turbo_rnnt"),
    ("fused_vocab", "triton", "turbo_rnnt"),
    # ("fused_vocab", "cutile", "turbo_rnnt"),
    ("fused", "triton", "turbo_rnnt"),
    # ("fused", "cutile", "turbo_rnnt"),  # not yet implemented
]

# Tensor shape / training-knob defaults from `_dev/flow/projects/turbo_transducer/main.md`.
BATCH_SIZE: int = 64
MAX_TIME: int = 150
MAX_TARGETS: int = 72
HIDDEN_DIM: int = 640
DROPOUT_P: float = 0.2


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-config RNN-T Joint benchmark sweep")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional path to write CSV")
    parser.add_argument("-wi", "--warmup-iterations", type=int, default=10)
    parser.add_argument("-bi", "--benchmark-iterations", type=int, default=100)
    parser.add_argument(
        "-fo", "--forward-only", action="store_true", help="Benchmark forward pass only (skip backward)"
    )
    parser.add_argument("--no-dropout", action="store_true", help="Disable dropout (enabled by default, p=0.2)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=VALID_DTYPES)
    parser.add_argument('--mm-full-precision', action='store_true', help='Use FP32 matmul operands in optimized Joint')
    parser.add_argument(
        '--use-high-precision-mm', action='store_true', help='Use FP32 operands and IEEE matmuls in Triton Joint'
    )
    parser.add_argument(
        "--fused-batch-size",
        type=int,
        default=0,
        help="Sub-batch size for standard Joint+loss computation; 0 disables fused batching (default: 0)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    dtype = get_dtype(args.dtype)
    dropout_p = 0.0 if args.no_dropout else DROPOUT_P

    print(f"\nGPU: {gpu_name}")
    print(
        f"Sweep: {len(VOCAB_SIZES)} vocab sizes x {len(EXPERIMENTS)} experiments "
        f"= {len(VOCAB_SIZES) * len(EXPERIMENTS)} runs"
    )
    print(
        f"Shape: enc=[{BATCH_SIZE}, {MAX_TIME}, {HIDDEN_DIM}], "
        f"pred=[{BATCH_SIZE}, {MAX_TARGETS + 1}, {HIDDEN_DIM}], dtype={args.dtype}, dropout_p={dropout_p}, "
        f"standard_fused_batch_size={args.fused_batch_size}"
    )

    torch.manual_seed(42)

    results: list[BenchmarkResults] = []
    for vocab_size in VOCAB_SIZES:
        for joint, joint_backend, loss in EXPERIMENTS:
            print(
                f"\n[{len(results) + 1}/{len(VOCAB_SIZES) * len(EXPERIMENTS)}] "
                f"vocab={vocab_size}, joint={joint}, backend={joint_backend}, loss={loss}"
            )
            result = run_benchmark(
                joint=joint,
                loss=loss,
                dtype=dtype,
                joint_backend=joint_backend,
                warmup_iters=args.warmup_iterations,
                bench_iters=args.benchmark_iterations,
                batch_size=BATCH_SIZE,
                max_time=MAX_TIME,
                hidden_dim=HIDDEN_DIM,
                vocab_size=vocab_size,
                max_targets=MAX_TARGETS,
                forward_only=args.forward_only,
                dropout_p=dropout_p,
                fused_batch_size=args.fused_batch_size if joint == "standard" else 0,
                mm_full_precision=args.mm_full_precision if joint != "standard" else False,
                use_high_precision_mm=args.use_high_precision_mm if joint != "standard" else False,
            )
            results.append(result)

    print()
    print_results_table(results)

    if args.csv_output:
        write_csv(results, args.csv_output)
        print(f"\nCSV written: {args.csv_output}")


if __name__ == "__main__":
    main()
