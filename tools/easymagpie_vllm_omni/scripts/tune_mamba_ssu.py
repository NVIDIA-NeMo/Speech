#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Tune vLLM's Triton Mamba selective-state-update kernel for EasyMagpie.

Run this while no serving process is using the GPU::

    source setenv.sh
    python scripts/tune_mamba_ssu.py --model converted_model

The output is written directly to ``VLLM_TUNED_CONFIG_FOLDER`` using the
filename consumed by vLLM. Restart the service after tuning because vLLM
caches the resolved launch configuration in each worker process.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import NamedTuple

import torch
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    get_ssm_config_file_name,
    get_ssm_device_name,
    override_ssm_config,
    selective_state_update,
)
from vllm.triton_utils import triton


class ModelShape(NamedTuple):
    nheads: int
    headdim: int
    dstate: int
    ngroups: int


def _model_shape(model_dir: Path, tensor_parallel_size: int) -> ModelShape:
    with (model_dir / "config.json").open() as config_file:
        config = json.load(config_file)

    global_heads = int(config["mamba_num_heads"])
    global_groups = int(config.get("n_groups", 1))
    if global_heads % tensor_parallel_size or global_groups % tensor_parallel_size:
        raise ValueError(
            "mamba_num_heads and n_groups must be divisible by tensor parallel size; "
            f"got heads={global_heads}, groups={global_groups}, tp={tensor_parallel_size}"
        )
    return ModelShape(
        nheads=global_heads // tensor_parallel_size,
        headdim=int(config["mamba_head_dim"]),
        dstate=int(config["ssm_state_size"]),
        ngroups=global_groups // tensor_parallel_size,
    )


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Not a torch dtype: {name}")
    return dtype


def _inputs(batch: int, shape: ModelShape, model_dtype: torch.dtype, cache_dtype: torch.dtype) -> dict:
    device = torch.device("cuda")
    nheads, headdim, dstate, ngroups = shape

    # Match MambaMixer2's decode path, including stride-0 broadcasts for A,
    # dt, D, and dt_bias and indexed reads/writes into the state cache.
    state = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=cache_dtype)
    x = torch.randn(batch, nheads, headdim, device=device, dtype=model_dtype)
    dt = torch.randn(batch, nheads, device=device, dtype=model_dtype)[..., None].expand(-1, -1, headdim)
    a = -torch.rand(nheads, device=device, dtype=torch.float32)
    A = a[:, None, None].expand(-1, headdim, dstate)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=model_dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=model_dtype)
    D = torch.randn(nheads, device=device, dtype=model_dtype)[:, None].expand(-1, headdim)
    dt_bias = torch.randn(nheads, device=device, dtype=model_dtype)[:, None].expand(-1, headdim)
    state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "dt_softplus": True,
        "state_batch_indices": state_indices,
        "dst_state_batch_indices": state_indices,
        "out": torch.empty_like(x),
    }


def _run(inputs: dict, config: tuple[int, int]) -> None:
    with override_ssm_config(config):
        selective_state_update(**inputs)


def _time_ms(inputs: dict, config: tuple[int, int], warmup: int, repeats: int, samples: int) -> float:
    for _ in range(warmup):
        _run(inputs, config)
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            _run(inputs, config)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / repeats)
    return statistics.median(timings)


def _validate(inputs: dict, config: tuple[int, int], reference: tuple[int, int] = (4, 4)) -> None:
    reference_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }
    candidate_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }
    _run(reference_inputs, reference)
    _run(candidate_inputs, config)
    torch.testing.assert_close(candidate_inputs["out"], reference_inputs["out"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(candidate_inputs["state"], reference_inputs["state"], rtol=2e-3, atol=2e-3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("converted_model"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(range(1, 33)))
    parser.add_argument("--model-dtype", default="float16")
    parser.add_argument("--cache-dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to VLLM_TUNED_CONFIG_FOLDER from setenv.sh.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to tune the Mamba SSU kernel")
    if args.tensor_parallel_size <= 0 or args.warmup < 0 or args.repeats <= 0 or args.samples <= 0:
        raise ValueError("tensor parallel size and repeats must be positive; warmup must be non-negative")
    if not args.batch_sizes or any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        configured = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
        if not configured:
            raise ValueError("Set VLLM_TUNED_CONFIG_FOLDER (source setenv.sh) or pass --output-dir")
        output_dir = Path(configured)

    shape = _model_shape(args.model, args.tensor_parallel_size)
    model_dtype = _torch_dtype(args.model_dtype)
    cache_dtype = _torch_dtype(args.cache_dtype)
    cache_dtype_name = str(cache_dtype).removeprefix("torch.")
    candidates = [(block_size, num_warps) for block_size in (2, 4, 8, 16, 32, 64) for num_warps in (1, 2, 4)]
    device_name = get_ssm_device_name()
    print(
        f"GPU={device_name} shape={shape} model_dtype={model_dtype} cache_dtype={cache_dtype}; "
        f"tuning batches {min(args.batch_sizes)}..{max(args.batch_sizes)}"
    )

    results: dict[str, dict[str, int]] = {"triton_version": triton.__version__}
    for batch in sorted(set(args.batch_sizes)):
        inputs = _inputs(batch, shape, model_dtype, cache_dtype)
        timings: list[tuple[float, tuple[int, int]]] = []
        for candidate in candidates:
            try:
                timings.append((_time_ms(inputs, candidate, args.warmup, args.repeats, args.samples), candidate))
            except Exception as exc:  # A candidate may exceed a device resource limit.
                print(f"batch={batch:2d} config={candidate}: skipped ({exc})")
        if not timings:
            raise RuntimeError(f"No valid SSU launch configuration for batch={batch}")
        best_ms, best = min(timings)
        _validate(inputs, best)
        default_ms = next((elapsed for elapsed, config in timings if config == (4, 4)), float("nan"))
        effective_batch = batch * shape.nheads
        results[str(effective_batch)] = {"BLOCK_SIZE_M": best[0], "num_warps": best[1]}
        print(
            f"batch={batch:2d} effective_batch={effective_batch:4d}: "
            f"BLOCK_SIZE_M={best[0]:2d} warps={best[1]} {best_ms * 1000:7.2f}us "
            f"(default {default_ms * 1000:7.2f}us, {default_ms / best_ms:5.2f}x)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = get_ssm_config_file_name(shape.headdim, shape.dstate, cache_dtype_name, device_name)
    output_path = output_dir / filename
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, indent=4) + "\n")
    temporary_path.replace(output_path)
    print(f"Wrote {output_path}")
    print("Restart vLLM-Omni so every stage-0 worker loads the tuned configuration.")


if __name__ == "__main__":
    main()
