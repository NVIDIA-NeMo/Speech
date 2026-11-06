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

"""Offline autotuner for the cuTile ``fused_vocab`` joint kernels.

cuTile has no ``@triton.autotune(key=...)`` equivalent: tuning is an explicit, synchronizing,
offline API (``ct.tune.exhaustive_search``), so the winning tile shapes and kernel hints must be
baked into ``rnnt_joint_fused_vocab_cutile._FORWARD_CONFIG`` / ``._BACKWARD_CONFIG`` by hand.
This script produces those numbers. It is a developer tool, not a test.

Usage::

    PYTHONPATH=. python tests/collections/asr/turbo_transducer/tune_rnnt_joint_fused_vocab_cutile.py \
        --pass-name backward --vocab-size 8193
"""

import argparse
import itertools

import cuda.tile as ct
import torch

from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_cutile import (
    _rnnt_joint_fused_vocab_bwd_cutile_kernel,
    _rnnt_joint_fused_vocab_fwd_cutile_kernel,
)

# Tile-shape candidates: (FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK, VOCAB_BLOCK).
_FLATTENED_BATCH_BLOCKS = (64, 128, 256)
_HIDDEN_BLOCKS = (32, 64, 128)
_VOCAB_BLOCKS = (64, 128, 256)
# A [FLATTENED_BATCH_BLOCK, VOCAB_BLOCK] fp32 accumulator above 32768 elements is 128 KB and spills.
_MAX_ACCUMULATOR_ELEMENTS = 32768

_OCCUPANCIES = (None, 1, 2, 4)
_NUM_WORKER_WARPS = (None, 4, 8)


def _build_inputs(batch_size, max_src_len, max_target_len, hidden_dim, vocab_size, dtype, device):
    max_tgt_len_plus_1 = max_target_len + 1
    flattened_batch_size = batch_size * max_src_len * max_tgt_len_plus_1
    blank_id = vocab_size - 1
    generator = torch.Generator(device=device).manual_seed(42)

    inputs = {
        "joint_hidden_2d": torch.randn(
            flattened_batch_size, hidden_dim, device=device, dtype=dtype, generator=generator
        ),
        "targets_flat": torch.randint(
            0, vocab_size - 1, (batch_size * max_target_len,), device=device, dtype=torch.long, generator=generator
        ),
        "src_lengths": torch.full([batch_size], max_src_len, device=device, dtype=torch.long),
        "tgt_lengths": torch.full([batch_size], max_target_len, device=device, dtype=torch.long),
        "weight": torch.randn(vocab_size, hidden_dim, device=device, dtype=dtype, generator=generator),
        "bias": torch.randn(vocab_size, device=device, dtype=dtype, generator=generator),
        "log_sum_exp": torch.randn(flattened_batch_size, device=device, dtype=torch.float32, generator=generator),
        "grad_target_scores": torch.randn(
            flattened_batch_size, device=device, dtype=torch.float32, generator=generator
        ),
        "grad_blank_scores": torch.randn(
            flattened_batch_size, device=device, dtype=torch.float32, generator=generator
        ),
    }
    inputs["flattened_batch_size"] = flattened_batch_size
    inputs["max_src_len"] = max_src_len
    inputs["max_tgt_len_plus_1"] = max_tgt_len_plus_1
    inputs["max_target_len"] = max_target_len
    inputs["hidden_dim"] = hidden_dim
    inputs["vocab_size"] = vocab_size
    inputs["blank_id"] = blank_id
    return inputs


def _search_space(tune_hints: bool, tile_shapes=None):
    space = []
    tile_candidates = (
        tile_shapes
        if tile_shapes is not None
        else itertools.product(_FLATTENED_BATCH_BLOCKS, _HIDDEN_BLOCKS, _VOCAB_BLOCKS)
    )
    for flattened_batch_block, hidden_block, vocab_block in tile_candidates:
        if flattened_batch_block * vocab_block > _MAX_ACCUMULATOR_ELEMENTS:
            continue
        occupancies = _OCCUPANCIES if tune_hints else (None,)
        worker_warps = _NUM_WORKER_WARPS if tune_hints else (None,)
        for occupancy, num_worker_warps in itertools.product(occupancies, worker_warps):
            space.append(
                {
                    "flattened_batch_block": flattened_batch_block,
                    "hidden_block": hidden_block,
                    "vocab_block": vocab_block,
                    "occupancy": occupancy,
                    "num_worker_warps": num_worker_warps,
                }
            )
    return space


def _hints_fn(config):
    hints = {}
    if config["occupancy"] is not None:
        hints["occupancy"] = config["occupancy"]
    if config["num_worker_warps"] is not None:
        hints["num_worker_warps"] = config["num_worker_warps"]
    return hints


def tune_backward(inputs, use_high_precision_mm, tune_hints, timeout_sec, tile_shapes=None):
    device = inputs["joint_hidden_2d"].device
    dtype = inputs["joint_hidden_2d"].dtype
    flattened_batch_size = inputs["flattened_batch_size"]
    hidden_dim = inputs["hidden_dim"]
    vocab_size = inputs["vocab_size"]

    def grid_fn(config):
        batch_blocks = ct.cdiv(flattened_batch_size, config["flattened_batch_block"])
        vocab_blocks = ct.cdiv(vocab_size, config["vocab_block"])
        return (batch_blocks * vocab_blocks, 1, 1)

    def args_fn(config):
        # Gradient buffers accumulate, so every measured config needs its own freshly zeroed copies.
        return (
            inputs["joint_hidden_2d"],
            inputs["targets_flat"],
            inputs["src_lengths"],
            inputs["tgt_lengths"],
            inputs["weight"],
            inputs["bias"],
            inputs["log_sum_exp"],
            inputs["grad_target_scores"],
            inputs["grad_blank_scores"],
            torch.zeros(flattened_batch_size, hidden_dim, device=device, dtype=dtype),
            torch.zeros(vocab_size, hidden_dim, device=device, dtype=dtype),
            torch.zeros(vocab_size, device=device, dtype=dtype),
            flattened_batch_size,
            inputs["max_src_len"],
            inputs["max_tgt_len_plus_1"],
            inputs["max_target_len"],
            hidden_dim,
            vocab_size,
            inputs["blank_id"],
            config["flattened_batch_block"],
            config["hidden_block"],
            config["vocab_block"],
            use_high_precision_mm,
        )

    return ct.tune.exhaustive_search(
        _search_space(tune_hints, tile_shapes),
        torch.cuda.current_stream(),
        grid_fn,
        _rnnt_joint_fused_vocab_bwd_cutile_kernel,
        args_fn,
        _hints_fn,
        single_run_timeout_sec=timeout_sec,
    )


def tune_forward(inputs, use_high_precision_mm, tune_hints, timeout_sec, tile_shapes=None):
    device = inputs["joint_hidden_2d"].device
    flattened_batch_size = inputs["flattened_batch_size"]

    def grid_fn(config):
        return (ct.cdiv(flattened_batch_size, config["flattened_batch_block"]), 1, 1)

    def args_fn(config):
        return (
            inputs["joint_hidden_2d"],
            inputs["targets_flat"],
            inputs["src_lengths"],
            inputs["tgt_lengths"],
            inputs["weight"],
            inputs["bias"],
            torch.zeros(flattened_batch_size, device=device, dtype=torch.float32),
            torch.zeros(flattened_batch_size, device=device, dtype=torch.float32),
            torch.zeros(flattened_batch_size, device=device, dtype=torch.float32),
            flattened_batch_size,
            inputs["max_src_len"],
            inputs["max_tgt_len_plus_1"],
            inputs["max_target_len"],
            inputs["hidden_dim"],
            inputs["vocab_size"],
            inputs["blank_id"],
            config["flattened_batch_block"],
            config["hidden_block"],
            config["vocab_block"],
            use_high_precision_mm,
        )

    return ct.tune.exhaustive_search(
        _search_space(tune_hints, tile_shapes),
        torch.cuda.current_stream(),
        grid_fn,
        _rnnt_joint_fused_vocab_fwd_cutile_kernel,
        args_fn,
        _hints_fn,
        single_run_timeout_sec=timeout_sec,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-name", choices=["forward", "backward"], default="backward")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-time", type=int, default=150)
    parser.add_argument("--max-targets", type=int, default=72)
    parser.add_argument("--hidden-dim", type=int, default=640)
    parser.add_argument("--vocab-size", type=int, default=8193)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--use-high-precision-mm", action="store_true")
    parser.add_argument("--tune-hints", action="store_true", help="also sweep occupancy / num_worker_warps")
    parser.add_argument(
        "--tile-shape",
        action="append",
        default=None,
        metavar="FB,HB,VB",
        help="pin the tile shape(s) instead of sweeping them; repeatable",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    inputs = _build_inputs(
        args.batch_size, args.max_time, args.max_targets, args.hidden_dim, args.vocab_size, dtype, device
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"pass={args.pass_name} vocab_size={args.vocab_size} hidden_dim={args.hidden_dim} "
        f"dtype={args.dtype} flattened_batch_size={inputs['flattened_batch_size']}"
    )

    tile_shapes = None
    if args.tile_shape:
        tile_shapes = [tuple(int(value) for value in shape.split(",")) for shape in args.tile_shape]

    tune = tune_backward if args.pass_name == "backward" else tune_forward
    result = tune(inputs, args.use_high_precision_mm, args.tune_hints, args.timeout_sec, tile_shapes)
    print(result.summary(top_k=args.top_k))


if __name__ == "__main__":
    main()
