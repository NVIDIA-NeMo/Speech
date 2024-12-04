# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
Test the LLaMA3 recipe with a smaller model.
"""

import argparse
import os

import nemo_run as run
import torch

from nemo.collections import llm
from nemo.lightning.pytorch.callbacks.debugging import ParameterDebugger
from tests.collections.llm.common import (
    MCoreModelAttributeValidator,
    MiscAttributeValidator,
    StopBeforeEnd,
    create_verify_precision,
    small_llama_cfg,
    train_data,
    verify_ckpt_dir,
)


def get_args():
    parser = argparse.ArgumentParser(prog="", description="")
    parser.add_argument('--devices', type=int, required=True, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, required=True, help="Number of steps to train for")
    parser.add_argument(
        '--early-stop',
        type=int,
        default=None,
        help="Stop training early at this global step (for testing resume training)",
    )
    parser.add_argument(
        '--experiment-dir', type=str, required=True, help="directory to write results and checkpoints to"
    )
    parser.add_argument(
        '--data-path', type=str, default=None, help="Path to data file. If not specified, uses mock data."
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default=None,
        help="Path to a sentencepiece tokenizer model file. If not specified, uses mock data.",
    )
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")
    parser.add_argument('--seq-length', type=int, default=8192, help="Sequence length. default is 8k")
    parser.add_argument('--tp', type=int, default=None, help="Override tensor parallelism")
    parser.add_argument('--pp', type=int, default=None, help="Override pipeline parallelism")
    parser.add_argument('--vp', type=int, default=None, help="Override virtual pipeline parallelism")
    parser.add_argument('--cp', type=int, default=None, help="Override context parallelism")
    parser.add_argument('--sp', type=int, choices=[0, 1], default=None, help="Override sequence parallel")
    parser.add_argument(
        '--precision', type=str, choices=['bf16', 'fp16', 'fp32'], default='bf16', help="Override recipe precision"
    )
    parser.add_argument('--fp8', action='store_true', help="Enable FP8")

    return parser.parse_args()


def main():
    args = get_args()

    exp_name = "L2_llama3_small_pretrain_test"
    pretrain_recipe = llm.llama3_8b.pretrain_recipe(
        dir=args.experiment_dir, name=exp_name, num_gpus_per_node=args.devices
    )

    pretrain_recipe.model = run.Config(llm.LlamaModel, small_llama_cfg(args.seq_length))

    if args.data_path and args.tokenizer_path:
        pretrain_recipe.data = train_data(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            index_mapping_dir=args.index_mapping_dir,
            seq_length=args.seq_length,
        )

    # Recipe Overrides
    pretrain_recipe.trainer.max_steps = args.max_steps
    pretrain_recipe.trainer.log_every_n_steps = 1
    pretrain_recipe.log.ckpt.every_n_train_steps = None
    pretrain_recipe.log.ckpt.train_time_interval = None
    pretrain_recipe.trainer.val_check_interval = 2
    pretrain_recipe.trainer.limit_val_batches = 2

    if args.early_stop:
        pretrain_recipe.trainer.callbacks.append(StopBeforeEnd(stop_on_step=args.early_stop))

    if not args.precision == 'bf16' or args.fp8:  # default case is bf16 without fp8
        import llm.recipes.precision.mixed_precision as mp_recipes

        key = (args.precision, args.fp8)
        precision_recipe = {
            ("fp16", False): mp_recipes.fp16_mixed,
            ("bf16", True): mp_recipes.bf16_with_fp8_mixed,
            ("fp16", True): mp_recipes.fp16_with_fp8_mixed,
            # Need fp32
        }[key]
        pretrain_recipe.trainer.plugins = precision_recipe()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    debugger_callback = ParameterDebugger(
        param_fn=create_verify_precision(dtype_map[args.precision]),
        grad_fn=create_verify_precision(torch.float32),
        log_on_hooks=["on_train_start", "on_train_end"],
    )
    pretrain_recipe.trainer.callbacks.append(debugger_callback)

    parallelisms = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "virtual_pipeline_model_parallel_size": args.vp,
        "context_parallel_size": args.cp,
        "sequence_parallel": bool(args.sp) if args.sp is not None else None,
    }
    for k, v in parallelisms.items():
        if v is not None:  # use recipe default if not specified
            setattr(pretrain_recipe.trainer.strategy, k, v)
        parallelisms[k] = getattr(pretrain_recipe.trainer.strategy, k)
    pretrain_recipe.trainer.callbacks.append(MCoreModelAttributeValidator(parallelisms))

    misc_checker = MiscAttributeValidator(
        {"max_steps": args.max_steps, "stop_on_step": args.early_stop or args.max_steps}
    )
    pretrain_recipe.trainer.callbacks.append(misc_checker)

    run.run(pretrain_recipe, direct=True)

    verify_ckpt_dir(
        pretrain_recipe.log.ckpt,
        args.early_stop or args.max_steps,
        pretrain_recipe.trainer.val_check_interval,
        os.path.join(args.experiment_dir, exp_name),
    )


if __name__ == '__main__':
    main()
