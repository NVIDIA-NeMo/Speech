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

from typing import Optional

import nemo_run as run
from nemo_run.config import NEMORUN_HOME
from utils import get_comm_overlap_callback_idx, hf_tokenizer, parse_cli_args, slurm_executor

from nemo.collections.llm.recipes.mixtral_8x7b import pretrain_recipe
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin
from nemo.utils import logging

NUM_NODES = 128
NUM_GPUS_PER_NODE = 8
MICRO_BATCH_SIZE = 1
GLOBAL_BATCH_SIZE = 256
TP_SIZE = 4
PP_SIZE = 4
CP_SIZE = 8
VP_SIZE = 14
EP_SIZE = 8
MAX_STEPS = 100


def mixtral_8x22b_performance_recipe(
    compute_dtype: str,
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: Optional[int],
    ep_size: int,
    max_steps: int,
):
    """
    mixtral 8x7b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe(performance_mode=True)

    # data module configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs
    recipe.data.num_train_samples = max_steps * gbs * mbs  # ensure only 1 epoch for whole run
    recipe.data.tokenizer = hf_tokenizer("mistralai/Mixtral-8x22B-v0.1")

    recipe.trainer.max_steps = max_steps
    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = vp_size
    recipe.trainer.strategy.expert_model_parallel_size = ep_size
    if tp_size > 1:
        recipe.trainer.strategy.sequence_parallel = True
    else:
        recipe.trainer.strategy.sequence_parallel = False

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)

    # compute dtype configs
    if compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
    recipe.trainer.plugins.grad_reduce_in_fp32 = False  # bf16 grad dtype

    # callback configs
    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=500,
    )
    recipe.trainer.callbacks.extend(
        [
            garbage_collection_callback,
        ]
    )
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1:
        if comm_overlap_callback_idx >= 0:
            recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = True

    # Misc. for overall faster experiment runtime
    recipe.log.ckpt = None
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.log_every_n_steps = 1

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    if args.log_dir != NEMORUN_HOME:
        import sys

        logging.error(f"Run `export NEMORUN_HOME={args.log_dir}` in your shell environment and rerun this script.")
        sys.exit(1)

    exp_name = "_".join(
        [
            f"mixtral_8x22b",
            args.compute_dtype,
            f"{NUM_NODES}nodes",
            f"tp{TP_SIZE}_pp{PP_SIZE}_cp{CP_SIZE}_vp{VP_SIZE}",
            f"{MICRO_BATCH_SIZE}mbs_{GLOBAL_BATCH_SIZE}gbs",
        ]
    )

    executor = slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        NUM_NODES,
        NUM_GPUS_PER_NODE,
        args.time_limit,
        args.container_image,
        custom_mounts=[],
        custom_env_vars={},
        retries=0,
    )

    recipe = mixtral_8x22b_performance_recipe(
        args.compute_dtype,
        NUM_NODES,
        NUM_GPUS_PER_NODE,
        MICRO_BATCH_SIZE,
        GLOBAL_BATCH_SIZE,
        TP_SIZE,
        PP_SIZE,
        CP_SIZE,
        VP_SIZE,
        EP_SIZE,
        MAX_STEPS,
    )

    if not args.tensorboard:  # tensorboard adds performance overhead.
        recipe.log.tensorboard = None
        recipe.trainer.logger = False
    else:
        # default path is NOT intuitive- `<log_dir>/code/nemo_experiments/tb_logs/default/<tfevents_file>`
        # following line ensures file is at- `<log_dir>/lightning_logs/tb_logs/default/<tfevents_file>`
        recipe.log.log_dir = "/nemo_run/lightning_logs"

    plugins = [PerfEnvPlugin(enable_vboost=True, nccl_pp_comm_chunksize=2097152)]
    if args.enable_profiling:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
