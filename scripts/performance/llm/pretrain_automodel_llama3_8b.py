# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from os.path import basename, splitext

import nemo_run as run

from nemo import lightning as nl
from nemo.collections.llm.gpt.data.hf_dataset import HFMockDataModule
from nemo.collections.llm.recipes import hf_auto_model_for_causal_lm
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin

from ..argument_parser import parse_additional_slurm_params, parse_cli_args
from ..executors import slurm_executor
from ..helpers import args_sanity_check, build_perf_env_plugin, get_user_configs

SEQ_LENGTH = 2048
NUM_GPUS_PER_NODE = 8


def override_recipe_configs(
    args: str,
    num_nodes: int,
    num_gpus_per_node: int,
    seq_length: int,
    global_batch_size: int,
    micro_batch_size: int = 1,
):
    """
    Use HFMockdataModule for benchmarking purposes

    """
    model_name = "meta-llama/Meta-Llama-3-8B"
    pretrain = hf_auto_model_for_causal_lm.pretrain_recipe(
        model_name=model_name, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )

    pretrain.trainer.max_steps = 10000
    pretrain.trainer.val_check_interval = 100
    pretrain.log.ckpt.save_top_k = -1
    pretrain.data = run.Config(
        HFMockDataModule,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
    )

    # data module configs
    pretrain.data.num_train_samples = (
        args.max_steps * global_batch_size * micro_batch_size
    )  # ensure only 1 epoch for whole run

    pretrain.trainer.strategy = run.Config(
        nl.FSDP2Strategy,
        data_parallel_size=num_gpus_per_node * num_nodes,
        tensor_parallel_size=1,
    )
    pretrain.trainer.accumulate_grad_batches = global_batch_size / num_gpus_per_node / num_nodes
    return pretrain


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)
    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "llama3", "8b", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size = kwargs[:8]

    recipe = override_recipe_configs(
        args,
        num_nodes,
        num_gpus_per_node=NUM_GPUS_PER_NODE,
        seq_length=SEQ_LENGTH,
        global_batch_size=gbs,
        micro_batch_size=mbs,
    )
    exp_config = f"{num_nodes}nodes_seq{SEQ_LENGTH}_gbs{gbs}"
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    executor = slurm_executor(
        args.gpu.lower(),
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        network='sharp' if args.use_sharp else None,
        additional_slurm_params=additional_slurm_params,
    )

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=args.detach)
        else:
            exp.dryrun()
