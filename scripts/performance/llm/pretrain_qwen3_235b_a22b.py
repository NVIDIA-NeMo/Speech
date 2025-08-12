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

from nemo.collections.llm.recipes.qwen3_235b_a22b import pretrain_recipe
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor
from ..helpers import (
    args_sanity_check,
    build_perf_env_plugin,
    get_user_configs,
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from ..utils import dump_config_diff_from_base_recipe


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    etp_size: int,
    enable_cuda_graphs: bool,
):
    """
    qwen3 30b_a3b pre-train recipe aimed at achieving best possible performance and faster
    overall runtime.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe()
    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        etp_size,
        enable_cuda_graphs=enable_cuda_graphs,
        use_mcore_fsdp=args.use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "qwen3", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    # set it to use token drop config
    recipe.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))
    recipe.trainer.callbacks.append(run.Config(MegatronCommOverlapCallback, tp_comm_overlap=True))

    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.moe_permute_fusion = True
    recipe.model.config.bias_dropout_fusion = True
    recipe.model.config.bias_activation_fusion = True

    # reset recompute args in the default recipe
    recipe.model.config.recompute_granularity = None
    recipe.model.config.recompute_method = None
    recipe.model.config.recompute_num_layers = None

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "qwen3", "235b_a22b", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, etp_size, enable_cuda_graphs, _, _, _ = kwargs[
        0:13
    ]

    recipe = override_recipe_configs(
        args, num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, etp_size, enable_cuda_graphs
    )

    exp_config = (
        f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_ep{ep_size}_etp{etp_size}_{mbs}mbs_{gbs}gbs"
    )
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
    )

    if args.gpu.lower() in ['b200', 'gb200'] and "PYTORCH_CUDA_ALLOC_CONF" in executor.env_vars:
        del executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"]

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6, gen_shape=True))
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
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()

    if args.dump_config_diff_from_base_recipe:
        output_dir = exp.jobs[0].executor.job_dir
        # dump difference from base recipe
        base_recipe = pretrain_recipe()
        file_name = f"diff_from_base_recipe_{args.compute_dtype}.diff"
        dump_config_diff_from_base_recipe(base_recipe, recipe, output_dir, file_name=file_name)
