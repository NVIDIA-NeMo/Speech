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

from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.vlm.recipes.qwen25vl_7b import finetune_recipe
from nemo.lightning.run.plugins import NsysPlugin

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor
from ..helpers import (
    args_sanity_check,
    build_perf_env_plugin,
    get_user_configs,
    set_exp_logging_configs,
    set_primary_perf_configs,
)


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
    enable_cuda_graphs: bool,
):
    """
    Qwen25VL-7B finetune recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=args.gpus_per_node)

    recipe = set_primary_perf_configs(
        recipe,
        "finetune",
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
        enable_cuda_graphs=enable_cuda_graphs,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        use_mcore_fsdp=args.use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_te_act_func=args.use_te_act_func,
        act_func_fp8_input_store=args.act_func_fp8_input_store,
    )

    recipe = set_exp_logging_configs(
        recipe,
        "finetune",
        "vlm",
        "qwen25vl",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.data.tokenizer = run.Config(
        get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=152064
    )

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "finetune", "qwen25vl", "7b", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, _, enable_cuda_graphs = kwargs[:10]

    recipe = override_recipe_configs(
        args, num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, enable_cuda_graphs
    )

    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"
    exp_name = f"qwen25vl_7b_finetune_{args.compute_dtype}_{exp_config}"

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
    )

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]

    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

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
