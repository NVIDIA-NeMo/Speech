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

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.collections.vlm.recipes.llama4_omni_e128 import pretrain_recipe
from nemo.lightning.run.plugins import NsysPlugin

from ..argument_parser import parse_additional_slurm_params, parse_cli_args
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
    etp_size: int,
    enable_cuda_graphs: bool,
):
    """
    Llama4 16-Experts (Scout) VLM pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe(performance_mode=True)
    recipe.data.tokenizer = run.Config(
        AutoTokenizer, pretrained_model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct'
    )

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
        compute_dtype=args.compute_dtype,
        use_mcore_fsdp=args.use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_te_act_func=args.use_te_act_func,
        act_func_fp8_input_store=args.act_func_fp8_input_store,
    )
    recipe = set_exp_logging_configs(
        recipe,
        "pre_train",
        "vlm",
        "vlm_llama4",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    recipe.model.config.language_transformer_config.cross_entropy_fusion_impl = "te"
    recipe.model.config.language_transformer_config.cross_entropy_loss_fusion = True
    recipe.model.config.language_transformer_config.apply_rope_fusion = True
    recipe.model.config.language_transformer_config.moe_permute_fusion = True
    recipe.model.config.vision_transformer_config.gradient_accumulation_fusion = True

    # enable cudagraph
    recipe.model.config.vision_transformer_config.enable_cuda_graph = True
    recipe.model.config.enable_cuda_graph = True
    recipe.trainer.strategy.use_te_rng_tracker = True

    recipe.model.config.language_transformer_config.enable_cuda_graph = enable_cuda_graphs

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)
    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "vlm_llama4", "e128", args)
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
        additional_slurm_params=additional_slurm_params,
    )

    if args.gpu.lower() in ['gb200'] and "PYTORCH_CUDA_ALLOC_CONF" in executor.env_vars:
        del executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"]

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]

    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=15, end_step=16, gen_shape=True))

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
