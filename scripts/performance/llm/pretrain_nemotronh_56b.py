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

from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.recipes.nemotronh_56b import pretrain_recipe
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
from ..utils import dump_config_diff_from_base_recipe, hf_tokenizer


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
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
):
    """
    NemotronH 56b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe(performance_mode=True)
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
        enable_cuda_graphs=enable_cuda_graphs,
        use_mcore_fsdp=use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        use_te_act_func=args.use_te_act_func,
        act_func_fp8_input_store=args.act_func_fp8_input_store,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "nemotronh", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer("nvidia/Nemotron-H-8B-Base-8K")
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=131072
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)
    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "nemotronh", "56b", args)
    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        _,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
    ) = kwargs[:13]

    recipe = override_recipe_configs(
        args,
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
    )

    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"
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

    if args.dump_config_diff_from_base_recipe:
        output_dir = exp.jobs[0].executor.job_dir
        # dump difference from base recipe
        base_recipe = pretrain_recipe(performance_mode=False)
        file_name = f"diff_from_base_recipe_{args.compute_dtype}.diff"
        dump_config_diff_from_base_recipe(base_recipe, recipe, output_dir, file_name=file_name)
        # dump difference from default perf recipe
        default_perf_recipe = pretrain_recipe(performance_mode=True)
        file_name = f"diff_from_default_perf_recipe_{args.compute_dtype}.diff"
        dump_config_diff_from_base_recipe(default_perf_recipe, recipe, output_dir, file_name=file_name)
