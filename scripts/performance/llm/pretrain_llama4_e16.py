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

from nemo.collections.llm.recipes.llama4_e16 import pretrain_recipe
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..utils import (
    args_sanity_check,
    get_user_configs,
    hf_tokenizer,
    set_exp_logging_configs,
    set_primary_perf_configs,
    slurm_executor,
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
    llama4 e16 pre-train recipe aimed at achieving best possible performance and faster
    overall runtime.

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
        etp_size,
        enable_cuda_graphs=enable_cuda_graphs,
        compute_dtype=args.compute_dtype,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "llama4", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer('meta-llama/Llama-4-Scout-17B-16E-Instruct')
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=202048
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.moe_permute_fusion = True

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "llama4", "e16", args)
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

    # Workaround for CUDA graph illegal memory access error
    if not enable_cuda_graphs:
        custom_env_vars = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    else:
        custom_env_vars = {}

    executor = slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=custom_env_vars,
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
    )

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        ),
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=15, end_step=16, gen_shape=True))
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
