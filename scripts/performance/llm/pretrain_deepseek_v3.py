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
from typing import List, Optional

import nemo_run as run

from nemo.collections.llm.recipes.deepseek_v3 import pretrain_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor
from ..helpers import args_sanity_check, get_user_configs, set_exp_logging_configs, set_primary_perf_configs
from ..utils import hf_tokenizer

HF_MODEL_URI = "deepseek-ai/DeepSeek-V3-Base"
USE_TOKEN_DROP = True  # Use token drop callback


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
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: Optional[List[str]] = None,
):
    """
    DeepSeek V3 pre-train recipe aimed at achieving best possible performance.
    """
    recipe = pretrain_recipe(performance_mode=True)
    recipe.model.config.moe_permute_fusion = True
    recipe.model.config.recompute_granularity = "selective"
    recipe.model.config.recompute_num_layers = None
    recipe.model.config.recompute_method = None
    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.moe_router_dtype = 'fp32'

    assert pp_size is None or pp_size == 1 or (64 % pp_size == 0), "pp_size must be 1 or a factor of 64"
    num_layers_in_middle_pipeline_stages = 64 // pp_size

    recipe.trainer.strategy.num_layers_in_first_pipeline_stage = num_layers_in_middle_pipeline_stages - 1
    recipe.trainer.strategy.num_layers_in_last_pipeline_stage = num_layers_in_middle_pipeline_stages - 2

    if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    if USE_TOKEN_DROP:
        recipe.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))

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
        use_mcore_fsdp=use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        recompute_modules=recompute_modules,
    )
    recipe = set_exp_logging_configs(
        recipe,
        "pre_train",
        "llm",
        "deepseekv3",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=129280
        )
    recipe.model.tokenizer = recipe.data.tokenizer

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        raise ValueError("Deepseek FP8 recipe requires subchannel scaling which will be supported soon.")

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "deepseek", "v3", args)
    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        recompute_modules,
    ) = kwargs

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
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        recompute_modules,
    )

    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_ep{ep_size}_{mbs}mbs_{gbs}gbs"
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

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        )
    ]
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
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
