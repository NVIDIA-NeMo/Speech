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

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run

from nemo.collections.llm.recipes.gpt3_175b import pretrain_recipe
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048,
    userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
    userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048,
    userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor
from ..helpers import args_sanity_check, get_user_configs, set_exp_logging_configs, set_primary_perf_configs
from ..utils import get_comm_overlap_callback_idx, hf_tokenizer


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
    gpt3 175b pre-train recipe aimed at achieving best possible performance.

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
        use_sharp=args.use_sharp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "gpt3", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    gpu_type = args.gpu.lower()
    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer("nvidia/megatron-gpt2-345m")
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=51200
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    ub_cfg = {
        "h100": {
            "bf16": userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
            "fp8": userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048,
            "fp8": userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048,
            "fp8": userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048,
        },
    }

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if args.fp8_recipe.lower() != "mxfp8":
        tp_comm_overlap_cfg = ub_cfg[gpu_type][args.compute_dtype]
        # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
        tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg
    if args.compute_dtype.lower() == "fp8" and args.fp8_recipe.lower() == "mxfp8":
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = None
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = False

    recipe.model.config.tp_only_amax_red = True

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "gpt3", "175b", args)
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
