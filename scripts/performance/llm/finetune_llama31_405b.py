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

from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.recipes.llama31_405b import finetune_recipe, model
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_fp8_h100_h16384_tp4_mbs1_seqlen2048_lora,
)
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..utils import (
    args_sanity_check,
    get_comm_overlap_callback_idx,
    get_user_configs,
    hf_tokenizer,
    import_ckpt_experiment,
    isfile_train_pack_metadata,
    set_exp_logging_configs,
    set_primary_perf_configs,
    slurm_executor,
)

HF_MODEL_URI = "meta-llama/Llama-3.1-405B"

# Set this to True if checkpoint is available at 'NEMO_HOME'. If set to False,
# extra Slurm job will be scheduled. In this case, if checkpoint is available
# at 'NEMO_HOME', fine-tuning job will use this checkpoint, else, it will be
# downloaded from HuggingFace
SKIP_IMPORT = False


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
    llama3.1 405b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning
    recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True)

    recipe = set_primary_perf_configs(
        recipe,
        finetuning_scheme,
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
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
    )
    recipe = set_exp_logging_configs(
        recipe,
        finetuning_scheme,
        "llm",
        "llama3",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    # data module configs
    recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
    if recipe.data.__fn_or_cls__ == SquadDataModule and not isfile_train_pack_metadata(HF_MODEL_URI, recipe.data):
        # flag is valid only for SquadDataModule
        recipe.data.force_redownload = True

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if (
        finetuning_scheme == "lora"
        and tp_size > 1
        and args.compute_dtype.lower() == "fp8"
        and args.fp8_recipe.lower() != "mxfp8"
    ):
        tp_comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp4_mbs1_seqlen2048_lora if tp_size == 4 else None
        if tp_comm_overlap_cfg:
            # Enable TP comm overlap with the given config
            recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = True
            tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
            recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

            # Disable this overlap to allow skipping an all-gather which is redundant for LoRA
            recipe.model.config.tp_comm_overlap_disable_qkv = True

            # Allow overlapping of dgrad reduce-scatter with dgrad GEMMs
            # (instead of wgrad GEMMs which are not done when using LoRA)
            recipe.model.config.tp_comm_bulk_dgrad = False
            recipe.model.config.tp_comm_overlap_rs_dgrad = True
    if args.compute_dtype.lower() == "fp8" and args.fp8_recipe.lower() == "mxfp8":
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = None
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = False

    recipe.optim.config.use_distributed_optimizer = True
    recipe.model.config.disable_parameter_transpose_cache = True

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), args.finetuning, "llama31", "405b", args)
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
    exp_name = f"{args.finetuning}_{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    executor = slurm_executor(
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

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        )
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

    with run.Experiment(exp_name) as exp:
        if not SKIP_IMPORT:
            assert args.hf_token is not None, "HF token is required for importing checkpoint from HuggingFace"
            exp.add(*import_ckpt_experiment(executor, model(), source=f"hf://{HF_MODEL_URI}"))
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
