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

from nemo.collections.llm.recipes.llama3_70b import finetune_recipe, model
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_fp8_h100_h8192_tp2_mbs1_seqlen4096_lora,
)
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
from ..utils import (
    get_comm_overlap_callback_idx,
    hf_tokenizer,
    import_ckpt_experiment,
    prepare_squad_dataset_experiment,
)

HF_MODEL_URI = "meta-llama/Meta-Llama-3-70B"

# Set this to True if checkpoint is available at 'NEMO_HOME'. If set to False,
# extra Slurm job will be scheduled. In this case, if checkpoint is available
# at 'NEMO_HOME', fine-tuning job will use this checkpoint, else, it will be
# downloaded from HuggingFace
SKIP_IMPORT = False

# Set this to True if dataset is already downloaded. If set to False,
# dataset will be downloaded from HuggingFace
SKIP_DATASET_DOWNLOAD = False


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
    llama3 70b fine-tuning recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning
    gpu_type = args.gpu.lower()
    if gpu_type in ["gb200"] and finetuning_scheme == "lora":
        # On GB200 for lora task, we need to enable Cuda Graph for optimal performance.
        # However, Cuda Graph increases memory usage, so in order to avoid OOM, we need
        # to reduce the sequence length.
        recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True, seq_length=2048)
    else:
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
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        use_sharp=args.use_sharp,
        use_te_op_fuser=args.use_te_op_fuser or use_mcore_fsdp,
        use_te_act_func=args.use_te_act_func,
        act_func_fp8_input_store=args.act_func_fp8_input_store,
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

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if finetuning_scheme == "lora" and tp_size > 1 and args.compute_dtype.lower() == "fp8":
        tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp2_mbs1_seqlen4096_lora if tp_size == 2 else None
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

    recipe.optim.config.use_distributed_optimizer = True
    recipe.model.config.disable_parameter_transpose_cache = True

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), args.finetuning, "llama3", "70b", args)
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

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    with run.Experiment(exp_name) as exp:
        if not SKIP_IMPORT:
            assert args.hf_token is not None, "HF token is required for importing checkpoint from HuggingFace"
            exp.add(*import_ckpt_experiment(executor, model(), source=f"hf://{HF_MODEL_URI}"))
        if not SKIP_DATASET_DOWNLOAD:
            exp.add(
                *prepare_squad_dataset_experiment(executor, HF_MODEL_URI, seq_length=4096, nemo_home=args.nemo_home)
            )

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
