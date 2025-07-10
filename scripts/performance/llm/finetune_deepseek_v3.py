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

from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.recipes.deepseek_v3 import finetune_recipe, model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.callbacks.megatron_enable_experimental_callback import MegatronEnableExperimentalCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor
from ..helpers import args_sanity_check, get_user_configs, set_primary_perf_configs
from ..utils import hf_tokenizer, import_ckpt_experiment, isfile_train_pack_metadata

HF_MODEL_URI = "deepseek-ai/DeepSeek-V3-Base"

# Set this to True if checkpoint is available at 'NEMO_HOME'. If set to False,
# extra Slurm job will be scheduled. In this case, if checkpoint is available
# at 'NEMO_HOME', fine-tuning job will use this checkpoint, else, it will be
# downloaded from HuggingFace
SKIP_IMPORT = True
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
    enable_cuda_graphs: bool,
):
    """
    deepseek v3 finetune recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning

    recipe = finetune_recipe(peft_scheme=finetuning_scheme, packed_sequence=False, performance_mode=True)

    # use mock data module for testing
    recipe.data = run.Config(MockDataModule, seq_length=4096, global_batch_size=gbs, micro_batch_size=1)

    if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []

    # Token dispatcher configs. For H100 we use deepEP and for Blackwell,
    # because deepEP is not supported yet, we use all-to-all dispatcher with
    # token drop. After deepEP is supported, we can use deepEP dispatcher.
    if args.gpu.lower() in ['h100']:
        recipe.model.config.moe_token_dispatcher_type = "flex"
        recipe.model.config.moe_enable_deepep = True
        recipe.model.config.moe_shared_expert_overlap = False  # not supported for deepEP
    else:
        recipe.model.config.moe_token_dispatcher_type = "alltoall"
        recipe.model.config.moe_shared_expert_overlap = True
        if USE_TOKEN_DROP:
            recipe.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))

    # Performance optimization knobs
    recipe.model.config.moe_permute_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.trainer.callbacks.append(run.Config(MegatronEnableExperimentalCallback))

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
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        use_mcore_fsdp=args.use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
    )

    # disable HF ckpt loading
    recipe.resume.restore_config = None

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=129280
        )
    recipe.model.tokenizer = recipe.data.tokenizer

    if recipe.data.__fn_or_cls__ == SquadDataModule and not isfile_train_pack_metadata(HF_MODEL_URI, recipe.data):
        # flag is valid only for SquadDataModule
        recipe.data.force_redownload = True

    recipe.model.config.recompute_granularity = 'full'
    recipe.model.config.recompute_method = 'uniform'
    recipe.model.config.recompute_num_layers = 1

    recipe.trainer.strategy.account_for_loss_in_pipeline_split = True
    recipe.trainer.strategy.account_for_embedding_in_pipeline_split = False  # embedding is not split
    recipe.trainer.strategy.num_layers_in_first_pipeline_stage = None
    recipe.trainer.strategy.num_layers_in_last_pipeline_stage = None
    recipe.trainer.strategy.sequence_parallel = False

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), args.finetuning, "deepseek", "v3", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, _, enable_cuda_graphs = kwargs[:10]

    recipe = override_recipe_configs(
        args, num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, enable_cuda_graphs
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

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        )
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=10, end_step=12, gen_shape=True))
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

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
