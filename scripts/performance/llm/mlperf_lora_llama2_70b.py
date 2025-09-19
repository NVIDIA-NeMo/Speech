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

import nemo_run as run
import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.data import MLPerfGovReportDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.gpt.model.llama import *
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin

from ..argument_parser import parse_additional_slurm_params, parse_cli_args
from ..executors import slurm_executor
from ..helpers import args_sanity_check, build_perf_env_plugin
from ..utils import import_ckpt_experiment

NUM_NODES = 1
NUM_GPUS_PER_NODE = 8
MICRO_BATCH_SIZE = 1
GLOBAL_BATCH_SIZE = 8
TP_SIZE = 4  # tp_comm_overlap_cfg may need to be re-tuned if altering
PP_SIZE = 1
CP_SIZE = 1
MAX_STEPS = 100

SEQ_LENGTH = 8192
HF_MODEL_URI = "meta-llama/Llama-2-70b-hf"
SKIP_IMPORT = False


def mlperf_lora_llama2_70b_recipe(
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    max_steps: int,
):
    """
    A recipe for llama2 70B LoRA training based on performance optimizations used in NVIDIA's MLPerf submissions.
    Some optimizations are experimental; use caution.
    Last updated for the 4.1 submission. Designed for 8x H100, tp4pp1cp1.
    """
    tokenizer = run.Config(
        AutoTokenizer,
        pretrained_model_name=HF_MODEL_URI,
    )

    packed_sequence_specs = run.Config(
        PackedSequenceSpecs,
        packed_sequence_size=SEQ_LENGTH,
    )

    datamodule = run.Config(
        MLPerfGovReportDataModule,
        seq_length=SEQ_LENGTH,
        tokenizer=tokenizer,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        persistent_workers=True,
        packed_sequence_specs=packed_sequence_specs,
        dataset_kwargs={
            "return_cu_seqlen": False,
        },
    )

    optimizer_config = run.Config(
        OptimizerConfig,
        lr=2e-4,
        clip_grad=0.3,
        weight_decay=0.0001,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )

    scheduler = run.Config(
        CosineAnnealingScheduler,
        max_steps=max_steps,
        warmup_steps=500,
        constant_steps=0,
        min_lr=0,
    )
    optimizer = run.Config(
        nl.MegatronOptimizerModule,
        config=optimizer_config,
        lr_scheduler=scheduler,
    )
    peft = run.Config(
        llm.peft.LoRA,
        dim=16,
        alpha=32,
        dropout=0.1,
        a2a_experimental=True,
        dropout_position="pre",
        lora_A_init_method="kaiming",
        target_modules=['linear_proj', 'linear_qkv'],
    )
    llama2_config = run.Config(
        llm.Llama2Config70B,
        seq_length=SEQ_LENGTH,
        tp_comm_overlap_disable_qkv=True,
        fp8_dot_product_attention=1,
        cross_entropy_loss_fusion=False,
        disable_parameter_transpose_cache=True,
        external_cuda_graph=False,
        enable_cuda_graph=False,
    )
    llama2_config.microbatch_group_size_per_vp_stage = 1
    model = run.Config(
        MLPerfLoRALlamaModel,
        llama2_config,
    )

    resume = llm.recipes.finetune_default.nemo_resume(HF_MODEL_URI)

    from megatron.core.distributed import DistributedDataParallelConfig

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        sequence_parallel=1 if (tp_size > 1) else 0,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_directly_on_device=False,
        ckpt_parallel_load=False,
        ckpt_load_strictness="log_all",
        gradient_as_bucket_view=True,
        use_te_rng_tracker=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            use_distributed_optimizer=True,
        ),
    )

    precision = run.Config(
        nl.MegatronMixedPrecision,
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=True,
        grad_reduce_in_fp32=False,
        fp8="hybrid",
        fp8_amax_history_len=32,
        fp8_amax_compute_algo='max',
        fp8_param_gather=True,
        fp8_dot_product_attention=1,
    )

    # Communication overlap settings
    tp_comm_overlap_cfg = None
    ub_tp_comm_overlap = tp_size > 1
    if ub_tp_comm_overlap:
        from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
            BulkOverlapCfg,
            PipelineOverlapCfg,
            RingExchangeOverlapCfg,
            TransformerLayerTPOverlapCfg,
        )

        tp_comm_overlap_cfg = TransformerLayerTPOverlapCfg(
            qkv_fprop=RingExchangeOverlapCfg(),
            fc1_fprop=RingExchangeOverlapCfg(),
            proj_dgrad=RingExchangeOverlapCfg(),
            fc2_dgrad=RingExchangeOverlapCfg(),
            proj_fprop=PipelineOverlapCfg(
                num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True, atomic_gemm=True, fp8_buf=True
            ),
            fc2_fprop=PipelineOverlapCfg(
                num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, atomic_gemm=True, fp8_buf=False
            ),
            qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
            fc1_dgrad=RingExchangeOverlapCfg(cga_size=2, set_sm_margin=True),
            qkv_wgrad=None,
            fc1_wgrad=None,
        )
    overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=ub_tp_comm_overlap,
        tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
    )

    # Disable garbage collection
    from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback

    gc_callback = run.Config(
        GarbageCollectionCallback,
        max_steps + 1,
        max_steps + 1,
    )

    # Log step times
    from nemo.utils.exp_manager import TimingCallback

    timing_callback = run.Config(TimingCallback)

    trainer = run.Config(
        nl.Trainer,
        max_steps=max_steps,
        val_check_interval=(max_steps - 1),
        limit_val_batches=0,
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        accelerator="gpu",
        strategy=strategy,
        plugins=precision,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        use_distributed_sampler=False,
        log_every_n_steps=10,
        callbacks=[overlap_callback, gc_callback, timing_callback],
    )

    recipe = run.Partial(
        llm.finetune,
        model=model,
        trainer=trainer,
        data=datamodule,
        optim=optimizer,
        resume=resume,
        peft=peft,
    )
    recipe.model.tokenizer = tokenizer
    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)
    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

    if args.finetuning.lower() != "lora" or args.compute_dtype != "fp8":
        raise ValueError(
            "This example assumes LoRA and fp8; instead got "
            f"--finetuning={args.finetuning} --compute_dtype={args.compute_dtype}"
        )
    if args.virtual_pipeline_parallel_size or args.expert_parallel_size:
        raise ValueError(
            "This example does not support virtual pipeline parallel or expert parallel; "
            f"got --virtual_pipeline_parallel_size={args.virtual_pipeline_parallel_size} "
            f"--expert_parallel_size={args.expert_parallel_size}"
        )

    num_gpus_per_node = NUM_GPUS_PER_NODE if args.gpus_per_node is None else args.gpus_per_node
    num_gpus = NUM_NODES * num_gpus_per_node if args.num_gpus is None else args.num_gpus
    num_nodes = -(num_gpus // -num_gpus_per_node)

    mbs = MICRO_BATCH_SIZE if args.micro_batch_size is None else args.micro_batch_size
    gbs = GLOBAL_BATCH_SIZE if args.global_batch_size is None else args.global_batch_size
    tp_size = TP_SIZE if args.tensor_parallel_size is None else args.tensor_parallel_size
    pp_size = PP_SIZE if args.pipeline_parallel_size is None else args.pipeline_parallel_size
    cp_size = CP_SIZE if args.context_parallel_size is None else args.context_parallel_size

    exp_name = "_".join(
        [
            args.finetuning.lower(),
            "llama2_70b",
            args.compute_dtype,
            f"{num_nodes}nodes",
            f"tp{tp_size}_pp{pp_size}_cp{cp_size}",
            f"{mbs}mbs_{gbs}gbs",
        ]
    )

    env_vars = {
        # pylint: disable=C0301
        "CUDA_DEVICE_MAX_CONNECTIONS": (
            "32" if args.gpu.lower() in ['b200', 'gb200'] else "1"
        ),  # Limit GPUs to one compute stream so that kernels will be executed in consistent order, for best performance with communication overlap configs
        # pylint: disable=C0301
        "CUBLAS_FORCE_XMMA_KERNEL_INIT": "DEVICE",  # Use a device kernel instead of memset for matrix multiply initialization, which can help reduce CPU-side overhead
        "CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT": "0",  # Reduce memory used by cuDNN attention
        "NVTE_FP8_DPA_BWD": "1",  # Enable TransformerEngine FP8 attention for bprop
        # pylint: disable=C0301
        "NVTE_RS_STRIDED_ATOMIC": "2",  # Reduce-scatter communication will be done as a single kernel (with userbuffer TP communication overlap)
        "NCCL_MIN_CTAS": "32",  # Increase CTA resources available to NCCL to improve communication perf
        "NCCL_MIN_P2P_NCHANNELS": "32",  # Likewise, increase P2P channels availabe to NCCL
        "NCCL_NCHANNELS_PER_NET_PEER": "32",  # Likewise, increase per-peer P2P channel limit
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
    }

    executor = slurm_executor(
        args.gpu.lower(),
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        num_gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=env_vars,
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        network='sharp' if args.use_sharp else None,
        additional_slurm_params=additional_slurm_params,
    )

    recipe = mlperf_lora_llama2_70b_recipe(
        num_nodes,
        num_gpus_per_node,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        args.max_steps,
    )

    if not args.tensorboard:
        recipe.log.tensorboard = None
        recipe.trainer.logger = False
    else:
        recipe.log.log_dir = "/nemo_run/lightning_logs"
    if args.wandb:
        assert args.wandb_prj_name is not None
        assert args.wandb_job_name is not None
        from nemo.collections.llm.recipes.log.default import wandb_logger

        recipe.log.wandb = wandb_logger(project=args.wandb_prj_name, name=args.wandb_job_name)

    plugins = [build_perf_env_plugin(args, pp_size=PP_SIZE)]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    with run.Experiment(exp_name) as exp:
        if not SKIP_IMPORT:
            assert args.hf_token is not None, "HF token is required for importing checkpoint from HuggingFace"
            exp.add(
                *import_ckpt_experiment(
                    executor, run.Config(LlamaModel, config=run.Config(Llama2Config70B)), source=f"hf://{HF_MODEL_URI}"
                )
            )
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
