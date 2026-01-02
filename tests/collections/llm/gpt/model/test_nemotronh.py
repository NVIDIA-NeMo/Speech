# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from typing import Type

import torch
import torch._dynamo
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.gpt.data import MockDataModule, PreTrainingDataModule
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback

torch._dynamo.config.suppress_errors = True

model_options: dict[str, Type[llm.SSMConfig]] = {
    "4B": llm.NemotronHConfig4B,
    "8B": llm.NemotronHConfig8B,
    "47B": llm.NemotronHConfig47B,
    "56B": llm.NemotronHConfig56B,
    "Nano9Bv2": llm.NemotronNano9Bv2,
    "Nano12Bv2": llm.NemotronNano12Bv2,
}


def parse_args():
    """Parse arguments for NMH model training."""
    parser = argparse.ArgumentParser(
        description="Train a nemotronh model using NeMo 2.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group()

    data_group.add_argument(
        "-d",
        "--dataset-config",
        type=str,
        default=None,
        help="Path to the blended / weighted training dataset configuration YAML.",
    )
    data_group.add_argument(
        "--mock-data",
        action="store_true",
        help="Train with Mock data (for testing/debugging), either set this or provide a dataset config.",
    )
    data_group.add_argument(
        "--sft",
        action="store_true",
        help="SFT with SQUAD.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Absolute path to the dataset directory. Defaults to using the absolute or relative paths (dataset_prefix) specified in the dataset config YAML.",
    )

    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for training, defaults to 1.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training, defaults to 1.")
    parser.add_argument("--seq-length", type=int, default=8192, help="Training sequence length")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable Wandb logging")
    parser.add_argument("--wandb-project", type=str, default="nemotronh", help="Wandb project name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Wandb run identifier")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable creating a ModelCheckpoint callback.",
    )
    parser.add_argument("--sequence-parallel", action="store_true", help="Set to enable sequence parallelism.")
    parser.add_argument("--fp8", action="store_true", help="Set to enable FP8")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro-batch size for data-parallel training.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters.",
    )
    parser.add_argument(
        "--grad-acc-batches", type=int, default=1, help="Number of batches to accumulate gradients over."
    )
    parser.add_argument("--max-steps", type=int, help="Number of training optimizer update steps.")
    parser.add_argument(
        "--val-check-interval", type=int, help="Number of steps between validation measurements and model checkpoints."
    )
    parser.add_argument("--grad-reduce-in-fp32", action="store_true", default=False, help="Gradient reduce in FP32.")
    parser.add_argument(
        "--fp8-wgrad",
        action="store_true",
        default=False,
        help="Faster option that is maybe less accurate (TBD) when using fp8.",
    )
    parser.add_argument(
        "--no-aligned-megatron-ddp", action="store_true", default=False, help="Do not do aligned gradient updates etc."
    )
    parser.add_argument(
        "--tp-comm-overlap-backend",
        type=str,
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="TP communication backend to use. Defaults to 'nccl'.",
    )
    parser.add_argument("--align-param-gather", action="store_true", default=False)
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(model_options.keys()),
        default="56B",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory to write model checkpoints and results to.",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=False,
        help="Path to tokenizer vocab file.",
    )
    parser.add_argument(
        "--hf-tokenizer-name",
        type=str,
        default="nvidia/Nemotron-H-8B-Base-8K",
        required=False,
        help="Path to tokenizer vocab file.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=20,
        help="Number of validation steps",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        required=False,
        help="Number of steps between logging.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory to restore an initial checkpoint from. Use this for supervised fine-tuning.",
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument(
        "--restore-optimizer-from-ckpt",
        action="store_true",
        help="Restore optimizer state from initial checkpoint. Defaults to False.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set random seed for training.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers to use for data loading.")
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps.",
    )
    parser.add_argument(
        "--enable-preemption",
        action="store_true",
        default=False,
        help="Enable preemption hooks. If enabled this will save a checkpoint whenver slurm exits.",
    )
    parser.add_argument(
        "--ckpt-async-save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated. Only use if "
        "resuming training from a zarr checkpoint.",
    )
    parser.add_argument(
        "--cross-entropy-loss-fusion",
        action="store_true",
        default=False,
        help="Use the faster, but maybe less accurate fused form of cross entropy, "
        "which also has bf16 grads internally.",
    )
    parser.add_argument(
        "--no-fp32-residual-connection",
        action="store_true",
        default=False,
        help="If set, turn off fp32 residual connections which may be faster but may impact accuracy.",
    )
    parser.add_argument(
        "--debug-ddp-parity-freq",
        type=int,
        default=0,
        help="Set to value > 0 to debug DDP weight parity between ranks.",
    )
    parser.add_argument(
        "--hybrid-override-pattern",
        type=str,
        help="Override the hybrid override pattern in the config (specifies mamba layer ordering and type).",
    )
    parser.add_argument(
        "--num-layers", type=int, help="If set, override the number of layers specified in the requested config."
    )
    parser.add_argument(
        "--log-parameters-and-shapes",
        action="store_true",
        default=False,
        help="Log training parameters shapes and dtypes for debugging.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Min learning rate in cosine annealing.")
    parser.add_argument("--warmup-steps", type=int, default=2500, help="Number of warmup steps in cosine annealing")
    # NSYS profiling/tooling arguments
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling"
        " output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true "
        "--capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )
    # start, end, rank
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    # rank as list of integers
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Grad clip value. Note that when using DDP this may need to be inflated.",
    )
    parser.add_argument(
        "--seq-len-interpolation-factor",
        type=float,
        help="Adjusts the linear scaling of ROPE (Rotary Position Embedding) for context extension. "
        "Set this factor relative to your base context length e.g., for an original context length of 8192 and "
        "an extended context length of 524288, use 524288/8192 = 64.",
    )
    parser.add_argument(
        "--overlap-param-gather",
        action="store_true",
        default=False,
        help="Overlap the parameter gather with the optimizer step. This is currently disabled due to a NeMo bug "
        "when using DDP. Making this an option defaulting to False is a temporary solution until the bug is fixed.",
    )
    parser.add_argument(
        "--overlap-grad-reduce",
        action="store_true",
        default=False,
        help="Overlap the gradient reduce with the optimizer step.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=1073741824,
        help="DDP bucket size.",
    )
    return parser.parse_args()


def main():
    """Main function to run NMH training."""
    args = parse_args()

    # Instantiate tokenizer.
    if args.vocab_file:
        tokenizer = get_nmt_tokenizer(
            library='tiktoken',
            model_name="TiktokenTokenizer",
            vocab_file=args.vocab_file,
            use_fast=True,
        )
    else:
        tokenizer = get_nmt_tokenizer(
            library='huggingface',
            model_name=args.hf_tokenizer_name,
            use_fast=True,
        )

    # Infer global batch size.
    global_batch_size = args.global_batch_size
    if args.mock_data:
        data = MockDataModule(
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=args.workers,
            tokenizer=tokenizer,
        )
    elif args.sft:
        data = llm.SquadDataModule(
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            tokenizer=tokenizer,
            num_workers=args.workers,
            dataset_kwargs={"pad_to_max_length": True},
        )
    else:
        # Instantiate pre-training module.
        data = PreTrainingDataModule(
            paths=args.dataset_dir,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            seed=args.seed,
            num_workers=args.workers,
            tokenizer=tokenizer,
            # eod_mask_loss=args.eod_pad_in_loss_mask,
        )

    # Retrieve model config.
    config_modifiers_init = {
        "seq_length": args.seq_length,
        "distribute_saved_activations": False if args.sequence_parallel else True,
        "cross_entropy_loss_fusion": args.cross_entropy_loss_fusion,
        "fp32_residual_connection": not args.no_fp32_residual_connection,
    }
    if args.hybrid_override_pattern:
        config_modifiers_init["hybrid_override_pattern"] = args.hybrid_override_pattern
    if args.num_layers:
        config_modifiers_init["num_layers"] = args.num_layers

    if args.model_size not in model_options:
        raise ValueError(f"Invalid model size: {args.model_size}")
    mamba_config = model_options[args.model_size](**config_modifiers_init)
    # Instantiate model.
    model = llm.MambaModel(mamba_config, tokenizer=data.tokenizer)

    # Setup callbacks.
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),
    ]
    if args.create_checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=args.val_check_interval,
            dirpath=args.experiment_dir,
            save_top_k=5,
            always_save_context=True,
            save_optim_on_train_end=True,
            save_context_on_train_end=True,
        )
        callbacks.append(checkpoint_callback)

    if args.enable_preemption:
        callbacks.append(nl_callbacks.PreemptionCallback())
    if args.debug_ddp_parity_freq > 0:
        callbacks.append(nl_callbacks.DdpParityChecker(interval=args.debug_ddp_parity_freq))
    if args.log_parameters_and_shapes:
        callbacks.append(nl_callbacks.ParameterDebugger())

    callbacks.append(
        MegatronCommOverlapCallback(
            bucket_size=args.bucket_size,
            tp_comm_bootstrap_backend=args.tp_comm_overlap_backend,
        )
    )

    if args.gc_interval > 0:
        callbacks.append(
            nl_callbacks.GarbageCollectionCallback(
                gc_interval_train=args.gc_interval, gc_interval_val=args.gc_interval
            )
        )
    if args.nsys_profiling:
        if args.nsys_end_step is None:
            nsys_end_step = args.max_steps
        else:
            nsys_end_step = args.nsys_end_step
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=args.nsys_start_step, end_step=nsys_end_step, ranks=args.nsys_ranks, gen_shape=True
            )
        )

    loggers = []
    nemo_logger_kwargs = {}
    if (not args.no_wandb) and args.wandb_project:
        wandb_logger = WandbLogger(
            name=(
                f"nemotronh-size-{args.model_size}-TP{args.tensor_parallel_size}-"
                f"PP{args.pipeline_model_parallel_size}-CP{args.context_parallel_size}"
                f"-GBS{global_batch_size}-MBS{args.micro_batch_size}"
                f"FP8{args.fp8}"
                f"-SEQLEN{args.seq_length}"
                f"-NODES{args.num_nodes}"
            ),
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            id=args.wandb_run_id,  # set this to use the same curve name for restarts.
            project=args.wandb_project,
            save_dir=args.experiment_dir,
        )
        loggers.append(wandb_logger)
        nemo_logger_kwargs["wandb"] = wandb_logger
    tb_logger = TensorBoardLogger(
        save_dir="dummy",  ## NOTE: this gets overwritten by default
    )
    nemo_logger_kwargs["tensorboard"] = tb_logger
    loggers.append(tb_logger)

    nemo_logger = NeMoLogger(log_dir=args.experiment_dir, **nemo_logger_kwargs)
    ddp: DistributedDataParallelConfig = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=args.overlap_param_gather,  # Verify that this works using
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
    )
    # Initialize Megatron Strategy and Trainer.
    strategy = nl.MegatronStrategy(
        ddp=ddp,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=args.sequence_parallel,
        ckpt_load_optimizer=True,
        ckpt_save_optimizer=True,
        ckpt_async_save=args.ckpt_async_save,
        save_ckpt_format=args.ckpt_format,
        ckpt_load_strictness="log_all",  # or rebasing to https://github.com/NVIDIA/NeMo/pull/11988/files#diff-7667eae242a8ef776bff78cd08e79bc81df4896a450f0a781f6ed317a3dfb7ffR139
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            grad_reduce_in_fp32=args.grad_reduce_in_fp32,
            fp8="hybrid" if args.fp8 else None,
            fp8_recipe="tensorwise",
            fp8_amax_history_len=1,
            fp8_amax_compute_algo="max" if args.fp8 else "most_recent",
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=2,
        ),
        val_check_interval=args.val_check_interval,
        enable_checkpointing=args.create_checkpoint_callback,
    )

    # Logger setup
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=False,
        resume_from_directory=args.experiment_dir,
        restore_config=(
            RestoreConfig(
                path=args.ckpt_dir,
                load_model_state=True,
                load_optim_state=args.restore_optimizer_from_ckpt,
            )
            if args.ckpt_dir
            else None
        ),
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=args.wd,
        clip_grad=args.clip_grad,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
    )

    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    # Start training
    trainer.fit(model, data)


if __name__ == "__main__":
    """ Example command to run the script, use --help for more options.:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 \
        /opt/NeMo/tests/collections/llm/gpt/model/test_nemotronh.py \
            --num-nodes=1 \
            --devices=8 \
            --max-steps=500000 \
            --val-check-interval=1000 \
            --experiment-dir=<EXP_DIR> \
            --vocab-file=<VOCAB_FILE> \
            --mock-data \
            --seq-length=8192 \
            --tensor-parallel-size=8 \
            --pipeline-model-parallel-size=1 \
            --context-parallel-size=1 \
            --global-batch-size=8 \
            --micro-batch-size=1 \
            --model-size=8B \
            --fp8 \
            --clip-grad 1 \
            --lr=0.0003 \
            --warmup-steps=2500 \
            --wandb-project=nemotronh
            
    """
    main()
