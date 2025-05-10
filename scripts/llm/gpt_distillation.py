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

import os
from argparse import ArgumentParser

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler

# Suppress lengthy HF warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    """Parse the command line arguments."""
    parser = ArgumentParser(description="""Run Knowledge Distillation from a teacher model to a student.""")

    parser.add_argument("--name", type=str, required=True, help="""Experiment name""")
    parser.add_argument("--teacher_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")
    parser.add_argument("--student_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")
    parser.add_argument("--tp_size", type=int, default=1, help="""Tensor parallel size""")
    parser.add_argument("--cp_size", type=int, default=1, help="""Context parallel size""")
    parser.add_argument("--pp_size", type=int, default=1, help="""Pipeline parallel size""")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="""Datatype for models and optimizer""")
    parser.add_argument("--devices", type=int, default=1, help="""Number of GPUs to use per node""")
    parser.add_argument("--num_nodes", type=int, default=1, help="""Number of nodes to use""")
    parser.add_argument("--log_dir", type=str, required=True, help="""Folder for logging and checkpoint saving""")
    parser.add_argument("--max_steps", type=int, required=True, help="""Number of global batches to process""")
    parser.add_argument("--gbs", type=int, required=True, help="""Global Batch Size""")
    parser.add_argument("--mbs", type=int, required=True, help="""Micro-batch Size""")
    parser.add_argument("--data_paths", nargs="+", required=True, help="""List of tokenized data paths to load from""")
    parser.add_argument("--split", type=str, default="99,1,0", help="""Train,Val,Test ratios to split data""")
    parser.add_argument("--index_mapping_dir", type=str, default=None, help="""Folder to write cached data indices""")
    parser.add_argument("--seq_length", type=int, required=True, help="""Number of tokens per input sample""")
    parser.add_argument("--tokenizer", type=str, default=None, help="""Name of tokenizer model to override default""")
    parser.add_argument("--lr", type=float, default=3e-5, help="""Base LR for Cosine-Annealing scheduler""")
    parser.add_argument("--min_lr", type=float, default=2e-7, help="""Minimum LR for Cosine-Annealing scheduler""")
    parser.add_argument("--warmup_steps", type=int, default=50, help="""Number of scheduler warmup steps""")
    parser.add_argument("--val_check_interval", type=int, default=100, help="""Validate + checkpoint every _ steps""")
    parser.add_argument("--limit_val_batches", type=int, default=32, help="""Number of batches per validation stage""")
    parser.add_argument("--log_interval", type=int, default=10, help="""Write to log every _ steps""")
    parser.add_argument("--legacy_ckpt", action="store_true", help="""Load ckpt saved with TE < 1.14""")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ## Initialize the strategy and trainer
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=args.cp_size,
        sequence_parallel=(args.tp_size > 1),
        ddp=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            check_for_nan_in_grad=True,
            average_in_collective=True,
        ),
        ckpt_load_strictness=StrictHandling.LOG_ALL if args.legacy_ckpt else None,
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        strategy=strategy,
        accelerator="gpu",
        plugins=nl.MegatronMixedPrecision(
            precision=args.precision,
            params_dtype=torch.bfloat16 if "bf16" in args.precision else torch.float32,
            autocast_enabled=False,
            grad_reduce_in_fp32=True,
        ),
    )

    # Set up dataset
    data = llm.PreTrainingDataModule(
        paths=args.data_paths,
        seq_length=args.seq_length,
        global_batch_size=args.gbs,
        micro_batch_size=args.mbs,
        split=args.split,
        index_mapping_dir=args.index_mapping_dir,
    )

    ## Set up optimizer
    optim_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        bf16=("bf16" in args.precision),
        use_distributed_optimizer=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        constant_steps=0,
        min_lr=args.min_lr,
    )
    optim = nl.MegatronOptimizerModule(optim_config, sched)

    # Set up checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=args.val_check_interval,
    )
    logger = nl.NeMoLogger(
        name=args.name,
        log_dir=args.log_dir,
        ckpt=checkpoint_callback,
        tensorboard=TensorBoardLogger(os.path.join(args.log_dir, args.name)),
        update_logger_directory=False,
    )

    # Set up resume and/or restore functionality
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        restore_config=nl.RestoreConfig(path=args.student_path),
    )

    llm.distill(
        student_model_path=args.student_path,
        teacher_model_path=args.teacher_path,
        data=data,
        trainer=trainer,
        log=logger,
        resume=resume,
        optim=optim,
        tokenizer=get_tokenizer(args.tokenizer) if args.tokenizer else None,
    )
