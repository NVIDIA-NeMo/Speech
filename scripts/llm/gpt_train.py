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
from nemo.collections.llm.gpt.data import ChatDataModule, MockDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.utils import logging

# Suppress lengthy HF warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    """Parse the command-line arguments."""
    parser = ArgumentParser(
        description="""
            Script for training GPT models. Supports 4 modes, with different arguments needed in addition to the required arguments:
            1. Pretrain: no additional arguments required
            2. SFT: --use-chat-data required
            3. Distillation: --teacher_path required
            4. SFT Distillation: --use-chat-data and --teacher_path required
            """
    )
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to NeMo 2 checkpoint. If only model_path is provided, the model will be trained (pretrain or SFT). If teacher_path is also provided, the model will be distilled.",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        required=False,
        help="Path to NeMo 2 checkpoint to use as a distillation teacher. Will trigger distillation mode if provided.",
    )
    parser.add_argument("--kd_config", type=str, help="""Path to Knowledge-Distillation config file""")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Datatype for models and optimizer")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--log_dir", type=str, required=True, help="Folder for logging and checkpoint saving")
    parser.add_argument("--max_steps", type=int, required=True, help="Number of global batches to process")
    parser.add_argument("--gbs", type=int, required=True, help="Global Batch Size")
    parser.add_argument("--mbs", type=int, required=True, help="Micro-batch Size")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        help="List of tokenized data paths to load from. If using chat data, provide a single path.",
    )
    parser.add_argument("--split", type=str, default="99,1,0", help="Train,Val,Test ratios to split data")
    parser.add_argument("--index_mapping_dir", type=str, help="Folder to write cached data indices")
    parser.add_argument("--use-chat-data", action="store_true", help="Use chat data for fine-tuning.")
    parser.add_argument(
        "--chat-template-path",
        type=str,
        help="Path to Chat template .txt file to use for chat data. Only provide if overriding default chat template in HuggingFace tokenizer.",
    )
    parser.add_argument(
        "--use_mock_data", action="store_true", help="Use mock data instead of custom data in --data_paths"
    )
    parser.add_argument("--seq_length", type=int, required=True, help="Number of tokens per input sample")
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of tokenizer model to override default. Required if using chat data (--use-chat-data).",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Base LR for Cosine-Annealing scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR for Cosine-Annealing scheduler")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of scheduler warmup steps")
    parser.add_argument("--val_check_interval", type=int, default=100, help="Validate + checkpoint every _ steps")
    parser.add_argument("--limit_val_batches", type=int, default=32, help="Number of batches per validation stage")
    parser.add_argument("--log_interval", type=int, default=10, help="Write to log every _ steps")
    parser.add_argument("--legacy_ckpt", action="store_true", help="Load ckpt saved with TE < 1.14")
    return parser.parse_args()


def _read_chat_template(template_path: str):
    # pylint: disable=C0116
    if not template_path:
        return None
    with open(template_path, 'r') as f:
        return f.read().strip()


if __name__ == "__main__":
    args = get_args()

    ## Initialize the strategy and trainer
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=args.cp_size,
        expert_model_parallel_size=args.ep_size,
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

    ## Set up dataset
    if not args.use_mock_data and not args.data_paths:
        raise ValueError("Must provide either custom dataset(s) in --data_paths or set --use_mock_data.")

    if args.use_mock_data:
        logging.warning("Using Mock Data for training!")
        data = MockDataModule(seq_length=args.seq_length, global_batch_size=args.gbs, micro_batch_size=args.mbs)
    elif args.use_chat_data:
        assert len(args.data_paths) == 1, "If using chat data, provide a single path."
        assert args.tokenizer is not None, "Tokenizer is required if using chat data."

        chat_template = _read_chat_template(args.chat_template_path)
        tokenizer = get_tokenizer(args.tokenizer, chat_template=chat_template)
        if '{% generation %}' not in tokenizer.tokenizer.chat_template:
            logging.warning(
                "The chat template does not contain a '{% generation %}' keyword, which will not produce proper assistant mask during training. Instead no tokens will be masked (all tokens contribute to the loss). See https://github.com/huggingface/transformers/pull/30650"
            )
        data = ChatDataModule(
            dataset_root=args.data_paths[0],
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            use_hf_tokenizer_chat_template=True,
        )
    else:
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

    ## Set up checkpointing and logging
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

    ## Set up resume and/or restore functionality
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        restore_config=nl.RestoreConfig(path=args.model_path),
    )

    if args.teacher_path:
        llm.distill(
            student_model_path=args.model_path,
            teacher_model_path=args.teacher_path,
            distillation_config_path=args.kd_config,
            data=data,
            trainer=trainer,
            log=logger,
            resume=resume,
            optim=optim,
            tokenizer=get_tokenizer(args.tokenizer) if args.tokenizer else None,
        )
    else:
        llm.train(
            model=args.model_path,
            data=data,
            trainer=trainer,
            optim=optim,
            log=logger,
            resume=resume,
            tokenizer="data" if args.use_chat_data else "model",
        )
