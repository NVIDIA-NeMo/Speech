# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Scripts to run gemma3 VLM SFT.

torchrun --nproc_per_node=1 gemma3vl_finetune.py --data_type=mock

torchrun --nproc_per_node=1 gemma3vl_finetune.py --data_type=energon \
--data_dir=<YOUR DATA DIR>
"""
from scripts.vlm import gemma3vl_utils as train_utils

# Need to run these filters before importing nemo.
train_utils.filter_warnings()
train_utils.filter_grad_bucket_logs()

import argparse
import time

import torch

torch.autograd.set_detect_anomaly(True)
import os

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from transformers import Gemma3ImageProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule
from nemo.collections.vlm.gemma3vl.data.mock import Gemma3VLMockDataModule
from nemo.collections.vlm.gemma3vl.data.task_encoder import TaskEncoder as Gemma3VLTaskEncoder
from nemo.collections.vlm.gemma3vl.data.task_encoder import TaskEncoderConfig as Gemma3VLTaskEncoderConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback


def main(args):
    """Gemma3 VL finetune"""
    num_nodes = int(os.getenv("SLURM_NNODES", "1"))
    if num_nodes == 1:
        train_utils.ignore_sigprof()

    tokenizer = AutoTokenizer(args.hf_model_id)
    language_transformer_config = llm.Gemma3Config4B(seq_length=args.max_sequence_length)
    # The default cross_entropy_fusion_impl is `te`, which will not calculate loss properly with label < 0.
    language_transformer_config.cross_entropy_fusion_impl = "native"
    vision_transformer_config = vlm.Gemma3VLVisionConfig()
    vision_projection_config = vlm.Gemma3VLMultimodalProjectorConfig(
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=vision_transformer_config.ffn_hidden_size,
    )
    gemma3vl_config = vlm.Gemma3VLConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        freeze_language_model=False,
        freeze_vision_model=True,
        freeze_vision_projection=True,
    )
    model = vlm.Gemma3VLModel(gemma3vl_config, tokenizer=tokenizer)

    image_processor = Gemma3ImageProcessor()
    # Data setup
    if args.data_type == "energon":
        if args.data_dir is None:
            raise ValueError("data_dir is required for energon data type.")
        # Initialize the data module
        use_packed_sequence = False
        data = EnergonMultiModalDataModule(
            path=args.data_dir,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=args.max_sequence_length,
            micro_batch_size=args.mbs,
            global_batch_size=args.gbs,
            num_workers=args.num_workers,
            task_encoder=Gemma3VLTaskEncoder(
                Gemma3VLTaskEncoderConfig(
                    hf_path=args.hf_model_id,
                    image_token_id=262144,
                    image_processor=image_processor,
                )
            ),
            packing_buffer_size=200 if use_packed_sequence else None,
        )
    elif args.data_type == "mock":
        data = Gemma3VLMockDataModule(
            seq_length=args.max_sequence_length,
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported.")

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        pipeline_dtype=torch.bfloat16,
        ckpt_async_save=False,
    )

    # Trainer setup
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", params_dtype=torch.bfloat16),
        callbacks=[TimingCallback()],
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        save_optim_on_train_end=True,
        filename="{val_loss:.2f}-{step}-{consumed_samples}",
        dirpath=args.log_dir,
    )

    # Logger setup
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.exp_name,
        ckpt=checkpoint_callback,
        tensorboard=TensorBoardLogger(save_dir="tensorboard", name=""),
        wandb=WandbLogger(project=args.wandb_project, name=args.exp_name) if args.wandb_project is not None else None,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=False,
        resume_ignore_no_checkpoint=True,
        restore_config=nl.RestoreConfig(path=args.resume_from_ckpt) if args.resume_from_ckpt is not None else None,
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )
    lr_scheduler = CosineAnnealingScheduler(
        max_steps=args.max_steps,
        warmup_steps=100,
        constant_steps=0,
        min_lr=1e-7,
    )
    opt = MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

    start_time = time.time()
    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )
    time_elapsed = time.time() - start_time
    logging.info(f"The training elapsed in {time_elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma3VL Model Training Script")

    parser.add_argument("--data_type", type=str, required=False, default="energon", help="mock | energon")
    parser.add_argument("--data_dir", type=str, required=False, default=None, help="Path to the dataset folder")
    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument("--log_dir", type=str, required=False, default="/logs", help="Path to the log folder")
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--cp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--max_steps", type=int, required=False, default=10)
    parser.add_argument("--val_check_interval", type=int, required=False, default=10)
    parser.add_argument("--limit_val_batches", type=float, required=False, default=1.0)
    parser.add_argument("--lr", type=float, required=False, default=2.0e-06, help="Learning rate")
    parser.add_argument(
        "--hf_model_id",
        type=str,
        required=False,
        default="google/gemma-3-4b-it",
        help="HuggingFace Gemma3VL model ids",
    )
    parser.add_argument("--gbs", type=int, required=False, default=32, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=1, help="Micro batch size")
    parser.add_argument("--save_top_k", type=int, required=False, default=1, help="Save top k")
    parser.add_argument(
        "--num_workers", type=int, required=False, default=2, help="The num of workers for data loader"
    )
    parser.add_argument("--max_sequence_length", type=int, required=False, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="Path to restore model from checkpoint",
    )
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--exp_name", type=str, required=False, default="gemma3vl_finetune")

    args = parser.parse_args()
    main(args)
