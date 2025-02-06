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

"""
Example:
  python scripts/vlm/clip_pretrain.py \
  --data_type=mock
"""

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.multimodal.data.energon.base import EnergonMultiModalDataModule
from nemo.collections.vlm.clip.data.clip_data_module import ClipTaskEncoder
from nemo.collections.vlm.clip.model import CLIPConfigB32, CLIPModel
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    max_steps = args.max_steps

    train_task_encoder = ClipTaskEncoder(max_length=args.decoder_seq_length)
    valid_task_encoder = ClipTaskEncoder(max_length=args.decoder_seq_length, is_train=False)
    if args.data_type == "energon":
        data = EnergonMultiModalDataModule(
            args.data_path,
            seq_length=args.decoder_seq_length,
            image_processor=None,
            micro_batch_size=args.mbs,
            global_batch_size=args.gbs,
            num_workers=args.num_workers,
            task_encoder=train_task_encoder,
            tokenizer=train_task_encoder.tokenizer,
            validation_task_encoder=valid_task_encoder,
            image_decode="pil",
            ignore_decoder_errors=True,
        )
    elif args.data_type == "mock":
        data = vlm.ClipMockDataModule(
            seq_length=args.decoder_seq_length,
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            tokenizer=None,
            num_train_samples=10_000_000_000,
            image_processor=None,
            num_workers=8,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    model = CLIPModel(
        CLIPConfigB32(),
        tokenizer=train_task_encoder.tokenizer,
        imagenet_val=args.imagenet_val,
        mbs=args.mbs,
        gbs=args.gbs,
        max_workers=8,
    )

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last="link",
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=2000,
        dirpath=os.path.join(args.log_dir, args.name),
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=2000,
        check_val_every_n_epoch=None,
        limit_val_batches=1,  # We limit validation batches as we are using imagenet validation set
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    # Logger setup
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=False,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=os.path.join(args.log_dir, args.name),
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.2,
    )

    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=2000,
        constant_steps=0,
        min_lr=1e-5,
    )
    opt = MegatronOptimizerModule(
        opt_config,
        sched,
    )

    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip Model Training Script")

    # Argument parsing
    parser.add_argument("--data_type", type=str, required=False, default="energon", help="mock | energon")
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the dataset director")

    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )

    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )

    parser.add_argument("--mbs", type=int, required=False, default=32, help="Micro batch size")
    parser.add_argument("--gbs", type=int, required=False, default=64, help="Global batch size")
    parser.add_argument(
        "--decoder_seq_length",
        type=int,
        required=False,
        default=80,
        help="Decoder" " sequence length for encoding the input text",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=8)

    parser.add_argument("--max_steps", type=int, required=False, default=375000)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--name", type=str, required=False, default="clip_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--lr", type=float, required=False, default=2.0e-06, help="Learning rate")
    parser.add_argument("--imagenet_val", type=str, required=False, default=None, help="imagenet path for val")

    args = parser.parse_args()
    main(args)
