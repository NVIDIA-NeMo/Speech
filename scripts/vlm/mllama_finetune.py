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

import argparse

import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.multimodal.data.energon.conversation import MLlamaTemplateConfig
from nemo.collections.vlm import ImageDataConfig
from nemo.collections.vlm.mllama.data.preloaded import MLlamaPreloadedDataModule
from nemo.collections.vlm.mllama.data.task_encoder import LlamaTaskEncoder
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

"""
Example:
  torchrun --nproc_per_node=8 scripts/vlm/mllama_finetune.py \
  --devices=8 --tp=4 --data_type=mock
"""


def main(args):
    """
    Main function for setting up and training the MLLama model.

    This function prepares the data module, model, training strategy,
    logger, checkpointing, and optimizer configuration. It then starts
    the training loop using PyTorch Lightning's trainer.

    Args:
        args (argparse.Namespace): The command-line arguments passed to the script.
    """
    # Setting gbs, mbs, and max_steps from arguments
    gbs = args.gbs
    mbs = args.mbs
    max_steps = args.max_steps
    num_workers = args.num_workers

    # encoder (vision) seq length
    # ((img_res / patch_size) ** 2 + cls_token) * num_tiles, = ((560 / 14) ** 2 + 1) * 4 = 6404
    seq_length = 6404
    decoder_seq_length = 1024  # decoder (llm) seq length

    if args.restore_path is not None and args.restore_path.startswith("nemo://"):
        model_id = args.restore_path[len("nemo://") :]
    else:
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor
    tokenizer = AutoTokenizer(model_id)

    model_configs = {
        "meta-llama/Llama-3.2-11B-Vision": vlm.MLlamaConfig11B,
        "meta-llama/Llama-3.2-11B-Vision-Instruct": vlm.MLlamaConfig11BInstruct,
        "meta-llama/Llama-3.2-90B-Vision": vlm.MLlamaConfig90B,
        "meta-llama/Llama-3.2-90B-Vision-Instruct": vlm.MLlamaConfig90BInstruct,
    }
    conf = model_configs[model_id]()
    if args.use_toy_model:
        conf.language_model_config.num_layers = 2
        num_workers = 0

    if args.data_type == "llava":
        # Data configuration
        data_config = ImageDataConfig(
            image_folder=args.image_folder,
            conv_template="mllama",
        )

        # Data module setup
        data = MLlamaPreloadedDataModule(
            paths=args.data_path,
            data_config=data_config,
            seq_length=seq_length,
            decoder_seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=num_workers,
        )
    elif args.data_type == "energon":
        # Data configuration
        from nemo.collections.multimodal.data.energon import (
            EnergonMultiModalDataModule,
            ImageToken,
            MultiModalSampleConfig,
        )

        # Configure multimodal samples
        config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200),
            ignore_place_holder=-100,
            conversation_template_config=MLlamaTemplateConfig(),
        )

        # Initialize the data module
        data = EnergonMultiModalDataModule(
            path=args.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=decoder_seq_length,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            num_workers=num_workers,
            multimodal_sample_config=config,
            task_encoder=LlamaTaskEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=config,
            ),
        )
    elif args.data_type == "mock":
        data = vlm.MLlamaMockDataModule(
            seq_length=seq_length,
            decoder_seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    model = vlm.MLlamaModel(conf, tokenizer=tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=6,
        every_n_train_steps=100,
        dirpath=args.log_dir,
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
        val_check_interval=min(500, max_steps),
        limit_val_batches=gbs,
        log_every_n_steps=1,
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
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=100,
        constant_steps=0,
        min_lr=args.lr,
    )
    opt = MegatronOptimizerModule(opt_config, sched)

    # PEFT setup
    if args.peft == 'lora':
        peft = vlm.peft.LoRA(
            freeze_vision_model=True,
            target_modules=[
                "linear_qkv",
                "linear_q",
                "linear_kv",
            ],
            dim=8,
            alpha=32,
            dropout=0.05,
            dropout_position="pre",
        )
    else:
        peft = None

    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        peft=peft,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mllama Model Training Script")

    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument("--data_type", type=str, required=False, default="mock", help="mock | llava | energon")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the dataset")
    parser.add_argument("--image_folder", type=str, required=False, help="Path to the image folder")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=False,
        default="/results",
        help="Directory for logging and checkpoints",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--max_steps", type=int, required=False, default=5190)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--name", type=str, required=False, default="neva_pretrain")
    parser.add_argument("--peft", type=str, default='none', help="none | lora")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=64, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=2, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=2.0e-06, help="Learning rate")
    parser.add_argument(
        "--use_toy_model",
        action="store_true",
        help="Toy size model used for testing",
    )
    args = parser.parse_args()
    main(args)
