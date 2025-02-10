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

"""
Example usage of NeMo pretraining commands.

    torchrun --nproc_per_node=8 /path/to/NeMo/examples/vlm/neva_pretrain.py  \
        --data_path "/path/to/dataset/blip_laion_cc_sbu_558k.json" \
        --image_folder "/path/to/dataset/images" \
        --log_dir "/path/to/experiments/neva_pretrain" \
        --devices=8 \
        --projector_type=mcore_mlp \
        --language_model_path "/path/to/models/vicuna-7b-v1.5" \
        --wandb_project=neva_demo \
        --name=neva_pretrain
"""

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.vlm import ImageDataConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # Global and micro batch sizes
    gbs = 256
    mbs = 2
    seq_length = 2048

    # Data configuration
    data_config = ImageDataConfig(
        image_folder=args.image_folder,
        conv_template="plain",
    )

    # Data module setup
    data = vlm.NevaLazyDataModule(
        paths=args.data_path,
        data_config=data_config,
        seq_length=seq_length,
        decoder_seq_length=None,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=None,
        image_processor=None,
        num_workers=8,
    )
    language_transformer_config = llm.Llama2Config7B(
        seq_length=seq_length,
    )
    vision_transformer_config = vlm.HFCLIPVisionConfig(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    )
    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type=args.projector_type,
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=language_transformer_config.hidden_size,
    )

    # NEVA model configuration
    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=args.language_model_path,
        freeze_language_model=True,
        freeze_vision_model=True,
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

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
        save_top_k=2,
        every_n_train_steps=500,
        dirpath=args.log_dir,
    )

    # Trainer setup
    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=2170,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=500,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Logger setup
    from lightning.pytorch.loggers import WandbLogger

    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.001,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=70,
        constant_steps=0,
        min_lr=2.0e-05,
    )
    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    # Start training
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEVA Model Training Script")

    # Argument parsing
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory for logging and checkpoints")
    parser.add_argument(
        "--language_model_path", type=str, required=False, default=None, help="Path to the pretrained language model"
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--projector_type", type=str, required=False, default="mlp2x_gelu")
    parser.add_argument("--name", type=str, required=False, default="neva_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)
