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
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm import ImageDataConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    max_steps = args.max_steps

    seq_length = 2048
    if args.use_packed_sequence:
        seq_length = 4096

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
    if args.use_toy_model:
        language_transformer_config.num_layers = 2

    # NEVA model configuration
    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=args.language_model_path,
        freeze_language_model=True,
        freeze_vision_model=True,
    )
    num_image_embeddings_per_tile = vision_transformer_config.num_image_embeddings_per_tile

    if args.data_type == "llava":
        # Data configuration
        data_config = ImageDataConfig(
            image_folder=args.image_folder,
            conv_template="plain",
        )

        # Data module setup
        data = vlm.NevaPreloadedDataModule(
            paths=args.data_path,
            data_config=data_config,
            seq_length=seq_length,
            decoder_seq_length=None,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=None,
            image_processor=None,
            num_workers=4,
            packed_sequence=args.use_packed_sequence,
            num_image_embeddings_per_tile=num_image_embeddings_per_tile,
        )
    elif args.data_type == "energon":
        from transformers import AutoProcessor
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        from nemo.collections.multimodal.data.energon import (
            EnergonMultiModalDataModule,
            ImageToken,
            LLaVATemplateConfig,
            MultiModalSampleConfig,
        )

        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        image_processor = processor.image_processor
        tokenizer = AutoTokenizer("llava-hf/llava-1.5-7b-hf")

        # Configure multimodal samples
        config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200),
            ignore_place_holder=-100,
            conversation_template_config=LLaVATemplateConfig(system="", chat_template=""),
        )

        # Initialize the data module
        data = EnergonMultiModalDataModule(
            path=args.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=seq_length,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            num_workers=0,
            multimodal_sample_config=config,
            task_encoder=MultiModalTaskEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=config,
                packed_sequence=args.use_packed_sequence,
                # leave some space for perf padding, otherwise after packing and padding,
                # it will go beyond max seq len, then it will need a truncation.
                packed_sequence_size=int(seq_length * 0.9),
                num_image_embeddings_per_tile=num_image_embeddings_per_tile,
            ),
            packing_buffer_size=200 if args.use_packed_sequence else None,
        )
    elif args.data_type == "mock":
        data = vlm.NevaMockDataModule(
            seq_length=seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=None,
            image_processor=None,
            num_workers=4,
            packed_sequence=args.use_packed_sequence,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    from megatron.core.distributed import DistributedDataParallelConfig

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        context_parallel_size=args.cp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
        ),
        ckpt_load_strictness="log_all",
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

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
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=max_steps,
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
        lr=args.lr,
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
    parser.add_argument("--data_type", type=str, required=False, default="mock", help="mock | llava | energon")
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the dataset JSON file")
    parser.add_argument("--image_folder", type=str, required=False, default=None, help="Path to the image folder")
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument(
        "--language_model_path", type=str, required=False, default=None, help="Path to the pretrained language model"
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--max_steps", type=int, required=False, default=5190)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--cp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--projector_type", type=str, required=False, default="mcore_mlp")
    parser.add_argument("--name", type=str, required=False, default="neva_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=128, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=2, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="Learning rate")
    parser.add_argument(
        "--use_packed_sequence",
        action="store_true",
    )
    parser.add_argument(
        "--use_toy_model",
        action="store_true",
        help="Toy size model used for testing",
    )
    args = parser.parse_args()
    main(args)
