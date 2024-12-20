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
Example:
  torchrun --nproc_per_node=8 scripts/vlm/llava_next_pretrain.py \
  --devices=8 --tp=4 --data_type=mock
  
  torchrun --nproc_per_node=8 scripts/vlm/llava_next_pretrain.py \
  --devices=8 --tp=4 --data_type=energon --data_path='' \ 
  --language_model_path=/root/.cache/nemo/models/lmsys/vicuna-7b-v1.5
"""

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    max_steps = args.max_steps

    decoder_seq_length = 4096

    if args.data_type == "energon":
        from transformers import AutoProcessor

        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule
        from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
        from nemo.collections.vlm import LlavaNextTaskEncoder

        data_path = args.data_path
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer(model_id)

        multimodal_sample_config = MultiModalSampleConfig()
        # Setting system prompt to empty string
        multimodal_sample_config.conversation_template_config.system = ''

        task_encoder = LlavaNextTaskEncoder(
            tokenizer=tokenizer.tokenizer,
            image_processor=processor.image_processor,
            multimodal_sample_config=multimodal_sample_config,
        )
        data = EnergonMultiModalDataModule(
            path=data_path,
            tokenizer=tokenizer,
            image_processor=processor.image_processor,
            num_workers=32,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            multimodal_sample_config=multimodal_sample_config,
            task_encoder=task_encoder,
        )

    elif args.data_type == "mock":
        data = vlm.LlavaNextMockDataModule(
            seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=None,
            image_processor=None,
            num_workers=4,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    # Submodules configurations
    language_transformer_config = llm.Llama2Config7B(
        seq_length=decoder_seq_length,
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

    # Llava Next model configuration
    llava_next_config = vlm.LlavaNextConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=args.language_model_path,
        freeze_language_model=True,
        freeze_vision_model=True,
    )

    model = vlm.LlavaNextModel(llava_next_config, tokenizer=data.tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=1000,
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
        warmup_steps=150,
        constant_steps=0,
        min_lr=2.0e-05,
    )
    opt = MegatronOptimizerModule(opt_config, sched)

    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Next Pretraining Script")

    # Argument parsing
    parser.add_argument("--data_type", type=str, required=False, default="mock", help="mock | energon")
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the dataset JSON file")
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument(
        "--language_model_path", type=str, required=False, default=None, help="Path to the pretrained language model"
    )
    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--max_steps", type=int, required=False, default=2100)
    parser.add_argument("--tp_size", type=int, required=False, default=2)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--projector_type", type=str, required=False, default="mlp2x_gelu")
    parser.add_argument("--name", type=str, required=False, default="llava_next_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=32, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=4, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="Learning rate")

    args = parser.parse_args()
    main(args)
