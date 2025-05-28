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
  # mock dataset:
  torchrun --nproc_per_node=8 scripts/vlm/qwen2vl_finetune.py \
  --devices=8 --tp=4 --data_type=mock

  # real dataset:
   torchrun --nproc_per_node=8 /path/to/NeMo/scripts/vlm/qwen2vl_finetune.py  \
     --data_type=qwen2vl \
     --data_path=/path/to/datasets/train.json \
     --image_folder "/path/to/dataset/images" \
     --video_folder "/path/to/dataset/video" \
     --num_nodes 1 \
     --log_dir "/path/to/experiments/qwen2vl_finetune" \
     --devices=8 \
     --tp_size 2 --pp_size 1 \
     --gbs 32 --mbs 1 \
     --wandb_project=qwen2vl_demo \
     --name=qwen2vl_finetune \
     --restore_path "/path/to/experiments/qwen2vl_checkpoint"
"""

import argparse

import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.vlm import Qwen2VLDataConfig
from nemo.collections.vlm.qwen2vl.data.task_encoder import Qwen2VLTaskEncoder
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    max_steps = args.max_steps

    SIZE_INFO_MAP = {
        "2B": {"hf_model_name": "Qwen/Qwen2-VL-2B-Instruct", "llmconfig_class": llm.Qwen2Config1P5B},
        "7B": {"hf_model_name": "Qwen/Qwen2-VL-7B-Instruct", "llmconfig_class": llm.Qwen2Config7B},
    }
    model_size = "2B"
    hf_model_name, llm_config_class = (
        SIZE_INFO_MAP[model_size]["hf_model_name"],
        SIZE_INFO_MAP[model_size]["llmconfig_class"],
    )

    max_sequence_length = args.max_sequence_length
    tokenizer = AutoTokenizer(hf_model_name)
    image_processor = Qwen2VLImageProcessor()
    if args.data_type == "qwen2vl":
        # Data configuration
        data_config = Qwen2VLDataConfig(
            image_folder=args.image_folder,
            video_folder=args.video_folder,
            conv_template="qwen2vl",
            image_process_mode="square",
        )
        # Data module setup
        data = vlm.Qwen2VLPreloadedDataModule(
            paths=args.data_path,
            data_config=data_config,
            seq_length=max_sequence_length,
            decoder_seq_length=None,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=1,
        )
    elif args.data_type == "energon":
        from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule

        # Initialize the data module
        use_packed_sequence = False
        data = EnergonMultiModalDataModule(
            path=args.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=max_sequence_length,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            num_workers=1,
            task_encoder=Qwen2VLTaskEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_padding_length=int(max_sequence_length * 0.9),
            ),
            packing_buffer_size=200 if use_packed_sequence else None,
        )
    elif args.data_type == "mock":
        data = vlm.Qwen2VLMockDataModule(
            seq_length=max_sequence_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,  # for mock data, we generate random token directly, here tokenizer could be none
            image_processor=image_processor,
            num_workers=1,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    # Submodules configurations
    language_transformer_config = llm_config_class(
        seq_length=max_sequence_length,
    )

    vision_transformer_config = vlm.Qwen2VLVisionConfig()

    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type=args.projector_type,
        input_size=vision_transformer_config.ffn_hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=vision_transformer_config.ffn_hidden_size,
    )

    # Qwen2VL model configuration
    qwen2vl_config = vlm.Qwen2VLConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=args.language_model_path,
        freeze_language_model=False,
        freeze_vision_model=True,
    )

    model = vlm.Qwen2VLModel(qwen2vl_config, tokenizer=data.tokenizer)

    from megatron.core.distributed import DistributedDataParallelConfig

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=args.enable_sp,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
        ckpt_load_strictness="log_all",
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_optim_on_train_end=False,
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
        val_check_interval=gbs,
        limit_val_batches=0.0,
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
        warmup_steps=0,
        constant_steps=1000,
        min_lr=1.0e-07,
    )
    opt = MegatronOptimizerModule(opt_config, sched)

    # PEFT setup
    if args.peft == 'lora':
        peft = vlm.peft.LoRA(
            target_modules=[
                "linear_qkv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
            ]
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
    parser = argparse.ArgumentParser(description="QWEN2VL Model Training Script")

    # Argument parsing
    parser.add_argument("--data_type", type=str, required=False, default="mock", help="mock | qwen2vl | energon")
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the dataset JSON file")
    parser.add_argument("--image_folder", type=str, required=False, default=None, help="Path to the image folder")
    parser.add_argument(
        "--video_folder",
        type=str,
        required=False,
        default=None,
        help="Path to the video folder, if not provided, use image_folder",
    )
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
    parser.add_argument("--max_steps", type=int, required=False, default=5190)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--projector_type", type=str, required=False, default="mcore_mlp")
    parser.add_argument("--name", type=str, required=False, default="qwen2vl_finetune")
    parser.add_argument("--peft", type=str, default='none', help="none | lora")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=64, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=2, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=2.0e-06, help="Learning rate")
    parser.add_argument('--enable_sp', action='store_true', help="enable sequence parallel")
    parser.add_argument(
        "--max_sequence_length", type=int, required=False, default=4096, help="Maximum sequence length"
    )

    args = parser.parse_args()
    main(args)
