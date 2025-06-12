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
  torchrun --nproc_per_node=8 scripts/vlm/avlm_pretrain.py \
  --devices=8 --tp=4 --data_type=mock
  
  torchrun --nproc_per_node=8 scripts/vlm/avlm_pretrain.py \
  --devices=8 --tp=4 --data_type=energon --data_path='' \ 
  --language_model_path=/root/.cache/nemo/models/lmsys/vicuna-7b-v1.5
"""

import argparse

import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import AttnBackend

from nemo import lightning as nl
from nemo.collections import avlm, llm, vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    num_workers = args.num_workers
    max_steps = args.max_steps
    if args.sequence_parallel == "true":
        args.sequence_parallel = True
    elif args.sequence_parallel == "false":
        args.sequence_parallel = False
    else:
        raise ValueError(f"Invalid sequence parallel value: {args.sequence_parallel}")
    if args.use_packed_sequence == "true":
        args.use_packed_sequence = True
    elif args.use_packed_sequence == "false":
        args.use_packed_sequence = False
    else:
        raise ValueError(f"Invalid use packed sequence value: {args.use_packed_sequence}")
    decoder_seq_length = args.seq_length
    if args.use_packed_sequence:
        decoder_seq_length = int(args.seq_length * 2)

    if args.data_type == "energon":
        from nemo.collections.avlm.data.energon import AVLMDataModule, AVLMSampleConfig, AVLMTaskEncoder

        data_path = args.data_path

        avlm_sample_config = AVLMSampleConfig(
            audio_encoder_config={  # whisper audio encoder
                "model_type": "whisper",
                "window_stride": 0.01,
                "sample_rate": 16000,
                "fixed_max_audio_length": 29.9999 * 16000,
                "encoder_down_sampling": 2,
                "num_mel_bins": None,
                "patch_size": None,
                "time_stride": None,
                "frequency_stride": None,
                "max_spectrogram_length": None,
            },
            image_encoder_config={
                "model_type": "vit",
                "img_width": 336,
                "img_height": 336,
                "patch_size": 14,
                "projection_downsample_factor": None,
            },
        )
        # Setting system prompt to empty string
        avlm_sample_config.conversation_template_config.system = ''

        task_encoder = AVLMTaskEncoder(
            multimodal_sample_config=avlm_sample_config,
            packed_sequence=args.use_packed_sequence,
            packed_sequence_size=decoder_seq_length,
        )
        data = AVLMDataModule(
            path=data_path,
            num_workers=num_workers,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            seq_length=decoder_seq_length,
            tokenizer=AutoTokenizer("llava-hf/llava-1.5-7b-hf"),
            multimodal_sample_config=avlm_sample_config,
            task_encoder=task_encoder,
            packing_buffer_size=200 if args.use_packed_sequence else None,
        )
    elif args.data_type == "mock":
        data = avlm.data.AVLMMockDataModule(
            seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=AutoTokenizer("llava-hf/llava-1.5-7b-hf"),
            image_processor=None,
            audio_processor=None,
            num_workers=num_workers,
            image_embedding_tokens=576,  # e.g. for CLIP-ViT-L-14-336
            audio_embedding_tokens=1500,  # e.g. for Whisper
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    # Submodules configurations
    language_transformer_config = llm.Llama2Config7B(
        seq_length=decoder_seq_length,
        attention_backend=AttnBackend.fused,
    )
    vision_transformer_config = vlm.HFCLIPVisionConfig(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    )
    vision_model_from_pretrained = None
    # vision_transformer_config = vlm.CLIPViTL_14_336_Config()
    # vision_model_from_pretrained = "/root/.cache/nemo/models/openai/clip-vit-large-patch14"
    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type=args.projector_type,
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=language_transformer_config.hidden_size,
    )

    # whisper audio encoder  # need update NeMo from Steve's branch
    audio_transformer_config = ASRModuleConfig(
        _target_="nemo.collections.speechlm.modules.asr_module.ASRModuleConfig",
        use_hf_auto_model=True,
        hf_trust_remote_code=False,
        hf_load_pretrained_weights=True,
        pretrained_model="openai/whisper-large-v3",
        hidden_size=1280,
        target_module="model.encoder",
    )
    audio_projection_config = vlm.MultimodalProjectorConfig(
        projector_type=args.projector_type,
        input_size=audio_transformer_config.hidden_size,  # need to set somehow?
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=language_transformer_config.hidden_size,
    )

    # AVLM model configuration
    avlm_config = avlm.AVLMConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        audio_transformer_config=audio_transformer_config,
        audio_projection_config=audio_projection_config,
        language_model_from_pretrained=args.language_model_path,
        vision_model_from_pretrained=vision_model_from_pretrained,
        audio_model_from_pretrained=None,
        freeze_language_model=True,
        freeze_vision_model=True,
        freeze_vision_projection=False,
        freeze_audio_model=True,
        freeze_audio_projection=False,
    )
    model = avlm.AVLMModel(avlm_config, tokenizer=data.tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        context_parallel_size=args.cp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=args.sequence_parallel,
        ckpt_async_save=True,
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=5,
        every_n_train_steps=5000,
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
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        # limit_val_batches=1.0,
        limit_val_batches=20,
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
        restore_config=(
            nl.RestoreConfig(path=args.restore_path, load_optim_state=False) if args.restore_path is not None else None
        ),
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
        clip_grad=1.0,
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
    parser = argparse.ArgumentParser(description="AVLM Pretraining Script")

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
    parser.add_argument("--max_steps", type=int, required=False, default=2000)
    parser.add_argument("--val_check_interval", type=int, required=False, default=500)
    parser.add_argument("--seq_length", type=int, required=False, default=8192)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--cp_size", type=int, required=False, default=1)
    parser.add_argument(
        "--sequence_parallel", type=str, required=False, default="false", help="Enable sequence parallel"
    )
    parser.add_argument(
        "--use_packed_sequence", type=str, required=False, default="false", help="Enable sequence packing"
    )
    parser.add_argument("--projector_type", type=str, required=False, default="mlp2x_gelu")
    parser.add_argument("--name", type=str, required=False, default="avlm_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=32, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=4, help="Micro batch size")
    parser.add_argument(
        "--num_workers", type=int, required=False, default=32, help="Number of workers for data loading"
    )
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="Learning rate")

    args = parser.parse_args()
    main(args)
