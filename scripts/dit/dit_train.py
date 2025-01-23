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

import os

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import AttnMaskType

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.data.diffusion_fake_datamodule import VideoLatentFakeDataModule
from nemo.collections.diffusion.data.diffusion_taskencoder import BasicDiffusionTaskEncoder
from nemo.collections.diffusion.models.model import (
    DiT7BConfig,
    DiTConfig,
    DiTLConfig,
    DiTLlama1BConfig,
    DiTLlama5BConfig,
    DiTLlama30BConfig,
    DiTModel,
    DiTXLConfig,
    ECDiTLlama1BConfig,
)
from nemo.collections.multimodal.data.energon.base import EnergonMultiModalDataModule
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, PreemptionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.callbacks.nsys import NsysCallback
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback


@run.cli.factory
@run.autoconvert
def multimodal_datamodule() -> pl.LightningDataModule:
    """Multimodal Datamodule Initialization"""
    data_module = DiffusionDataModule(
        seq_length=2048,
        task_encoder=run.Config(BasicDiffusionTaskEncoder, seq_length=2048),
        micro_batch_size=1,
        global_batch_size=32,
    )
    return data_module


@run.cli.factory
@run.autoconvert
def simple_datamodule() -> pl.LightningDataModule:
    """Simple Datamodule Initialization"""
    data_module = EnergonMultiModalDataModule(
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=32,
        num_workers=16,
        tokenizer=None,
        image_processor=None,
        task_encoder=run.Config(BasicDiffusionTaskEncoder, seq_length=2048),
    )
    return data_module


@run.cli.factory
@run.autoconvert
def multimodal_fake_datamodule() -> pl.LightningDataModule:
    """Multimodal Mock Datamodule Initialization"""
    data_module = VideoLatentFakeDataModule(
        seq_length=None,  # Set None to dectect the sequence length automatically.
        task_encoder=run.Config(BasicDiffusionTaskEncoder, seq_length=2048),
        micro_batch_size=1,
        global_batch_size=32,
    )
    return data_module


@run.cli.factory
@run.autoconvert
def peft(args) -> ModelTransform:
    """Parameter Efficient Fine Tuning"""
    return llm.peft.LoRA(
        target_modules=['linear_qkv', 'linear_proj'],  # , 'linear_fc1', 'linear_fc2'],
        dim=args.lora_dim,
    )


@run.cli.factory(target=llm.train)
def pretrain() -> run.Partial:
    """Base Pretraining Config"""
    return run.Partial(
        llm.train,
        model=run.Config(
            DiTModel,
            config=run.Config(DiTConfig),
        ),
        data=multimodal_datamodule(),
        trainer=run.Config(
            nl.Trainer,
            devices='auto',
            num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                ),
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=1,
            val_check_interval=1000,
            max_epochs=10000,
            log_every_n_steps=1,
            callbacks=[
                run.Config(
                    ModelCheckpoint,
                    monitor='global_step',
                    filename='{global_step}',
                    every_n_train_steps=1000,
                    save_top_k=3,
                    mode='max',
                ),
                run.Config(PreemptionCallback),
                run.Config(TimingCallback),
                run.Config(
                    MegatronCommOverlapCallback,
                    tp_comm_overlap=False,
                ),
            ],
        ),
        log=nl.NeMoLogger(wandb=(WandbLogger() if "WANDB_API_KEY" in os.environ else None)),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                weight_decay=0,
            ),
        ),
        tokenizer=None,
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        model_transform=None,
    )


@run.cli.factory(target=llm.train)
def pretrain_xl() -> run.Partial:
    """DiT-XL Pretraining Recipe"""
    recipe = pretrain()
    recipe.model.config = run.Config(DiTXLConfig)
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_l() -> run.Partial:
    """DiT-L Pretraining Recipe"""
    recipe = pretrain()
    recipe.model.config = run.Config(DiTLConfig)
    return recipe


@run.cli.factory(target=llm.train)
def train_mock() -> run.Partial:
    """DiT Mock Pretraining Recipe"""
    recipe = pretrain()
    recipe.model.config = run.Config(DiTLlama5BConfig, max_frames=1)
    recipe.data = multimodal_fake_datamodule()
    recipe.model.config.num_layers = 16
    recipe.data.seq_length = 73728
    recipe.data.task_encoder.seq_length = 73728
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.context_parallel_size = 2
    recipe.data.micro_batch_size = 1
    recipe.data.global_batch_size = 1
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.val_check_interval = 1.0
    recipe.data.model_config = recipe.model.config
    recipe.log.log_dir = 'nemo_experiments/train_mock'

    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = 'MODEL_AND_OPTIMIZER_STATES'
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    return recipe


@run.cli.factory(target=llm.train)
def mock_ditllama5b_8k() -> run.Partial:
    """DiT-5B mock Recipe"""
    recipe = pretrain()
    recipe.model.config = run.Config(DiTLlama5BConfig, max_frames=1)
    recipe.data = multimodal_fake_datamodule()
    recipe.data.seq_length = recipe.data.task_encoder.seq_length = 8192
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.data.micro_batch_size = 1
    recipe.data.global_batch_size = 32
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.val_check_interval = 1.0
    recipe.data.model_config = recipe.model.config
    recipe.log.log_dir = 'nemo_experiments/mock_ditllama5b_8k'
    recipe.model.config.attn_mask_type = AttnMaskType.no_mask
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = 'MODEL_AND_OPTIMIZER_STATES'
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    recipe.trainer.max_steps = 15
    recipe.trainer.callbacks.pop(0)
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=10,
            end_step=11,
        )
    )
    recipe.resume = None
    return recipe


@run.cli.factory(target=llm.train)
def mock_dit7b_8k() -> run.Partial:
    """DiT-7B mock Recipe"""
    recipe = mock_ditllama5b_8k()
    recipe.model.config = run.Config(DiT7BConfig, max_frames=1)
    recipe.data.model_config = recipe.model.config
    recipe.model.config.attn_mask_type = AttnMaskType.no_mask
    recipe.model.config.use_cpu_initialization = True
    recipe.log.log_dir = 'nemo_experiments/mock_dit7b_8k'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_7b() -> run.Partial:
    """DiT-7B Pretraining Recipe"""
    recipe = pretrain()
    recipe.model.config = run.Config(DiT7BConfig)
    recipe.data.global_batch_size = 4608
    recipe.data.micro_batch_size = 9
    recipe.data.num_workers = 15
    recipe.data.use_train_split_for_val = True
    recipe.data.seq_length = 260
    recipe.data.task_encoder.seq_length = 260
    recipe.trainer.val_check_interval = 1000
    recipe.log.log_dir = 'nemo_experiments/dit7b'
    recipe.optim.lr_scheduler = run.Config(nl.lr_scheduler.WarmupHoldPolicyScheduler, warmup_steps=100, hold_steps=1e9)
    recipe.optim.config.weight_decay = 0.1
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.95

    return recipe


@run.cli.factory(target=llm.train)
def pretrain_7b_pack() -> run.Partial:
    """DiT-7B Pretraining Recipe with Packing"""
    recipe = pretrain_7b()
    recipe.data.global_batch_size = 4608 // 9
    recipe.data.micro_batch_size = 1
    recipe.data.num_workers = 15
    recipe.data.use_train_split_for_val = True
    recipe.data.seq_length = 256 * 9
    recipe.data.packing_buffer_size = 1000
    recipe.data.task_encoder.seq_length = None
    recipe.data.task_encoder.max_seq_length = recipe.data.seq_length
    recipe.model.config.qkv_format = 'thd'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_7b_256p_joint() -> run.Partial:
    """DiT-7B Pretraining Recipe 256p Stage 1"""
    recipe = pretrain_7b()
    recipe.data.global_batch_size = 256  # 768
    recipe.data.micro_batch_size = 1
    recipe.data.seq_length = 8192
    recipe.data.task_encoder.seq_length = 8192
    recipe.model.config.seq_length = 8192

    recipe.optim.config.lr = 6e-5
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    # recipe.resume.restore_config = run.Config(RestoreConfig, path='', load_optim_state=True)
    recipe.log.log_dir = 'nemo_experiments/pretrain_7b_256p_joint'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_7b_256p_joint_pack() -> run.Partial:
    """DiT-7B Pretraining Recipe 256p Stage 1 with Packing"""
    recipe = pretrain_7b_256p_joint()
    recipe.data.global_batch_size = 128
    recipe.data.micro_batch_size = 1
    recipe.data.num_workers = 10
    recipe.data.seq_length = recipe.model.config.seq_length = recipe.data.task_encoder.max_seq_length = 10240
    recipe.data.task_encoder.seq_length = None
    recipe.data.packing_buffer_size = 1000
    recipe.data.virtual_epoch_length = 0
    recipe.model.config.qkv_format = 'thd'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama5b() -> run.Partial:
    """MovieGen 5B Training"""
    recipe = pretrain_7b()
    recipe.data.micro_batch_size = 12
    recipe.model.config = run.Config(DiTLlama5BConfig)
    recipe.log.log_dir = 'nemo_experiments/ditllama5b'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama30b() -> run.Partial:
    """MovieGen 30B Stage 1 Training"""
    recipe = pretrain_ditllama5b()
    recipe.model.config = run.Config(DiTLlama30BConfig)
    recipe.data.global_batch_size = 9216
    recipe.data.micro_batch_size = 6
    recipe.data.task_encoder.aethetic_score = 4.0
    recipe.data.seq_length = 256
    recipe.data.task_encoder.seq_length = 256
    recipe.data.virtual_epoch_length = 0
    recipe.log.log_dir = 'nemo_experiments/ditllama30b_stage1_mock'
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = 'MODEL_AND_OPTIMIZER_STATES'
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama30b_stage2_mock() -> run.Partial:
    """MovieGen 30B Stage 2 Training"""
    recipe = pretrain_ditllama5b()
    recipe.model.config = run.Config(DiTLlama30BConfig)
    recipe.data = multimodal_fake_datamodule()
    recipe.data.model_config = recipe.model.config
    recipe.data.seq_length = 8192
    recipe.data.task_encoder.seq_length = 8192
    recipe.data.global_batch_size = 256
    recipe.data.micro_batch_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.context_parallel_size = 4
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.val_check_interval = 1.0
    recipe.data.model_config = recipe.model.config
    recipe.log.log_dir = 'nemo_experiments/ditllama30b_stage2_mock'
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = 'MODEL_AND_OPTIMIZER_STATES'
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama30b_stage3_mock() -> run.Partial:
    """MovieGen 30B Stage 3 Training"""
    recipe = pretrain_ditllama5b()
    recipe.model.config = run.Config(DiTLlama30BConfig)
    recipe.data = multimodal_fake_datamodule()
    recipe.data.model_config = recipe.model.config
    recipe.data.seq_length = 73728
    recipe.data.task_encoder.seq_length = 73728
    recipe.data.global_batch_size = 256
    recipe.data.micro_batch_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.context_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.val_check_interval = 1.0
    recipe.data.model_config = recipe.model.config
    recipe.log.log_dir = 'nemo_experiments/ditllama30b_stage3_mock'
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = 'MODEL_AND_OPTIMIZER_STATES'
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama5b_stage3_mock_with_pp() -> run.Partial:
    """MovieGen 30B Stage 3 Training"""
    recipe = pretrain_ditllama5b()
    recipe.data = multimodal_fake_datamodule()
    recipe.data.model_config = recipe.model.config
    recipe.data.seq_length = 8192
    recipe.data.task_encoder.seq_length = 8192
    recipe.data.global_batch_size = 1
    recipe.data.micro_batch_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.pipeline_model_parallel_size = 2
    recipe.trainer.strategy.context_parallel_size = 2
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.val_check_interval = 1.0
    recipe.data.model_config = recipe.model.config
    recipe.log.log_dir = 'nemo_experiments/ditllama30b_stage5_mock_with_pp'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama30b_stage3_mock_with_pp() -> run.Partial:
    """MovieGen 30B Stage 3 Training with Pipeline Parallelism"""
    recipe = pretrain_ditllama5b()
    recipe.model.config = run.Config(DiTLlama30BConfig)
    recipe.data = multimodal_fake_datamodule()
    recipe.data.model_config = recipe.model.config
    recipe.data.seq_length = 73728
    recipe.data.task_encoder.seq_length = 73728
    recipe.data.global_batch_size = 256
    recipe.data.micro_batch_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 4
    recipe.trainer.strategy.context_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.val_check_interval = 1.0
    recipe.data.model_config = recipe.model.config
    recipe.log.log_dir = 'nemo_experiments/ditllama30b_stage3_mock_with_pp'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama1b() -> run.Partial:
    """MovieGen 1B Stage 1 Training"""
    recipe = pretrain_ditllama5b()
    recipe.model.config = run.Config(DiTLlama1BConfig)
    recipe.data.task_encoder.aethetic_score = 4.0
    recipe.data.seq_length = 256
    recipe.data.task_encoder.seq_length = 256
    recipe.model.config.seq_length = 256
    recipe.data.global_batch_size = 1536
    recipe.data.micro_batch_size = 96
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.log.log_dir = 'nemo_experiments/ditllama1b'
    recipe.trainer.val_check_interval = 3000
    recipe.trainer.callbacks[0].every_n_train_steps = 3000
    recipe.trainer.callbacks[0].monitor = 'global_step'
    recipe.trainer.callbacks[0].save_top_k = 3
    recipe.trainer.callbacks[0].mode = 'max'
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ditllama3b() -> run.Partial:
    """MovieGen 3B Stage 1 Training"""
    recipe = pretrain_ditllama1b()
    recipe.data.micro_batch_size = 48
    recipe.model.config = run.Config(
        DiTLlama1BConfig,
        hidden_size=3072,
        num_layers=28,
        num_attention_heads=24,
        ffn_hidden_size=8192,
    )
    recipe.log.log_dir = 'nemo_experiments/ditllama3b'

    return recipe


@run.cli.factory(target=llm.train)
def pretrain_ecditllama1b() -> run.Partial:
    """EC-DiT 1B Training"""
    recipe = pretrain_ditllama1b()
    recipe.data.task_encoder.aethetic_score = 5.0
    recipe.data.micro_batch_size = 72
    recipe.data.global_batch_size = 2304
    recipe.model.config = run.Config(ECDiTLlama1BConfig)
    recipe.log.log_dir = 'nemo_experiments/ecditllama1b'
    recipe.trainer.val_check_interval = 3000

    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = 'MODEL_AND_OPTIMIZER_STATES'
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    return recipe


@run.cli.factory(target=llm.train)
def dreambooth() -> run.Partial:
    """Dreambooth Fine Tuning"""
    recipe = pretrain()
    recipe.optim.config.lr = 1e-6
    recipe.data = multimodal_datamodule()
    recipe.model.config = run.Config(DiTConfig)
    recipe.trainer.max_steps = 1000
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True
    recipe.resume.restore_config = run.Config(RestoreConfig)
    recipe.resume.resume_if_exists = False
    return recipe


if __name__ == "__main__":
    OOM_DEBUG = False
    if OOM_DEBUG:
        torch.cuda.memory._record_memory_history(
            True,
            # Keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # Record stack information for the trace events
            trace_alloc_record_context=True,
        )
    run.cli.main(llm.train, default_factory=dreambooth)
