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

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.data.diffusion_mock_datamodule import MockDataModule
from nemo.collections.diffusion.data.diffusion_taskencoder import RawImageDiffusionTaskEncoder
from nemo.collections.diffusion.models.flux.model import (
    ClipConfig,
    FluxConfig,
    FluxModelParams,
    MegatronFluxModel,
    T5Config,
)
from nemo.collections.diffusion.vae.autoencoder import AutoEncoderConfig
from nemo.utils.exp_manager import TimingCallback


@run.cli.factory
@run.autoconvert
def flux_datamodule(dataset_dir) -> pl.LightningDataModule:
    """Flux Datamodule Initialization"""
    data_module = DiffusionDataModule(
        dataset_dir,
        seq_length=4096,
        task_encoder=run.Config(
            RawImageDiffusionTaskEncoder,
        ),
        micro_batch_size=1,
        global_batch_size=8,
        num_workers=23,
        use_train_split_for_val=True,
    )
    return data_module


@run.cli.factory
@run.autoconvert
def flux_mock_datamodule() -> pl.LightningDataModule:
    """Mock Datamodule Initialization"""
    data_module = MockDataModule(
        image_h=1024,
        image_w=1024,
        micro_batch_size=1,
        global_batch_size=2,
        image_precached=True,
        text_precached=True,
    )
    return data_module


@run.cli.factory(target=llm.train)
def flux_training() -> run.Partial:
    """Flux Controlnet Training Config"""
    return run.Partial(
        llm.train,
        model=run.Config(MegatronFluxModel, flux_params=run.Config(FluxModelParams), seed=42),
        data=flux_mock_datamodule(),
        trainer=run.Config(
            nl.Trainer,
            devices=1,
            num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                gradient_accumulation_fusion=True,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    use_custom_fsdp=True,
                    data_parallel_sharding_strategy='optim_grads_params',
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_param_gather=True,
                    overlap_grad_reduce=True,
                ),
                fsdp='megatron',
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=1,
            val_check_interval=1000,
            max_epochs=10000,
            log_every_n_steps=1,
            callbacks=[
                run.Config(
                    nl.ModelCheckpoint,
                    monitor='global_step',
                    filename='{global_step}',
                    every_n_train_steps=1000,
                    save_top_k=3,
                    mode='max',
                    save_last=False,
                ),
                run.Config(TimingCallback),
            ],
        ),
        log=nl.NeMoLogger(wandb=(WandbLogger() if "WANDB_API_KEY" in os.environ else None)),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
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
def convergence_test(custom_fsdp=True) -> run.Partial:
    '''
    A convergence recipe with real data loader.
    Image and text embedding calculated on the fly.
    '''
    recipe = flux_training()
    recipe.model.flux_params.t5_params = run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = run.Config(
        AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1, 2, 4, 4], attn_resolutions=[]
    )
    recipe.model.flux_params.device = 'cuda'
    recipe.trainer.devices = 8
    recipe.data = flux_datamodule('/dataset/fill50k/fill50k_tarfiles/')
    recipe.trainer.max_steps = 30000
    if custom_fsdp is True:
        configure_custom_fsdp(recipe)
    else:
        configure_ddp(recipe)

    return recipe


@run.cli.factory(target=llm.train)
def full_model_tp2_dp4_mock() -> run.Partial:
    '''
    An example recipe uses tp 2 dp 4 with mock dataset.
    '''
    recipe = flux_training()
    recipe.model.flux_params.t5_params = None  # run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None  # run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = (
        None  # run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    )
    recipe.model.flux_params.device = 'cuda'
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.devices = 8
    recipe.data.global_batch_size = 8

    return recipe


@run.cli.factory(target=llm.train)
def fp8_test(custom_fsdp=True) -> run.Partial:
    '''
    Basic functional test, with mock dataset,
    text/vae encoders not initialized, ddp strategy,
    frozen and trainable layers both set to 1
    '''
    recipe = flux_training()
    recipe.trainer.devices = 1
    recipe.model.flux_params.t5_params = None  # run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None  # run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = (
        None  # run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    )
    recipe.model.flux_params.device = 'cuda'
    recipe.model.flux_params.flux_config = run.Config(
        FluxConfig,
        num_joint_layers=5,
        num_single_layers=10,
        calculate_per_token_loss=False,
        gradient_accumulation_fusion=False,
    )
    recipe.data.global_batch_size = 8
    if custom_fsdp:
        configure_custom_fsdp(recipe)
    else:
        configure_ddp(recipe)
    recipe.trainer.plugins = run.Config(
        nl.MegatronMixedPrecision,
        precision="bf16-mixed",
        fp8='hybrid',
        fp8_margin=0,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        fp8_params=False,
    )
    recipe.trainer.max_steps = 100
    return recipe


def configure_custom_fsdp(recipe) -> run.Partial:
    recipe.trainer.strategy.ddp = run.Config(
        DistributedDataParallelConfig,
        use_custom_fsdp=True,
        data_parallel_sharding_strategy='optim_grads_params',  # Custom FSDP
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_param_gather=True,  # Custom FSDP requires this
        overlap_grad_reduce=True,  # Custom FSDP requires this
    )
    recipe.trainer.strategy.fsdp = 'megatron'
    return recipe


def configure_ddp(recipe) -> run.Partial:
    recipe.trainer.strategy.ddp = run.Config(
        DistributedDataParallelConfig,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )
    recipe.trainer.strategy.fsdp = None
    return recipe


@run.cli.factory(target=llm.train)
def unit_test(custom_fsdp=True) -> run.Partial:
    '''
    Basic functional test, with mock dataset,
    text/vae encoders not initialized, ddp strategy,
    frozen and trainable layers both set to 1
    '''
    recipe = flux_training()
    if custom_fsdp:
        recipe = configure_custom_fsdp(recipe)
    else:
        recipe = configure_ddp(recipe)
    recipe.model.flux_params.t5_params = None  # run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None  # run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = (
        None  # run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    )
    recipe.model.flux_params.device = 'cuda'
    recipe.model.flux_params.flux_config = run.Config(
        FluxConfig,
        num_joint_layers=1,
        num_single_layers=1,
        calculate_per_token_loss=False,
        gradient_accumulation_fusion=False,
    )
    recipe.data.global_batch_size = 1
    recipe.trainer.max_steps = 100
    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=unit_test)
