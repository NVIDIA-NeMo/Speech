# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import lightning.pytorch as pl
import nemo_run as run
from scripts.flux.flux_controlnet_training import unit_test as flux_control_training_unit_test
from scripts.flux.flux_training import unit_test as flux_training_unit_test

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_mock_datamodule import MockDataModule
from nemo.collections.diffusion.models.flux.model import FluxConfig, FluxModelParams, MegatronFluxModel, T5Config
from nemo.collections.diffusion.models.flux_controlnet.model import FluxControlNetConfig, MegatronFluxControlNetModel
from nemo.collections.llm.recipes.log.default import tensorboard_logger


@run.cli.factory
@run.autoconvert
def flux_mock_datamodule() -> pl.LightningDataModule:
    """Mock Datamodule Initialization"""
    data_module = MockDataModule(
        image_h=1024,
        image_w=1024,
        micro_batch_size=1,
        global_batch_size=1,
        image_precached=False,
        text_precached=False,
    )
    return data_module


@run.cli.factory(target=llm.train, name='flux_controlnet_training_fsdp_test')
def flux_controlnet_training_fsdp() -> run.Partial:
    """Flux Controlnet Training Config with FSDP"""
    recipe = flux_control_training_unit_test(megatron_fsdp=True)
    recipe.trainer.num_sanity_val_steps = 0
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.max_steps = 10
    recipe.trainer.callbacks = [
        run.Config(
            nl.ModelCheckpoint,
            save_last=False,
        )
    ]
    recipe.log = run.Config(
        nl.NeMoLogger,
        ckpt=None,
        name='flux_controlnet_training_test',
        tensorboard=tensorboard_logger(name='flux_controlnet_training'),
        log_dir='/tmp/flux_controlnet_training',
    )
    return recipe


@run.cli.factory(target=llm.train, name='flux_controlnet_training_ddp_test')
def flux_controlnet_training_ddp(
    flux_num_joint_layers=1,
    flux_num_single_layers=1,
    flux_controlnet_num_joint_layers=1,
    flux_controlnet_num_single_layers=1,
) -> run.Partial:
    """Flux Controlnet Training Config with DDP"""
    recipe = flux_control_training_unit_test(megatron_fsdp=False)
    recipe.model = run.Config(
        MegatronFluxControlNetModel,
        flux_params=run.Config(
            FluxModelParams,
            flux_config=run.Config(
                FluxConfig,
                num_joint_layers=flux_num_joint_layers,
                num_single_layers=flux_num_single_layers,
            ),
            t5_params=run.Config(T5Config, load_config_only=True),
        ),
        flux_controlnet_config=run.Config(
            FluxControlNetConfig,
            num_joint_layers=flux_controlnet_num_joint_layers,
            num_single_layers=flux_controlnet_num_single_layers,
        ),
    )
    recipe.trainer.num_sanity_val_steps = 0
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.max_steps = 10
    recipe.trainer.callbacks = [
        run.Config(
            nl.ModelCheckpoint,
            save_last=False,
        )
    ]
    recipe.data = flux_mock_datamodule()
    recipe.log = run.Config(
        nl.NeMoLogger,
        ckpt=None,
        name='flux_controlnet_training_test',
        tensorboard=tensorboard_logger(name='flux_controlnet_training'),
        log_dir='/tmp/flux_controlnet_training',
    )
    return recipe


@run.cli.factory(target=llm.train, name='flux_training_ddp_test')
def flux_training_ddp(
    flux_num_joint_layers=1,
    flux_num_single_layers=1,
) -> run.Partial:
    """Flux Training Config with DDP"""
    recipe = flux_training_unit_test(megatron_fsdp=False)
    recipe.data = flux_mock_datamodule()
    recipe.model = run.Config(
        MegatronFluxModel,
        flux_params=run.Config(
            FluxModelParams,
            flux_config=run.Config(
                FluxConfig,
                num_joint_layers=flux_num_joint_layers,
                num_single_layers=flux_num_single_layers,
            ),
            t5_params=run.Config(T5Config, load_config_only=True),
        ),
    )
    recipe.trainer.num_sanity_val_steps = 0
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.max_steps = 10
    recipe.trainer.callbacks = [
        run.Config(
            nl.ModelCheckpoint,
            save_last=False,
        )
    ]
    recipe.log = run.Config(
        nl.NeMoLogger,
        ckpt=None,
        name='flux_training_test',
        tensorboard=tensorboard_logger(name='flux_training'),
        log_dir='/tmp/flux_training',
    )
    return recipe


@run.cli.factory(target=llm.train, name='flux_training_fsdp_test')
def flux_training_fsdp() -> run.Partial:
    """Flux Training Config with custom FSDP"""
    recipe = flux_training_unit_test(megatron_fsdp=True)
    recipe.trainer.num_sanity_val_steps = 0
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.max_steps = 10
    recipe.trainer.callbacks = [
        run.Config(
            nl.ModelCheckpoint,
            save_last=False,
        )
    ]
    recipe.log = run.Config(
        nl.NeMoLogger,
        ckpt=None,
        name='flux_training_test',
        tensorboard=tensorboard_logger(name='flux_training'),
        log_dir='/tmp/flux_training',
    )
    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=flux_controlnet_training_ddp)
