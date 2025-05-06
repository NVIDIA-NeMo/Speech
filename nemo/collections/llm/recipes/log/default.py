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
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from nemo_run import Config, cli

from nemo import lightning as nl
from nemo.lightning.ckpt_utils import CONTEXT_PATH, WEIGHTS_PATH


def tensorboard_logger(name: str, save_dir: str = "tb_logs") -> Config[TensorBoardLogger]:
    """Factory function to configure TensorBoard Logger."""
    return Config(TensorBoardLogger, save_dir=save_dir, name=name)


def wandb_logger(project: str, name: str, entity: Optional[str] = None) -> Config[WandbLogger]:
    """Factory function to configure W&B Logger."""
    cfg = Config(
        WandbLogger,
        project=project,
        name=name,
        config={},
    )

    if entity:
        cfg.entity = entity

    return cfg


@cli.factory(is_target_default=True)
def default_log(
    dir: Optional[str] = None,
    name: str = "default",
    tensorboard_logger: Optional[Config[TensorBoardLogger]] = None,
    wandb_logger: Optional[Config[WandbLogger]] = None,
) -> Config[nl.NeMoLogger]:
    """Factory function to configure NemoLogger."""
    ckpt = Config(
        nl.ModelCheckpoint,
        save_last=True,
        save_top_k=10,
        train_time_interval=Config(timedelta, minutes=15),
        filename="{model_name}--{val_loss:.2f}-{step}-{consumed_samples}",
    )

    return Config(
        nl.NeMoLogger,
        ckpt=ckpt,
        name=name,
        tensorboard=tensorboard_logger,
        wandb=wandb_logger,
        log_dir=dir,
    )


@cli.factory(is_target_default=True)
def default_resume(resume_if_exists=True, resume_ignore_no_checkpoint=True) -> Config[nl.AutoResume]:
    """Factory function to configure AutoResume."""
    return Config(
        nl.AutoResume,
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=resume_ignore_no_checkpoint,
    )


def get_global_step_from_global_checkpoint_path(path: Union[str, Path]) -> int:
    """Extract global step based on formatted path.

    Args:
        path (Union(str, Path)): Directory name should be formatted as
            {model_name}--{val_loss:.2f}-{step}-{consumed_samples}
    """
    # Get directory name from path
    if isinstance(path, Path):
        # Get parent directory if the path is a subdirectory of the checkpoint directory
        if path.name == WEIGHTS_PATH or path.name == CONTEXT_PATH:
            dir_name = path.parent.name
        else:
            dir_name = path.name
    else:
        norm_path = os.path.normpath(path)
        # Get parent directory if the path is a subdirectory of the checkpoint directory
        if os.path.basename(norm_path) == WEIGHTS_PATH or os.path.basename(norm_path) == CONTEXT_PATH:
            dir_name = os.path.basename(os.path.dirname(norm_path))
        else:
            dir_name = os.path.basename(norm_path)

    # Format from above is {model_name}--{val_loss:.2f}-{step}-{consumed_samples}
    assert "--" in dir_name, f"Unexpected path format found for {path}"

    # Find the parts after '--'
    double_dash_index = dir_name.index('--')
    remaining = dir_name[double_dash_index + 2 :]  # Skip the '--' itself
    parts = remaining.split('-')
    assert len(parts) > 1, f"Unexpected path format found for {path}"
    # Global step should be at index 1
    step_part = parts[1]

    if "=" in step_part:
        # Handle "step=value" format (if metric name is auto-inserted)
        step_value = step_part.split("=")
        assert len(step_value) == 2, f"Expected 'step=<integer>' but found '{step_value}'. Full path: {path}"
        step_value = step_value[1]
        if not step_value.isdigit():
            raise ValueError(
                f"Could not parse the step number from '{step_part}'. "
                f"Expected 'step=<integer>' but found '{step_value}'. Full path: {path}"
            )
        step = int(step_value)
    else:
        # Handle plain value format (if metric name is not auto-inserted)
        if not step_part.isdigit():
            raise ValueError(f"Expected an integer but found '{step_part}'. Full path: {path}")
        step = int(step_part)

    return step
