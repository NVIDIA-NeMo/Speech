# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a DFlash or DFlash2 draft against a frozen audio-conditioned SALM target."""

import os
from datetime import timedelta

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils.callbacks.training_stats import TrainingStatsCallback
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(config_path="conf", config_name="dflash")
def train(cfg):
    """Build the frozen target, draft, and data module, then train or validate."""
    OmegaConf.resolve(cfg)
    if not cfg.model.get("use_nemo_automodel", False):
        raise ValueError("DFlash training requires model.use_nemo_automodel=true")
    from nemo.collections.speechlm2 import SALMAutomodel
    from nemo.collections.speechlm2.parts.dflash import SALMDFlashModule

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_kwargs = {}
        if timeout := _process_group_timeout(cfg):
            init_kwargs["timeout"] = timeout
        torch.distributed.init_process_group(backend="nccl", **init_kwargs)
    seed_everything(cfg.data.train_ds.seed)
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    # Save current data counters before a timer callback checkpoints the step.
    trainer.callbacks.insert(0, TrainingStatsCallback())
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        target = SALMAutomodel(OmegaConf.to_container(cfg.model, resolve=True))
    model = SALMDFlashModule(target, OmegaConf.to_container(cfg, resolve=True))
    # DFlash expands and packs audio inside SALMDFlashModule._prepare_batch.
    dataset = SALMDataset(
        tokenizer=target.tokenizer,
        multispeaker_cfg=cfg.data.get("multispeaker_cfg", None),
        batch_tokens=cfg.data.train_ds.get("batch_tokens", None),
    )
    datamodule = DataModule(cfg.data, tokenizer=target.tokenizer, dataset=dataset)
    if cfg.get("run_validate_only", False):
        trainer.validate(model, datamodule)
    else:
        trainer.fit(model, datamodule)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _process_group_timeout(cfg):
    """Match the target strategy timeout before initializing its default NCCL group."""
    minutes = OmegaConf.select(cfg, "trainer.strategy.timeout_minutes", default=None)
    if minutes is not None:
        return timedelta(minutes=float(minutes))
    seconds = os.environ.get("TORCH_NCCL_TIMEOUT_S") or os.environ.get("TORCH_NCCL_TIMEOUT_SEC")
    return timedelta(seconds=float(seconds)) if seconds is not None else None


if __name__ == "__main__":
    train()
