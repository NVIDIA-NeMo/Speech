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
import multiprocessing as mp
import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict

from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset, DuplexSTTModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Set multiprocessing start method to 'spawn' for CUDA compatibility with DataLoader workers
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Start method already set

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="s2s_duplex_stt")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    # avoid using `=` in the checkpoint name
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.CHECKPOINT_EQUALS_CHAR = "-"

    with trainer.init_module():
        model = DuplexSTTModel(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = DuplexSTTDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        aug_by_swap_role=cfg.data.get("aug_by_swap_role", False),
        cfg=cfg.data,
        model_cfg=cfg.model,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
