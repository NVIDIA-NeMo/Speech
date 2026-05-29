# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from copy import deepcopy

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.speechlm2 import DataModule, StreamingSTTDataset, StreamingSTTModel
from nemo.core.classes.common import Serialization
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="streaming_stt")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    dataset_cfg = cfg.data.dataset

    # Build the optional validation dataset config (train dataset config with
    # val_dataset_overrides applied on top, e.g. pinning a single chunk_size).
    # It is passed to BOTH the model and the DataModule: under online forced
    # alignment the model runs get_batch_data (so it needs the val config), while
    # without it the DataModule's val dataset __getitem__ runs get_batch_data.
    val_dataset_overrides = cfg.data.get("val_dataset_overrides", None)
    if val_dataset_overrides is not None:
        val_dataset_cfg = deepcopy(dataset_cfg)
        if not isinstance(val_dataset_cfg, DictConfig):
            val_dataset_cfg = OmegaConf.create(val_dataset_cfg)
        with open_dict(val_dataset_cfg):
            val_dataset_cfg.update(val_dataset_overrides)
    else:
        val_dataset_cfg = None

    forced_aligner_cfg = cfg.get("forced_aligner", None)
    if forced_aligner_cfg is not None:
        forced_aligner = Serialization.from_config_dict(forced_aligner_cfg)
        defer_get_batch = True
        logging.info(f"Using online forced alignment: {forced_aligner_cfg}")
    else:
        forced_aligner = None
        defer_get_batch = False

    with trainer.init_module():
        model = StreamingSTTModel(
            OmegaConf.to_container(cfg.model, resolve=True),
            forced_aligner=forced_aligner,
            data_cfg=dataset_cfg,
            val_data_cfg=val_dataset_cfg,
            dataset_cls=StreamingSTTDataset,
        )

    dataset = StreamingSTTDataset(cfg=dataset_cfg, tokenizer=model.tokenizer, defer_get_batch=defer_get_batch)

    if val_dataset_cfg is not None:
        val_dataset = StreamingSTTDataset(
            cfg=val_dataset_cfg, tokenizer=model.tokenizer, defer_get_batch=defer_get_batch
        )
    else:
        val_dataset = None
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset, val_dataset=val_dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
