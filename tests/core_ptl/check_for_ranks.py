# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import shutil

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only

from nemo.core import ModelPT
from nemo.utils import logging
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg, trainer=kwargs.get('trainer', None))
        # dummy parameter in order to allow DDP to execute
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def predict_dataloader(self):
        dataset = OnesDataset(2)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def forward(self, batch):
        return batch.mean()

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.validation_step_outputs.append(loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self(batch)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack(self.validation_step_outputs).mean())
        self.validation_step_outputs.clear()  # free memory


def instantiate_multinode_ddp_if_possible():
    num_gpus = torch.cuda.device_count()
    ## Change logger=None to logger=False to support PTL 2.0
    trainer = Trainer(devices=num_gpus, accelerator='gpu', strategy='ddp', logger=False, enable_checkpointing=False)
    exp_manager_cfg = ExpManagerConfig(exp_dir='./ddp_check/', use_datetime_version=False, version="")
    exp_manager(trainer, cfg=OmegaConf.structured(exp_manager_cfg))
    return trainer


def setup_model(trainer: Trainer):
    model = ExampleModel(trainer=trainer)

    logging.info(f"M.Global Rank:{model.global_rank}")
    logging.info(f"M.Local Rank:{model.local_rank}")
    logging.info(f"M.World Size:{model.trainer.world_size}")

    trainer.predict(model)
    return model


def get_rank_info(texts: list, rank_key: str) -> int:
    for line in texts:
        if rank_key in line:
            rank_value = line.split(":")[-1]
            rank_value = int(rank_value)
            return rank_value

    print("Could not find the correct rank key !")
    exit(1)


@rank_zero_only
def check_model_ranks(model: ExampleModel):
    basedir = os.path.join('./ddp_check/', 'default', 'version_0')
    file_template = "nemo_log_globalrank-{rank}_localrank-{rank}.txt"

    world_size = torch.cuda.device_count()
    for rank in range(world_size):
        filename = file_template.format(rank=rank)
        filepath = os.path.join(basedir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            texts = [t.replace("\n", "") for t in texts]

        log_global_rank = get_rank_info(texts, rank_key='M.Global Rank')
        log_world_size = get_rank_info(texts, rank_key='M.World Size')

        if log_global_rank != rank:
            print("Logged global rank is not equal to trainer.global_rank !")
            exit(1)

        if log_world_size != world_size:
            print("Logged world size if not equal to trainer.world_size !")
            exit(1)


@rank_zero_only
def cleanup():
    if os.path.exists('./ddp_check'):
        shutil.rmtree('./ddp_check', ignore_errors=True)


def run_checks():
    cleanup()

    trainer = instantiate_multinode_ddp_if_possible()
    model = setup_model(trainer)
    check_model_ranks(model)

    print("DDP checks passed !")

    cleanup()


if __name__ == '__main__':
    run_checks()
