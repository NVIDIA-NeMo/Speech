# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import sys

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.core.classes import ModelPT
from nemo.utils.exp_manager import exp_manager

try:
    # `ptl_resiliency` is included in `gwe_resiliency_pkg` package
    from ptl_resiliency import StragglerDetectionCallback

    HAVE_STRAGGLER_DET = True
except (ImportError, ModuleNotFoundError):
    HAVE_STRAGGLER_DET = False


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, log_dir, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg)
        pl.seed_everything(1234)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)
        self.log_dir = log_dir

    def on_train_start(self):
        super().on_train_start()
        rank = torch.distributed.get_rank()

    def train_dataloader(self):
        dataset = OnesDataset(1024 * 1024)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)

    def val_dataloader(self):
        dataset = OnesDataset(128 * 1024)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)

    def forward(self, batch):
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))
        return output

    def validation_step(self, batch, batch_idx):
        self.loss = self(batch)
        return self.loss

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self, *args, **kwargs):
        pass

    def setup_training_data(self, *args, **kwargs):
        pass

    def setup_validation_data(self, *args, **kwargs):
        pass

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack([self.loss]).mean())


@pytest.mark.skipif(not HAVE_STRAGGLER_DET, reason="requires resiliency package to be installed.")
class TestStragglerDetection:

    @pytest.mark.run_only_on('GPU')
    def test_prints_perf_scores(self, tmp_path):
        # Run dummy 1 rank DDP training
        # Training time is limited to 3 seconds and straggler reporting is set to 1 second
        # Check if there are straggler related logs in the captured log
        max_steps = 1_000_000
        tmp_path = tmp_path / "test_1"
        print("TMP PATH", tmp_path)

        trainer = pl.Trainer(
            strategy='ddp',
            devices=1,
            accelerator='gpu',
            enable_checkpointing=False,
            logger=False,
            max_steps=max_steps,
            val_check_interval=0.33,
        )
        exp_manager(
            trainer,
            {
                "max_time_per_run": "00:00:00:03",
                "explicit_log_dir": str(tmp_path),
                "create_checkpoint_callback": False,
                "create_straggler_detection_callback": True,
                "straggler_detection_params": {
                    "report_time_interval": 1.0,
                    "calc_relative_gpu_perf": True,
                    "calc_individual_gpu_perf": True,
                    "num_gpu_perf_scores_to_log": 1,
                },
            },
        )
        model = ExampleModel(log_dir=tmp_path)
        trainer.fit(model)

        # assume that NeMo logs are written into "nemo_log_globalrank-0_localrank-0.txt"
        rank0_log_content = None
        with open(tmp_path / "nemo_log_globalrank-0_localrank-0.txt") as f:
            rank0_log_content = f.read()

        assert "GPU relative performance" in rank0_log_content
        assert "GPU individual performance" in rank0_log_content
