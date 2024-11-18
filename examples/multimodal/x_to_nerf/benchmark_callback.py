# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import time
from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer

from nemo.utils import logging


class BenchmarkCallback(Callback):
    def __init__(
        self,
        start_benchmark_at_step: int = 0,
        stop_benchmark_at_step: Optional[int] = None,
        log_every_n_steps: int = 10,
    ):
        super().__init__()
        self.start_benchmark_at_step = start_benchmark_at_step
        self.stop_benchmark_at_step = stop_benchmark_at_step
        self.log_every_n_steps = log_every_n_steps
        self.train_times = []
        self.val_times = []
        self.train_steps_times = []
        self.val_steps_times = []

    def should_benchmark(self, trainer: Trainer):
        if self.stop_benchmark_at_step is None:
            return trainer.global_step >= self.start_benchmark_at_step
        return self.start_benchmark_at_step <= trainer.global_step <= self.stop_benchmark_at_step

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.should_benchmark(trainer):
            epoch_time = time.time() - self.epoch_start_time
            self.train_times.append(epoch_time)
            logging.info(f'Training-Epoch-{trainer.current_epoch}-Time: {epoch_time} [sec]')

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int):
        self.step_start_time = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int):
        if self.should_benchmark(trainer):
            step_time = time.time() - self.step_start_time
            self.train_steps_times.append(step_time)
            if trainer.global_step % self.log_every_n_steps == 0:
                logging.info(f'Training-Step-{trainer.global_step}-Time: {step_time} [sec]')

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self.val_start_time = time.time()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.should_benchmark(trainer):
            val_time = time.time() - self.val_start_time
            self.val_times.append(val_time)
            logging.info(f'Validation-Epoch-{trainer.current_epoch}-Time: {val_time} [sec]')

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int, dataloader_idx: int
    ):
        self.val_step_start_time = time.time()

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int
    ):
        if self.should_benchmark(trainer):
            val_step_time = time.time() - self.val_step_start_time
            self.val_steps_times.append(val_step_time)
            if trainer.global_step % self.log_every_n_steps == 0:
                logging.info(f'Validation-Step-{trainer.global_step}-Time: {val_step_time} [sec]')

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.should_benchmark(trainer):
            avg_train_time = sum(self.train_times) / len(self.train_times)
            avg_val_time = sum(self.val_times) / len(self.val_times)
            avg_train_step_time = sum(self.train_steps_times) / len(self.train_steps_times)
            avg_val_step_time = sum(self.val_steps_times) / len(self.val_steps_times)

            logging.info(f'Average-Training-Epoch-Time: {avg_train_time} [sec]')
            logging.info(f'Average-Validation-Epoch-Time: {avg_val_time} [sec]')
            logging.info(f'Average-Training-Step-Time: {avg_train_step_time} [sec]')
            logging.info(f'Average-Validation-Step-Time: {avg_val_step_time} [sec]')
