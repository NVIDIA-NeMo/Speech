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

import gc
from typing import Any

import pytorch_lightning as pl
from nemo.utils import logging


class GarbageCollectionCallback(pl.Callback):
    """Callback for synchronized manual Garbage Collection. This is required for distributed training
    as all processes on different rank need to synchronize to garbage collect at the same time, without which
    one process might hog or straggle all the rest of the processes.

    Migration from NeMo 1.0:
        When mitrating from NeMo1,
            - gc_interval = 0 implied no GC, simply do not add this callback to the trainer
            - gc_interval > 0, this config is maps => gc_interval_train

            - env-var:NEMO_MANUAL_GC_IN_VALIDATION=0 or doesn't exist => Set gc_interval_val to a very high value that it does not practically run.
            - env-var:NEMO_MANUAL_GC_IN_VALIDATION=1 => Set gc_interval_val to the same value as gc_interval

        Moving from boolean flag (NEMO_MANUAL_GC_IN_VALIDATION) to integer is to allow user to set a specific value based on the size of the
        validation datasets.

    Note: This callback does not run gc at the start or the end of training or validation.
    """

    def __init__(self, gc_interval_train, gc_interval_val) -> None:
        """_summary_

        Args:
            gc_interval (int, mandatory): Number of global train steps at which garbage collection is done.
            gc_interval_val (int, mandatory): Number of global validation steps at which garbage collection is done.
        """
        assert gc_interval_train > 0, "gc_interval_train should be an integer value larger than 0."
        assert gc_interval_val > 0, "gc_interval_val should be an integer value larger than 0."

        super().__init__()
        self.gc_interval_train = gc_interval_train
        self.gc_interval_val = gc_interval_val
        # As garbage collection is manually controlled, disable automatic garbage collector.
        gc.disable()
        # This counter is required as pl does not have a native way to track the validation step counter.
        self.validation_global_step = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.gc_interval_train == 0:
            logging.info(f"Running garbage collection at train global_step: {trainer.global_step}")
            gc.collect()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.validation_global_step += 1
        if self.validation_global_step % self.gc_interval_val == 0:
            logging.info(f"Running garbage collection at validation step: {self.validation_global_step}")
            gc.collect()
