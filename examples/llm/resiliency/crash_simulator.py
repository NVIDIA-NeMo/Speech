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

"""Module for simulating crashes during training to test model resiliency.

This module provides a PyTorch Lightning callback that can simulate crashes at specified
training steps to test model checkpoint and recovery mechanisms.
"""

from typing import Any, Dict, Optional

from lightning.pytorch.callbacks import Callback
from nemo.utils import logging


class CrashSimulationCallback(Callback):
    """Callback that simulates a crash at a specified training step.

    This callback is useful for testing model checkpoint and recovery mechanisms
    by simulating crashes at predetermined points during training.

    Args:
        crash_step (int, optional): The training step at which to simulate a crash.
            Defaults to 17.
    """

    def __init__(self, crash_step: int = 17):
        self.crash_step = crash_step
        self.has_simulated_crash_happened = False
        logging.info(
            f"Setup to simulate a crash if step == {self.crash_step} " "and a crash hasn't been simulated before"
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[Any] = None,
        batch: Optional[Any] = None,
        batch_idx: Optional[int] = None,
    ) -> None:
        if self.crash_step and trainer.global_step == self.crash_step and not self.has_simulated_crash_happened:
            raise RuntimeError(f"Simulating a crash at step {self.crash_step}!")

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        if not self.has_simulated_crash_happened:
            self.has_simulated_crash_happened = True
            logging.info("Resuming from checkpoint, setting has_simulated_crash_happened to True!")
