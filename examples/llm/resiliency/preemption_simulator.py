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

"""Module for simulating preemption during training to test model resiliency.

This module provides a PyTorch Lightning callback that can simulate preemption signals
at specified training steps to test model checkpoint and recovery mechanisms.
"""

import signal
from typing import Any, Optional

from lightning.pytorch.callbacks import Callback
from nemo.utils import logging


class PreemptionSimulationCallback(Callback):
    """Callback that simulates a preemption signal at a specified training step.

    This callback is useful for testing model checkpoint and recovery mechanisms
    by simulating preemption signals at predetermined points during training.

    Args:
        preemption_step (int, optional): The training step at which to simulate
            a preemption signal. Defaults to 4.
    """

    def __init__(self, preemption_step: int = 4):
        self.preemption_step = preemption_step
        logging.info(f"Setup to simulate a preemption if step == {self.preemption_step}")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[Any] = None,
        batch: Optional[Any] = None,
        batch_idx: Optional[int] = None,
    ) -> None:
        if self.preemption_step and trainer.global_step == self.preemption_step:
            logging.info(f"Simulating preemption by raising a SIGTERM at step {self.preemption_step}!")
            signal.raise_signal(signal.SIGTERM)
