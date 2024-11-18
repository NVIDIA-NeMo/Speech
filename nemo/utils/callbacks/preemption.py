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

import signal
import sys

import torch
from lightning.pytorch.callbacks import Callback

from nemo.utils import logging


class PreemptionCallback(Callback):
    """
    PreemptionCallback class creates a callback that checks for preemption during training at the end of every step.
    Upon preemption the callback provides a function to gracefully exit the training immediately and also saves the current state in a checkpoint as *last.ckpt.
    (to be able to start from the same step without wasting any compute while resuming the next time).

    PreemptionCallback is always enabled by default via the arg create_preemption_callback under ExpManagerConfig. To disable please pass
    create_preemption_callback: False in your config file.
    """

    def __init__(self, checkpoint_callback, sig=None):
        self.sig = sig
        if self.sig is None:
            self.sig = signal.SIGTERM
        self.checkpoint_callback = checkpoint_callback
        self.preemption_enabled = False

    @property
    def interrupted(self):
        interrupted = torch.tensor(self._interrupted, device=torch.cuda.current_device(), dtype=torch.int32)
        torch.distributed.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def on_train_start(self, trainer, pl_module):
        """
        Defines custom handlers at the beginning of training to be executed when the
        preemption signal is received.
        """

        # Check if torch distributed is initialised, as its needed for broadcasting the preemption signal to all the ranks
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            logging.info("Preemption requires torch distributed to be initialized, disabling preemption")
        else:
            self.preemption_enabled = True
            # Bool var that's initialized to false and made True upon receving the preemption signal
            self._interrupted = False
            self.released = False
            self.original_handler = signal.getsignal(self.sig)

            # Master handler executed only by rank 0 when the preemption siganal is received, to avoid deadlock conditions
            def master_handler(signum, frame):
                self.release()
                self._interrupted = True

            # Handler executed by the non zero ranks
            def ignoring_handler(signum, frame):
                self.release()

            self.private_rank = torch.distributed.get_rank()
            if self.private_rank == 0:
                signal.signal(self.sig, master_handler)
            else:
                signal.signal(self.sig, ignoring_handler)

        return self

    def on_train_end(self, trainer, pl_module):
        if self.preemption_enabled:
            self.release()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        if self.preemption_enabled:
            # check if the job was preempted at the end of every training step/iteration
            # NOTE: "self.interrupted" is a property which triggers a
            # distributed broadcast of "_interrupted" flag from rank 0 to all other
            # ranks, to avoid performance overheads it's best to store the result in
            # a regular local variable
            interrupted = self.interrupted
            if interrupted:
                logging.info("Received SIGTERM, saving checkpoint and exiting")
                monitor_candidates = self.checkpoint_callback._monitor_candidates(trainer)
                self.checkpoint_callback._save_last_checkpoint(trainer, monitor_candidates)
                sys.exit(0)

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
