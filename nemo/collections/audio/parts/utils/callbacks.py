# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import List, Optional, Type

import einops
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.wandb import WandbLogger

from nemo.utils import logging
from nemo.utils.decorators import experimental

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


def _get_logger(loggers: List[Logger], logger_type: Type[Logger]):
    for logger in loggers:
        if isinstance(logger, logger_type):
            if hasattr(logger, "experiment"):
                return logger.experiment
            else:
                return logger
    raise ValueError(f"Could not find {logger_type} logger in {loggers}.")


@experimental
class SpeechEnhancementLoggingCallback(Callback):
    """
    Callback which can log artifacts (eg. model predictions, graphs) to local disk, Tensorboard, and/or WandB.

    Args:
        data_loader: Data to log artifacts for.
        output_dir: Optional local directory. If provided, artifacts will be saved in output_dir.
        loggers: Optional list of loggers to use if logging to tensorboard or wandb.
        log_tensorboard: Whether to log artifacts to tensorboard.
        log_wandb: Whether to log artifacts to WandB.
    """

    def __init__(
        self,
        data_loader,
        data_loader_idx: int,
        loggers: Optional[List[Logger]] = None,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        sample_rate: int = 16000,
        max_utts: Optional[int] = None,
    ):
        self.data_loader = data_loader
        self.data_loader_idx = data_loader_idx
        self.loggers = loggers if loggers else []
        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb
        self.sample_rate = sample_rate
        self.max_utts = max_utts

        if log_tensorboard:
            logging.info('Creating tensorboard logger')
            self.tensorboard_logger = _get_logger(self.loggers, TensorBoardLogger)
        else:
            logging.debug('Not using tensorbord logger')
            self.tensorboard_logger = None

        if log_wandb:
            if not HAVE_WANDB:
                raise ValueError("Wandb not installed.")
            logging.info('Creating wandb logger')
            self.wandb_logger = _get_logger(self.loggers, WandbLogger)
        else:
            logging.debug('Not using wandb logger')
            self.wandb_logger = None

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tlog_tensorboard: %s', self.log_tensorboard)
        logging.debug('\tlog_wandb:       %s', self.log_wandb)

    def _log_audio(self, audios: torch.Tensor, lengths: torch.Tensor, step: int, label: str = "input"):

        num_utts = audios.size(0)
        for audio_idx in range(num_utts):
            length = lengths[audio_idx]
            if self.tensorboard_logger:
                self.tensorboard_logger.add_audio(
                    tag=f"{label}_{audio_idx}",
                    snd_tensor=audios[audio_idx, :length],
                    global_step=step,
                    sample_rate=self.sample_rate,
                )

            if self.wandb_logger:
                wandb_audio = (
                    wandb.Audio(audios[audio_idx], sample_rate=self.sample_rate, caption=f"{label}_{audio_idx}"),
                )
                self.wandb_logger.log({f"{label}_{audio_idx}": wandb_audio})

    def on_validation_epoch_end(self, trainer: Trainer, model: LightningModule):
        """Log artifacts at the end of an epoch."""
        epoch = 1 + model.current_epoch
        output_signal_list = []
        output_length_list = []
        num_examples_uploaded = 0

        logging.info(f"Logging processed speech for validation dataset {self.data_loader_idx}...")
        for batch in self.data_loader:
            if isinstance(batch, dict):
                # lhotse batches are dictionaries
                input_signal = batch['input_signal']
                input_length = batch['input_length']
                target_signal = batch.get('target_signal', input_signal.clone())
            else:
                input_signal, input_length, target_signal, _ = batch

            if self.max_utts is None:
                num_examples = input_signal.size(0)  # batch size
                do_upload = True
            else:
                do_upload = num_examples_uploaded < self.max_utts
                num_examples = min(self.max_utts - num_examples_uploaded, input_signal.size(0))
                num_examples_uploaded += num_examples

            if do_upload:
                # Only pick the required numbers of speech to the logger
                input_signal = input_signal[:num_examples, ...]
                target_signal = target_signal[:num_examples, ...]
                input_length = input_length[:num_examples]

                # For consistency, the model uses multi-channel format, even if the channel dimension is 1
                if input_signal.ndim == 2:
                    input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
                if target_signal.ndim == 2:
                    target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

                input_signal = input_signal.to(model.device)
                input_length = input_length.to(model.device)

                output_signal, output_length = model(input_signal=input_signal, input_length=input_length)
                output_signal_list.append(output_signal.to(target_signal.device))
                output_length_list.append(output_length.to(target_signal.device))

        if len(output_signal_list) == 0:
            logging.debug('List are empty, no artifacts to log at epoch %d.', epoch)
            return

        output_signals = torch.concat(output_signal_list, dim=0)
        output_lengths = torch.concat(output_length_list, dim=0)
        if output_signals.size(1) != 1:
            logging.error(
                f"Currently only supports single-channel audio! Current output shape: {output_signals.shape}"
            )
            raise NotImplementedError

        output_signals = einops.rearrange(output_signals, "B 1 T -> B T")

        self._log_audio(
            audios=output_signals,
            lengths=output_lengths,
            step=model.global_step,
            label=f"dataloader_{self.data_loader_idx}_processed",
        )
