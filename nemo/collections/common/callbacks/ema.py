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
import os.path
import warnings
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nemo.utils import logging

try:
    import amp_C

    apex_available = True
except Exception:
    apex_available = False


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps: Apply EMA every n global steps.
        start_step: Start applying EMA from ``start_step`` global step onwards.
        evaluate_ema_weights_instead: Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the validation metrics are calculated with the EMA weights.
        save_ema_weights_in_callback_state: Enable saving ema weights in callback state.
            This is not required when using NeMo as the experiment manager handles saving weights.
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        if not apex_available:
            rank_zero_warn(
                "EMA has better performance when Apex is installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logging.info('Creating EMA weights copy.')
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.state_dict().values()]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        if apex_available and pl_module.device.type == "cuda":
            return self.apply_multi_tensor_ema(pl_module)
        return self.apply_ema(pl_module)

    def apply_multi_tensor_ema(self, pl_module: "pl.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        amp_C.multi_tensor_axpby(
            65536,  # todo (sean): chunk size, should we expose?
            self._overflow_buf,
            [self._ema_model_weights, model_weights, self._ema_model_weights],
            self.decay,
            1 - self.decay,
            -1,
        )

    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._ema_model_weights):
            diff = ema_weight.data - orig_weight.data
            diff.mul_(1.0 - self.decay)
            ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cur_step = state_dict['cur_step']
        # when loading using NeMo, ema weights will be loaded by the experiment manager separately.
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get('ema_weights')

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None and 'NeMo' in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f'-EMA{ext}'):
                logging.info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = trainer.ckpt_path.replace(ext, f'-EMA{ext}')
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device('cpu'))
                self._ema_model_weights = ema_state_dict['state_dict'].values()
                del ema_state_dict
                logging.info("EMA weights have been loaded successfully. Continuing training with saved EMA weights.")
            else:
                warnings.warn(
                    "we were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        self._weights_buffer = [p.detach().clone().to('cpu') for p in pl_module.state_dict().values()]
        new_state_dict = {k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)}
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)
