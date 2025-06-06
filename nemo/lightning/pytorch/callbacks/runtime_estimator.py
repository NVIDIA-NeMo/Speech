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

import time
from typing import Any, Optional

import lightning.pytorch as pl

from nemo.utils import logging


class RuntimeEstimator(pl.Callback):
    """
    Estimates total training time.

    The training time is computed by taking the time elapsed for the current duration and multiplying
    out to the full extended length of the training run.

    This callback provides a best attempt estimate. This estimate may be inaccurate if throughput
    changes through training or other significant changes are made to the model or dataloader.

    Example:
        import nemo_run as run
        from nemo.lightning.pytorch.callbacks import RuntimeEstimator

        recipe.trainer.callbacks.append(
            run.Config(RuntimeEstimator)
        )

    +-----------------------------+-------------------------------+
    | Key                         | Logged data                   |
    +=============================+===============================+
    | `time/remaining_estimate`   | Estimated time to completion  |
    +-----------------------------+-------------------------------+
    | `time/tokens`               | Number of consumed tokens     |
    +-----------------------------+-------------------------------+
    | `time/samples`              | Number of consumed samples    |
    +-----------------------------+-------------------------------+
    | `time/batches`              | Number of consumed batches    |
    +-----------------------------+-------------------------------+
    | `time/total`                | Total training time           |
    +-----------------------------+-------------------------------+

    Args:
        time_unit (str, optional): Time unit to use for `time` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(self, time_unit: str = 'hours') -> None:
        self._enabled = True
        self.start_time = None
        self.start_dur = None

        self.time_unit = time_unit
        self.divider = 1
        if time_unit == 'seconds':
            self.divider = 1
        elif time_unit == 'minutes':
            self.divider = 60
        elif time_unit == 'hours':
            self.divider = 60 * 60
        elif time_unit == 'days':
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".',
            )

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.eval_wct_per_label: dict[str, list[float]] = {}
        # How often eval is called as fraction of total training time
        self.eval_frequency_per_label: dict[str, float] = {}
        self.num_tokens: int = 0
        self.num_samples: int = 0
        self.num_batches: int = 0

    def _get_elapsed_duration(self, trainer) -> Optional[float]:
        """Get the elapsed duration.

        This method computes fractional progress in an epoch
        provided at least 1 epoch has passed by recording how many batches were in each epoch.
        """
        # Use max_steps if defined
        if trainer.max_steps and trainer.max_steps != -1:
            if trainer.global_step is not None:
                return trainer.global_step / trainer.max_steps
            return None

        # Use max_epochs if defined
        if trainer.max_epochs and trainer.max_epochs != -1:
            if trainer.current_epoch >= 1:
                if trainer.num_training_batches is not None:
                    batches_per_epoch = trainer.num_training_batches
                    total_batches = trainer.max_epochs * batches_per_epoch
                    return trainer.global_step / total_batches
            elif trainer.num_training_batches is not None:
                total_batches = trainer.max_epochs * trainer.num_training_batches
                return trainer.global_step / total_batches

        # Fallback: estimate from steps if available
        if trainer.estimated_stepping_batches:
            return trainer.global_step / trainer.estimated_stepping_batches

        return None

    def on_train_start(self, trainer, pl_module):
        """ """
        self.start_train_time = time.time()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
    ) -> None:
        """ """
        if self._enabled and self.start_time is None:
            self.start_time = time.time()
            self.start_dur = self._get_elapsed_duration(trainer)
            if self.start_dur is None:
                logging.warning('`max_duration` is not set. Cannot estimate remaining time.')
                self._enabled = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """ """
        if not self._enabled:
            return

        elapsed_dur = self._get_elapsed_duration(trainer)
        assert elapsed_dur is not None, 'elapsed_dur should be not None. Please, make sure that training has started.'

        assert self.start_dur is not None
        assert self.start_time is not None

        time_metrics = {}
        if elapsed_dur > self.start_dur:
            elapsed_time = time.time() - self.start_time
            elapsed_time -= self.total_eval_wct  # Subtract time spent evaluating
            rate = elapsed_time / (elapsed_dur - self.start_dur)
            remaining_time = rate * (1 - elapsed_dur)

            # Add remaining time from each evaluator using known frequencies. We explicitly compute
            # frequency instead of using time interpolation to avoid saw tooth pattern in estimates
            for dataloader_label, eval_wcts in self.eval_wct_per_label.items():
                # Discard first eval_wct if possible as it is often slower due to dataset downloading
                eval_wct_avg = None
                num_evals_finished = len(eval_wcts)
                if num_evals_finished > 1:
                    eval_wct_avg = sum(eval_wcts[1:]) / (num_evals_finished - 1)
                else:
                    eval_wct_avg = sum(eval_wcts) / num_evals_finished
                eval_rate = self.eval_frequency_per_label[dataloader_label]
                num_total_evals = 1 / eval_rate * (1 - self.start_dur)
                remaining_calls = num_total_evals - num_evals_finished
                remaining_time += eval_wct_avg * remaining_calls

            time_metrics['time/remaining_estimate'] = remaining_time / self.divider

        batch_size = trainer.train_dataloader.batch_sampler.global_batch_size
        self.num_tokens += batch['tokens'].size()[1] * (batch_size)
        time_metrics['time/tokens'] = self.num_tokens
        self.num_samples += batch_size
        time_metrics['time/samples'] = self.num_samples
        self.num_batches += 1
        time_metrics['time/batches'] = self.num_batches
        time_metrics['time/total'] = (time.time() - self.start_train_time) / self.divider

        for metric, value in time_metrics.items():
            self.log(metric, value)
