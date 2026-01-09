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

from collections import defaultdict
from typing import Any

from lightning.pytorch.callbacks.progress import ProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

from typing_extensions import override


class ProgressPrinter(ProgressBar):
    """
    Callback for logging progress in Megatron. Prints status in terms of global batches rather than microbatches.
    Recommended over MegatronProgressBar for non-interactive settings

    Args:
        log_interval (int): determines how frequently (in steps) to print the progress.
        skip_accumulate_metrics (list[str]): for all metrics in this list, value logged will
            simply reflect the latest value rather than averaging over the log interval.
        exclude_metrics (list[str]): any metrics to exclude from logging.
    """

    def __init__(
        self,
        log_interval: int = 1,
        skip_accumulate_metrics: list[str] = ["global_step"],
        exclude_metrics: list[str] = ["v_num"],
    ):
        self._train_description = "Training"
        self._validation_description = "Validation"
        self._test_description = "Testing"
        self._log_interval = int(log_interval)
        # most recent "global_step" will be logged
        # rather than averaging over last log_interval steps
        self.skip_accumulate_metrics = skip_accumulate_metrics
        self.exclude_metrics = exclude_metrics
        self.total_metrics_dict = defaultdict(lambda: 0.0)
        self._is_disabled = log_interval <= 0

        super().__init__()

    def format_string(self, prefix, metrics):
        log_string = prefix
        for metric, val in metrics.items():
            if isinstance(val, (float)) and val.is_integer():
                val = int(val)
                log_string += f' | {metric}: {val}'
            else:
                log_string += f' | {metric}: {val:.4}'
        return log_string

    def disable(self):
        self._is_disabled = True

    def enable(self):
        self._is_disabled = False

    @property
    def is_disabled(self) -> bool:
        return self._is_disabled

    @property
    def average_metrics_dict(self):
        average_dict = {}
        for key in self.total_metrics_dict:
            if key in self.skip_accumulate_metrics or not isinstance(self.total_metrics_dict[key], (int, float)):
                average_dict[key] = self.total_metrics_dict[key]
            else:
                average_dict[key] = self.total_metrics_dict[key] / self.log_interval
        return average_dict

    @property
    def train_description(self):
        return self._train_description

    @property
    def validation_description(self):
        return self._validation_description

    @property
    def test_description(self):
        return self._test_description

    @property
    def log_interval(self):
        return self._log_interval

    @log_interval.setter
    def log_interval(self, val):
        self._log_interval = val

    @override
    def on_sanity_check_start(self, *_: Any) -> None:
        self._validation_description = "Sanity checking " + self.validation_description

    @override
    def on_sanity_check_end(self, *_: Any) -> None:
        self._validation_description = "Validation"

    @override
    def on_train_start(self, trainer, *_):
        if trainer.max_steps > 0:
            # while resuming from a ckpt use trainer.max_steps as the total for progress bar as trainer.num_training_batches
            # is truncated to max_steps - step being resumed at
            self.total = trainer.max_steps
        else:
            self.total = trainer.num_training_batches

    ## TODO(ashors): handle nan losses
    @override
    def on_train_batch_end(self, trainer, pl_module, *_, **__):
        n = trainer.strategy.current_epoch_step

        if self.is_disabled:
            return

        metrics = self.get_metrics(trainer, pl_module)
        for key in metrics:
            if key in self.exclude_metrics:
                continue
            if key in self.skip_accumulate_metrics or not isinstance(metrics[key], (int, float)):
                self.total_metrics_dict[key] = metrics[key]
            else:
                self.total_metrics_dict[key] += metrics[key]

        if self.should_log(n):
            prefix = self.train_description + f" epoch {trainer.current_epoch}, iteration {n-1}/{self.total-1}"
            log_string = self.format_string(prefix, self.average_metrics_dict)
            print(log_string)
            if getattr(trainer.strategy, "timers", None):
                timers = trainer.strategy.timers
                megatron_log_string = self.log_megatron_timers(timers)

                if megatron_log_string:
                    print(megatron_log_string, flush=True)

            self.total_metrics_dict = defaultdict(lambda: 0.0)

    @override
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        if float(self.total_val_batches_current_dataloader) == float('inf'):
            self.total_validation_steps = float('inf')
        else:
            self.total_validation_steps = int(self.total_val_batches_current_dataloader / get_num_microbatches())

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled:
            return
        n = (batch_idx + 1) / get_num_microbatches()
        if self.should_log(n):
            print(self.validation_description + f": iteration {int(n)}/{self.total_validation_steps}")

    @override
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return
        self.total_test_steps = int(self.total_test_batches_current_dataloader / get_num_microbatches())

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled:
            return
        n = int((batch_idx + 1) / get_num_microbatches())
        if self.should_log(n):
            print(self.test_description + f": iteration {n}/{self.total_validation_steps}")

    def should_log(self, n):
        return n % self.log_interval == 0

    def log_megatron_timers(self, timers):
        output_string = timers.get_all_timers_string(names=None, normalizer=self.log_interval)
        if output_string is not None:
            return output_string + "\n"
        return None
