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

import dataclasses
import logging
from typing import List, Literal, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from nemo.lightning.megatron_parallel import MegatronStep


class DataSampler:
    """Abstract interface for data sampling and dataloader transformation.

    Implementations can prepare state in ``setup`` and wrap/transform a
    ``torch.utils.data.DataLoader`` in ``transform_dataloader`` to inject the
    appropriate sampler for the active strategy.
    """

    def connect(self, trainer: pl.Trainer):
        """Attach the Lightning ``trainer`` to this sampler instance."""
        self.trainer = trainer

    def setup(self, global_rank: int) -> None:
        """Initialize any sampler-related state for the given ``global_rank``."""
        raise NotImplementedError()

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        """Transform the dataloader."""
        raise NotImplementedError()


class MegatronDataSampler(DataSampler):
    """Megatron-LM data sampler.

    Handles batch ramp-up, logging of consumed samples, and wiring Megatron's
    microbatch/global-batch calculations into NeMo Lightning training.
    """

    def __init__(
        self,
        seq_len: int,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        dataloader_type: Literal["single", "cyclic", "batch"] = "single",
        init_consumed_samples: int = 0,
        init_global_step: int = 0,
        output_log: bool = True,
        decoder_seq_len: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.decoder_seq_len = decoder_seq_len
        self.output_log = output_log
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.rampup_batch_size = rampup_batch_size
        self.dataloader_type = dataloader_type
        self.init_consumed_samples = init_consumed_samples
        self.prev_consumed_samples = self.init_consumed_samples
        self.if_first_step = 0
        self.prev_global_batch_size = None
        self.init_global_step = init_global_step

    def setup(self, global_rank: int) -> None:
        """Initialize Megatron microbatch calculator for this process."""
        from nemo.lightning.data import setup_microbatch_calculator

        setup_microbatch_calculator(global_rank, self.micro_batch_size, self.global_batch_size, self.rampup_batch_size)

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        """Wrap the dataloader with a Megatron-aware sampler.

        The sampler accounts for data-parallel rank/size, ramp-up schedule, and
        train/validation/test modes.
        """
        from megatron.core import parallel_state

        from nemo.lightning.data import add_megatron_sampler

        mode = getattr(dataloader, 'mode', 'train')

        data_parallel_rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        return add_megatron_sampler(
            dataloader,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            consumed_samples=self.init_consumed_samples if mode == 'train' else 0,
            dataloader_type=self.dataloader_type,
            drop_last=mode not in ["test", "predict"],  # don't drop the incomplete batch in test and predict methods
            dataloader_mode=mode,  # dataloader wrapped with nemo.lightning.data.WrappedDataLoader has mode attribute
            rank=data_parallel_rank,
            world_size=data_parallel_size,
        )

    def compute_consumed_samples(self, steps_since_resume=0) -> int:
        """Compute the number of consumed samples since training start or resume.

        If a ramp-up schedule is active, the value uses the previous and current
        global batch sizes. Otherwise it is derived from
        ``data_parallel_size * micro_batch_size * num_microbatches`` times the
        number of steps since resume.
        """
        from nemo.lightning.pytorch.strategies import MegatronStrategy
        from nemo.utils import AppState

        if not hasattr(self, "trainer") or not isinstance(self.trainer.strategy, MegatronStrategy):
            return 0

        app_state = AppState()
        if self.rampup_batch_size is not None:
            consumed_samples = self.prev_consumed_samples + self.if_first_step * self.current_global_batch_size
        else:
            consumed_samples = (
                self.init_consumed_samples
                + steps_since_resume * app_state.data_parallel_size * self.micro_batch_size * self.num_microbatches
            )

        return int(consumed_samples)

    # Megatron callbacks

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Inject Megatron step configuration such as sequence length and batch sizes."""
        return dataclasses.replace(
            step,
            seq_length=self.seq_len,
            micro_batch_size=self.micro_batch_size,
            num_microbatches=self.num_microbatches,
            decoder_seq_length=self.decoder_seq_len,
        )

    def on_megatron_microbatches_start(self, step: MegatronStep) -> None:
        """Trigger a validation/checkpoint boundary when global batch size changes.

        During batch-size ramp-up we stop the trainer at the boundary so that a
        checkpoint can be saved and validation can run with the new batch size.
        """
        if not step.trainer:
            return

        # do validation and save the checkpoint when gbs is changed
        if (
            self.rampup_batch_size is not None
            and self.prev_global_batch_size != self.current_global_batch_size
            and self.prev_global_batch_size
        ):
            step.trainer.should_stop = True

    def on_megatron_step_end(self, step: MegatronStep) -> None:
        """Log training metrics and update Megatron's microbatch calculator.

        Logs ``consumed_samples`` and ``global_batch_size`` (GPU-friendly) and
        updates Megatron's internal number of microbatches for the next step.
        """
        trainer = step.trainer
        pl_module = step.pl_module

        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        self.prev_global_batch_size = self.current_global_batch_size

        if step.step_i:
            consumed_samples = self.compute_consumed_samples(step.step_i + 1 - self.init_global_step)
            if self.output_log and trainer and getattr(trainer, "training", False):
                # You may need to turn off logging, for example when doing trainer.predict(model, data)
                # pl_module.log () will trigger pageable H2D Memcpy which stalls CPU. Use pin_memory=True to avoid it
                consumed_samples = (
                    consumed_samples
                    if (torch.is_tensor(consumed_samples) and consumed_samples.is_cuda)
                    else torch.tensor(consumed_samples, pin_memory=True).to("cuda", non_blocking=True)
                )
                pl_module.log(
                    'consumed_samples',
                    consumed_samples,
                    prog_bar=True,
                    batch_size=1,
                )

            self.prev_consumed_samples = consumed_samples

            update_num_microbatches(
                consumed_samples=consumed_samples,
                consistency_check=False,
            )
        if self.output_log and trainer:
            # You may need to turn off logging, for example when doing trainer.predict(model, data)
            current_global_batch_size = (
                self.current_global_batch_size
                if (torch.is_tensor(self.current_global_batch_size) and self.current_global_batch_size.is_cuda)
                else torch.tensor(self.current_global_batch_size, pin_memory=True).to("cuda", non_blocking=True)
            )
            pl_module.log(
                "global_batch_size",
                current_global_batch_size,
                prog_bar=True,
                batch_size=1,
            )
        self.if_first_step = 1

    @property
    def num_microbatches(self) -> int:
        """Return the current number of microbatches from Megatron."""
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        return get_num_microbatches()

    @property
    def current_global_batch_size(self) -> int:
        """Return the current effective global batch size (fallback to 1)."""
        try:
            from megatron.core.num_microbatches_calculator import get_current_global_batch_size

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_current_global_batch_size

        if get_current_global_batch_size():
            current_global_batch_size = get_current_global_batch_size()
        else:
            current_global_batch_size = 1

        return current_global_batch_size
