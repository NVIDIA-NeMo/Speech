import abc
import logging
import os
from itertools import chain
from typing import List, Literal, Optional

import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: "Dataset", drop_last: bool = True, pad_samples_to_global_batch_size=False, **kwargs
) -> DataLoader:
    output = DataLoader(dataset, collate_fn=dataset.collate_fn, **kwargs)

    output._drop_last = drop_last  # noqa: SLF001
    output._pad_samples_to_global_batch_size = pad_samples_to_global_batch_size  # noqa: SLF001

    return output


def setup_microbatch_calculator(
    global_rank: int,
    micro_batch_size: int,
    global_batch_size: int,
    rampup_batch_size: Optional[List[int]] = None,
) -> None:
    """
    Initializes the data for distributed training by setting up the microbatch calculator
    based on the provided global rank and data configuration.

    This function checks if the microbatch calculator has already been initialized. If it has,
    the function validates that the current configuration matches the initialized settings. If the
    calculator has not been initialized, it sets up a new one with the provided configuration.

    Args:
        global_rank (int): The global rank of the current process.
        config (DataConfig): The data configuration object containing settings for global batch size,
            micro batch size, data parallel size, and optional ramp-up batch size.

    Raises
    ------
        Exception: If the microbatch calculator has already been initialized with different settings.

    """
    from nemo.lightning._strategy_lib import NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE
    from nemo.utils import AppState

    app_state = AppState()

    if os.environ.get(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, "false").lower() == "true":
        init_global_rank = app_state.global_rank
    else:
        init_global_rank = global_rank

    from apex.transformer.microbatches import ConstantNumMicroBatches
    from apex.transformer.pipeline_parallel.utils import (
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR,
        setup_microbatch_calculator,
    )

    if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        setup_microbatch_calculator(
            rank=init_global_rank,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=app_state.data_parallel_size,
            rampup_batch_size=rampup_batch_size,
        )
    else:
        if isinstance(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, ConstantNumMicroBatches):
            assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.current_global_batch_size == global_batch_size
            assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size == micro_batch_size
            assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.num_micro_batches == global_batch_size // (
                micro_batch_size * app_state.data_parallel_size
            )
        else:
            raise Exception("Microbatch calculator already initialized.")


def add_megatron_sampler(
    dataloader: DataLoader,
    micro_batch_size: int,
    global_batch_size: int,
    rampup_batch_size: Optional[List[int]] = None,
    consumed_samples: int = 0,
    dataloader_type: Literal["single", "cyclic"] = "single",
    # data_sharding: bool = False
) -> DataLoader:
    from megatron.core import parallel_state

    if dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataloader.dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=getattr(dataloader, "_drop_last", False),
            pad_samples_to_global_batch_size=getattr(dataloader, "_pad_samples_to_global_batch_size", False),
        )
    elif dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataloader.dataset,
            total_samples=len(dataloader.dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            pad_samples_to_global_batch_size=getattr(dataloader, "_pad_samples_to_global_batch_size", False),
            # data_sharding=data_sharding
        )
    else:
        raise Exception(f'{dataloader_type} dataloader type is not supported.')

    return DataLoader(
        dataloader.dataset,
        batch_sampler=batch_sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        persistent_workers=dataloader.persistent_workers,
        collate_fn=dataloader.collate_fn,
    )


# TODO: Replace this with megatron.core.data.data_samplers after we upgrade
class BaseMegatronSampler:
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        rampup_batch_size: Optional[list] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        # Sanity checks.
        if total_samples <= 0:
            raise RuntimeError(f"no sample to consume: {total_samples}")
        if consumed_samples >= total_samples:
            raise RuntimeError(f"no samples left to consume: {consumed_samples}, {total_samples}")
        if micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {micro_batch_size}")
        if data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise RuntimeError(
                f"data_parallel_rank should be smaller than data size, but {data_parallel_rank} >= {data_parallel_size}"
            )
        if global_batch_size is not None and rampup_batch_size is None:
            if global_batch_size % (micro_batch_size * data_parallel_size) != 0:
                raise RuntimeError(
                    f"`global_batch_size` ({global_batch_size}) is not divisible by "
                    f"`micro_batch_size ({micro_batch_size}) x data_parallel_size "
                    f"({data_parallel_size})`"
                )
        if pad_samples_to_global_batch_size and global_batch_size is None:
            raise RuntimeError(
                "`pad_samples_to_global_batch_size` can be `True` only when "
                "`global_batch_size` is set to an integer value"
            )

        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.global_batch_size = global_batch_size
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size

        logging.info(
            f"Instantiating MegatronPretrainingSampler with total_samples: {total_samples} and"
            f" consumed_samples: {consumed_samples}"
        )

    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    @abc.abstractmethod
    def __iter__(self): ...


class MegatronPretrainingSampler(BaseMegatronSampler):
    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        indices = range(self.consumed_samples, self.total_samples)
        if (not self.drop_last) and self.pad_samples_to_global_batch_size:
            pad_samples_num = -len(indices) % self.global_batch_size
            pad_indices = range(-1, -pad_samples_num - 1, -1)
            indices = chain(indices, pad_indices)

        for idx in indices:
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            assert (
                not self.pad_samples_to_global_batch_size
            ), "with pad_samples_to_global_batch_size all batches should be complete"
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomSampler(BaseMegatronSampler):
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        assert (
            not pad_samples_to_global_batch_size
        ), "`MegatronPretrainingRandomSampler` does not support sample padding"
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch
