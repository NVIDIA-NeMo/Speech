# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Dataloaders."""

import abc
from itertools import chain
from typing import Optional

import torch

from nemo.utils import logging


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
            raise RuntimeError("no sample to consume: {}".format(total_samples))
        if micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {micro_batch_size}")
        if data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise RuntimeError(
                "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                    data_parallel_rank, data_parallel_size
                )
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
                f"`pad_samples_to_global_batch_size` can be `True` only when "
                f"`global_batch_size` is set to an integer value"
            )

        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.global_batch_size = global_batch_size
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size

        logging.info(
            f'Instantiating MegatronPretrainingSampler with total_samples: {total_samples} and consumed_samples: {consumed_samples}'
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
    def __iter__(self):
        ...


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
            ), 'with pad_samples_to_global_batch_size all batches should be complete'
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
            pad_samples_to_global_batch_size == False
        ), "`MegatronPretrainingRandomSampler` does not support sample padding"
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

    def __len__(self):
        num_available_samples: int = self.total_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size

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
