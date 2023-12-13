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
from multiprocessing import Value

import torch

from nemo.utils import logging

try:
    from webdataset.pytorch import IterableDataset

except (ImportError, ModuleNotFoundError):
    from nemo.core.classes import IterableDataset

    logging.warning("Webdataset import failed! We recommend use `webdataset==0.2.48`.")


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class WDSUrlsRandomSampler(IterableDataset):
    def __init__(
        self,
        urls,
        total_urls: int,
        chunk_size: int,
        consumed_samples: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        num_workers: int,
        drop_last: bool,
        data_sharding: bool,
    ):
        r"""Sampler for WebDataset Urls with data parallelism.
        Args:
            urls : The urls of the tar files from which to sample.
            total_urls (int): Total number of urls in the dataset.
            chunk_size (int): Number of objects per tar file.
            consumed_samples (int): Number of samples consumed so far by the training process.
                **Note samples here is not urls.**
            data_parallel_rank (int): Rank of the current data parallel process.
            data_parallel_size (int): Number of data parallel processes.
            drop_last (bool): If True, drop the remaining urls if the number is smaller than `data_parallel_size`.
                If False, pad the urls until its size is divisible by `data_parallel_size`.
            data_sharding (bool): If True, use data sharding before data shuffling, i.e. only shuffle within the data parallel group.
        """
        super().__init__()
        self.urls = urls
        self.total_urls = total_urls
        self.chunk_size = chunk_size

        if consumed_samples % data_parallel_size == 0:
            logging.warning("Multimodal data resuming will be approximate!")
        self.consumed_urls = (
            consumed_samples // (data_parallel_size * num_workers) // chunk_size * (data_parallel_size * num_workers)
        )
        self.consumed_samples = self.consumed_urls * chunk_size

        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.drop_last = drop_last
        self.data_sharding = data_sharding
        self.epoch = SharedEpoch()

        self.remaining_urls = self.total_urls % self.data_parallel_size

    def __len__(self):
        if self.drop_last:
            return self.total_urls // self.data_parallel_size
        else:
            return (self.total_urls + self.data_parallel_size - 1) // self.data_parallel_size

    def __iter__(self):
        worker_id, num_workers = 0, 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        self.consumed_urls = (
            self.consumed_samples
            // (self.data_parallel_size * num_workers)
            // self.chunk_size
            * (self.data_parallel_size * num_workers)
        )

        if self.drop_last or self.remaining_urls == 0:
            active_total_urls = self.total_urls - self.remaining_urls
        else:
            active_total_urls = self.total_urls + self.data_parallel_size - self.remaining_urls

        self.epoch.set_value(self.consumed_urls // active_total_urls)
        current_epoch_urls = self.consumed_urls % active_total_urls

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = active_total_urls // self.data_parallel_size
            bucket_offset = current_epoch_urls // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch.get_value())
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = active_total_urls
            full_bucket_offset = current_epoch_urls
            g = torch.Generator()
            g.manual_seed(self.epoch.get_value())
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        # Use additional permutation to replace out-of-range indices when drop_last is False
        additional_random_idx = torch.randperm(self.total_urls, generator=g).tolist()
        for n, idx in enumerate(idx_range):
            self.consumed_samples += self.data_parallel_size * self.chunk_size
            if worker_info is not None and n % num_workers != worker_id:
                continue
            if idx < self.total_urls:
                yield dict(url=self.urls[idx])
            else:
                yield dict(url=self.urls[additional_random_idx[idx - self.total_urls]])
