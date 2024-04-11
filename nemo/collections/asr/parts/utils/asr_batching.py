# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset, AudioToCharDataset
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.utils import logging


class SemiSortBatchSampler(DistributedSampler):
    def __init__(
        self,
        global_rank: int,
        world_size: int,
        durations: List[int],
        batch_size: int,
        batch_shuffle: bool = True,
        drop_last: bool = False,
        randomization_factor: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        """
        Semi Sorted Batching, as proposed in _SSB ("Speed up training with variable 
        length inputs by efficient batching strategies.", Zhenhao Ge et al. (2021).).

        The Semi Sorted Batch Sampler (SSB) samples the indices by their duration 
        with the addition of pseudo noise that is sampled from the uniform 
        distribution \mathbb{U}\left[ -delta * r, delta * r \right], where delta is 
        defined as the difference between the maximum and minimum duration and r is 
        the randomization factor that controls the strength of the noise (when r = 0, 
        there will be a strong sorting). The heuristic value of the r according to 
        the experiments from paper is 0.2. 

        The torch calls the set_epoch method from the distributed data loader sampler 
        at the end of each epoch to shuffle the samples according to the seed and 
        epoch number. So the SSB is passed to the dataloader as a sampler with the 
        dataloader's batch size options and the batch_sampler option set to None to 
        disable automatical batching. In this case, the sampler has become an iterator 
        that returns a list of batch indices.

        Args:
            global_rank: Rank among all GPUs.
            world_size: The number of GPUs used.
            durations: Sample durations parsed from `dataset.manifest_processor`.
            batch_size: Micro batch size or batch size per singe gpu.
            batch_shuffle: Batch sort before each epoch.
            drop_last: Drop the last batch if the number of samples is less than batch 
                size. Defaults to False.
            randomization_factor: The strength of noise that will be added to the sample
                duration. If no value is passed, the value 0.2 will be used.
            seed: Seed for batch shuffleling. Defaults to 42.

        Raises:
            ValueError: Wrong randomization factor value.
            RuntimeError: Unexpected behavior.

        .. SSB_: 
            https://www.isca-speech.org/archive/pdfs/interspeech_2021/ge21_interspeech.pdf
        """
        if randomization_factor is None:
            randomization_factor = 0.1
            logging.info("Randomization factor not found in config, default value 0.1 will be set.")
        else:
            logging.info(f"A randomization factor {randomization_factor} will be used.")

        if randomization_factor < 0.0:
            raise ValueError(f'Randomization factor must be non-negative but found {randomization_factor}.')

        self.rank: List = global_rank
        self.num_replicas: int = world_size

        self.durations: np.array = np.array(durations, dtype=np.float32)

        self.shuffle: bool = batch_shuffle
        self.micro_batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.epoch: int = 0
        self.seed: int = seed
        self.randomization_factor: float = randomization_factor

        self.local_num_batches: int = self._calculate_local_num_batches()

        logging.info(f"Semi Sorted Batch Sampler will be used")

    def _calculate_local_num_batches(self) -> int:
        init_num_samples = len(self.durations)

        # delete batches with a non-integer number of samples
        if self.drop_last:
            init_num_samples -= init_num_samples % self.micro_batch_size

        # calculate the number of batches according to the counted number of samples
        global_num_batches = math.ceil(init_num_samples / self.micro_batch_size)

        # add extra batches to make it divisible by world size (num replicas)
        num_batches_pad = (self.num_replicas - global_num_batches % self.num_replicas) % self.num_replicas
        global_num_batches += num_batches_pad

        # calculate the number of batches per rank
        local_num_batches = global_num_batches // self.num_replicas

        return local_num_batches

    def _make_batches(self) -> List[np.array]:
        max_duration: float = np.max(self.durations)
        min_duration: float = np.min(self.durations)
        bound: float = (max_duration - min_duration) * self.randomization_factor / 2

        # generate pseudo noise
        noise: np.array = np.random.uniform(low=-bound, high=bound, size=len(self.durations))

        # sort indices accroding to pseudo noise
        sorted_indices: np.array = np.argsort(self.durations + noise)

        # delete batches with a non-integer number of samples
        tail = 0
        if self.drop_last:
            tail: int = len(sorted_indices) % self.micro_batch_size
            exclude = np.random.choice(len(sorted_indices), tail, replace=False)
            sorted_indices = np.delete(sorted_indices, exclude)
            logging.warning(f"Drop last is set to True, so {len(exclude)} samples will be dropped.")

        global_num_batches: int = math.ceil(len(sorted_indices) / self.micro_batch_size)

        # if the global_num_batches is zero than return empty list
        if global_num_batches == 0:
            logging.warning(
                f"The number of all batches is {global_num_batches}, than dataloader will "
                "be empty. To avoid this try to decrease batch size or world size or set "
                "drop_last to False."
            )
            return []

        # add extra batches to make it divisible by world size (num replicas)
        pad_batches_num: int = (self.num_replicas - global_num_batches % self.num_replicas) % self.num_replicas
        if global_num_batches < self.num_replicas:
            logging.warning(
                f"The number of all batches is {global_num_batches}, which is less than the "
                f"world size of {self.num_replicas}. SSB Sampler will add {pad_batches_num} "
                "batches. To avoid this try to decrease batch size or world size."
            )

        if pad_batches_num != 0:
            # randomly select batch indeces to pad and concatenate them
            batch_indeces_pad: np.array = np.random.randint(
                low=0, high=len(sorted_indices), size=pad_batches_num * self.micro_batch_size,
            )
            sorted_indices: np.array = np.concatenate(
                (sorted_indices, sorted_indices[batch_indeces_pad]), axis=0,
            )

        # local indeces are selected by world size and local rank
        local_indices: np.array = sorted_indices[self.rank :: self.num_replicas]

        # split local batches
        size_mask = range(self.micro_batch_size, len(local_indices), self.micro_batch_size)
        local_batches = np.split(local_indices, size_mask, axis=0)

        if len(local_batches) != self.local_num_batches:
            raise RuntimeError(
                f'Number of calculated indices {len(local_batches)} is not equal to calculated '
                f'number of local batches {self.local_num_batches}.'
            )

        return local_batches

    def __iter__(self) -> Iterator[List[int]]:
        local_batches = self._make_batches()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            indices = torch.randperm(self.local_num_batches, generator=g)
        else:
            indices = torch.arange(0, self.local_num_batches)

        for _, index in enumerate(indices):
            yield local_batches[index]

    def __len__(self) -> int:
        return self.local_num_batches


def get_semi_sorted_batch_sampler(
    model: ASRModel, dataset: Union[AudioToCharDataset, AudioToBPEDataset], config: dict
) -> SemiSortBatchSampler:
    """
    Instantiates a Semi Sorted (Batch) Sampler.

    Args:
        model: ASR Model.
        dataset: Dataset which allow iterate over all object and parse durations.
        config: Train, Vaidation or Test dataset config.

    Raises:
        ValueError: Wrong dataset type.

    Returns:
        SemiSortBatchSampler: Semi Sorted Batch Sampler class.
    """
    if not (isinstance(dataset, AudioToCharDataset) or isinstance(dataset, AudioToBPEDataset)):
        raise ValueError(
            "Only AudioToCharDataset or AudioToBPEDataset supported with semi sorted batching, "
            f"but found {type(dataset)}."
        )

    durations = [sample.duration for sample in dataset.manifest_processor.collection.data]

    sampler = SemiSortBatchSampler(
        global_rank=model.global_rank,
        world_size=model.world_size,
        durations=durations,
        batch_size=config['batch_size'],
        batch_shuffle=config.get('shuffle', True),
        drop_last=config.get('drop_last', False),
        randomization_factor=config.get('randomization_factor', None),
        seed=config.get('semi_sort_sampler_seed', 42),
    )

    return sampler
