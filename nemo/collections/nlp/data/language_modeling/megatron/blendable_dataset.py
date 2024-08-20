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

"""Blendable dataset."""

import time

import numpy as np
import torch

from nemo.utils import logging
from nemo.utils.app_state import AppState


class BlendableDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights, size):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        app_state = AppState()
        try:
            if app_state.local_rank == 0:
                from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper

                compile_helper()
            torch.distributed.barrier()
            from nemo.collections.nlp.data.language_modeling.megatron import helpers
        except ImportError:
            raise ImportError(
                f'Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file.'
            )

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )
        logging.info(
            '> elapsed time for building blendable dataset indices: ' '{:.2f} (sec)'.format(time.time() - start_time)
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        dataset_size = len(self.datasets[dataset_idx])
        # Ensure the sample index doesn't exceed the dataset size
        if sample_idx >= dataset_size:
            logging.warning(f"Index {sample_idx} out of bounds for dataset {dataset_idx}. Reusing existing examples.")
            sample_idx = sample_idx % dataset_size
            logging.warning(f"Reusing index {sample_idx} for dataset {dataset_idx}.")

        return self.datasets[dataset_idx][sample_idx]

    def create_data_mmap(self):
        for dataset in self.datasets:
            dataset.create_data_mmap()


class MemoryEfficientBlendableDataset(torch.utils.data.Dataset):
    """
    A BlendableDataset implementation that uses less memory than the original implementation.
    Indices are computed algorithmically instead of storing them in memory.

    To test call: MemoryEfficientBlendableDataset.test_index_blending()
    """

    def __init__(self, datasets, weights, size, weight_bins=100):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        weight_bins = min(weight_bins, size)

        self.size = size
        self.weight_bins = weight_bins

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        assert (weights > 0.0).all()
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        self.weights = weights / sum_weights

        # create ds index based on weights
        ds_index = []
        ds_bias = []
        for i, w in enumerate(self.weights):
            n = int(w * weight_bins)
            ds_index.extend([i] * n)
            ds_bias.extend(range(n))
        # make sure arrays have length of weight_bins
        n = weight_bins - len(ds_index)
        ds_index.extend([i] * n)
        ds_bias.extend(range(ds_bias[-1], ds_bias[-1] + n))

        self.ds_index = np.array(ds_index, dtype=np.uint32)
        self.ds_index_size = np.array([(self.ds_index == i).sum() for i in range(num_datasets)], dtype=np.uint32)
        assert (
            self.ds_index_size > 0
        ).all(), f"Some datasets have no samples in the blendable dataset, increase weight_bins or the offending weight. ds_index_size = {self.ds_index_size}"
        self.ds_bias = np.array(ds_bias, dtype=np.uint32)

        self.ds_size = np.array([len(ds) for ds in datasets], dtype=np.uint32)

    def get_ds_sample_idx(self, idx):
        """Returns ds index and sample index (within the ds) for the given index in the blendable dataset."""

        bin = idx % self.weight_bins
        ds_idx = self.ds_index[bin]
        sample_idx = (self.ds_bias[bin] + (idx // self.weight_bins) * self.ds_index_size[ds_idx]) % self.ds_size[
            ds_idx
        ]

        return ds_idx, sample_idx

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.get_ds_sample_idx(idx)

        return self.datasets[ds_idx][sample_idx]

    @classmethod
    def test_index_blending(cls):
        """Visualize indices of blended dataset"""

        import matplotlib.pyplot as plt

        plt.ion()

        class DS(torch.utils.data.Dataset):
            def __init__(self, size, data):
                self.size = size
                self.data = data

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx]

        for weight_bins in [10, 100]:
            blend_ds = MemoryEfficientBlendableDataset(
                [DS(10, "a"), DS(10, "b"), DS(10, "c")], [0.5, 0.3, 0.2], 50, weight_bins=weight_bins
            )

            ds_sample_idx_list = [blend_ds.get_ds_sample_idx(i) for i in range(50)]
            ds_list = list(zip(*ds_sample_idx_list))[0]
            sample_list = list(zip(*ds_sample_idx_list))[1]

            plt.figure()
            plt.plot(ds_list, label="ds idx")
            plt.plot(sample_list, label="sample")
            plt.legend()
            plt.grid()
            plt.title(f"weight_bins={weight_bins}")
