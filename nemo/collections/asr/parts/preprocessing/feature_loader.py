# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from typing import Any, Optional

import numpy as np
import torch


class ExternalFeatureLoader(object):
    """Load numeric NumPy arrays and PyTorch tensors without object deserialization."""

    def __init__(
        self,
        augmentor: Optional[Any] = None,
    ):
        """
        Feature loader
        """
        self.augmentor = augmentor

    def load_feature_from_file(self, file_path: str):
        """Load samples from file_path and convert it to be of type float32
        file_path (str) is the path of the file that stores feature/sample.
        """

        if file_path.endswith(".pt") or file_path.endswith(".pth"):
            samples = torch.load(file_path, map_location="cpu", weights_only=True).float().numpy()
            return samples
        else:
            # A feature file stores one numeric array, not a pickled object or an archive.
            samples = np.load(file_path, allow_pickle=False)
            if isinstance(samples, np.lib.npyio.NpzFile):
                samples.close()
                raise ValueError("Expected a single NumPy array; save the features as a .npy file.")
            return self._convert_samples_to_float32(samples)
            # TODO load other type of files such as kaldi io ark

    @staticmethod
    def _convert_samples_to_float32(samples: np.ndarray) -> np.ndarray:
        """Convert sample type to float32.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in (np.int8, np.int16, np.int32, np.int64):
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in (np.float16, np.float32, np.float64):
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    def process(self, file_path: str) -> torch.Tensor:
        """Processes the features from the provided `file_path`."""
        features = self.load_feature_from_file(file_path)
        features = self.process_segment(features)
        return features

    def process_segment(self, feature_segment):
        """Processes the provided feature segment."""
        if self.augmentor:
            # augmentor for external features. Here possible augmentor for
            # external embedding feature is Diaconis Augmentation and might
            # be implemented later
            self.augmentor.perturb(feature_segment)
            return torch.tensor(feature_segment, dtype=torch.float)

        return torch.tensor(feature_segment, dtype=torch.float)
