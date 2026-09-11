# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import torch

from nemo.utils import logging


class FallbackDataset(torch.utils.data.Dataset):
    """
    FallbackDataset is a wrapper on an existing map-style ``torch.utils.data.Dataset``.
    It's used to return the previous item (or batch, depending on Dataset) whenever
    the underlying ``Dataset`` returns ``None``.
    This is useful when ``Dataset`` returns a full batch (as e.g. Lhotse datasets typically do),
    and wasn't able to read any of the items in that batch.

    Example::

        >>> dataset = AudioToTextLhotseDataset(...)
        ... dataset = FallbackDataset(dataset)
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._fallback = None

    def __getitem__(self, item):
        ans = self.dataset[item]
        if ans is None:
            if self._fallback is None:
                logging.warning(
                    f"FallbackDataset received None from {self.dataset} on the first call to __getitem__, "
                    f"and must return None instead of an actual batch."
                    f"This indicates an issue with data reading."
                )
            ans = self._fallback
        self._fallback = ans
        return ans

    def __len__(self):
        return len(self.dataset)
