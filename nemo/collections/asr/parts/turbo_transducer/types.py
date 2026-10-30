# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import NamedTuple, Optional

import torch


class TransducerLossInputs(NamedTuple):
    """
    Per-position quantities extracted from the Joint output, sufficient to compute the transducer losses.

    All tensors here are small (no vocabulary dimension), which allows the loss functions to operate
    without materializing (or keeping gradients for) the full `[B, T, U+1, V]` Joint output.

    Fields:
        target_logprobs: log-probabilities for target labels, `[B, T, U+1]`. Zero for non-existent targets.
        blank_logprobs: log-probabilities for the blank label, `[B, T, U+1]`.
        consistency_kl: optional per-position consistency (KL) loss, `[B // 2, T, U+1]`, computed between
            the first (teacher) and the second (student) half of a double batch. `None` if not requested.
    """

    target_logprobs: torch.Tensor
    blank_logprobs: torch.Tensor
    consistency_kl: Optional[torch.Tensor] = None
