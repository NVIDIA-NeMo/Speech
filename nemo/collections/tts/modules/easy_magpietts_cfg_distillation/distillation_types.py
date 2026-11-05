# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Shared output types for EasyMagpieTTS CFG distillation."""

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class TeacherOutput:
    """Outputs produced by the teacher rollout used as distillation targets."""

    codes: Tensor
    logits: Tensor
    lens: Tensor
    sample_weights: Optional[Tensor]
    rejected: Optional[Tensor]

    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]

    # CPU-observed wall times in seconds, without explicit CUDA synchronization.
    timing_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class StudentOutput:
    """Outputs produced by the student forward pass for distillation."""

    logits: Tensor
    logits_lt: Optional[Tensor]
    logits_phonemes: Optional[Tensor]
    audio_codes_gt_lt: Optional[Tensor]
    audio_codes_lens_gt_lt: Optional[Tensor]
