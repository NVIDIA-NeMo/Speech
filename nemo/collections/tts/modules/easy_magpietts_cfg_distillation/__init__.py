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

"""Public components for EasyMagpieTTS classifier-free guidance distillation."""

from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.distillation_types import (
    StudentOutput,
    TeacherOutput,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.logging_keys import (
    MONITORED_KEYS_TRAIN,
    MONITORED_KEYS_VALID,
    ON_EPOCH_TRAIN_KEYS,
    ROLLOUT_TIMING_KEYS,
    LossKey,
    LossMode,
    MetricKey,
    get_loss_key,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.teacher_inference_engine import (
    TeacherInferenceEngine,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.validation_inference_engine import (
    ValidationAudioPaths,
    ValidationInferenceEngine,
    ValidationInferenceParams,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.validation_metrics import (
    aggregate_language_metrics,
    collect_validation_outputs,
    compute_validation_metrics,
)

__all__ = [
    "StudentOutput",
    "TeacherOutput",
    "LossKey",
    "LossMode",
    "MetricKey",
    "get_loss_key",
    "MONITORED_KEYS_TRAIN",
    "MONITORED_KEYS_VALID",
    "ON_EPOCH_TRAIN_KEYS",
    "ROLLOUT_TIMING_KEYS",
    "TeacherInferenceEngine",
    "ValidationAudioPaths",
    "ValidationInferenceEngine",
    "ValidationInferenceParams",
    "aggregate_language_metrics",
    "collect_validation_outputs",
    "compute_validation_metrics",
]
