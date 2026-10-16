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
"""Validated initial and partial SpeechLM perception weight loading."""

from collections.abc import Mapping

import torch


def load_perception_state(
    perception: torch.nn.Module,
    weights: Mapping[str, torch.Tensor],
    *,
    require_complete: bool,
) -> set[str]:
    """Validate all supplied state before mutation and optionally require full coverage.

    A successful initial exact load establishes the serving encoder architecture.
    Subsequent standard vLLM weight loads may contain only updated tensors; their
    names and shapes must still match that architecture.
    """
    expected = perception.state_dict()
    missing = (
        sorted(key for key in expected if key not in weights and not key.endswith("._extra_state"))
        if require_complete
        else []
    )
    unexpected = sorted(key for key in weights if key not in expected)
    wrong_shapes = [
        key
        for key, tensor in weights.items()
        if key in expected and isinstance(expected[key], torch.Tensor) and expected[key].shape != tensor.shape
    ]
    if missing or unexpected or wrong_shapes:
        raise RuntimeError(
            "Speech encoder checkpoint does not exactly match its exported architecture: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={wrong_shapes}."
        )
    perception.load_state_dict(weights, strict=False)
    return {"perception." + key for key in weights}
