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
import pytest
import torch

from nemo.collections.speechlm2.parts.weight_loading import load_perception_state


def test_initial_complete_load_then_partial_update_preserves_frozen_state():
    perception = torch.nn.Linear(3, 2)
    seed = {key: value.clone() for key, value in perception.state_dict().items()}
    with pytest.raises(RuntimeError, match="missing"):
        load_perception_state(perception, {"weight": seed["weight"] + 1}, require_complete=True)
    torch.testing.assert_close(perception.weight, seed["weight"])
    load_perception_state(perception, seed, require_complete=True)
    update = torch.full_like(perception.weight, 0.5)
    assert load_perception_state(perception, {"weight": update}, require_complete=False) == {"perception.weight"}
    torch.testing.assert_close(perception.weight, update)
    torch.testing.assert_close(perception.bias, seed["bias"])
    assert load_perception_state(perception, {}, require_complete=False) == set()


@pytest.mark.parametrize("defect", ["unexpected", "shape"])
def test_invalid_partial_update_fails_before_mutation(defect):
    perception = torch.nn.Linear(3, 2)
    before = perception.weight.detach().clone()
    weights = {"weight": before + 1}
    weights["wrong" if defect == "unexpected" else "bias"] = torch.zeros(5)
    with pytest.raises(RuntimeError, match="does not exactly match"):
        load_perception_state(perception, weights, require_complete=False)
    torch.testing.assert_close(perception.weight, before)
