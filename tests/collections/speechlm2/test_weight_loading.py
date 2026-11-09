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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo.collections.speechlm2.parts.weight_loading import checkpoint_load_state


def test_real_checkpoint_wrapper_names_map_to_seed_names():
    model = torch.nn.Module()
    model.layer = checkpoint_wrapper(torch.nn.Linear(3, 2))
    model.mtp = torch.nn.Linear(2, 2)
    model.mtp.requires_grad_(False)
    model.register_buffer("generated_positions", torch.arange(4), persistent=False)
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    state = checkpoint_load_state(model, shapes)
    assert set(state) == set(shapes)
    assert state["layer.weight"] is model.layer.weight
    assert state["mtp.weight"] is model.mtp.weight


@pytest.mark.parametrize("defect", ["missing_frozen", "extra", "shape"])
def test_checkpoint_coverage_errors_precede_mutation(defect):
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    before = model.weight.detach().clone()
    if defect == "missing_frozen":
        del shapes["bias"]
    elif defect == "extra":
        shapes["mtp.weight"] = (2, 2)
    else:
        shapes["weight"] = (2, 4)
    with pytest.raises(ValueError):
        checkpoint_load_state(model, shapes)
    torch.testing.assert_close(before, model.weight)


def test_missing_persistent_router_buffer_fails():
    model = torch.nn.Linear(3, 2)
    model.register_buffer("e_score_correction_bias", torch.zeros(2))
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    del shapes["e_score_correction_bias"]
    with pytest.raises(ValueError, match="e_score_correction_bias"):
        checkpoint_load_state(model, shapes)


def test_tied_parameter_aliases_have_complete_coverage():
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(3, 2)
    model.head = torch.nn.Linear(2, 3, bias=False)
    model.head.weight = model.embedding.weight
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    state = checkpoint_load_state(model, shapes)
    assert state["embedding.weight"] is state["head.weight"]


def test_native_flat_dcp_loads_without_lightning_namespace(tmp_path):
    import torch.distributed.checkpoint as dcp

    from nemo.collections.speechlm2.parts.weight_loading import load_nemo_rl_checkpoint

    original = torch.nn.Linear(3, 2)
    original.bias.requires_grad_(False)
    dcp.save(original.state_dict(), checkpoint_id=tmp_path)
    loaded = torch.nn.Linear(3, 2)
    load_nemo_rl_checkpoint(loaded, str(tmp_path))
    for key, tensor in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)


class ExtraStateLinear(torch.nn.Linear):
    """Exercise the actual PyTorch module protocol used by TE without a CUDA dependency."""

    def __init__(self, state):
        super().__init__(3, 2)
        self.extra = state

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state):
        self.extra = state


def test_declared_empty_extra_state_loads_narrowly():
    model = ExtraStateLinear(None)
    shapes = {"weight": (2, 3), "bias": (2,), "_extra_state": (0,)}
    state = checkpoint_load_state(model, shapes)
    assert state["_extra_state"].shape == (0,)
    assert state["_extra_state"].dtype == torch.uint8
    shapes["_extra_state"] = (1,)
    with pytest.raises(ValueError, match="shape mismatch"):
        checkpoint_load_state(model, shapes)
    with pytest.raises(ValueError, match="unexpected tensors"):
        checkpoint_load_state(torch.nn.Linear(3, 2), {**shapes, "_extra_state": (0,)})


def test_native_dcp_restores_nonempty_declared_extra_state(tmp_path):
    import torch.distributed.checkpoint as dcp

    from nemo.collections.speechlm2.parts.weight_loading import load_nemo_rl_checkpoint

    original = ExtraStateLinear(torch.tensor([12, 34], dtype=torch.uint8))
    dcp.save(original.state_dict(), checkpoint_id=tmp_path)
    loaded = ExtraStateLinear(torch.zeros(2, dtype=torch.uint8))
    load_nemo_rl_checkpoint(loaded, str(tmp_path))
    torch.testing.assert_close(loaded.get_extra_state(), original.get_extra_state())
    shapes = {"weight": (2, 3), "bias": (2,)}
    with pytest.raises(ValueError, match="missing parameters/buffers"):
        checkpoint_load_state(loaded, shapes)
