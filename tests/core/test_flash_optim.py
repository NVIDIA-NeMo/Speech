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

import pytest
import torch

from nemo.core.optim.flash_optim import patch_flashoptim_uneven_shard_support


class DummyFlashOptimizer:
    @staticmethod
    def _wrap_state_as_dtensor(state, param):
        state["original"] = True


class DummyOptimizer:
    pass


class DummyParam:
    device_mesh = object()
    placements = object()
    shape = torch.Size([5])

    @staticmethod
    def stride():
        return (1,)


@pytest.mark.unit
def test_patch_flashoptim_helper_is_noop_for_other_optimizers():
    optimizer = DummyOptimizer()
    patch_flashoptim_uneven_shard_support(optimizer)
    assert not hasattr(DummyOptimizer, "_nemo_patched_uneven_shard")


@pytest.mark.unit
def test_patch_flashoptim_helper_is_idempotent():
    optimizer = DummyFlashOptimizer()
    patch_flashoptim_uneven_shard_support(optimizer)
    patched = DummyFlashOptimizer._wrap_state_as_dtensor
    patch_flashoptim_uneven_shard_support(DummyFlashOptimizer())
    assert DummyFlashOptimizer._nemo_patched_uneven_shard is True
    assert DummyFlashOptimizer._wrap_state_as_dtensor is patched


@pytest.mark.unit
def test_patch_flashoptim_helper_wraps_state_with_shape_and_stride(monkeypatch):
    calls = {}

    class FakeDTensor:
        def __init__(self, value=None):
            self.value = value

        @staticmethod
        def from_local(local, mesh, placements, shape, stride):
            calls["local"] = local
            calls["mesh"] = mesh
            calls["placements"] = placements
            calls["shape"] = shape
            calls["stride"] = stride
            return FakeDTensor(local)

    import torch.distributed.tensor

    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)

    optimizer = DummyFlashOptimizer()
    patch_flashoptim_uneven_shard_support(optimizer)

    state = {"exp_avg": torch.ones(5)}
    DummyFlashOptimizer._wrap_state_as_dtensor(state, DummyParam())

    assert isinstance(state["exp_avg"], FakeDTensor)
    assert torch.equal(calls["local"], torch.ones(5))
    assert calls["shape"] == torch.Size([5])
    assert calls["stride"] == (1,)
