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

from types import SimpleNamespace

import torch

from nemo.utils.callbacks.training_stats import TrainingStatsCallback


class _DummyModule:
    device = torch.device("cpu")
    _last_batch_num_tokens = 5
    _last_batch_num_examples = 2

    def __init__(self, device_mesh=None):
        self._device_mesh = device_mesh
        self.logged = {}

    def log_dict(self, values, **kwargs):
        self.logged.update(values)


class _FakeSubMesh:
    def __init__(self, group):
        self._group = group

    def get_group(self):
        return self._group


class _FakeMesh:
    mesh_dim_names = ("data_parallel", "tensor_parallel")

    def __init__(self, group):
        self._group = group

    def __getitem__(self, item):
        if item != "data_parallel":
            raise KeyError(item)
        return _FakeSubMesh(self._group)


def test_training_stats_callback_reduces_with_device_mesh_dp_group(monkeypatch):
    dp_group = object()
    module = _DummyModule(device_mesh=_FakeMesh(dp_group))
    callback = TrainingStatsCallback()
    seen = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(tensor, op=None, group=None):
        seen.append(group)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    callback.on_train_batch_end(SimpleNamespace(), module, outputs=None, batch={}, batch_idx=0)

    assert seen == [dp_group]
    assert callback.num_tokens_total == 5
    assert callback.num_examples_total == 2
    assert module.logged["num_tokens_total"] == 5.0
    assert module.logged["num_examples_total"] == 2.0


def test_training_stats_callback_plain_ddp_uses_default_group(monkeypatch):
    module = _DummyModule()
    callback = TrainingStatsCallback()
    seen = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(tensor, op=None, group=None):
        seen.append(group)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    callback.on_train_batch_end(SimpleNamespace(), module, outputs=None, batch={}, batch_idx=0)

    assert seen == [None]
