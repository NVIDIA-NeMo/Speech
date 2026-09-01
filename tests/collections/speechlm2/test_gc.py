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

from nemo.collections.speechlm2.parts.gc import GarbageCollectionManager


def test_garbage_collection_manager_owns_step_state(monkeypatch):
    calls = []

    class FakeGarbageCollection:
        def __init__(self, gc_every_steps):
            calls.append(("init", gc_every_steps))

        def run(self, step_count):
            calls.append(("run", step_count))

    import nemo_automodel.components.training.garbage_collection as gc_module

    monkeypatch.setattr(gc_module, "GarbageCollection", FakeGarbageCollection)
    manager = GarbageCollectionManager(gc_every_steps=10)

    manager.on_fit_start()
    manager.on_optimizer_step()
    manager.on_optimizer_step()

    assert calls == [("init", 10), ("run", 1), ("run", 2)]


def test_garbage_collection_manager_is_noop_when_disabled():
    manager = GarbageCollectionManager(gc_every_steps=None)

    manager.on_fit_start()
    manager.on_optimizer_step()

    assert manager._collector is None
    assert manager._optimizer_step_count == 0


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "10"])
def test_garbage_collection_manager_rejects_invalid_interval(value):
    with pytest.raises(ValueError, match="gc_every_steps"):
        GarbageCollectionManager(gc_every_steps=value)
