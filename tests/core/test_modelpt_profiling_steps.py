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
"""ModelPT's nsys / CUDA memory profiling windows count training batches, not the hook's ``batch_idx``.

With a stateful train dataloader, ModelPT trains with ``training_step(dataloader_iter)``. Lightning then passes
``on_train_batch_start`` the previous step's ``batch_idx``. ``batch_idx`` also restarts every epoch, and
``trainer.global_step`` counts optimizer updates, which can advance several times in one training batch with
manual optimization. The window therefore follows Lightning's running count of training batches.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo.core.classes.modelPT import ModelPT


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["batch", "dataloader_iter"])
@pytest.mark.parametrize("updates_per_batch", [[1], [2], [1, 2]])  # optimizer updates, cycled over batches
def test_profiling_window_covers_the_configured_training_batches(mode, updates_per_batch):
    events = _run(start=3, end=5, mode=mode, updates_per_batch=updates_per_batch)
    # Each tuple: (event, index of the training batch whose hook made the call).
    assert events == [("nsys_start", 3), ("mem_start", 3), ("nsys_stop", 5), ("mem_stop", 5)]


@pytest.mark.unit
def test_profiling_single_batch_window_with_two_updates_per_batch():
    events = _run(start=1, end=1, mode="batch", updates_per_batch=[2])
    assert events == [("nsys_start", 1), ("mem_start", 1), ("nsys_stop", 1), ("mem_stop", 1)]


@pytest.mark.unit
def test_profiling_resumed_inside_the_window_starts_at_the_first_batch():
    events = _run(start=3, end=5, mode="batch", updates_per_batch=[1], first_batch=4)
    assert events == [("nsys_start", 4), ("mem_start", 4), ("nsys_stop", 5), ("mem_stop", 5)]


@pytest.mark.unit
def test_profiling_resumed_after_the_window_never_starts_or_stops():
    assert _run(start=3, end=5, mode="batch", updates_per_batch=[1], first_batch=6) == []


def _run(start: int, end: int, mode: str, updates_per_batch: list, first_batch: int = 0) -> list:
    model = _ProfilingStub(start, end)
    progress = model.trainer.fit_loop.epoch_loop.batch_progress.total
    progress.ready = progress.completed = first_batch
    current = lambda: progress.ready - 1  # noqa: E731
    events = []
    cudart = mock.MagicMock()
    cudart.return_value.cudaProfilerStart.side_effect = lambda: events.append(("nsys_start", current()))
    cudart.return_value.cudaProfilerStop.side_effect = lambda: events.append(("nsys_stop", current()))

    with (
        mock.patch.object(torch.cuda, "cudart", cudart),
        mock.patch.object(
            torch.cuda.memory,
            "_record_memory_history",
            # Called with ``max_entries`` to start recording and with ``enabled=None`` to stop it.
            side_effect=lambda **kw: "max_entries" in kw and events.append(("mem_start", current())),
        ),
        mock.patch.object(
            torch.cuda.memory, "_dump_snapshot", side_effect=lambda path: events.append(("mem_stop", current()))
        ),
        mock.patch("nemo.core.classes.modelPT.get_rank", return_value=0),
    ):
        for i, epoch_batch_idx in enumerate(range(first_batch, 10)):
            progress.ready += 1  # Lightning counts the batch before calling on_train_batch_start
            # ``batch_idx`` Lightning passes to on_train_batch_start: the current batch's index for the default
            # ``training_step(batch, batch_idx)``, the previous batch's index for ``training_step(dataloader_iter)``.
            start_idx = epoch_batch_idx if mode == "batch" else max(epoch_batch_idx - 1, 0)
            ModelPT.on_train_batch_start(model, batch=None, batch_idx=start_idx % 4)  # % 4: epochs of 4 batches
            model.trainer.global_step += updates_per_batch[i % len(updates_per_batch)]
            ModelPT.on_train_batch_end(model, outputs=None, batch=None, batch_idx=epoch_batch_idx % 4)
            progress.completed += 1
    return events


class _ProfilingStub(SimpleNamespace):
    """Just the ModelPT state that the profiling branches of the batch hooks read, on a fake CUDA device."""

    def __init__(self, start: int, end: int):
        progress = SimpleNamespace(total=SimpleNamespace(ready=0, completed=0))
        super().__init__(
            device=torch.device("cuda"),
            training=True,
            trainer=SimpleNamespace(
                global_step=0, fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(batch_progress=progress))
            ),
            _nsys_profile_enabled=True,
            _nsys_profile_started=False,
            _nsys_profile_complete=False,
            _nsys_profile_start_step=start,
            _nsys_profile_end_step=end,
            _nsys_profile_ranks=[0],
            _nsys_profile_gen_shape=False,
            _memory_profile_enabled=True,
            _memory_profile_started=False,
            _memory_profile_complete=False,
            _memory_profile_start_step=start,
            _memory_profile_end_step=end,
            _memory_profile_rank=0,
            _memory_profile_output_path="/nonexistent",
        )

    def _train_batch_counter(self) -> int:
        return ModelPT._train_batch_counter(self)
