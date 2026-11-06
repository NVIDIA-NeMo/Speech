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
"""ModelPT's nsys / CUDA memory profiling windows follow ``trainer.global_step``, not the hook's ``batch_idx``.

With a stateful train dataloader, ModelPT trains with ``training_step(dataloader_iter)``. Lightning then passes
``on_train_batch_start`` the previous step's ``batch_idx``, so ``batch_idx`` can't be used to find the current step.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo.core.classes.modelPT import ModelPT

START, END = 3, 5


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["batch", "dataloader_iter"])
def test_nsys_and_memory_profiling_follow_global_step(mode):
    model = _ProfilingStub()
    cudart = mock.MagicMock()
    events = []
    cudart.return_value.cudaProfilerStart.side_effect = lambda: events.append(
        ("nsys_start", model.trainer.global_step)
    )
    cudart.return_value.cudaProfilerStop.side_effect = lambda: events.append(("nsys_stop", model.trainer.global_step))

    with (
        mock.patch.object(torch.cuda, "cudart", cudart),
        mock.patch.object(
            torch.cuda.memory,
            "_record_memory_history",
            # Called with ``max_entries`` to start recording and with ``enabled=None`` to stop it.
            side_effect=lambda **kw: "max_entries" in kw and events.append(("mem_start", model.trainer.global_step)),
        ),
        mock.patch.object(
            torch.cuda.memory,
            "_dump_snapshot",
            side_effect=lambda path: events.append(("mem_stop", model.trainer.global_step)),
        ),
        mock.patch("nemo.core.classes.modelPT.get_rank", return_value=0),
    ):
        for step in range(8):
            # ``batch_idx`` Lightning passes to on_train_batch_start: the current batch's index for the default
            # ``training_step(batch, batch_idx)``, the previous batch's index for ``training_step(dataloader_iter)``.
            start_idx = step if mode == "batch" else max(step - 1, 0)
            ModelPT.on_train_batch_start(model, batch=None, batch_idx=start_idx)
            model.trainer.global_step += 1  # the optimizer step
            ModelPT.on_train_batch_end(model, outputs=None, batch=None, batch_idx=step)

    # global_step before the step runs == index of the step; on_train_batch_end sees it already incremented.
    assert events == [
        ("nsys_start", START),
        ("mem_start", START),
        ("nsys_stop", END + 1),
        ("mem_stop", END + 1),
    ]


class _ProfilingStub(SimpleNamespace):
    """Just the ModelPT state that the profiling branches of the batch hooks read, on a fake CUDA device."""

    def __init__(self):
        super().__init__(
            device=torch.device("cuda"),
            training=True,
            trainer=SimpleNamespace(global_step=0),
            _nsys_profile_enabled=True,
            _nsys_profile_started=False,
            _nsys_profile_complete=False,
            _nsys_profile_start_step=START,
            _nsys_profile_end_step=END,
            _nsys_profile_ranks=[0],
            _nsys_profile_gen_shape=False,
            _memory_profile_enabled=True,
            _memory_profile_started=False,
            _memory_profile_complete=False,
            _memory_profile_start_step=START,
            _memory_profile_end_step=END,
            _memory_profile_rank=0,
            _memory_profile_output_path="/nonexistent",
        )
