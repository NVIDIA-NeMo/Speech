# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback

from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging
from nemo.utils.get_rank import get_rank


def trace_handler(prof, chakra_device_trace_path):
    """
    Handles the export of profiling traces to a specified directory.

    Args:
        prof: The profiler object containing the trace data.
        chakra_device_trace_path: The path where the trace file will be saved.
    """
    rank = get_rank()
    trace_file = chakra_device_trace_path / f"rank-{rank}.json.gz"
    prof.export_chrome_trace(str(trace_file))
    logging.info(f"Kineto trace saved: {trace_file}")


class PytorchProfilerCallback(Callback, IOMixin):
    """
    A PyTorch Lightning callback for profiling with PyTorch's built-in Profiler and ExecutionTraceObserver.

    This callback enables profiling for specific steps during training using PyTorch's profiler.
    It also captures detailed execution traces using `ExecutionTraceObserver`.
    It ensures proper cleanup, preventing memory leaks or duplicate profiling instances.

    Args:
        start_step (int): Global batch step to start profiling.
        end_step (int): Global batch step to end profiling.
        warmup_steps (int): Number of warmup steps before profiling starts.
        active_steps (int): Number of active profiling steps.
        trace_dir (str): Directory where traces will be saved.
        collect_et (bool): Collect Execution Trace (host trace).
        profiler_kwargs (dict, optional): Additional keyword args to pass to torch.profiler.profile
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        warmup_steps: int = 2,
        trace_dir: str = None,
        collect_et: bool = False,
        profiler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if trace_dir is None:
            trace_dir = os.path.join(os.getcwd(), 'traces')
            os.makedirs(trace_dir, exist_ok=True)

        if not isinstance(start_step, int) or not isinstance(end_step, int):
            raise TypeError(f"start_step and end_step must be integers. Got {type(start_step)}, {type(end_step)}")
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step.")

        if not os.path.isdir(trace_dir):
            raise ValueError(f"Chakra trace output path ({trace_dir}) does not exist.")

        self.start_step = start_step
        self.end_step = end_step
        self.warmup_steps = warmup_steps
        self.active_steps = max(self.end_step - self.start_step, 1)

        wait_steps = max(self.start_step - self.warmup_steps, 0)

        self.trace_dir = Path(trace_dir)
        self.chakra_host_trace_path = self.trace_dir / "host"
        self.chakra_device_trace_path = self.trace_dir / "torch_profiler"

        self.chakra_device_trace_path.mkdir(parents=True, exist_ok=True)

        self.trace_observer = None
        if collect_et:
            self.chakra_host_trace_path.mkdir(parents=True, exist_ok=True)
            self.trace_observer = torch.profiler.ExecutionTraceObserver()
            trace_file = self.chakra_host_trace_path / f"rank-{get_rank()}.json.gz"
            self.trace_observer.register_callback(str(trace_file))

        base_kwargs = {
            "activities": [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "schedule": torch.profiler.schedule(
                wait=wait_steps, warmup=self.warmup_steps, active=self.active_steps, repeat=1
            ),
            "on_trace_ready": lambda prof: trace_handler(prof, self.chakra_device_trace_path),
            "execution_trace_observer": self.trace_observer,
        }
        if profiler_kwargs is not None:
            if not isinstance(profiler_kwargs, dict):
                raise TypeError(f"profiler_kwargs must be a dict if provided. Got {type(profiler_kwargs)}")
            base_kwargs.update(profiler_kwargs)

        self.profiler = torch.profiler.profile(**base_kwargs)

        logging.info(
            "PyTorch profiling initialized:\n"
            f" - Start Step: {self.start_step}\n"
            f" - End Step: {self.end_step}\n"
            f" - Warmup Steps: {self.warmup_steps}\n"
            f" - Active Steps: {self.active_steps}\n"
            f" - Trace Directory: {self.trace_dir}\n"
            f" - Collect Execution Trace: {collect_et}\n"
            f" - Extra profiler kwargs: {profiler_kwargs or {}}"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        """Chakra trace collection ends."""
        # Step the profiler after each training batch
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self.trace_observer:
            try:
                logging.info("Unregistering ExecutionTraceObserver...")
                self.trace_observer.unregister_callback()
            except RuntimeError as e:
                logging.warning(f"ExecutionTraceObserver cleanup failed: {e}")


# Add an alias, ideally we want camelcase for PyTorch
PyTorchProfilerCallback = PytorchProfilerCallback
