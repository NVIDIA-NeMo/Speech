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

from nemo.utils import logging


class GarbageCollectionManager:
    """Manage deterministic Python garbage collection during distributed training.

    When enabled, automatic garbage collection is replaced at fit start by
    NeMo Automodel's generation-1 collector. The manager owns the optimizer-step
    counter so model implementations only need to forward lifecycle events.
    """

    def __init__(self, gc_every_steps: int | None) -> None:
        if gc_every_steps is not None and (
            isinstance(gc_every_steps, bool) or not isinstance(gc_every_steps, int) or gc_every_steps <= 0
        ):
            raise ValueError(f"model.gc_every_steps must be a positive integer or null, got {gc_every_steps!r}")
        self.gc_every_steps = gc_every_steps
        self._collector = None
        self._optimizer_step_count = 0

    def on_fit_start(self) -> None:
        """Disable automatic GC and initialize the configured manual collector."""
        if self.gc_every_steps is None:
            return

        from nemo_automodel.components.training.garbage_collection import GarbageCollection

        self._collector = GarbageCollection(gc_every_steps=self.gc_every_steps)
        self._optimizer_step_count = 0
        logging.info(
            "Automatic Python GC disabled; generation-1 collection will run every %d optimizer steps",
            self.gc_every_steps,
        )

    def on_optimizer_step(self) -> None:
        """Advance the manual collector after a completed optimizer step."""
        if self._collector is None:
            return
        self._optimizer_step_count += 1
        self._collector.run(self._optimizer_step_count)
