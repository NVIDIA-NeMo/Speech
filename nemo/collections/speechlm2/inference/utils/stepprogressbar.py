# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Per-inference-step progress bar for S2S streaming pipelines."""

from __future__ import annotations

import math

from tqdm import tqdm

from nemo.collections.speechlm2.inference.utils.audio_data import calculate_durations_incl_padding


class StepProgressBar:
    """Tracks per-step inference progress across one or more streams.

    Each call to :meth:`step` advances the bar by one and updates the
    per-stream postfix (e.g. ``stream 2: 45/127``).

    Create via :meth:`from_audio_filepaths`.
    """

    def __init__(self, total_steps: int, steps_per_stream: dict[int, int] | None = None):
        self._bar = tqdm(total=total_steps, desc="Inference", unit="step", dynamic_ncols=True)
        self._steps_per_stream = steps_per_stream or {}
        self._stream_progress: dict[int, int] = {}

    def step(self, stream_id: int) -> None:
        """Record one inference step for *stream_id* and advance the bar."""
        self._stream_progress[stream_id] = self._stream_progress.get(stream_id, 0) + 1
        stream_total = self._steps_per_stream.get(stream_id)
        if stream_total is not None:
            self._bar.set_postfix_str(
                f"stream {stream_id}: {self._stream_progress[stream_id]}/{stream_total}",
                refresh=False,
            )
        self._bar.update(1)

    def finish(self) -> None:
        """Close the underlying tqdm bar."""
        self._bar.close()

    @classmethod
    def from_audio_filepaths(
        cls,
        audio_filepaths: list[str],
        chunk_size_in_secs: float,
        pad_audio_to_sec: float | None = None,
        pad_silence_ratio: float | None = None,
        pad_audio_by_sec: float | None = None,
    ) -> StepProgressBar:
        durations = calculate_durations_incl_padding(
            audio_filepaths, pad_audio_to_sec, pad_silence_ratio, pad_audio_by_sec,
        )
        steps_per_stream = {
            idx: math.ceil(dur / chunk_size_in_secs) for idx, dur in enumerate(durations)
        }
        return cls(
            total_steps=sum(steps_per_stream.values()),
            steps_per_stream=steps_per_stream,
        )
