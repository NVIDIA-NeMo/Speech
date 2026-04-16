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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class S2SRequestOptions:
    """Immutable per-stream options for S2S inference.

    Attached to the first ``Frame`` of each stream via the ``options``
    field so that the pipeline can read per-stream configuration at the
    start of every new audio stream.  Frozen so that options cannot be
    accidentally modified after the stream is initialised.

    All fields default to ``None``, which means "use the pipeline-level
    default".  Call :meth:`augment_with_defaults` to fill ``None`` fields
    with pipeline-level values — this mirrors the
    ``ASRRequestOptions.augment_with_defaults()`` pattern used by the
    ASR streaming pipelines.
    """

    system_prompt: str | None = None

    top_p: float | None = None          # (0, 1]
    temperature: float | None = None    # >= 0
    repetition_penalty: float | None = None  # > 0

    def __post_init__(self) -> None:
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")

    @staticmethod
    def _with_default(value: Any, default: Any) -> Any:
        """Return *value* when it is not ``None``, otherwise *default*."""
        return default if value is None else value

    def augment_with_defaults(
        self,
        default_system_prompt: str | None = None,
        default_top_p: float | None = None,
        default_temperature: float | None = None,
        default_repetition_penalty: float | None = None,
    ) -> S2SRequestOptions:
        """Return a new options instance with ``None`` fields filled from defaults.

        This is the S2S equivalent of ``ASRRequestOptions.augment_with_defaults``.
        """
        return S2SRequestOptions(
            system_prompt=self._with_default(self.system_prompt, default_system_prompt),
            top_p=self._with_default(self.top_p, default_top_p),
            temperature=self._with_default(self.temperature, default_temperature),
            repetition_penalty=self._with_default(self.repetition_penalty, default_repetition_penalty),
        )
