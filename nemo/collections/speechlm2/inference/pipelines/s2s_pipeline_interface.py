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

from typing import Any

from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions


class S2SPipelineInterface:
    """Base class for all streaming S2S pipelines.

    This class is intentionally kept minimal and mirrors the behaviour of
    ``BasePipeline`` that is used for streaming ASR pipelines.  It
    provides a small in-memory *state pool* that stores per-stream objects
    (cache, running buffers, etc.) required by a concrete pipeline
    implementation.  Sub-classes are expected to implement
    :py:meth:`create_state` to construct a fresh state object.
    """

    def __init__(self) -> None:
        # Pool that holds per-stream state, keyed by ``stream_id``
        self._state_pool: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def get_state(self, stream_id: int):
        """Return the state object for *stream_id* or *None* if it does not exist."""
        return self._state_pool.get(stream_id, None)

    def delete_state(self, stream_id: int) -> None:
        """Delete the state associated with *stream_id* (noop if missing)."""
        if stream_id in self._state_pool:
            del self._state_pool[stream_id]

    def create_state(self, options: S2SRequestOptions | None = None):
        """Create and return a *new*, *empty* state object.

        Args:
            options: Per-stream request options (system prompt, sampling
                overrides, etc.).  Stored on the state so they can be
                consulted throughout the stream's lifetime.

        Must be implemented by concrete pipelines.
        """
        raise NotImplementedError("`create_state()` must be implemented in a subclass.")

    def get_or_create_state(self, stream_id: int, options: S2SRequestOptions | None = None):
        """Return existing state for *stream_id* or create a new one via :py:meth:`create_state`."""
        if stream_id not in self._state_pool:
            self._state_pool[stream_id] = self.create_state(options)
        return self._state_pool[stream_id]

    # ------------------------------------------------------------------
    # Session helpers – identical to *BasePipeline*
    # ------------------------------------------------------------------
    def reset_session(self) -> None:
        """Clear the internal *state pool* – effectively resetting the pipeline."""
        self._state_pool.clear()

    def open_session(self) -> None:
        """Alias for :py:meth:`reset_session` to start a fresh streaming session."""
        self.reset_session()

    def close_session(self) -> None:
        """Alias for :py:meth:`reset_session` to end the current streaming session."""
        self.reset_session()
