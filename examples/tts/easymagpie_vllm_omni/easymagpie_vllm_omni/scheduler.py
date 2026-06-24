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
"""Streaming-aware scheduler for the single-stage EasyMagpieTTS engine.

vLLM-Omni's stage-0 streaming session update
(:meth:`OmniARScheduler._update_request_as_session`) extends the prompt token
ids from each ``StreamingInput`` chunk but **never updates**
``session.additional_information``. For EasyMagpie's streaming-text path that
silently drops the per-chunk ``text_token`` payload on the scheduler side: the
runner only ever sees the initial request's ``additional_information``, so every
decode step reads ``text_token=None``, the text channel is masked off, and the
model emits audio-EOS almost immediately (a handful of frames instead of the
full utterance).

:class:`EasyMagpieARAsyncScheduler` restores the missing propagation. It is a
drop-in replacement for ``OmniARAsyncScheduler``; wire it in via the stage's
``scheduler_cls``::

    "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"
"""
from __future__ import annotations

from vllm.v1.request import Request, StreamingUpdate

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """``OmniARAsyncScheduler`` that forwards per-chunk ``additional_information``.

    Replace (not merge) is the correct session-level semantics: the session field
    is just a courier for the latest chunk's payload to ``OmniNewRequestData``.
    Per-key accumulation, where a model needs it, is handled by the runner's
    ``_update_streaming_input_additional_info`` against the model's
    ``streaming_accumulated_keys`` set, so the merge policy stays a per-model
    concern. ``None`` is treated as "this chunk omitted the field" (keep the prior
    value) rather than "clear the session", so a client may keep pumping
    placeholder chunks (e.g. the masking ``text_token=-1`` sentinel still sets a
    value; a truly empty chunk leaves the previous payload intact).
    """

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        super()._update_request_as_session(session, update)

        # ``check_stop`` decides segment termination on ``session.max_tokens``,
        # a value cached once at request creation. The base session update swaps
        # ``session.sampling_params`` for each chunk but never refreshes the
        # cached ``session.max_tokens`` (even though ``StreamingUpdate`` carries
        # one). Without this, a chunk that raises ``max_tokens`` — e.g. handing
        # the request off to a free-running acoustic tail once the text stream is
        # exhausted — is silently capped at the request's *initial* ``max_tokens``
        # (1 in the one-frame-per-chunk streaming-text path), so every segment,
        # the tail included, stops after a single decoded frame.
        new_max_tokens = getattr(update, "max_tokens", None)
        if new_max_tokens is not None:
            session.max_tokens = new_max_tokens

        # At stage_id != 0 the base class already routed through
        # ``_replace_session_with_streaming_update`` (which sets
        # ``additional_information``); only stage 0 drops it.
        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info
