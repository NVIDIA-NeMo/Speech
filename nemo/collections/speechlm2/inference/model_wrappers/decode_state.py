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

"""State and result types for streaming S2S inference.

These dataclasses define the *model wrapper's* interface contract:

* :class:`StreamingDecodeState` — mutable per-stream decode state
  (KV caches, token workspaces, perception/codec caches).  Created by
  the wrapper, mutated in-place by ``infer_one_step``, and held between
  steps by the pipeline's context manager.

* :class:`InferenceStepResult` — immutable per-step outputs returned
  by ``infer_one_step`` (predicted tokens, text strings, audio).

Defined here (in ``model_wrappers/``) because the wrapper is the
component that knows what state it needs.  The context manager and
pipeline import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from nemo.utils import logging
from nemo.utils.timers import NamedTimer

if TYPE_CHECKING:
    from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import PerceptionCacheState


class TimingSummary(NamedTimer):
    """Accumulates per-stage wall-clock times across inference steps.

    Extends :class:`~nemo.utils.timers.NamedTimer` (``sync_cuda=True``)
    with a :meth:`log_summary` that prints a compact min/mean/max table
    at ``logging.info`` level once a stream finishes.

    Usage::

        timing.start("perception")
        # ... run perception ...
        timing.stop("perception")

        timing.log_summary(label="Stream 0", chunk_ms=240)
    """

    def __init__(self):
        super().__init__(reduction="none", sync_cuda=True)

    def stop(self, name: str = "") -> None:
        """Stop the named timer and log its duration at DEBUG level."""
        super().stop(name)
        dt_ms = self.timers[name]["dt"][-1] * 1000
        logging.debug(f"[timing] {name}: {dt_ms:.1f}ms")

    def log_summary(self, label: str = "Timing", chunk_ms: float | None = None) -> None:
        header = f"{label} timing"
        if chunk_ms is not None:
            header += f" (chunk={chunk_ms:.0f}ms)"
        parts = []
        for name, data in self.timers.items():
            times = data.get("dt", [])
            if not times:
                continue
            mean_ms = sum(times) / len(times) * 1000
            min_ms = min(times) * 1000
            max_ms = max(times) * 1000
            parts.append(f"{name}: mean={mean_ms:.1f}ms min={min_ms:.1f}ms max={max_ms:.1f}ms")
        if parts:
            logging.info(f"{header}:\n  " + "\n  ".join(parts))


class NullTimingSummary:
    """No-op stand-in for :class:`TimingSummary`."""

    def start(self, name: str = "") -> None:
        pass

    def stop(self, name: str = "") -> None:
        pass

    def log_summary(self, label: str = "Timing", chunk_ms: float | None = None) -> None:
        pass


@dataclass
class StreamingDecodeState:
    """Per-stream model-level decode state for streaming S2S inference.

    Holds KV caches, token workspaces, perception cache, and codec state
    that persist across inference steps within a single stream.
    """

    frame_idx: int
    gen_text: torch.Tensor
    gen_asr_text: torch.Tensor
    input_embeds_history: list[torch.Tensor]
    llm_cache: Any  # DynamicCache for supported native transformer backbones, otherwise None.
    tts_past_key_values: Any
    tts_code: torch.Tensor | None
    subword_mask: torch.Tensor | None
    perception_cache: "PerceptionCacheState" | None = None
    tts_codec_cache: Any = None
    llm_cache_position_offset: int = 0
    timing: TimingSummary | NullTimingSummary = field(default_factory=NullTimingSummary)


@dataclass
class InferenceStepResult:
    """Output from a single ``infer_one_step`` call.

    State mutations (caches, token workspaces, frame_idx) are applied
    in-place on :class:`StreamingDecodeState`.  This dataclass carries
    only the per-step *outputs* needed by the pipeline.
    """

    predicted_text_tokens: torch.Tensor
    asr_predicted_text_tokens: torch.Tensor
    predicted_text_strs: list[str]
    asr_predicted_text_strs: list[str]
    decoded_audio: torch.Tensor | None = None
    debug: dict | None = None


class IntermediateResultLogger:
    """Records per-frame debug data (logits, embeddings, indices) during inference.

    Tensors are kept on their original device until :meth:`build_debug_dict`
    is called, which performs a single bulk copy to CPU.
    """

    def __init__(self):
        self.text_logits: list[torch.Tensor] = []
        self.asr_logits: list[torch.Tensor] = []
        self.input_embeds: list[torch.Tensor] = []
        self.selected_frame_indices: list[int] = []

    def log_input_embeds(self, emb: torch.Tensor):
        self.input_embeds.append(emb.detach())

    def log_text_logits(self, logits: torch.Tensor):
        self.text_logits.append(logits.detach())

    def log_asr_logits(self, logits: torch.Tensor | None):
        if logits is not None:
            self.asr_logits.append(logits.detach())

    def log_selected_frame_index(self, idx: int):
        self.selected_frame_indices.append(idx)

    def build_debug_dict(
        self, source_encoded: torch.Tensor, gen_text: torch.Tensor, gen_asr_text: torch.Tensor | None
    ) -> dict:
        return {
            "source_encoded": source_encoded.detach().cpu(),
            "selected_frame_indices": self.selected_frame_indices,
            "input_embeds": torch.cat(self.input_embeds, dim=1).cpu() if self.input_embeds else None,
            "gen_text": gen_text.detach().cpu(),
            "gen_asr": gen_asr_text.detach().cpu() if gen_asr_text is not None else None,
            "text_logits": torch.stack(self.text_logits, dim=1).cpu() if self.text_logits else None,
            "asr_logits": torch.stack(self.asr_logits, dim=1).cpu() if self.asr_logits else None,
        }


class NullIntermediateResultLogger:
    """No-op stand-in for :class:`IntermediateResultLogger`."""

    def log_input_embeds(self, emb):
        pass

    def log_text_logits(self, logits):
        pass

    def log_asr_logits(self, logits):
        pass

    def log_selected_frame_index(self, idx):
        pass

    def build_debug_dict(self, *args, **kwargs):
        return None
