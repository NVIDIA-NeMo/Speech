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
"""Helpers for pulling decoded audio from vLLM-Omni stage outputs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


def extract_audio_from_stage_output(stage_output) -> Optional[tuple[Any, int]]:
    """Return ``(waveform, sample_rate)`` from a final-stage output."""
    import torch

    if isinstance(stage_output, (list, tuple)):
        for item in reversed(stage_output):
            extracted = extract_audio_from_stage_output(item)
            if extracted is not None:
                return extracted
        return None

    mm = getattr(stage_output, "multimodal_output", None)
    ro = getattr(stage_output, "request_output", stage_output)
    if not isinstance(mm, Mapping):
        outputs = getattr(ro, "outputs", None)
        if not outputs:
            return None
        mm = getattr(outputs[0], "multimodal_output", None)
    if not isinstance(mm, Mapping):
        return None

    audio_data = mm.get("audio")
    if audio_data is None:
        audio_data = mm.get("model_outputs")
    if audio_data is None:
        return None

    if isinstance(audio_data, list):
        chunks = [t for t in audio_data if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not chunks:
            return None
        try:
            wav_t = torch.cat([t.reshape(-1) for t in chunks], dim=0)
        except RuntimeError:
            wav_t = chunks[-1].reshape(-1)
    elif isinstance(audio_data, torch.Tensor):
        wav_t = audio_data.reshape(-1)
    else:
        wav_t = torch.as_tensor(audio_data).reshape(-1)

    wav_t = wav_t.detach().float().cpu()
    if wav_t.numel() == 0:
        return None

    sr = mm.get("sr")
    sr_val = sr[-1] if isinstance(sr, (list, tuple)) and sr else sr
    if isinstance(sr_val, torch.Tensor):
        sr_val = int(sr_val.reshape(-1)[0].item())
    return wav_t.numpy(), int(sr_val or 22050)
