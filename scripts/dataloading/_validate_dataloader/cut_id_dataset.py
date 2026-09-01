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
"""No-op dataset that materializes stable per-example validation identities.

The sampler/dataloader machinery decides *which* examples each call gets,
which is exactly the question the validator answers. Indexed examples carry
the graph-origin token used by checkpoint/restore; that token, rather than a
source-provided semantic ``id`` field, is the authoritative identity. Some
valid corpora reuse semantic IDs (for example, class labels ``0`` through
``6``) across many examples.
"""

import json

import torch.utils.data
from lhotse.lazy import get_graph_origin


def _validation_identity(cut) -> str:
    """Return a canonical identity for partition/resume comparisons."""
    token = get_graph_origin(cut)
    if token is None:
        # Preserve usefulness for legacy/non-indexed validator runs while
        # keeping this namespace distinct from indexed graph tokens.
        return "semantic:" + json.dumps(str(cut.id), ensure_ascii=False)
    try:
        encoded = json.dumps(
            token, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except TypeError as error:
        raise TypeError(
            "Indexed example graph-origin token is not JSON-serializable: "
            f"type={type(token).__name__}"
        ) from error
    return f"graph:{encoded}"


def _nonnegative_float(value, *, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0.0 else default


def _nonnegative_int(value, *, default: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _declared_audio_duration(example) -> float:
    """Return declared audio seconds without loading any payload.

    Raw cuts expose ``duration`` directly. Prompt-formatted conversation
    examples instead expose their component cuts through ``list_cuts()``;
    text-only conversations legitimately return an empty list.
    """
    direct = getattr(example, "duration", None)
    if direct is not None:
        return _nonnegative_float(direct, default=0.0)
    list_cuts = getattr(example, "list_cuts", None)
    if not callable(list_cuts):
        return 0.0
    return sum(
        _nonnegative_float(getattr(cut, "duration", None), default=0.0)
        for cut in list_cuts()
    )


class CutIdDataset(torch.utils.data.Dataset):
    """Return graph identities and worker metadata without realizing audio.

    ``cut_ids`` is retained as the validator JSON schema field name, but its
    values are canonical graph identities when graph-origin metadata is
    available. ``semantic_cut_ids`` preserves source IDs for diagnostics.
    """

    def __getitem__(self, cuts):
        info = torch.utils.data.get_worker_info()
        return {
            "cut_ids": [_validation_identity(cut) for cut in cuts],
            "semantic_cut_ids": [str(cut.id) for cut in cuts],
            "declared_duration_seconds": [
                _declared_audio_duration(cut)
                for cut in cuts
            ],
            "sampled_num_tokens": [
                _nonnegative_int(getattr(cut, "num_tokens", None), default=-1)
                for cut in cuts
            ],
            "source_groups": [
                str(getattr(cut, "validation_source_group", "")) for cut in cuts
            ],
            "source_ids": [
                str(getattr(cut, "validation_source_id", "")) for cut in cuts
            ],
            "worker_id": int(info.id) if info is not None else 0,
            "num_workers": int(info.num_workers) if info is not None else 1,
        }
