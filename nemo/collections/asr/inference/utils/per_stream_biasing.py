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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import Tensor

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.parts.context_biasing.biasing_multi_model import GPUBiasingMultiModelBase


def build_multi_biasing_ids_np(
    states: Sequence[Any],
    biasing_multi_model: GPUBiasingMultiModelBase,
    tokenizer: TokenizerSpec,
) -> np.ndarray:
    """Build per-stream biasing model ids; ``-1`` means no biasing for that stream."""
    ids_np = np.full([len(states)], fill_value=-1, dtype=np.int64)
    for i, state in enumerate(states):
        if not state.has_biasing_request():
            continue

        biasing_cfg = state.options.biasing_cfg
        model_id = biasing_cfg.multi_model_id
        if model_id is not None and not biasing_multi_model.model2active[model_id].item():
            biasing_cfg.multi_model_id = None
            model_id = None

        if model_id is None:
            if biasing_cfg.auto_manage_multi_model:
                with torch.inference_mode():
                    biasing_cfg.add_to_multi_model(tokenizer=tokenizer, biasing_multi_model=biasing_multi_model)
                model_id = biasing_cfg.multi_model_id
            else:
                logging.warning("Biasing request is not empty, not auto managed and not compiled. Skipping")
                continue

        ids_np[i] = model_id
    return ids_np


def multi_biasing_ids_tensor_from_states(
    states: Sequence[Any],
    device: torch.device,
    *,
    per_stream_biasing_enabled: bool,
) -> Tensor | None:
    """Build decode-time biasing ids from ``state.options.biasing_cfg`` (after registration)."""
    if not per_stream_biasing_enabled:
        return None

    ids_np = np.full([len(states)], fill_value=-1, dtype=np.int64)
    for i, state in enumerate(states):
        if not state.has_biasing_request():
            continue
        model_id = state.options.biasing_cfg.multi_model_id
        if model_id is None:
            logging.warning(f"Boosting tree requested in index {i}, not compiled, skipping")
            continue
        ids_np[i] = model_id

    if (ids_np < 0).all():
        return None
    return torch.from_numpy(ids_np).to(device=device)


def release_all_biasing_models(biasing_multi_model: GPUBiasingMultiModelBase, states: Sequence[Any]) -> None:
    """Remove every active biasing model and clear per-stream ``multi_model_id`` bookkeeping."""
    active_model_ids = [
        model_id
        for model_id in range(biasing_multi_model.num_models)
        if biasing_multi_model.model2active[model_id].item()
    ]
    with torch.inference_mode():
        for model_id in sorted(active_model_ids, reverse=True):
            biasing_multi_model.remove_model(model_id)
    for state in states:
        if state.has_biasing_request():
            state.options.biasing_cfg.multi_model_id = None


def release_auto_managed_stream_biasing(state: Any, biasing_multi_model: GPUBiasingMultiModelBase) -> None:
    """Drop an auto-managed biasing model when a single stream ends."""
    if not state.has_biasing_request():
        return
    if state.options.biasing_cfg.auto_manage_multi_model:
        state.options.biasing_cfg.remove_from_multi_model(biasing_multi_model)
