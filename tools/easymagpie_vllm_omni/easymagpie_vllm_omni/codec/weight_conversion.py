# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Weight conversion helpers shared by the CLI and parity tests."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Materialize PyTorch weight norm with its default output-channel dim."""
    norm_dims = tuple(range(1, v.dim()))
    return v * (g / torch.linalg.vector_norm(v, dim=norm_dims, keepdim=True))


def convert_decoder_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the decoder, fold weight norm, and rename NeMo activations."""
    prefix = "audio_decoder."
    decoder = {name: tensor for name, tensor in state.items() if name.startswith(prefix)}
    converted: dict[str, torch.Tensor] = {}

    suffix_v = ".parametrizations.weight.original1"
    suffix_g = ".parametrizations.weight.original0"
    for name, value in decoder.items():
        if name.endswith(suffix_g):
            continue
        if name.endswith(suffix_v):
            base = name[: -len(suffix_v)]
            g_name = base + suffix_g
            if g_name not in decoder:
                raise KeyError(f"missing weight-norm magnitude {g_name}")
            converted[base + ".weight"] = fold_weight_norm(decoder[g_name], value).contiguous()
            continue

        renamed = name.replace(".activation.activation.snake_act.alpha", ".activation.alpha")
        renamed = renamed.replace(".output_activation.activation.snake_act.alpha", ".output_activation.alpha")
        converted[renamed] = value.contiguous()

    return converted


def convert_encoder_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the codec encoder and materialize all weight-normalized convolutions."""
    prefix = "audio_encoder."
    encoder = {name: tensor for name, tensor in state.items() if name.startswith(prefix)}
    if not encoder:
        raise KeyError(f"checkpoint does not contain any {prefix} weights")

    converted: dict[str, torch.Tensor] = {}
    suffix_v = ".parametrizations.weight.original1"
    suffix_g = ".parametrizations.weight.original0"
    for name, value in encoder.items():
        if name.endswith(suffix_g):
            continue
        if name.endswith(suffix_v):
            base = name[: -len(suffix_v)]
            g_name = base + suffix_g
            if g_name not in encoder:
                raise KeyError(f"missing weight-norm magnitude {g_name}")
            converted[base + ".weight"] = fold_weight_norm(encoder[g_name], value).contiguous()
            continue
        converted[name] = value.contiguous()

    return converted
