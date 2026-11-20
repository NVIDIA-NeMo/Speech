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
"""Transformers configuration for the native EasyMagpie codec."""

from __future__ import annotations

from math import prod

from transformers import PretrainedConfig


class EasyMagpieCodecConfig(PretrainedConfig):
    """Configuration of the 25-fps causal spectral codec decoder.

    The model consumes one packed row per EasyMagpie acoustic frame. Each row
    contains ``num_codebooks * frame_stacking_factor`` scalar FSQ indices.
    """

    model_type = "easymagpie_codec"

    def __init__(
        self,
        *,
        input_dim: int = 40,
        input_filters: int = 768,
        hidden_filters: int = 1536,
        num_hidden_layers: int = 6,
        pre_upsample_rates: list[int] | None = None,
        pre_upsample_filters: list[int] | None = None,
        resblock_upsample_rates: list[int] | None = None,
        resblock_upsample_filters: list[int] | None = None,
        kernel_size: int = 3,
        resblock_kernel_size: int = 7,
        activation: str = "half_snake",
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        num_levels_per_group: list[int] | None = None,
        frame_stacking_factor: int = 2,
        output_sample_rate: int = 22050,
        **kwargs,
    ) -> None:
        kwargs.setdefault("architectures", ["EasyMagpieCodecForConditionalGeneration"])
        kwargs.setdefault("torch_dtype", "float32")
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.input_filters = int(input_filters)
        self.hidden_filters = int(hidden_filters)
        self.num_hidden_layers = int(num_hidden_layers)
        self.pre_upsample_rates = list(pre_upsample_rates or [2])
        self.pre_upsample_filters = list(pre_upsample_filters or [768])
        self.resblock_upsample_rates = list(resblock_upsample_rates or [9, 7, 7])
        self.resblock_upsample_filters = list(resblock_upsample_filters or [384, 128, 32])
        self.kernel_size = int(kernel_size)
        self.resblock_kernel_size = int(resblock_kernel_size)
        self.activation = activation
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.num_levels_per_group = list(num_levels_per_group or [4, 4, 4, 4, 4])
        self.frame_stacking_factor = int(frame_stacking_factor)
        self.output_sample_rate = int(output_sample_rate)
        # Minimal language-model-shaped fields used by generic vLLM input allocation.
        self.vocab_size = max(self.codebook_size + 1, 2)
        self.hidden_size = 1
        self.validate()

    def validate(self) -> None:
        """Reject codec architectures the native decoder does not implement."""
        positive_fields = (
            "input_dim",
            "input_filters",
            "hidden_filters",
            "kernel_size",
            "resblock_kernel_size",
            "num_codebooks",
            "codebook_size",
            "frame_stacking_factor",
            "output_sample_rate",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.num_hidden_layers < 0:
            raise ValueError(f"num_hidden_layers cannot be negative, got {self.num_hidden_layers}")
        if len(self.pre_upsample_rates) != len(self.pre_upsample_filters):
            raise ValueError("pre_upsample_rates and pre_upsample_filters must have the same length")
        if len(self.resblock_upsample_rates) != len(self.resblock_upsample_filters):
            raise ValueError("resblock_upsample_rates and resblock_upsample_filters must have the same length")
        for name in (
            "pre_upsample_rates",
            "pre_upsample_filters",
            "resblock_upsample_rates",
            "resblock_upsample_filters",
            "num_levels_per_group",
        ):
            if any(value <= 0 for value in getattr(self, name)):
                raise ValueError(f"all {name} values must be positive")
        if self.input_dim != self.num_codebooks * len(self.num_levels_per_group):
            raise ValueError(
                "input_dim must equal num_codebooks * len(num_levels_per_group), got "
                f"{self.input_dim} and {self.num_codebooks} * {len(self.num_levels_per_group)}"
            )
        if any(level <= 1 for level in self.num_levels_per_group):
            raise ValueError("all num_levels_per_group values must be greater than 1")
        if prod(self.num_levels_per_group) != self.codebook_size:
            raise ValueError(
                "codebook_size must equal the product of num_levels_per_group, got "
                f"{self.codebook_size} and {prod(self.num_levels_per_group)}"
            )
        if self.activation != "half_snake":
            raise ValueError(
                "activation must be 'half_snake' because the native codec currently hard-codes HalfSnake; "
                f"implement the matching activation in packed.py to support '{self.activation}'."
            )

        channels = self.input_filters
        for name, filters in [("pre_upsample_filters", value) for value in self.pre_upsample_filters] + [
            ("resblock_upsample_filters", value) for value in self.resblock_upsample_filters
        ]:
            if channels % filters != 0:
                raise ValueError(
                    f"decoder input channels {channels} must be divisible by {name} stage filters {filters} "
                    "because the native causal ConvTranspose1d uses groups=out_channels"
                )
            channels = filters

    @property
    def num_stacked_codebooks(self) -> int:
        return self.num_codebooks * self.frame_stacking_factor

    @property
    def samples_per_codec_frame(self) -> int:
        factor = 1
        for rate in self.pre_upsample_rates + self.resblock_upsample_rates:
            factor *= rate
        return factor

    @property
    def samples_per_frame(self) -> int:
        return self.frame_stacking_factor * self.samples_per_codec_frame
