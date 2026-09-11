# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Transformers configuration for the EasyMagpie codec encoder."""

from __future__ import annotations

from math import prod

from transformers import PretrainedConfig


class EasyMagpieCodecEncoderConfig(PretrainedConfig):
    """Configuration of the EasyMagpie multi-resolution STFT codec encoder."""

    model_type = "easymagpie_codec_encoder"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        samples_per_frame: int = 640,
        output_dim: int = 40,
        resolutions: list[list[int]] | None = None,
        resolution_filters: list[int] | None = None,
        downsample_filters: list[int] | None = None,
        downsample_rates: list[int] | None = None,
        kernel_size: int = 3,
        activation: str = "lrelu",
        pad_mode: str = "replicate",
        original_num_codebooks: int = 5,
        original_num_levels_per_group: list[int] | None = None,
        num_codebooks: int = 8,
        num_levels_per_group: list[int] | None = None,
        frame_stacking_factor: int = 2,
        audio_bos_id: int = 1024,
        audio_eos_id: int = 1025,
        context_audio_bos_id: int = 1026,
        context_audio_eos_id: int = 1027,
        embedding_dim: int = 1536,
        reference_speaker_encoder_n_layers: int = 1,
        reference_speaker_encoder_d_ffn: int = 3072,
        reference_speaker_encoder_n_heads: int = 12,
        reference_speaker_encoder_kernel_size: int = 1,
        reference_speaker_encoder_max_length: int = 4096,
        **kwargs,
    ) -> None:
        kwargs.setdefault("architectures", ["EasyMagpieCodecEncoder"])
        kwargs.setdefault("torch_dtype", "float32")
        super().__init__(**kwargs)
        self.sample_rate = int(sample_rate)
        self.samples_per_frame = int(samples_per_frame)
        self.output_dim = int(output_dim)
        if resolutions is None:
            resolutions = _default_encoder_resolutions()
        if resolution_filters is None:
            resolution_filters = [256, 384, 512, 640, 768]
        if downsample_filters is None:
            downsample_filters = [768]
        if downsample_rates is None:
            downsample_rates = [2] * len(downsample_filters)
        self.resolutions = [list(value) for value in resolutions]
        self.resolution_filters = list(resolution_filters)
        self.downsample_filters = list(downsample_filters)
        self.downsample_rates = list(downsample_rates)
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.pad_mode = pad_mode
        self.original_num_codebooks = int(original_num_codebooks)
        self.original_num_levels_per_group = list(
            [4] * 8 if original_num_levels_per_group is None else original_num_levels_per_group
        )
        self.num_codebooks = int(num_codebooks)
        self.num_levels_per_group = list([4] * 5 if num_levels_per_group is None else num_levels_per_group)
        self.frame_stacking_factor = int(frame_stacking_factor)
        self.audio_bos_id = int(audio_bos_id)
        self.audio_eos_id = int(audio_eos_id)
        self.context_audio_bos_id = int(context_audio_bos_id)
        self.context_audio_eos_id = int(context_audio_eos_id)
        self.embedding_dim = int(embedding_dim)
        self.reference_speaker_encoder_n_layers = int(reference_speaker_encoder_n_layers)
        self.reference_speaker_encoder_d_ffn = int(reference_speaker_encoder_d_ffn)
        self.reference_speaker_encoder_n_heads = int(reference_speaker_encoder_n_heads)
        self.reference_speaker_encoder_kernel_size = int(reference_speaker_encoder_kernel_size)
        self.reference_speaker_encoder_max_length = int(reference_speaker_encoder_max_length)
        self.validate()

    def validate(self) -> None:
        positive_fields = (
            "sample_rate",
            "samples_per_frame",
            "output_dim",
            "kernel_size",
            "original_num_codebooks",
            "num_codebooks",
            "frame_stacking_factor",
            "embedding_dim",
            "reference_speaker_encoder_n_layers",
            "reference_speaker_encoder_d_ffn",
            "reference_speaker_encoder_n_heads",
            "reference_speaker_encoder_max_length",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.reference_speaker_encoder_kernel_size <= 0 or self.reference_speaker_encoder_kernel_size % 2 == 0:
            raise ValueError("reference_speaker_encoder_kernel_size must be a positive odd integer")
        if self.embedding_dim % self.reference_speaker_encoder_n_heads:
            raise ValueError("embedding_dim must be divisible by reference_speaker_encoder_n_heads")
        if self.context_audio_bos_id == self.context_audio_eos_id:
            raise ValueError("context audio BOS and EOS ids must be distinct")
        if not self.resolutions or any(len(resolution) != 3 for resolution in self.resolutions):
            raise ValueError("resolutions must contain at least one [n_fft, hop_length, win_length] triple")
        if len(self.resolutions) != len(self.resolution_filters):
            raise ValueError("resolutions and resolution_filters must have the same length")
        if len(self.downsample_filters) != len(self.downsample_rates):
            raise ValueError("downsample_filters and downsample_rates must have the same length")
        sequence_fields = (
            "resolution_filters",
            "downsample_filters",
            "downsample_rates",
            "original_num_levels_per_group",
            "num_levels_per_group",
        )
        for name in sequence_fields:
            if any(value <= 0 for value in getattr(self, name)):
                raise ValueError(f"all {name} values must be positive")
        if self.activation != "lrelu":
            raise ValueError(f"codec encoder activation must be 'lrelu', got {self.activation!r}")
        if self.pad_mode != "replicate":
            raise ValueError(f"codec encoder pad_mode must be 'replicate', got {self.pad_mode!r}")
        original_dim = self.original_num_codebooks * len(self.original_num_levels_per_group)
        target_dim = self.num_codebooks * len(self.num_levels_per_group)
        if self.output_dim != original_dim or self.output_dim != target_dim:
            raise ValueError(
                "output_dim must equal both FSQ layouts, got "
                f"{self.output_dim}, {self.original_num_codebooks} * "
                f"{len(self.original_num_levels_per_group)}, and "
                f"{self.num_codebooks} * {len(self.num_levels_per_group)}"
            )
        if any(level <= 1 for level in self.original_num_levels_per_group + self.num_levels_per_group):
            raise ValueError("all FSQ levels must be greater than 1")
        if len(set(self.original_num_levels_per_group + self.num_levels_per_group)) != 1:
            raise ValueError("direct FSQ repacking requires the same scalar level count in every dimension")
        encoder_stride = self.resolutions[0][1]
        encoder_stride *= 2 ** (len(self.resolutions) - 1)
        encoder_stride *= prod(self.downsample_rates)
        if encoder_stride != self.samples_per_frame:
            raise ValueError(
                f"samples_per_frame must match the encoder stride, got {self.samples_per_frame} and {encoder_stride}"
            )

    @property
    def codebook_size(self) -> int:
        return prod(self.num_levels_per_group)

    @property
    def num_stacked_codebooks(self) -> int:
        return self.num_codebooks * self.frame_stacking_factor


def _default_encoder_resolutions() -> list[list[int]]:
    return [[80, 20, 80], [160, 40, 160], [320, 80, 320], [640, 160, 640], [1280, 320, 1280]]
