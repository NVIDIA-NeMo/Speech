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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lhotse.cut import Cut


@dataclass(frozen=True)
class ConvSubsamplingSpec:
    """One repeated convolution/pooling length transform.

    ``padding`` is the padding on one side, matching PyTorch's scalar
    ``padding`` argument. Each repetition applies the same transform.
    """

    kernel_size: int
    stride: int
    padding: int
    repeat: int = 1
    ceil_mode: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ConvSubsamplingSpec:
        required = {"kernel_size", "stride", "padding"}
        missing = required - set(config)
        if missing:
            raise ValueError(f"audio_token_estimator.subsampling is missing: {sorted(missing)}")
        ans = cls(
            kernel_size=int(config["kernel_size"]),
            stride=int(config["stride"]),
            padding=int(config["padding"]),
            repeat=int(config.get("repeat", 1)),
            ceil_mode=bool(config.get("ceil_mode", False)),
        )
        if ans.kernel_size <= 0 or ans.stride <= 0 or ans.padding < 0 or ans.repeat <= 0:
            raise ValueError(f"Invalid audio_token_estimator.subsampling values: {config}")
        return ans

    def __call__(self, length: int) -> int:
        for _ in range(self.repeat):
            numerator = length + 2 * self.padding - self.kernel_size
            quotient = -(-numerator // self.stride) if self.ceil_mode else numerator // self.stride
            length = quotient + 1
        return length


@dataclass(frozen=True)
class FeatureStackingSubsamplingSpec:
    """Feature-stacking length transform used by PEE encoders."""

    factor: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FeatureStackingSubsamplingSpec:
        if "factor" not in config:
            raise ValueError("audio_token_estimator.subsampling is missing: ['factor']")
        ans = cls(factor=int(config["factor"]))
        if ans.factor <= 0:
            raise ValueError(f"Invalid audio_token_estimator.subsampling values: {config}")
        return ans

    def __call__(self, length: int) -> int:
        return (length + self.factor - 1) // self.factor


SubsamplingSpec = ConvSubsamplingSpec | FeatureStackingSubsamplingSpec


def _subsampling_spec_from_config(config: Mapping[str, Any]) -> SubsamplingSpec:
    if not isinstance(config, Mapping):
        raise TypeError(
            "Each audio_token_estimator.subsampling stage must be a mapping, " f"got {type(config).__name__}"
        )
    stage_type = config.get("type", "conv")
    if stage_type == "conv":
        return ConvSubsamplingSpec.from_config(config)
    if stage_type == "feature_stacking":
        return FeatureStackingSubsamplingSpec.from_config(config)
    raise ValueError(
        "audio_token_estimator.subsampling.type must be 'conv' or " f"'feature_stacking', got {stage_type!r}"
    )


@dataclass(frozen=True)
class AudioTokenEstimator:
    """Sample-exact audio-to-model-token length estimator.

    This mirrors the integer length arithmetic of a centered STFT-style
    preprocessor followed by one or more temporal subsampling stages.
    When ``chunk_size_seconds`` is set, each audio is split exactly like SALM's
    encoder chunking path and the per-chunk output lengths are summed.
    """

    sample_rate: int
    n_fft: int
    hop_length: int
    stft_pad_amount: int
    subsampling: tuple[SubsamplingSpec, ...]
    chunk_size_seconds: float | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | AudioTokenEstimator | None,
        *,
        sample_rate: int,
    ) -> AudioTokenEstimator | None:
        if config is None or isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError(
                "audio_token_estimator must be a mapping with 'preprocessor' and "
                f"'subsampling' entries, got {type(config).__name__}"
            )

        preprocessor = config.get("preprocessor")
        if not isinstance(preprocessor, Mapping):
            raise TypeError("audio_token_estimator.preprocessor must be a mapping")
        required = {"n_fft", "hop_length", "stft_pad_amount"}
        missing = required - set(preprocessor)
        if missing:
            raise ValueError(f"audio_token_estimator.preprocessor is missing: {sorted(missing)}")

        raw_subsampling = config.get("subsampling")
        if isinstance(raw_subsampling, Mapping):
            raw_subsampling = [raw_subsampling]
        if not isinstance(raw_subsampling, Sequence) or isinstance(raw_subsampling, (str, bytes)):
            raise TypeError("audio_token_estimator.subsampling must be a mapping or list of mappings")
        subsampling = tuple(_subsampling_spec_from_config(stage) for stage in raw_subsampling)

        chunk_size_seconds = config.get("chunk_size_seconds")
        if chunk_size_seconds is not None:
            chunk_size_seconds = float(chunk_size_seconds)
            if chunk_size_seconds <= 0:
                raise ValueError("audio_token_estimator.chunk_size_seconds must be positive or null")

        ans = cls(
            sample_rate=int(sample_rate),
            n_fft=int(preprocessor["n_fft"]),
            hop_length=int(preprocessor["hop_length"]),
            stft_pad_amount=int(preprocessor["stft_pad_amount"]),
            subsampling=subsampling,
            chunk_size_seconds=chunk_size_seconds,
        )
        if ans.sample_rate <= 0 or ans.n_fft <= 0 or ans.hop_length <= 0 or ans.stft_pad_amount < 0:
            raise ValueError(f"Invalid audio_token_estimator.preprocessor values: {preprocessor}")
        return ans

    def estimate_cut(self, cut: Cut) -> int:
        if cut.sampling_rate != self.sample_rate:
            raise ValueError(
                "Exact audio token estimation requires audio to be resampled first: "
                f"cut {cut.id!r} has sampling_rate={cut.sampling_rate}, expected {self.sample_rate}."
            )
        return self.estimate_samples(cut.num_samples)

    def estimate_samples(self, num_samples: int) -> int:
        num_samples = int(num_samples)
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}")
        chunk_size = self._chunk_size_samples()
        if chunk_size is not None:
            chunk_size = max(chunk_size, self._min_chunk_size_samples())
        if chunk_size is None or num_samples <= chunk_size:
            return self._estimate_single_pass(num_samples)

        spans = [(begin, min(begin + chunk_size, num_samples)) for begin in range(0, num_samples, chunk_size)]
        min_chunk_size = self._min_chunk_size_samples()
        if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_chunk_size:
            spans[-2] = (spans[-2][0], spans[-1][1])
            spans.pop()
        return sum(self._estimate_single_pass(end - begin) for begin, end in spans)

    def _estimate_single_pass(self, num_samples: int) -> int:
        length = (num_samples + 2 * self.stft_pad_amount - self.n_fft) // self.hop_length
        for stage in self.subsampling:
            length = stage(length)
        return max(1, length)

    def _chunk_size_samples(self) -> int | None:
        if self.chunk_size_seconds is None:
            return None
        return max(1, round(self.chunk_size_seconds * self.sample_rate))

    def _min_chunk_size_samples(self) -> int:
        # Keep this in lockstep with speechlm2.parts.encoder_chunking:
        # find the first hop-aligned waveform producing at least two feature frames.
        samples = self.hop_length
        for _ in range(16):
            feature_frames = (samples + 2 * self.stft_pad_amount - self.n_fft) // self.hop_length
            if feature_frames >= 2:
                return samples
            samples += self.hop_length
        return max(samples, 2 * self.hop_length)
