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
"""Self-contained Torch implementation of the EasyMagpie codec encoder."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig
from easymagpie_vllm_omni.codec.reference_speaker_encoder import EasyMagpieReferenceSpeakerEncoder
from torch import nn
from torch.nn import functional as F


def _mask_sequence(inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(inputs.shape[-1], device=inputs.device)
    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 2) + (inputs.shape[-1],)
    mask = positions.view((1,) * (inputs.ndim - 1) + (-1,)) < lengths.view(-1, *([1] * (inputs.ndim - 1)))
    return inputs * mask.expand(shape).to(inputs.dtype)


def _without_autocast(device_type: str):
    if device_type in {"cpu", "cuda", "xpu", "mps"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


class Conv1dNorm(nn.Module):
    """Weight-norm-free equivalent of NeMo's inference-time Conv1dNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        activation: bool = False,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_mode,
        )
        self.activation = nn.LeakyReLU() if activation else nn.Identity()

    def forward(self, inputs: torch.Tensor, input_lens: torch.Tensor) -> torch.Tensor:
        return _mask_sequence(self.activation(self.conv(inputs)), input_lens)


class STFTProcessor(nn.Module):
    """Log-magnitude STFT with the exact window and explicit padding used by NeMo."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, pad_mode: str) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_amount = (n_fft - hop_length) // 2
        self.pad_mode = pad_mode
        self.register_buffer("window", torch.hann_window(win_length, periodic=False))

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec_lens = audio_lens // self.hop_length
        padded = F.pad(audio, (self.pad_amount, self.pad_amount), self.pad_mode)
        spectrum = torch.stft(
            padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=False,
        )
        return _mask_sequence(torch.log(torch.abs(spectrum) + 1.0), spec_lens), spec_lens


class ResidualBlockV2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, pad_mode: str) -> None:
        super().__init__()
        self.input_conv = Conv1dNorm(channels, channels, kernel_size, activation=True, pad_mode=pad_mode)
        self.skip_conv = Conv1dNorm(channels, channels, kernel_size, pad_mode=pad_mode)
        self.output_activation = nn.LeakyReLU()

    def forward(self, inputs: torch.Tensor, input_lens: torch.Tensor) -> torch.Tensor:
        residual = self.input_conv(inputs, input_lens)
        residual = self.skip_conv(residual, input_lens)
        return _mask_sequence(self.output_activation(inputs + residual), input_lens)


class STFTResidualBlock(nn.Module):
    def __init__(
        self,
        resolution: list[int],
        input_dim: int,
        filters: int,
        kernel_size: int,
        pad_mode: str,
    ) -> None:
        super().__init__()
        n_fft, hop_length, win_length = resolution
        self.down_sample_rate = 2
        self.down_sample_conv = Conv1dNorm(
            input_dim,
            filters,
            self.down_sample_rate * 2 + 1,
            stride=self.down_sample_rate,
            activation=True,
            pad_mode=pad_mode,
        )
        self.spec_processor = STFTProcessor(n_fft, hop_length, win_length, pad_mode)
        self.spec_conv = Conv1dNorm(n_fft // 2 + 1, filters, kernel_size, pad_mode=pad_mode)
        self.spec_act = nn.LeakyReLU()
        self.res_block = ResidualBlockV2(filters, kernel_size, pad_mode)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lens: torch.Tensor,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_lens = input_lens // self.down_sample_rate
        output = self.down_sample_conv(inputs, output_lens)
        spectrum, _ = self.spec_processor(audio, audio_lens)
        output = self.spec_act(output + self.spec_conv(spectrum, output_lens))
        return self.res_block(output, output_lens), output_lens


class DownSampleResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int,
        downsample_rate: int,
        pad_mode: str,
    ) -> None:
        super().__init__()
        self.down_sample_rate = downsample_rate
        self.down_sample_conv = Conv1dNorm(
            channels,
            filters,
            downsample_rate * 2 + 1,
            stride=downsample_rate,
            activation=True,
            pad_mode=pad_mode,
        )
        self.res_block = ResidualBlockV2(filters, kernel_size, pad_mode)

    def forward(self, inputs: torch.Tensor, input_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output_lens = input_lens // self.down_sample_rate
        output = self.down_sample_conv(inputs, output_lens)
        return self.res_block(output, output_lens), output_lens


class MultiResolutionSTFTEncoder(nn.Module):
    def __init__(self, config: EasyMagpieCodecEncoderConfig) -> None:
        super().__init__()
        n_fft, hop_length, win_length = config.resolutions[0]
        filters = config.resolution_filters[0]
        self.pre_spec_processor = STFTProcessor(n_fft, hop_length, win_length, config.pad_mode)
        self.pre_conv = Conv1dNorm(
            n_fft // 2 + 1, filters, config.kernel_size, activation=True, pad_mode=config.pad_mode
        )
        self.pre_res_block = ResidualBlockV2(filters, config.kernel_size, config.pad_mode)

        blocks = []
        for resolution, output_filters in zip(config.resolutions[1:], config.resolution_filters[1:]):
            blocks.append(STFTResidualBlock(resolution, filters, output_filters, config.kernel_size, config.pad_mode))
            filters = output_filters
        self.stft_blocks = nn.ModuleList(blocks)

        downsample_blocks = []
        for output_filters, rate in zip(config.downsample_filters, config.downsample_rates):
            downsample_blocks.append(
                DownSampleResidualBlock(filters, output_filters, config.kernel_size, rate, config.pad_mode)
            )
            filters = output_filters
        self.down_sample_blocks = nn.ModuleList(downsample_blocks)
        self.post_conv = Conv1dNorm(filters, config.output_dim, config.kernel_size, pad_mode=config.pad_mode)

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_lens = self.pre_spec_processor(audio, audio_lens)
        encoded = self.pre_conv(encoded, encoded_lens)
        encoded = self.pre_res_block(encoded, encoded_lens)
        for block in self.stft_blocks:
            encoded, encoded_lens = block(encoded, encoded_lens, audio, audio_lens)
        for block in self.down_sample_blocks:
            encoded, encoded_lens = block(encoded, encoded_lens)
        return self.post_conv(encoded, encoded_lens), encoded_lens


@dataclass
class EasyMagpieCodecEncoderOutput:
    """Batched acoustic tokens plus optional reference speaker embeddings.

    Iteration yields the acoustic tensors so callers written for the original
    two-tensor encoder return value keep working. Speaker outputs are exposed
    through their named attributes.
    """

    acoustic_codes: torch.Tensor
    acoustic_lens: torch.Tensor
    reference_speaker_embeddings: torch.Tensor | None = None
    reference_speaker_embedding_lens: torch.Tensor | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.acoustic_codes
        yield self.acoustic_lens


class EasyMagpieCodecEncoder(nn.Module):
    """Encode batched audio into acoustic tokens and reference speaker embeddings."""

    def __init__(self, config: EasyMagpieCodecEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = MultiResolutionSTFTEncoder(config)
        self.reference_speaker_encoder = EasyMagpieReferenceSpeakerEncoder(
            n_layers=config.reference_speaker_encoder_n_layers,
            d_model=config.embedding_dim,
            d_ffn=config.reference_speaker_encoder_d_ffn,
            n_heads=config.reference_speaker_encoder_n_heads,
            kernel_size=config.reference_speaker_encoder_kernel_size,
            max_length=config.reference_speaker_encoder_max_length,
        )
        levels = torch.tensor(config.num_levels_per_group, dtype=torch.int32)
        scalar_levels = levels.repeat(config.num_codebooks)
        target_bases = torch.cumprod(torch.tensor([1, *levels[:-1].tolist()]), dim=0, dtype=torch.int32)
        self.register_buffer("target_levels", scalar_levels.view(1, -1, 1), persistent=False)
        self.register_buffer("target_bases", target_bases.view(1, 1, -1, 1), persistent=False)

    def _pad_audio(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim != 2 or audio_lens.ndim != 1 or audio.shape[0] != audio_lens.shape[0]:
            raise ValueError("audio must be [batch, samples] and audio_lens must be [batch]")
        if torch.any(audio_lens <= 0) or torch.any(audio_lens > audio.shape[1]):
            raise ValueError("audio_lens must be positive and no larger than the input waveform")
        frame = self.config.samples_per_frame
        padded_lens = torch.div(audio_lens + frame - 1, frame, rounding_mode="floor") * frame
        max_audio_len = int(padded_lens.max().item())
        audio = _mask_sequence(audio, audio_lens)
        return F.pad(audio[:, :max_audio_len], (0, max(0, max_audio_len - audio.shape[1]))), padded_lens

    def _quantize_and_repack(self, encoded: torch.Tensor, encoded_lens: torch.Tensor) -> torch.Tensor:
        scalar_levels = self.target_levels.to(encoded.device)
        output_scale = (scalar_levels - 1).to(encoded.dtype) / 2
        output_scale = output_scale * (1 - 1e-3)
        output_offset = torch.where(scalar_levels % 2 == 0, 0.5, 0.0)
        input_shift = torch.tan(output_offset / output_scale)
        rounded = torch.round(output_scale * torch.tanh(encoded + input_shift) - output_offset)
        nonnegative = rounded.to(torch.int32) + scalar_levels // 2
        batch, _, time = nonnegative.shape
        grouped = nonnegative.view(batch, self.config.num_codebooks, -1, time)
        indices = torch.sum(grouped * self.target_bases, dim=2, dtype=torch.int32)
        return _mask_sequence(indices, encoded_lens)

    def encode(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unstacked target-FSQ codes shaped ``[B, 8, T]`` and their lengths."""
        with _without_autocast(audio.device.type):
            padded_audio, padded_lens = self._pad_audio(audio.float(), audio_lens.long())
            encoded, encoded_lens = self.audio_encoder(padded_audio, padded_lens)
            return self._quantize_and_repack(encoded.float(), encoded_lens), encoded_lens

    def stack_codes(self, codes: torch.Tensor, code_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the checkpoint's time-to-channel stacking, including EOS tail padding."""
        factor = self.config.frame_stacking_factor
        if factor == 1:
            return codes, code_lens
        batch, codebooks, time = codes.shape
        tail = (-time) % factor
        if tail:
            codes = F.pad(codes, (0, tail), value=self.config.audio_eos_id)
        output_time = codes.shape[-1] // factor
        stacked = codes.view(batch, codebooks, output_time, factor)
        stacked = stacked.permute(0, 1, 3, 2).reshape(batch, codebooks * factor, output_time)
        stacked_lens = torch.div(code_lens + factor - 1, factor, rounding_mode="floor")
        return stacked, stacked_lens

    def stack_context_codes(
        self,
        codes: torch.Tensor,
        code_lens: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add context BOS/EOS and preserve BOS outside time-to-channel stacking."""
        batch, codebooks, time = codes.shape
        if torch.any(code_lens < 0) or torch.any(code_lens > time):
            raise ValueError("context code lengths must be within the input time dimension")
        with_specials = codes.new_zeros((batch, codebooks, time + 2))
        with_specials[:, :, 0] = bos_id
        with_specials[:, :, 1 : time + 1] = codes
        for item_index, code_len in enumerate(code_lens.tolist()):
            with_specials[item_index, :, int(code_len) + 1] = eos_id

        factor = self.config.frame_stacking_factor
        special_lens = code_lens + 2
        if factor == 1:
            return with_specials, special_lens

        bos = codes.new_full((batch, codebooks * factor, 1), bos_id)
        body = with_specials[:, :, 1:]
        body_lens = special_lens - 1
        tail = (-body.shape[-1]) % factor
        if tail:
            body = F.pad(body, (0, tail), value=eos_id)
        output_time = body.shape[-1] // factor
        body = body.view(batch, codebooks, output_time, factor)
        body = body.permute(0, 1, 3, 2).reshape(batch, codebooks * factor, output_time)
        stacked_lens = 1 + torch.div(body_lens + factor - 1, factor, rounding_mode="floor")
        return torch.cat((bos, body), dim=-1), stacked_lens

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        *,
        audio_frame_embedder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> EasyMagpieCodecEncoderOutput:
        """Run one codec batch and optionally derive speaker conditioning for every item.

        ``audio_frame_embedder`` is the TTS checkpoint's shared acoustic-code
        embedding function. Keeping that table shared avoids duplicating its
        weights while the private speaker Transformer remains part of this
        codec tower.
        """
        codes, code_lens = self.encode(audio, audio_lens)
        acoustic_codes, acoustic_lens = self.stack_codes(codes, code_lens)
        output = EasyMagpieCodecEncoderOutput(acoustic_codes=acoustic_codes, acoustic_lens=acoustic_lens)
        if audio_frame_embedder is None:
            return output
        max_frames = int(code_lens.max().item())
        context_codes, context_lens = self.stack_context_codes(
            codes[..., :max_frames],
            code_lens,
            bos_id=self.config.context_audio_bos_id,
            eos_id=self.config.context_audio_eos_id,
        )
        batch, channels, time = context_codes.shape
        context_rows = context_codes.permute(0, 2, 1).reshape(batch * time, channels).long()
        context_embeddings = audio_frame_embedder(context_rows).reshape(batch, time, -1).float()
        reference_speaker_embeddings = self.reference_speaker_encoder(context_embeddings, context_lens)
        output.reference_speaker_embeddings = reference_speaker_embeddings
        output.reference_speaker_embedding_lens = context_lens
        return output


__all__ = ["EasyMagpieCodecEncoder", "EasyMagpieCodecEncoderOutput"]
