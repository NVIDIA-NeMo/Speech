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
"""Packed decoder layers backed by vLLM-managed fixed-size state pages.

The CPU fallback intentionally loops over sequence boundaries in eager PyTorch.
CUDA execution uses packed batched Triton and cuDNN kernels without changing
the model definition, weight names, or scheduler integration.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packing import unstack_acoustic_codes
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba1_attn import (
    Mamba1AttentionBackend,
    Mamba1AttentionMetadata,
    Mamba1AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

# Largest real history is 6 * 768 = 4608 values (a k=7 skip conv).
# Every codec state layer advertises the same allocation so vLLM can place all
# layers in uniform Mamba-style cache groups.
CODEC_STATE_ELEMENTS = 4608


@dataclass
class CodecStateMetadata(Mamba1AttentionMetadata):
    codec_uniform: bool = False
    codec_max_query_len: int | None = None


class CodecStateMetadataBuilder(Mamba1AttentionMetadataBuilder):
    """Annotate vLLM's CPU-known uniform shape for the codec fast path."""

    metadata_cls = CodecStateMetadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> CodecStateMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build, **kwargs)
        uniform = metadata.num_decodes == 0 or metadata.num_prefills == 0
        max_query_len = None
        if metadata.num_prefills:
            starts = common_attn_metadata.query_start_loc_cpu[-metadata.num_prefills - 1 :]
            lengths = torch.diff(starts)
            uniform = uniform and bool(torch.all(lengths == lengths[0]).item())
            max_query_len = int(lengths.max().item())
        return replace(metadata, codec_uniform=uniform, codec_max_query_len=max_query_len)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> CodecStateMetadata:
        lengths = torch.diff(common_attn_metadata.query_start_loc_cpu)
        if lengths.numel() and not torch.all(lengths == lengths[0]).item():
            raise ValueError("EasyMagpie codec CUDA graphs require a uniform chunk size")
        return self.build(0, common_attn_metadata)


class CodecStateBackend(Mamba1AttentionBackend):
    """Mamba metadata/allocation semantics, extended to the codec's fp32 state."""

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16", "float32"]

    @staticmethod
    def get_name() -> str:
        return "EASYMAGPIE_CODEC_STATE"

    @staticmethod
    def get_builder_cls() -> type[CodecStateMetadataBuilder]:
        return CodecStateMetadataBuilder


class PackedFiniteScalarDequantizer(nn.Module):
    def __init__(self, config: EasyMagpieCodecConfig) -> None:
        super().__init__()
        levels = torch.tensor(config.num_levels_per_group, dtype=torch.int64)
        bases = torch.cumprod(
            torch.tensor([1, *config.num_levels_per_group[:-1]], dtype=torch.int64),
            dim=0,
        )
        self.num_groups = config.num_codebooks
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("bases", bases, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        nonnegative = torch.div(indices.unsqueeze(-1), self.bases, rounding_mode="floor") % self.levels
        scale = torch.div(self.levels, 2, rounding_mode="floor")
        return ((nonnegative - scale) / scale).flatten(start_dim=1)


class PackedHalfSnake(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.snake_channels = channels // 2
        # Preserve the NeMo checkpoint shape.
        self.alpha = nn.Parameter(torch.ones(1, self.snake_channels, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.is_cuda:
            from easymagpie_vllm_omni.codec.kernels import packed_half_snake

            return packed_half_snake(inputs, self.alpha)
        snake_in = inputs[:, : self.snake_channels]
        alpha = self.alpha.reshape(1, -1)
        snake_out = snake_in + torch.sin(alpha * snake_in).square() / (alpha + 1e-9)
        return torch.cat((snake_out, F.leaky_relu(inputs[:, self.snake_channels :])), dim=-1)


class CodecStateLayer(nn.Module, AttentionLayerBase):
    """Base class for a cache-owning packed codec layer."""

    def __init__(self, *, time_factor: int, dtype: torch.dtype, prefix: str) -> None:
        super().__init__()
        self.time_factor = int(time_factor)
        self.dtype = dtype
        self.kv_cache = [torch.tensor([])]

        compilation = get_current_vllm_config().compilation_config
        # vLLM's cache binder requires one unique integer in every state-layer
        # name. Architectural paths contain zero or repeated local indices.
        layer_index = sum(isinstance(layer, CodecStateLayer) for layer in compilation.static_forward_context.values())
        prefix = f"easymagpie_codec_state.{layer_index}"
        self.prefix = prefix
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CodecStateBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.mamba_block_size
        if block_size is None:
            raise ValueError("EasyMagpie codec requires a resolved mamba_block_size")
        return MambaSpec(
            block_size=block_size,
            shapes=((CODEC_STATE_ELEMENTS,),),
            dtypes=(self.dtype,),
            page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
            mamba_type=MambaAttentionBackendEnum.MAMBA1,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=0,
        )

    def _metadata(self) -> Mamba1AttentionMetadata | None:
        raw = get_forward_context().attn_metadata
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError(f"expected per-layer attention metadata, got {type(raw)}")
        metadata = raw[self.prefix]
        if not isinstance(metadata, Mamba1AttentionMetadata):
            raise TypeError(f"expected Mamba1AttentionMetadata, got {type(metadata)}")
        if metadata.num_decode_tokens != metadata.num_decodes:
            raise NotImplementedError(
                "the codec supports one frame per decode request; speculative multi-frame decode is not supported"
            )
        return metadata

    @staticmethod
    def _decode_state_indices(metadata: Mamba1AttentionMetadata) -> torch.Tensor:
        state_indices = metadata.state_indices_tensor_d
        if state_indices is None:
            raise RuntimeError("incomplete codec decode metadata")
        if state_indices.dim() == 2:
            state_indices = state_indices[:, 0]
        if state_indices.dim() != 1 or state_indices.numel() != metadata.num_decodes:
            raise RuntimeError(f"invalid codec decode state indices: {tuple(state_indices.shape)}")
        return state_indices.contiguous()

    @staticmethod
    def _prefill_max_query_len(metadata: Mamba1AttentionMetadata) -> int:
        explicit = getattr(metadata, "codec_max_query_len", None)
        if explicit is not None:
            return int(explicit)

        nums_dict = metadata.nums_dict
        if nums_dict and 8 in nums_dict:
            # vLLM builds this dictionary from query_start_loc_p_cpu specifically
            # to avoid D2H synchronization in causal-convolution launch grids.
            nums = nums_dict[8].get("nums")
            if isinstance(nums, torch.Tensor) and nums.numel():
                return int(nums.max().item()) * 8

        # Safe graph-capturable upper bound for metadata produced outside the
        # vLLM builder. It can overlaunch for a ragged batch, but sequence masks
        # preserve correctness. Benchmarks/tests set codec_max_query_len.
        return metadata.num_prefill_tokens

    def _iter_sequences(
        self,
        inputs: torch.Tensor,
        metadata: Mamba1AttentionMetadata,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor, bool]]:
        cache = self.kv_cache[0]
        offset = 0
        if metadata.num_decodes:
            state_indices_d = self._decode_state_indices(metadata)
            rows = self.time_factor
            for seq_idx in range(metadata.num_decodes):
                page = int(state_indices_d[seq_idx].item())
                if page < 0:
                    raise RuntimeError("codec decode request has no state page")
                yield inputs[offset : offset + rows], cache[page], True
                offset += rows

        if metadata.num_prefills:
            query_start_loc = metadata.query_start_loc_p
            state_indices = metadata.state_indices_tensor_p
            has_initial = metadata.has_initial_states_p
            if query_start_loc is None or state_indices is None or has_initial is None:
                raise RuntimeError("incomplete codec prefill metadata")
            for seq_idx in range(metadata.num_prefills):
                start = offset + int(query_start_loc[seq_idx].item()) * self.time_factor
                end = offset + int(query_start_loc[seq_idx + 1].item()) * self.time_factor
                page = int(state_indices[seq_idx].item())
                if page < 0:
                    raise RuntimeError("codec prefill request has no state page")
                yield inputs[start:end], cache[page], bool(has_initial[seq_idx].item())


class PackedCausalConv1d(CodecStateLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        activate: bool,
        time_factor: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__(time_factor=time_factor, dtype=dtype, prefix=prefix)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.history = kernel_size - 1
        self.activation = PackedHalfSnake(out_channels) if activate else nn.Identity()
        if self.history * self.in_channels > CODEC_STATE_ELEMENTS:
            raise ValueError("codec convolution history exceeds the uniform state page")

    def _one(self, inputs: torch.Tensor, page: torch.Tensor, has_initial: bool) -> torch.Tensor:
        state = page[: self.history * self.in_channels].view(self.history, self.in_channels)
        if not has_initial:
            state.zero_()
        joined = torch.cat((state, inputs), dim=0)
        outputs = self.conv(joined.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        state.copy_(joined[-self.history :])
        return self.activation(outputs)

    def _uniform_cuda(
        self,
        inputs: torch.Tensor,
        metadata: Mamba1AttentionMetadata,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """Run a uniform packed batch through one batched cuDNN convolution."""
        from easymagpie_vllm_omni.codec.kernels import gather_packed_state_inputs, update_packed_state

        inputs = inputs.contiguous()
        if is_decode:
            state_indices = self._decode_state_indices(metadata)
            query_start_loc = state_indices
            has_initial = state_indices
        else:
            state_indices = metadata.state_indices_tensor_p
            query_start_loc = metadata.query_start_loc_p
            has_initial = metadata.has_initial_states_p
            if state_indices is None or query_start_loc is None or has_initial is None:
                raise RuntimeError("incomplete codec prefill metadata")

        joined = gather_packed_state_inputs(
            inputs,
            self.kv_cache[0],
            state_indices,
            has_initial,
            history=self.history,
            is_decode=is_decode,
        )
        outputs = self.conv(joined.transpose(1, 2)).transpose(1, 2).contiguous()

        update_packed_state(
            inputs,
            self.kv_cache[0],
            query_start_loc,
            state_indices,
            has_initial,
            channels=self.in_channels,
            history=self.history,
            time_factor=self.time_factor,
            is_decode=is_decode,
        )
        return outputs.reshape(-1, self.conv.out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        metadata = self._metadata()
        if metadata is None:
            channels_first = inputs.transpose(0, 1).unsqueeze(0)
            outputs = self.conv(F.pad(channels_first, (self.history, 0)))
            return self.activation(outputs.squeeze(0).transpose(0, 1))
        expected_rows = (metadata.num_decode_tokens + metadata.num_prefill_tokens) * self.time_factor
        if inputs.shape[0] != expected_rows:
            raise RuntimeError(f"codec metadata describes {expected_rows} rows, got {inputs.shape[0]}")
        if inputs.is_cuda:
            from easymagpie_vllm_omni.codec.kernels import packed_causal_conv1d

            if getattr(metadata, "codec_uniform", False):
                if metadata.num_decodes and metadata.num_prefills:
                    raise NotImplementedError("uniform codec batches cannot mix prefill and decode")
                outputs = self._uniform_cuda(inputs, metadata, is_decode=bool(metadata.num_decodes))
                return self.activation(outputs)

            parts = []
            decode_rows = metadata.num_decode_tokens * self.time_factor
            if metadata.num_decodes:
                state_indices_d = self._decode_state_indices(metadata)
                parts.append(
                    packed_causal_conv1d(
                        inputs[:decode_rows],
                        self.conv.weight,
                        self.conv.bias,
                        self.kv_cache[0],
                        state_indices_d,
                        state_indices_d,
                        state_indices_d,
                        time_factor=self.time_factor,
                        is_decode=True,
                    )
                )
            if metadata.num_prefills:
                if (
                    metadata.query_start_loc_p is None
                    or metadata.state_indices_tensor_p is None
                    or metadata.has_initial_states_p is None
                ):
                    raise RuntimeError("incomplete codec prefill metadata")
                parts.append(
                    packed_causal_conv1d(
                        inputs[decode_rows:],
                        self.conv.weight,
                        self.conv.bias,
                        self.kv_cache[0],
                        metadata.query_start_loc_p,
                        metadata.state_indices_tensor_p,
                        metadata.has_initial_states_p,
                        time_factor=self.time_factor,
                        max_query_len=self._prefill_max_query_len(metadata),
                    )
                )
            if len(parts) == 1:
                outputs = parts[0]
            else:
                outputs = torch.cat(parts, dim=0) if parts else inputs.new_empty((0, self.conv.out_channels))
            return self.activation(outputs)
        outputs = [
            self._one(sequence, page, has_initial)
            for sequence, page, has_initial in self._iter_sequences(inputs, metadata)
        ]
        return torch.cat(outputs, dim=0) if outputs else inputs.new_empty((0, self.conv.out_channels))


class PackedCausalConvTranspose1d(CodecStateLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        time_factor: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__(time_factor=time_factor, dtype=dtype, prefix=prefix)
        self.in_channels = in_channels
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            groups=out_channels,
        )
        self.activation = PackedHalfSnake(out_channels)

    def _one(self, inputs: torch.Tensor, page: torch.Tensor, has_initial: bool) -> torch.Tensor:
        state = page[: self.in_channels].view(1, self.in_channels)
        if not has_initial:
            state.zero_()
        joined = torch.cat((state, inputs), dim=0)
        outputs = self.conv(joined.transpose(0, 1).unsqueeze(0)).squeeze(0)
        outputs = outputs[:, self.stride : -self.stride].transpose(0, 1)
        state.copy_(joined[-1:])
        return self.activation(outputs)

    def _uniform_cuda(
        self,
        inputs: torch.Tensor,
        metadata: Mamba1AttentionMetadata,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """Run a uniform packed batch through one batched cuDNN deconvolution."""
        from easymagpie_vllm_omni.codec.kernels import gather_packed_state_inputs, update_packed_state

        inputs = inputs.contiguous()
        if is_decode:
            state_indices = self._decode_state_indices(metadata)
            query_start_loc = state_indices
            has_initial = state_indices
        else:
            state_indices = metadata.state_indices_tensor_p
            query_start_loc = metadata.query_start_loc_p
            has_initial = metadata.has_initial_states_p
            if state_indices is None or query_start_loc is None or has_initial is None:
                raise RuntimeError("incomplete codec prefill metadata")

        joined = gather_packed_state_inputs(
            inputs,
            self.kv_cache[0],
            state_indices,
            has_initial,
            history=1,
            is_decode=is_decode,
        )
        outputs = self.conv(joined.transpose(1, 2))[:, :, self.stride : -self.stride]
        outputs = outputs.transpose(1, 2).contiguous()

        update_packed_state(
            inputs,
            self.kv_cache[0],
            query_start_loc,
            state_indices,
            has_initial,
            channels=self.in_channels,
            history=1,
            time_factor=self.time_factor,
            is_decode=is_decode,
        )
        return outputs.reshape(-1, self.conv.out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        metadata = self._metadata()
        if metadata is None:
            outputs = self.conv(inputs.transpose(0, 1).unsqueeze(0)).squeeze(0)
            return self.activation(outputs[:, : -self.stride].transpose(0, 1))
        expected_rows = (metadata.num_decode_tokens + metadata.num_prefill_tokens) * self.time_factor
        if inputs.shape[0] != expected_rows:
            raise RuntimeError(f"codec metadata describes {expected_rows} rows, got {inputs.shape[0]}")
        if inputs.is_cuda:
            from easymagpie_vllm_omni.codec.kernels import packed_causal_conv_transpose1d

            if getattr(metadata, "codec_uniform", False):
                if metadata.num_decodes and metadata.num_prefills:
                    raise NotImplementedError("uniform codec batches cannot mix prefill and decode")
                outputs = self._uniform_cuda(inputs, metadata, is_decode=bool(metadata.num_decodes))
                return self.activation(outputs)

            parts = []
            decode_rows = metadata.num_decode_tokens * self.time_factor
            if metadata.num_decodes:
                state_indices_d = self._decode_state_indices(metadata)
                parts.append(
                    packed_causal_conv_transpose1d(
                        inputs[:decode_rows],
                        self.conv.weight,
                        self.conv.bias,
                        self.kv_cache[0],
                        state_indices_d,
                        state_indices_d,
                        state_indices_d,
                        stride=self.stride,
                        time_factor=self.time_factor,
                        output_channels=self.conv.out_channels,
                        is_decode=True,
                    )
                )
            if metadata.num_prefills:
                if (
                    metadata.query_start_loc_p is None
                    or metadata.state_indices_tensor_p is None
                    or metadata.has_initial_states_p is None
                ):
                    raise RuntimeError("incomplete codec prefill metadata")
                parts.append(
                    packed_causal_conv_transpose1d(
                        inputs[decode_rows:],
                        self.conv.weight,
                        self.conv.bias,
                        self.kv_cache[0],
                        metadata.query_start_loc_p,
                        metadata.state_indices_tensor_p,
                        metadata.has_initial_states_p,
                        stride=self.stride,
                        time_factor=self.time_factor,
                        output_channels=self.conv.out_channels,
                        max_query_len=self._prefill_max_query_len(metadata),
                    )
                )
            if len(parts) == 1:
                outputs = parts[0]
            else:
                outputs = torch.cat(parts, dim=0) if parts else inputs.new_empty((0, self.conv.out_channels))
            return self.activation(outputs)
        outputs = [
            self._one(sequence, page, has_initial)
            for sequence, page, has_initial in self._iter_sequences(inputs, metadata)
        ]
        return torch.cat(outputs, dim=0) if outputs else inputs.new_empty((0, self.conv.out_channels))


class PackedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int,
        *,
        time_factor: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.input_conv = PackedCausalConv1d(
            channels,
            filters,
            kernel_size,
            activate=True,
            time_factor=time_factor,
            dtype=dtype,
            prefix=f"{prefix}.input_conv",
        )
        self.skip_conv = PackedCausalConv1d(
            filters,
            channels,
            kernel_size,
            activate=False,
            time_factor=time_factor,
            dtype=dtype,
            prefix=f"{prefix}.skip_conv",
        )
        self.output_activation = PackedHalfSnake(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_activation(inputs + self.skip_conv(self.input_conv(inputs)))


class PackedResNetDecoder(nn.Module):
    def __init__(self, config: EasyMagpieCodecConfig, *, dtype: torch.dtype, prefix: str) -> None:
        super().__init__()
        factor = config.frame_stacking_factor
        self.pre_conv = PackedCausalConv1d(
            config.input_dim,
            config.input_filters,
            config.kernel_size,
            activate=False,
            time_factor=factor,
            dtype=dtype,
            prefix=f"{prefix}.pre_conv",
        )

        channels = config.input_filters
        self.pre_resblocks = nn.ModuleList()
        self.pre_up_sample_layers = nn.ModuleList()
        for index, (rate, filters) in enumerate(zip(config.pre_upsample_rates, config.pre_upsample_filters)):
            self.pre_resblocks.append(
                PackedResidualBlock(
                    channels,
                    2 * channels,
                    config.kernel_size,
                    time_factor=factor,
                    dtype=dtype,
                    prefix=f"{prefix}.pre_resblocks.{index}",
                )
            )
            self.pre_up_sample_layers.append(
                PackedCausalConvTranspose1d(
                    channels,
                    filters,
                    rate,
                    time_factor=factor,
                    dtype=dtype,
                    prefix=f"{prefix}.pre_up_sample_layers.{index}",
                )
            )
            factor *= rate
            channels = filters

        self.conv_layers = nn.ModuleList(
            PackedResidualBlock(
                channels,
                config.hidden_filters,
                config.kernel_size,
                time_factor=factor,
                dtype=dtype,
                prefix=f"{prefix}.conv_layers.{index}",
            )
            for index in range(config.num_hidden_layers)
        )

        self.resblock_up_sample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for index, (rate, filters) in enumerate(zip(config.resblock_upsample_rates, config.resblock_upsample_filters)):
            self.resblock_up_sample_layers.append(
                PackedCausalConvTranspose1d(
                    channels,
                    filters,
                    rate,
                    time_factor=factor,
                    dtype=dtype,
                    prefix=f"{prefix}.resblock_up_sample_layers.{index}",
                )
            )
            factor *= rate
            self.resblocks.append(
                PackedResidualBlock(
                    filters,
                    2 * filters,
                    config.resblock_kernel_size,
                    time_factor=factor,
                    dtype=dtype,
                    prefix=f"{prefix}.resblocks.{index}",
                )
            )
            channels = filters

        self.post_conv = PackedCausalConv1d(
            channels,
            1,
            config.resblock_kernel_size,
            activate=False,
            time_factor=factor,
            dtype=dtype,
            prefix=f"{prefix}.post_conv",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.pre_conv(inputs)
        for block, upsample in zip(self.pre_resblocks, self.pre_up_sample_layers):
            hidden = upsample(block(hidden))
        for block in self.conv_layers:
            hidden = block(hidden)
        for upsample, block in zip(self.resblock_up_sample_layers, self.resblocks):
            hidden = block(upsample(hidden))
        return self.post_conv(hidden).squeeze(-1).clamp(-1.0, 1.0)


class PackedEasyMagpieCodec(nn.Module):
    def __init__(self, config: EasyMagpieCodecConfig, *, dtype: torch.dtype, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.dequantizer = PackedFiniteScalarDequantizer(config)
        decoder_prefix = f"{prefix}.audio_decoder" if prefix else "audio_decoder"
        self.audio_decoder = PackedResNetDecoder(config, dtype=dtype, prefix=decoder_prefix)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 2 or codes.shape[-1] != self.config.num_stacked_codebooks:
            raise ValueError(
                f"expected packed [BT, {self.config.num_stacked_codebooks}] codes, got {tuple(codes.shape)}"
            )
        # [BT,C*S] -> [BT*S,C]. Sequence boundaries remain contiguous, and
        # every downstream layer scales metadata by the same S.
        unstacked = unstack_acoustic_codes(
            codes,
            num_codebooks=self.config.num_codebooks,
            frame_stacking_factor=self.config.frame_stacking_factor,
        )
        latent = self.dequantizer(unstacked.clamp(0, self.config.codebook_size - 1)).to(self.dtype)
        return self.audio_decoder(latent)
