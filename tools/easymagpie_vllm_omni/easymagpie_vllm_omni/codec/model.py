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
"""vLLM/vLLM-Omni model adapter for the stateful packed codec."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packed import PackedEasyMagpieCodec
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm_omni.model_executor.models.output_templates import OmniOutput


class EasyMagpieCodecForConditionalGeneration(nn.Module):
    """One-placeholder-per-frame vLLM model producing packed waveform chunks.

    Real code matrices arrive through ``runtime_additional_information``. The
    scheduled ``input_ids`` are only placeholders, but their per-request counts
    must equal the number of EasyMagpie model frames so vLLM metadata is at the
    base time resolution used by every cache layer.
    """

    input_modalities = "audio"
    has_inner_state = True
    is_attention_free = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int]]:
        from easymagpie_vllm_omni.codec.packed import CODEC_STATE_ELEMENTS

        return ((CODEC_STATE_ELEMENTS,),)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype]:
        return (vllm_config.model_config.dtype,)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.linear_attention_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, EasyMagpieCodecConfig):
            # AutoConfig registration is process-global and some inspection
            # paths may deserialize a generic PretrainedConfig first.
            config = EasyMagpieCodecConfig(**config.to_dict())
        self.config = config
        self.vllm_config = vllm_config
        self.codec = PackedEasyMagpieCodec(
            config,
            dtype=vllm_config.model_config.dtype,
            prefix=prefix,
        )
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), dtype=torch.float32, device=input_ids.device)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _payload_codes(
        self,
        runtime_infos: list[dict[str, Any]],
        device: torch.device,
        request_token_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        q = self.config.num_stacked_codebooks
        packed: list[torch.Tensor] = []
        frame_counts: list[int] = []
        if request_token_spans is not None and len(request_token_spans) != len(runtime_infos):
            raise ValueError(f"got {len(request_token_spans)} request spans for {len(runtime_infos)} codec payloads")
        for index, info in enumerate(runtime_infos):
            scheduled_frames = None
            if request_token_spans is not None:
                start, end = request_token_spans[index]
                scheduled_frames = end - start
                if scheduled_frames == 0:
                    continue
            codes = info.get("codes", {}) if isinstance(info, dict) else {}
            audio = codes.get("audio") if isinstance(codes, dict) else None
            tensor = torch.as_tensor(audio, dtype=torch.long, device=device)
            if tensor.ndim == 2:
                if tensor.shape[1] != q:
                    raise ValueError(f"expected codec payload [frames, {q}], got {tuple(tensor.shape)}")
                rows = tensor.contiguous()
            elif tensor.ndim == 1:
                flat = tensor.reshape(-1)
                if flat.numel() % q:
                    raise ValueError(f"codec payload length {flat.numel()} is not divisible by {q}")
                frames = flat.numel() // q
                rows = flat.view(q, frames).transpose(0, 1).contiguous()
            else:
                raise ValueError(f"expected a 1-D or 2-D codec payload, got {tuple(tensor.shape)}")
            if scheduled_frames is not None and rows.shape[0] != scheduled_frames:
                raise ValueError(
                    f"scheduled {scheduled_frames} placeholders for request {index}, "
                    f"but its codec payload has {rows.shape[0]} frames"
                )
            packed.append(rows)
            frame_counts.append(int(rows.shape[0]))
        if not packed:
            return torch.empty((0, q), dtype=torch.long, device=device), []
        return torch.cat(packed, dim=0), frame_counts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        codec_codes: torch.Tensor | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        **_: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            input_ids = torch.empty((0,), dtype=torch.long, device=self.vllm_config.device_config.device)

        if codec_codes is not None:
            codes = codec_codes.to(device=input_ids.device, dtype=torch.long)
            frame_counts = [int(codes.shape[0])]
        elif runtime_additional_information:
            codes, frame_counts = self._payload_codes(
                runtime_additional_information,
                input_ids.device,
                request_token_spans,
            )
        else:
            # Profile runs do not carry connector payloads. One scheduled
            # placeholder still represents one model frame.
            frames = int(input_ids.numel())
            codes = torch.zeros(
                (frames, self.config.num_stacked_codebooks),
                dtype=torch.long,
                device=input_ids.device,
            )
            frame_counts = [frames]

        if codes.shape[0] != input_ids.numel():
            raise ValueError(
                "EasyMagpie native codec requires one scheduled placeholder per model frame: "
                f"got {input_ids.numel()} placeholders and {codes.shape[0]} code frames"
            )
        packed_audio = self.codec(codes)
        outputs: list[torch.Tensor] = []
        frame_offset = 0
        offset = 0
        for frames in frame_counts:
            samples = frames * self.config.samples_per_frame
            valid_samples = samples
            if frames > 0:
                last = codes[frame_offset + frames - 1].view(-1, self.config.frame_stacking_factor)
                control = (last >= self.config.codebook_size).any(dim=0)
                if control.any():
                    valid_subframes = int(control.to(torch.int64).argmax().item())
                    valid_samples -= (self.config.frame_stacking_factor - valid_subframes) * (
                        self.config.samples_per_codec_frame
                    )
            outputs.append(packed_audio[offset : offset + valid_samples].float())
            frame_offset += frames
            offset += samples
        sample_rate = torch.tensor(self.config.output_sample_rate, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": [sample_rate for _ in outputs],
            },
        )

    def make_omni_output(self, model_outputs: OmniOutput | tuple, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"unexpected EasyMagpie codec output: {type(model_outputs)}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapped = ((f"codec.{name}" if name.startswith("audio_decoder.") else name, tensor) for name, tensor in weights)
        return AutoWeightsLoader(self, skip_prefixes=["codec.dequantizer."]).load_weights(mapped)
