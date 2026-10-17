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

"""Tensor interface for complete offline greedy TDT decoding."""

from pathlib import Path
from typing import NamedTuple

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules.transducer_decoding.tdt_label_looping import (
    GreedyBatchedTDTLabelLoopingComputer,
)
from nemo.core.classes.common import Serialization


class TDTDecoderOutput(NamedTuple):
    """Padded CUDA or CPU outputs; only the first ``lengths[b]`` entries are valid.

    Timestamps and durations are in encoder frames, not seconds.
    Padding is -1 for tokens/timestamps and zero for durations.
    """

    tokens: torch.Tensor
    lengths: torch.Tensor
    timestamps: torch.Tensor
    durations: torch.Tensor


class TDTDecoder:
    """Decode a complete batch with NeMo's cached label loop and optional CUDA graphs.

    This component owns prediction, joint evaluation and the autoregressive loop.
    It uses PyTorch execution and does not invoke a TensorRT single-step engine.
    Calls must be serialized on one CUDA stream per instance.
    Each call starts fresh utterances; no state carries between requests.
    Language-model fusion, beam search and streaming are outside this interface.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        joint: torch.nn.Module,
        blank_id: int,
        durations: list[int],
        max_symbols_per_step: int = 10,
        use_cuda_graphs: bool = True,
    ) -> None:
        if not decoder.blank_as_pad:
            raise ValueError("Complete TDT decoding requires blank_as_pad=True.")
        if not durations or any(type(d) is not int or d < 0 for d in durations):
            raise ValueError("TDT durations must be nonnegative integers.")
        if max_symbols_per_step < 1:
            raise ValueError("max_symbols_per_step must be positive.")
        if blank_id != decoder.blank_idx or joint.num_classes_with_blank != blank_id + 1 + len(durations):
            raise ValueError("Prediction vocabulary, blank ID and TDT joint dimensions disagree.")
        self.decoder = decoder.eval()
        self.joint = joint.eval()
        self.blank_id = blank_id
        self.durations = list(durations)
        self.max_symbols_per_step = max_symbols_per_step
        self.device = next(joint.parameters()).device
        self.dtype = next(joint.parameters()).dtype
        if next(decoder.parameters()).device != self.device or next(decoder.parameters()).dtype != self.dtype:
            raise ValueError("Prediction and joint networks must use the same device and dtype.")
        self._stream = None
        self.computer = GreedyBatchedTDTLabelLoopingComputer(
            decoder=decoder,
            joint=joint,
            blank_index=blank_id,
            durations=durations,
            max_symbols_per_step=max_symbols_per_step,
            include_duration=True,
            allow_cuda_graphs=use_cuda_graphs,
        )

    def validate_inputs(self, encoded: torch.Tensor, lengths: torch.Tensor) -> None:
        """Validate batch shapes and frame lengths before entering the decoding loop."""
        if encoded.ndim != 3 or encoded.shape[0] < 1 or encoded.shape[2] < 1:
            raise ValueError("Expected nonempty encoder storage [batch, channels, time].")
        if encoded.shape[1] != self.joint.enc.in_features:
            raise ValueError("Encoder channels do not match the joint projection.")
        if lengths.ndim != 1 or lengths.shape[0] != encoded.shape[0]:
            raise ValueError("Expected one encoded length per batch item.")
        if not encoded.is_floating_point() or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected floating-point encoder output and integer lengths.")
        if bool(((lengths < 0) | (lengths > encoded.shape[2])).any()):
            raise ValueError("Encoded lengths must be between zero and the padded time dimension.")

    @torch.inference_mode()
    def __call__(self, encoded: torch.Tensor, lengths: torch.Tensor) -> TDTDecoderOutput:
        """Decode [B, C, T] features and [B] lengths into independent padded tensors.

        Inputs may reside on CPU or the component's device.
        Dtype conversion and transfer happen once, before the loop.
        Zero-length utterances produce empty transcripts.
        Output storage has at least one column, including for all-empty batches.
        """
        self.validate_inputs(encoded, lengths)
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            if self._stream is not None and stream != self._stream:
                raise ValueError("Use a separate TDTDecoder instance for each CUDA stream.")
            self._stream = stream
        encoded = encoded.to(device=self.device, dtype=self.dtype).transpose(1, 2)
        lengths = lengths.to(device=self.device, dtype=torch.long)
        with torch.autocast(device_type=self.device.type, enabled=False):
            hyps, _ = self.computer(encoded, lengths)
        batch = encoded.shape[0]
        out_lengths = hyps.current_lengths[:batch].clone()
        width = max(1, int(out_lengths.max()))
        mask = torch.arange(width, device=self.device)[None, :] < out_lengths[:, None]
        return TDTDecoderOutput(
            hyps.transcript[:batch, :width].masked_fill(~mask, -1),
            out_lengths,
            hyps.timestamps[:batch, :width].masked_fill(~mask, -1),
            hyps.token_durations[:batch, :width].masked_fill(~mask, 0),
        )

    def save(self, path: str | Path) -> None:
        """Save only prediction/joint parameters and their configuration, without the encoder."""
        payload = {
            "format_version": 1,
            "blank_id": self.blank_id,
            "durations": self.durations,
            "max_symbols_per_step": self.max_symbols_per_step,
        }
        for name in ("decoder", "joint"):
            module = getattr(self, name)
            payload[f"{name}_config"] = OmegaConf.to_container(module.to_config_dict(), resolve=True)
            payload[f"{name}_state"] = {key: value.detach().cpu() for key, value in module.state_dict().items()}
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda", use_cuda_graphs: bool = True) -> "TDTDecoder":
        """Load a trusted decoder artifact, using FP32 network parameters."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("Unsupported TDT decoder artifact version.")
        modules = {}
        for name in ("decoder", "joint"):
            module = Serialization.from_config_dict(OmegaConf.create(payload[f"{name}_config"]))
            module.load_state_dict(payload[f"{name}_state"], strict=True)
            modules[name] = module.float().eval().to(device)
        return cls(
            **modules,
            blank_id=payload["blank_id"],
            durations=payload["durations"],
            max_symbols_per_step=payload["max_symbols_per_step"],
            use_cuda_graphs=use_cuda_graphs,
        )
