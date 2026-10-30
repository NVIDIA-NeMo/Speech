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

"""TensorRT execution for offline ASR encoder exports."""

from pathlib import Path

import torch

from nemo.collections.asr.parts.submodules.tensorrt_plan import TensorRTPlan


class TensorRTEncoder:
    """Run an offline ASR encoder using PyTorch-owned CUDA tensors.

    Requires TensorRT 10.x and an engine with inputs audio_signal (B, features, T)
    and length (B,), and outputs outputs (B, channels, T') and encoded_lengths (B,).
    All I/O must use linear device tensors.
    Output dimensions must be inferable from input shapes.
    Optimization profile zero is used.

    Execution is asynchronous on the current PyTorch stream.
    Direct TensorRT bindings avoid the separate stream and per-call synchronization in Polygraphy's TrtRunner.
    Each instance is bound to one device and, after its first call, one stream.
    Calls must be serialized; use separate instances for concurrent execution.
    The caller must establish dependencies if inputs were produced on another stream.
    Keep the instance alive until its queued work completes.
    Autograd, CUDA graph capture and streaming encoder caches are not supported.

    Args:
        engine_path: Serialized TensorRT engine built for the target GPU and runtime.
        device: CUDA device on which to deserialize and execute the engine.
            Defaults to the current CUDA device.
    """

    def __init__(self, engine_path: str | Path, device: str | torch.device = "cuda") -> None:
        # The shared plan owns execution; this class supplies the encoder's typed interface.
        self._plan = TensorRTPlan(engine_path, device)
        self.device = self._plan.device
        if set(self._plan.inputs) != {"audio_signal", "length"} or set(self._plan.outputs) != {
            "outputs",
            "encoded_lengths",
        }:
            raise ValueError(
                "Expected offline ASR encoder inputs audio_signal/length and outputs outputs/encoded_lengths."
            )
        for name, dtype in self._plan.dtypes.items():
            allowed = (
                (torch.int32, torch.int64) if name in ("length", "encoded_lengths") else (torch.float16, torch.float32)
            )
            if dtype not in allowed:
                raise ValueError(f"Unsupported TensorRT I/O dtype for {name}: {dtype}")
        self._stream: torch.cuda.Stream | None = None

    @torch.inference_mode()
    def __call__(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode CUDA features and integer lengths, returning newly allocated CUDA outputs.

        Input dtypes are converted to the engine's I/O dtypes on the same device.
        Noncontiguous inputs are copied.
        Values in length must be valid frame counts for audio_signal.
        Subsequent calls do not overwrite previously returned output tensors.
        """
        if audio_signal.ndim != 3 or length.ndim != 1 or audio_signal.shape[0] != length.shape[0]:
            raise ValueError("Expected features (B, features, T) and lengths (B,) with matching batch sizes.")
        if not audio_signal.is_floating_point() or length.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected floating-point features and int32 or int64 lengths.")
        if audio_signal.device != self.device or length.device != self.device:
            raise ValueError(f"Both inputs must be on {self.device}.")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("TensorRTEncoder does not support CUDA graph capture.")
            if self._stream is not None and self._stream != stream:
                raise ValueError("Use a separate TensorRTEncoder instance for each CUDA stream.")
            self._stream = stream
            outputs = self._plan(
                audio_signal=audio_signal.to(dtype=self._plan.dtypes["audio_signal"]),
                length=length.to(dtype=self._plan.dtypes["length"]),
            )
        return outputs["outputs"], outputs["encoded_lengths"]
