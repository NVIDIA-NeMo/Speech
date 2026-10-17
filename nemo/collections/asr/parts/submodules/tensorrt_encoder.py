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
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRTEncoder requires TensorRT 10.x. Install NeMo with pip install 'nemo-toolkit[asr,trt]'."
            ) from exc
        if int(trt.__version__.split(".")[0]) < 10:
            raise ImportError("TensorRTEncoder requires TensorRT 10 or newer.")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRTEncoder requires a CUDA device.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
            if self._engine is None:
                raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
            self._validate_engine()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("Cannot create TensorRT execution context.")
        self._stream: torch.cuda.Stream | None = None
        self._buffers: dict[str, torch.Tensor] = {}

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
            buffers = self._bind_inputs(audio_signal, length)
            missing = self._context.infer_shapes()
            if missing:
                raise RuntimeError(f"Insufficient shape inputs: {missing}")
            for name in ("outputs", "encoded_lengths"):
                shape = tuple(self._context.get_tensor_shape(name))
                if any(size < 0 for size in shape):
                    raise RuntimeError(f"Output shape cannot be inferred: {name} {shape}")
                buffers[name] = torch.empty(shape, dtype=self._dtypes[name], device=self.device)
                if not self._context.set_tensor_address(name, buffers[name].data_ptr()):
                    raise RuntimeError(f"Cannot bind TensorRT output: {name}")
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT encoder execution failed.")
            # TensorRT does not register external uses with PyTorch's caching allocator.
            for tensor in buffers.values():
                tensor.record_stream(stream)
            self._buffers = buffers
        return buffers["outputs"], buffers["encoded_lengths"]

    def _validate_engine(self) -> None:
        trt = self._trt
        modes = {
            "audio_signal": trt.TensorIOMode.INPUT,
            "length": trt.TensorIOMode.INPUT,
            "outputs": trt.TensorIOMode.OUTPUT,
            "encoded_lengths": trt.TensorIOMode.OUTPUT,
        }
        names = {self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)}
        if names != set(modes):
            raise ValueError(f"Expected offline ASR encoder I/O names {set(modes)}, got {names}.")
        types = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
        }
        self._dtypes: dict[str, torch.dtype] = {}
        for name, mode in modes.items():
            if self._engine.get_tensor_mode(name) != mode:
                raise ValueError(f"Incorrect TensorRT I/O mode for {name}.")
            if self._engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Only device I/O is supported: {name}")
            if self._engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Only linear I/O is supported: {name}")
            dtype = self._engine.get_tensor_dtype(name)
            allowed = (trt.int32, trt.int64) if name in ("length", "encoded_lengths") else (trt.float16, trt.float32)
            if dtype not in allowed:
                raise ValueError(f"Unsupported TensorRT I/O dtype for {name}: {dtype}")
            self._dtypes[name] = types[dtype]

    def _bind_inputs(self, audio_signal: torch.Tensor, length: torch.Tensor) -> dict[str, torch.Tensor]:
        buffers = {}
        for name, tensor in (("audio_signal", audio_signal), ("length", length)):
            tensor = tensor.to(dtype=self._dtypes[name]).contiguous()
            bounds = self._engine.get_tensor_profile_shape(name, 0)
            if any(size < lo or size > hi for size, lo, hi in zip(tensor.shape, bounds[0], bounds[2])):
                raise ValueError(f"Input shape is outside the engine profile: {name} {tuple(tensor.shape)}")
            if not self._context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"Invalid TensorRT input shape: {name} {tuple(tensor.shape)}")
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Cannot bind TensorRT input: {name}")
            buffers[name] = tensor
        return buffers
