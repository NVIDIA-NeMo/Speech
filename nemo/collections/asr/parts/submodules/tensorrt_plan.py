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

"""Shared TensorRT execution for ASR network components."""

from pathlib import Path

import torch


class TensorRTPlan:
    """Execute linear device tensors on the current CUDA stream.

    Calls must be serialized and callers must establish cross-stream dependencies.
    Optimization profile zero is used; output sizes must depend only on input shapes.
    Outputs own their storage.
    Keep this instance alive until queued work completes.
    For CUDA capture, enable context caching and warm each input shape outside capture; keep this object alive through all replays.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cuda", cache_contexts: bool = False) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT execution requires TensorRT 10.x. Install NeMo with pip install 'nemo-toolkit[asr,trt]'."
            ) from exc
        if int(trt.__version__.split(".")[0]) < 10:
            raise ImportError("TensorRT execution requires TensorRT 10 or newer.")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT requires a CUDA device.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.cache_contexts = cache_contexts
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self._runtime = trt.Runtime(self._logger)
            self.engine = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize {path}.")
        types = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }
        self.inputs, self.outputs, self.dtypes = [], [], {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Only device tensors are supported: {name}.")
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Only linear tensors are supported: {name}.")
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in types:
                raise ValueError(f"Unsupported dtype {dtype} for {name}.")
            self.dtypes[name] = types[dtype]
            names = self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs
            names.append(name)
        self._contexts = {}
        self._buffers = {}

    @torch.inference_mode()
    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Expected inputs {self.inputs}, received {list(inputs)}.")
        with torch.cuda.device(self.device):
            capturing = torch.cuda.is_current_stream_capturing()
            buffers = {}
            for name in self.inputs:
                tensor = inputs[name]
                if tensor.device != self.device or tensor.dtype != self.dtypes[name]:
                    raise ValueError(f"{name} must be {self.dtypes[name]} on {self.device}.")
                shape = tuple(tensor.shape)
                low, _, high = self.engine.get_tensor_profile_shape(name, 0)
                if len(shape) != len(low) or any(n < lo or n > hi for n, lo, hi in zip(shape, low, high)):
                    raise ValueError(f"{name} shape {shape} is outside the engine profile.")
                buffers[name] = tensor.contiguous()
            shapes = tuple(tuple(buffers[name].shape) for name in self.inputs)
            key = shapes if self.cache_contexts else None
            entry = self._contexts.get(key)
            if entry is None or entry[1] != shapes:
                if capturing:
                    raise RuntimeError("Warm this TensorRT input shape outside CUDA capture first.")
                context = self.engine.create_execution_context() if entry is None else entry[0]
                if context is None:
                    raise RuntimeError("Cannot create TensorRT execution context.")
                for name, shape in zip(self.inputs, shapes):
                    if not context.set_input_shape(name, shape):
                        raise ValueError(f"Invalid TensorRT shape for {name}: {shape}.")
                if context.infer_shapes():
                    raise ValueError("TensorRT output shapes must depend only on input shapes.")
                output_shapes = {name: tuple(context.get_tensor_shape(name)) for name in self.outputs}
                if any(n < 0 for shape in output_shapes.values() for n in shape):
                    raise ValueError("Data-dependent output sizes are unsupported.")
                entry = context, shapes, output_shapes
                self._contexts[key] = entry
            context, _, output_shapes = entry
            for name, shape in output_shapes.items():
                buffers[name] = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
            for name, tensor in buffers.items():
                if not context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Cannot bind TensorRT tensor {name}.")
            stream = torch.cuda.current_stream(self.device)
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")
            if not capturing:
                # Conditional graph child streams do not survive capture.
                # Their allocations belong to the graph's memory pool.
                for tensor in buffers.values():
                    tensor.record_stream(stream)
            self._buffers = buffers
            return {name: buffers[name] for name in self.outputs}
