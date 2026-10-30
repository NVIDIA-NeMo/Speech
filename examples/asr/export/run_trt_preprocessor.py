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

"""Run a waveform preprocessor plan with NumPy, TensorRT and cuda-python, without PyTorch."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def preprocess(engine_path: Path, waveforms: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accept zero-padded mono float32 waveforms and int64 sample counts at the checkpoint sample rate.

    This minimal example copies inputs and outputs between host and device on each call.
    A serving implementation should retain the engine, context and buffers and pass device outputs to the encoder.
    """
    if waveforms.dtype != np.float32 or lengths.dtype != np.int64:
        raise TypeError("Expected float32 waveforms and int64 lengths.")
    if waveforms.ndim != 2 or lengths.shape != (waveforms.shape[0],):
        raise ValueError("Expected waveforms [batch, samples] and lengths [batch].")
    if np.any(lengths < 0) or np.any(lengths > waveforms.shape[1]):
        raise ValueError("Sample counts must lie within the padded input width.")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the preprocessing engine.")
    context = engine.create_execution_context()
    arrays = {"input_signal": np.ascontiguousarray(waveforms), "length": np.ascontiguousarray(lengths)}
    for name, array in arrays.items():
        if not context.set_input_shape(name, array.shape):
            raise ValueError(f"Input {name} shape {array.shape} is outside the engine profile.")
    stream = _cuda(cudart.cudaStreamCreate())
    allocations = []
    try:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if not is_input:
                shape = tuple(context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(f"Unresolved output shape for {name}: {shape}")
                arrays[name] = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(name)))
            array = arrays[name]
            if engine.get_tensor_location(name) == trt.TensorLocation.HOST:
                pointer = array.ctypes.data
            else:
                pointer = _cuda(cudart.cudaMalloc(max(1, array.nbytes)))
                allocations.append(pointer)
                if is_input:
                    _cuda(
                        cudart.cudaMemcpy(
                            pointer, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                        )
                    )
            if not context.set_tensor_address(name, int(pointer)):
                raise RuntimeError(f"Could not bind {name}.")
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("TensorRT preprocessing execution failed.")
        _cuda(cudart.cudaStreamSynchronize(stream))
        for name in ("processed_signal", "processed_length"):
            if engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                array = arrays[name]
                _cuda(
                    cudart.cudaMemcpy(
                        array.ctypes.data,
                        context.get_tensor_address(name),
                        array.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )
                )
        return arrays["processed_signal"], arrays["processed_length"]
    finally:
        for pointer in allocations:
            _cuda(cudart.cudaFree(pointer))
        _cuda(cudart.cudaStreamDestroy(stream))


def _cuda(result):
    error, *values = result
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime error: {error}")
    return values[0] if len(values) == 1 else tuple(values)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with input_signal and length arrays.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if not output.name.endswith(".npz"):
        output = output.with_name(output.name + ".npz")
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.input, allow_pickle=False) as inputs:
        signal, lengths = preprocess(args.engine, inputs["input_signal"], inputs["length"])
    with output.open("xb") as output_file:
        np.savez(output_file, processed_signal=signal, processed_length=lengths)


if __name__ == "__main__":
    main()
