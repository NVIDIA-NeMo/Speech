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

"""Exercise a real dynamic TensorRT engine without downloading model weights."""

import sys
from pathlib import Path

import pytest
import torch

from nemo.collections.asr.parts.submodules.tensorrt_encoder import TensorRTEncoder


@pytest.mark.unit
def test_missing_tensorrt(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(ImportError, match="TensorRTEncoder requires"):
        TensorRTEncoder("unused.plan")


@pytest.fixture(scope="module", params=["float32", "float16"])
def engine_path(request, tmp_path_factory) -> Path:
    pytest.importorskip("tensorrt")
    pytest.importorskip("polygraphy")
    onnx = pytest.importorskip("onnx")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT execution requires CUDA.")
    from nemo.utils.trt_utils import build_engine

    folder = tmp_path_factory.mktemp("encoder")
    dtype = onnx.TensorProto.FLOAT if request.param == "float32" else onnx.TensorProto.FLOAT16
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["audio_signal", "audio_signal"], ["outputs"]),
            onnx.helper.make_node("Identity", ["length"], ["encoded_lengths"]),
        ],
        "encoder",
        [
            onnx.helper.make_tensor_value_info("audio_signal", dtype, ["batch", 4, "time"]),
            onnx.helper.make_tensor_value_info("length", onnx.TensorProto.INT64, ["batch"]),
        ],
        [
            onnx.helper.make_tensor_value_info("outputs", dtype, ["batch", 4, "time"]),
            onnx.helper.make_tensor_value_info("encoded_lengths", onnx.TensorProto.INT64, ["batch"]),
        ],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 9
    source = folder / "encoder.onnx"
    onnx.save(model, source)
    target = folder / "encoder.plan"
    build_engine(
        str(source),
        str(target),
        fp16=request.param == "float16",
        tf32=False,
        input_profile={"audio_signal": [(1, 4, 8), (2, 4, 16), (4, 4, 64)], "length": [(1,), (2,), (4,)]},
        workspace_size=64 * 1024**2,
        timing_cache=str(folder / "timing.cache"),
    )
    assert target.stat().st_size > 0 and (folder / "timing.cache").stat().st_size > 0
    return target


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
def test_dynamic_shapes_and_output_lifetime(engine_path):
    encoder = TensorRTEncoder(engine_path)
    retained = []
    for batch, frames in [(1, 8), (4, 64), (2, 17), (1, 9)]:
        # Integer-valued floats have exact FP16 parity, isolating I/O from arithmetic precision.
        features = torch.arange(batch * frames * 4, device="cuda", dtype=torch.float32).reshape(batch, frames, 4)
        features = features.transpose(1, 2)
        lengths = torch.full((batch,), frames, dtype=torch.int32, device="cuda")
        lengths[-1] = 0
        actual, actual_lengths = encoder(features, lengths)
        expected = (features * 2).to(actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_lengths, lengths.to(torch.int64), rtol=0, atol=0)
        retained.append((actual, expected))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
def test_stream_ownership(engine_path):
    encoder = TensorRTEncoder(engine_path)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        features = torch.ones((2, 4, 16), device="cuda")
        lengths = torch.tensor([16, 9], device="cuda")
        actual, actual_lengths = encoder(features, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, (features * 2).to(actual.dtype))
    torch.testing.assert_close(actual_lengths, lengths)
    with pytest.raises(ValueError, match="separate TensorRTEncoder"):
        encoder(features, lengths)
    independent = TensorRTEncoder(engine_path)
    result, _ = independent(features, lengths)
    torch.testing.assert_close(result, actual)


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("shape", [(1, 4, 7), (1, 4, 65), (5, 4, 16), (1, 5, 16)])
def test_profile_rejection(engine_path, shape):
    encoder = TensorRTEncoder(engine_path)
    with pytest.raises(ValueError, match="outside the engine profile"):
        encoder(torch.ones(shape, device="cuda"), torch.ones(shape[0], dtype=torch.int64, device="cuda"))


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
def test_invalid_inputs(engine_path):
    encoder = TensorRTEncoder(engine_path)
    features = torch.ones((2, 4, 16), device="cuda")
    lengths = torch.tensor([16, 9], device="cuda")
    with pytest.raises(ValueError, match="matching batch"):
        encoder(features, lengths[:1])
    with pytest.raises(ValueError, match="int32 or int64"):
        encoder(features, lengths.float())
    with pytest.raises(ValueError, match="Both inputs"):
        encoder(features.cpu(), lengths)
    with pytest.raises(ValueError, match="CUDA device"):
        TensorRTEncoder(engine_path, device="cpu")


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
def test_invalid_engine(tmp_path):
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT execution requires CUDA.")
    path = tmp_path / "invalid.plan"
    path.write_bytes(b"not an engine")
    with pytest.raises(RuntimeError, match="Cannot deserialize"):
        TensorRTEncoder(path)
