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

"""Build a small recurrent decoder/joint engine and compare successive steps without model downloads."""

import sys
from pathlib import Path

import pytest
import torch

from examples.asr.export.build_trt_decoder_joint import main


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
def test_recurrent_decoder_joint_engine(tmp_path: Path, monkeypatch) -> None:
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("polygraphy")
    pytest.importorskip("onnx")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT execution requires CUDA.")
    torch.cuda.init()
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = _DecoderJoint().eval().cuda()
    source, plan = tmp_path / "decoder.onnx", tmp_path / "decoder.plan"
    inputs = (
        torch.randn(2, 12, 1, device="cuda"),
        torch.zeros(2, 1, dtype=torch.long, device="cuda"),
        torch.ones(2, dtype=torch.long, device="cuda"),
        torch.zeros(2, 2, 8, device="cuda"),
        torch.zeros(2, 2, 8, device="cuda"),
    )
    input_names = ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"]
    output_names = ["outputs", "prednet_lengths", "output_states_1", "output_states_2"]
    dynamic_axes = {name: {1 if "states" in name else 0: "batch"} for name in input_names + output_names}
    torch.onnx.export(
        model,
        inputs,
        str(source),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_trt_decoder_joint.py",
            "--onnx",
            str(source),
            "--engine",
            str(plan),
            "--opt-batch",
            "2",
            "--max-batch",
            "4",
            "--workspace-gib",
            "0.125",
        ],
    )
    main()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    dtypes = {trt.float32: torch.float32, trt.int32: torch.int32, trt.int64: torch.int64}
    with torch.inference_mode():
        for batch in (1, 4, 2):
            native_states = [torch.zeros(2, batch, 8, device="cuda") for _ in range(2)]
            exported_states = [state.clone() for state in native_states]
            for step in range(16):
                encoded = torch.randn(batch, 12, 1, device="cuda")
                tokens = torch.randint(0, 9, (batch, 1), device="cuda")
                if step == 0:
                    tokens.fill_(8)
                lengths = torch.ones(batch, dtype=torch.long, device="cuda")
                expected = model(encoded, tokens, lengths, *native_states)
                buffers = dict(zip(input_names, (encoded, tokens, lengths, *exported_states)))
                for name, value in buffers.items():
                    assert context.set_input_shape(name, tuple(value.shape))
                    assert context.set_tensor_address(name, value.data_ptr())
                assert not context.infer_shapes()
                for name in output_names:
                    buffers[name] = torch.empty(
                        tuple(context.get_tensor_shape(name)),
                        device="cuda",
                        dtype=dtypes[engine.get_tensor_dtype(name)],
                    )
                    assert context.set_tensor_address(name, buffers[name].data_ptr())
                assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()
                for name, reference in zip(output_names, expected):
                    tolerance = 0 if name == "prednet_lengths" else 1e-4
                    torch.testing.assert_close(buffers[name], reference, atol=tolerance, rtol=tolerance)
                native_states = list(expected[2:])
                exported_states = [buffers[name] for name in output_names[2:]]


class _DecoderJoint(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 8, padding_idx=8)
        self.predictor = torch.nn.LSTM(8, 8, num_layers=2, batch_first=True)
        self.encoder_projection = torch.nn.Linear(12, 8)
        self.joint = torch.nn.Linear(8, 9)

    def forward(self, encoded, tokens, lengths, hidden, cell):
        predicted, (hidden, cell) = self.predictor(self.embedding(tokens), (hidden, cell))
        scores = self.joint(torch.relu(self.encoder_projection(encoded.transpose(1, 2)) + predicted)).unsqueeze(2)
        return scores, lengths, hidden, cell
