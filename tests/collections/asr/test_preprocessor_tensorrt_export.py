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

"""TensorRT preprocessing parity, including low-energy normalization sensitivity."""

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.utils.trt_utils import build_engine


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_preprocessor_tensorrt_numerics(tmp_path, record_property, monkeypatch):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")
    trt_backend = pytest.importorskip("polygraphy.backend.trt")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(17)
    model = AudioToMelSpectrogramPreprocessor(features=128, n_fft=512, pad_to=0, dither=0.0).float().eval()
    graph_path, plan_path = tmp_path / "preprocessor.onnx", tmp_path / "preprocessor.plan"
    model.export(
        str(graph_path),
        input_example=(torch.zeros(1, 3200), torch.tensor([3200])),
        onnx_opset_version=17,
        dynamic_axes={
            "input_signal": {0: "batch", 1: "samples"},
            "length": {0: "batch"},
            "processed_signal": {0: "batch", 2: "frames"},
            "processed_length": {0: "batch"},
        },
    )
    # Expose log-mel values to distinguish frontend errors from normalization conditioning.
    graph = onnx.load(graph_path)
    log_nodes = [node for node in graph.graph.node if node.op_type == "Log"]
    assert len(log_nodes) == 1
    log_output = log_nodes[0].output[0]
    graph.graph.output.append(
        onnx.helper.make_tensor_value_info(log_output, onnx.TensorProto.FLOAT, ["batch", 128, "frames"])
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, graph_path)
    build_engine(
        onnx_path=str(graph_path),
        output_path=str(plan_path),
        fp16=False,
        tf32=False,
        input_profile={
            "input_signal": [(1, 800), (3, 3200), (3, 3200)],
            "length": [(1,), (3,), (3,)],
        },
        workspace_size=1 << 30,
    )
    model.cuda()
    raw_model = (
        AudioToMelSpectrogramPreprocessor(features=128, n_fft=512, pad_to=0, dither=0.0, normalize=None)
        .float()
        .eval()
        .cuda()
    )
    raw_model.load_state_dict(model.state_dict())
    with trt_backend.TrtRunner(trt_backend.EngineFromBytes(plan_path.read_bytes())) as runner:
        for name, signal, lengths in (
            ("ordinary", torch.randn(3, 3200), torch.tensor([3200, 2400, 0])),
            ("quiet", 1e-7 * torch.sin(torch.arange(3200)[None, :] * 0.13), torch.tensor([3200])),
            ("silence", torch.zeros(2, 800), torch.tensor([800, 0])),
        ):
            signal[torch.arange(signal.shape[1])[None, :] >= lengths[:, None]] = 0
            with torch.inference_mode():
                expected, expected_lengths = model(input_signal=signal.cuda(), length=lengths.cuda())
                expected_logs, _ = raw_model(input_signal=signal.cuda(), length=lengths.cuda())
            actual = runner.infer({"input_signal": signal.numpy(), "length": lengths.numpy()})
            np.testing.assert_array_equal(actual["processed_length"], expected_lengths.cpu().numpy())
            features = actual["processed_signal"]
            assert np.isfinite(features).all()
            logs = actual[log_output]
            valid = np.arange(logs.shape[-1])[None, None, :] < expected_lengths.cpu().numpy()[:, None, None]
            error = np.where(valid, logs - expected_logs.cpu().numpy()[..., : logs.shape[-1]], 0)
            # Use the existing feature tolerance before potentially ill-conditioned normalization.
            np.testing.assert_allclose(
                np.where(valid, logs, 0),
                np.where(valid, expected_logs.cpu().numpy()[..., : logs.shape[-1]], 0),
                rtol=1e-4,
                atol=1e-4,
            )
            record_property(name + "_max_log_error", float(np.abs(error).max()))
            record_property(name + "_max_feature_error", float(np.abs(features - expected.cpu().numpy()).max()))
            if name != "quiet":
                np.testing.assert_allclose(features, expected.cpu().numpy(), rtol=1e-4, atol=1e-4)
            for index, length in enumerate(expected_lengths.cpu().tolist()):
                assert np.count_nonzero(features[index, :, length:]) == 0
