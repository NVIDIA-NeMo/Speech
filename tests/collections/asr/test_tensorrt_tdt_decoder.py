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

"""Parity checks for cached TensorRT TDT network boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from examples.asr.export import build_trt_tdt_decoder as builder
from omegaconf import OmegaConf

from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.submodules.tdt_decoder_artifact import build_tdt_decoder
from nemo.collections.asr.parts.submodules.tensorrt_tdt_decoder import load_tensorrt_tdt_decoder


def _model():
    torch.manual_seed(31)
    decoder = RNNTDecoder(prednet={"pred_hidden": 8, "pred_rnn_layers": 1}, vocab_size=2).eval()
    joint = RNNTJoint(
        jointnet={"encoder_hidden": 4, "pred_hidden": 8, "joint_hidden": 8, "activation": "relu"},
        num_classes=2,
        num_extra_outputs=3,
        log_softmax=False,
    ).eval()
    return SimpleNamespace(decoder=decoder, joint=joint, cfg=OmegaConf.create({"decoding": {"durations": [0, 1, 3]}}))


@pytest.mark.unit
def test_cached_components_onnx(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    model = _model()
    monkeypatch.setattr(builder, "build_engine", lambda **kwargs: None)
    output = tmp_path / "decoder"
    builder.build(model, output, max_batch=4, max_frames=16, max_symbols=2)
    sessions = {
        name: ort.InferenceSession(str(output / (name + ".onnx")), providers=["CPUExecutionProvider"])
        for name in ("predictor", "encoder_projection", "joint")
    }
    with torch.inference_mode():
        for batch in (1, 4, 2):
            encoded = torch.randn(batch, 7, 4)
            expected_f = model.joint.project_encoder(encoded)
            actual_f = sessions["encoder_projection"].run(None, {"encoded": encoded.numpy()})[0]
            np.testing.assert_allclose(actual_f, expected_f.numpy(), atol=1e-5, rtol=1e-5)
            native_state = model.decoder.initialize_state(encoded)
            state = tuple(x.numpy() for x in native_state)
            for _ in range(8):
                labels = torch.randint(0, 3, (batch, 1))
                pred, native_state = model.decoder.predict(labels, native_state, add_sos=False)
                expected_g = model.joint.project_prednet(pred)
                g, h, c = sessions["predictor"].run(
                    None, {"labels": labels.numpy(), "hidden": state[0], "cell": state[1]}
                )
                state = h, c
                for actual, expected in zip((g, h, c), (expected_g, *native_state)):
                    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-5)
                actual = sessions["joint"].run(None, {"encoder": actual_f[:, :1].copy(), "predictor": g})[0]
                expected = model.joint.joint_after_projection(expected_f[:, :1], expected_g)
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_complete_tensorrt_tdt(tmp_path):
    pytest.importorskip("tensorrt")
    model = _model()
    # Deterministic nonblank output exercises recurrent updates and max-symbol termination.
    with torch.no_grad():
        model.joint.joint_net[-1].bias[0] += 10
        model.joint.joint_net[-1].bias[3] += 10
    output = tmp_path / "decoder"
    builder.build(model, output, max_batch=4, max_frames=512, max_symbols=2)
    reference = build_tdt_decoder(model.decoder.cuda(), model.joint.cuda(), 2, [0, 1, 3], 2, False)
    eager = load_tensorrt_tdt_decoder(output, use_cuda_graphs=False)
    graphed = load_tensorrt_tdt_decoder(output, use_cuda_graphs=True)
    partial = load_tensorrt_tdt_decoder(output, use_cuda_graphs=True)
    partial.decoding_computer.force_cuda_graphs_mode(partial.decoding_computer.CudaGraphsMode.NO_WHILE_LOOPS)
    for frames, lengths in ((4, [4, 2, 0]), (2, [1]), (400, [4, 3, 2, 0]), (5, [0, 0]), (6, [6, 3])):
        encoded = torch.randn(len(lengths), 4, frames, device="cuda")
        lengths = torch.tensor(lengths, device="cuda")
        expected = reference.decode_tensors(encoded, lengths)
        saved = eager.decode_tensors(encoded, lengths)
        for result in (saved, graphed.decode_tensors(encoded, lengths), partial.decode_tensors(encoded, lengths)):
            for actual, wanted in zip(result, expected):
                torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
        snapshots = tuple(t.clone() for t in saved)
        eager.decode_tensors(encoded, lengths)
        for actual, wanted in zip(saved, snapshots):
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    other = load_tensorrt_tdt_decoder(output, use_cuda_graphs=True)
    other.decode_tensors(encoded, lengths)
    torch.cuda.empty_cache()
    for decoder in (graphed, partial):
        for actual, wanted in zip(
            decoder.decode_tensors(encoded, lengths), reference.decode_tensors(encoded, lengths)
        ):
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)


@pytest.mark.unit
def test_plan_module_uses_fp32():
    from nemo.collections.asr.parts.submodules.tensorrt_tdt_decoder import _PlanModule

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        module = _PlanModule("cpu")
    finally:
        torch.set_default_dtype(previous)
    assert next(module.parameters()).dtype == torch.float32


@pytest.mark.unit
def test_checkpoint_symbol_limit(tmp_path, monkeypatch):
    import json

    model = _model()
    model.cfg.decoding.greedy = {"max_symbols": 3}
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "build_engine", lambda **kwargs: None)
    builder.build(model, tmp_path / "decoder")
    assert json.loads((tmp_path / "decoder/config.json").read_text())["max_symbols_per_step"] == 3
    model.cfg.decoding.greedy.max_symbols = None
    with pytest.raises(ValueError, match="max_symbols"):
        builder.build(model, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
