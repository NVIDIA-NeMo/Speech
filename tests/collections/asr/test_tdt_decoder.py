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

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder
from nemo.core.classes.common import Serialization


def make_decoder(token: int = 0, duration: int = 0, device: str = "cpu", graphs: bool = False) -> TDTDecoder:
    decoder = Serialization.from_config_dict(
        OmegaConf.create(
            {
                "_target_": "nemo.collections.asr.modules.RNNTDecoder",
                "prednet": {"pred_hidden": 8, "pred_rnn_layers": 1},
                "vocab_size": 2,
            }
        )
    ).to(device)
    joint = Serialization.from_config_dict(
        OmegaConf.create(
            {
                "_target_": "nemo.collections.asr.modules.RNNTJoint",
                "jointnet": {"encoder_hidden": 4, "pred_hidden": 8, "joint_hidden": 8, "activation": "relu"},
                "num_classes": 2,
                "num_extra_outputs": 3,
                "log_softmax": False,
            }
        )
    ).to(device)
    with torch.no_grad():
        for parameter in joint.parameters():
            parameter.zero_()
        joint.joint_net[-1].bias[token] = 10
        joint.joint_net[-1].bias[3 + duration] = 10
    return TDTDecoder(decoder, joint, 2, [0, 1, 3], 2, graphs)


@pytest.mark.unit
@pytest.mark.parametrize("token,duration,expected", [(0, 0, [8, 4, 0]), (2, 0, [0, 0, 0]), (1, 2, [2, 1, 0])])
def test_complete_loop(token, duration, expected):
    decoder = make_decoder(token, duration)
    result = decoder(torch.randn(3, 4, 4), torch.tensor([4, 2, 0]))
    assert result.lengths.tolist() == expected
    for b, length in enumerate(expected):
        assert result.tokens[b, :length].tolist() == [token] * length
        assert result.tokens[b, length:].eq(-1).all()
        if length:
            timestamps = (
                [t for t in range([4, 2, 0][b]) for _ in range(2)]
                if duration == 0
                else list(range(0, [4, 2, 0][b], 3))
            )
            assert result.timestamps[b, :length].tolist() == timestamps
            assert result.durations[b, :length].tolist() == [[0, 1, 3][duration]] * length
    snapshot = tuple(t.clone() for t in result)
    decoder(torch.randn(1, 4, 2), torch.tensor([1]))
    for value, saved in zip(result, snapshot):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


@pytest.mark.unit
def test_artifact_roundtrip(tmp_path):
    original = make_decoder(1, 2)
    path = tmp_path / "decoder.pt"
    original.save(path)
    restored = TDTDecoder.load(path, "cpu", False)
    encoded, lengths = torch.randn(2, 4, 6), torch.tensor([6, 3])
    for actual, expected in zip(restored(encoded, lengths), original(encoded, lengths)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any("encoder" in key for key in torch.load(path, weights_only=True)["decoder_state"])
    with pytest.raises(FileExistsError):
        original.save(path)


@pytest.mark.unit
@pytest.mark.parametrize("lengths", [[-1, 2], [5, 2]])
def test_reject_invalid_lengths(lengths):
    with pytest.raises(ValueError, match="Encoded lengths"):
        make_decoder()(torch.randn(2, 4, 4), torch.tensor(lengths))


@pytest.mark.unit
def test_triton_batches_once_and_isolates_invalid_request(monkeypatch):
    class Tensor:
        def __init__(self, name, tensor):
            self.name, self.tensor = name, tensor

        def to_dlpack(self):
            return torch.utils.dlpack.to_dlpack(self.tensor)

        @staticmethod
        def from_dlpack(name, tensor):
            assert tensor.is_contiguous()
            return Tensor(name, torch.from_dlpack(tensor))

    class TritonError(str):
        INVALID_ARG = "INVALID_ARG"

        def __new__(cls, message, code=None):
            error = super().__new__(cls, message)
            error.code = code
            return error

    utils = SimpleNamespace(
        Tensor=Tensor,
        get_input_tensor_by_name=lambda request, name: request[name],
        InferenceResponse=SimpleNamespace,
        TritonError=TritonError,
    )
    monkeypatch.setitem(__import__("sys").modules, "triton_python_backend_utils", utils)
    module = import_module("examples.asr.export.transducer.tdt_triton_model")
    backend = module.TritonPythonModel()
    original = make_decoder(1, 2)
    calls = []

    class CountedDecoder:
        device = original.device
        validate_inputs = original.validate_inputs

        def __call__(self, encoded, lengths):
            calls.append(encoded.shape[0])
            return original(encoded, lengths)

    backend.decoder = CountedDecoder()

    def request(lengths):
        return {
            "ENCODER_OUTPUT": Tensor("ENCODER_OUTPUT", torch.randn(len(lengths), 4, 6)),
            "ENCODED_LENGTH": Tensor("ENCODED_LENGTH", torch.tensor(lengths)[:, None]),
        }

    responses = backend.execute([request([6]), request([8]), request([3, 0])])
    assert calls == [3]
    assert "Encoded lengths" in responses[1].error
    assert responses[1].error.code == TritonError.INVALID_ARG
    assert responses[0].output_tensors[1].tensor.tolist() == [[2]]
    assert responses[2].output_tensors[1].tensor.tolist() == [[1], [0]]


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_reuse_and_batch_changes():
    decoder = make_decoder(device="cuda", graphs=True)
    for lengths in ([4, 2, 0], [1], [0, 0], [5, 3, 2, 1]):
        out = decoder(torch.randn(len(lengths), 4, max(1, max(lengths)), device="cuda"), torch.tensor(lengths))
        assert out.lengths.tolist() == [2 * length for length in lengths]
    assert decoder.computer.cuda_graphs_mode is not None


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("force_partial", [False, True])
def test_graph_replay_after_another_instance_captures(force_partial):
    first = make_decoder(device="cuda", graphs=True)
    second = make_decoder(token=1, device="cuda", graphs=True)
    if force_partial:
        for decoder in (first, second):
            decoder.computer.force_cuda_graphs_mode(decoder.computer.CudaGraphsMode.NO_WHILE_LOOPS)

    def check(decoder, frames, token):
        lengths = torch.tensor([frames, frames // 2], device="cuda")
        result = decoder(torch.randn(2, 4, frames, device="cuda"), lengths)
        assert result.lengths.tolist() == [2 * frames, 2 * (frames // 2)]
        for batch, count in enumerate(result.lengths.tolist()):
            assert result.tokens[batch, :count].tolist() == [token] * count
        return result

    check(first, 4, 0)
    if force_partial or not hasattr(torch.cuda.CUDAGraph, "begin_capture_to_while_node"):
        assert first.computer.cuda_graphs_mode is first.computer.CudaGraphsMode.NO_WHILE_LOOPS
    saved = check(first, 400, 0)
    snapshot = tuple(value.clone() for value in saved)
    check(second, 4, 1)
    torch.cuda.empty_cache()
    check(first, 4, 0)
    check(second, 450, 1)
    torch.cuda.empty_cache()
    check(first, 400, 0)
    for value, expected in zip(saved, snapshot):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
