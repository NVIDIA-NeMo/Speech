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

from nemo.collections.asr.parts.submodules.tdt_decoder_artifact import (
    build_tdt_decoder,
    load_tdt_decoder,
    save_tdt_decoder,
)
from tests.collections.asr.decoding.test_rnnt_decoding import make_tdt_tensor_decoder


@pytest.mark.unit
def test_artifact_roundtrip(tmp_path):
    original = make_tdt_tensor_decoder(1, 2)
    path = tmp_path / "decoder.pt"
    save_tdt_decoder(original, path)
    restored = load_tdt_decoder(path, "cpu", False)
    encoded, lengths = torch.randn(2, 4, 6), torch.tensor([6, 3])
    for actual, expected in zip(restored.decode_tensors(encoded, lengths), original.decode_tensors(encoded, lengths)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any("encoder" in key for key in torch.load(path, weights_only=True)["decoder_state"])
    with pytest.raises(FileExistsError):
        save_tdt_decoder(original, path)


@pytest.mark.unit
def test_build_checks_networks_and_limits():
    infer = make_tdt_tensor_decoder()
    decoder, joint = infer.decoder, infer.joint
    with pytest.raises(ValueError, match="blank ID"):
        build_tdt_decoder(decoder, joint, 1, [0, 1, 3], 2, False)
    with pytest.raises(ValueError, match="durations"):
        build_tdt_decoder(decoder, joint, 2, [0, 1, 3.0], 2, False)
    with pytest.raises(ValueError, match="max_symbols_per_step"):
        build_tdt_decoder(decoder, joint, 2, [0, 1, 3], 0, False)
    assert not decoder.training and not joint.training


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
    original = make_tdt_tensor_decoder(1, 2)
    calls = []

    class CountedDecoder:
        validate_tensor_inputs = original.validate_tensor_inputs

        def decode_tensors(self, encoded, lengths):
            calls.append(encoded.shape[0])
            return original.decode_tensors(encoded, lengths)

    backend.device = torch.device("cpu")
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
