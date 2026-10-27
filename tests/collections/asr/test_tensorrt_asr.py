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

"""Input boundaries and request isolation for waveform TensorRT serving."""

import importlib
import re
import sys
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoderOutput
from nemo.collections.asr.parts.submodules.tensorrt_asr import TensorRTASR


def _pipeline():
    model = object.__new__(TensorRTASR)
    model.config = {
        "max_batch": 4,
        "max_samples": 1000,
        "min_samples": 800,
        "min_valid_samples": 320,
        "min_feature_frames": 8,
    }
    model.device = torch.device("cpu")
    return model


@pytest.mark.unit
@pytest.mark.parametrize("lengths", [[-1], [801], [1], [319]])
def test_waveform_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        _pipeline().validate_inputs(torch.zeros(1, 800), torch.tensor(lengths))


@pytest.mark.unit
def test_waveform_padding_and_empty_item():
    model = _pipeline()
    seen = {}

    def preprocess(**inputs):
        seen.update(inputs)
        return {"processed_signal": torch.zeros(2, 4, 6), "processed_length": torch.tensor([2, 0])}

    def encoder(features, lengths):
        assert features.shape == (2, 4, 8)
        assert lengths.tolist() == [2, 0]
        return features, lengths

    model.preprocessor, model.encoder = preprocess, encoder
    model.decoder = lambda features, lengths: (features, lengths)
    model(torch.ones(2, 400), torch.tensor([320, 0]))
    assert seen["input_signal"].shape == (2, 800)
    assert seen["input_signal"][0, :320].eq(1).all()
    assert seen["input_signal"][0, 320:].eq(0).all()
    assert seen["input_signal"][1].eq(0).all()


@pytest.mark.unit
def test_waveform_dtype_and_finiteness():
    model = _pipeline()
    for audio, lengths in (
        (torch.zeros(1, 800, dtype=torch.float64), torch.tensor([800])),
        (torch.zeros(1, 800), torch.tensor([800.0])),
        (torch.full((1, 800), float("nan")), torch.tensor([800])),
        (torch.zeros(1, 1001), torch.tensor([800])),
    ):
        with pytest.raises(ValueError):
            model.validate_inputs(audio, lengths)


@pytest.mark.unit
def test_text_policy():
    model = _pipeline()
    model.tokenizer = SimpleNamespace(decode=lambda ids: "<en-US> Hello , world")
    model._punctuation = re.compile(r"(\s+)([,])")
    model._language = re.compile(r"<[a-z]{2}-[A-Z]{2}>")
    result = TDTDecoderOutput(torch.tensor([[1, 2, -1]]), torch.tensor([2]), torch.zeros(1, 3), torch.zeros(1, 3))
    assert model.texts(result) == ["Hello, world"]


@pytest.mark.unit
def test_waveform_requests_batch_and_recover(monkeypatch):
    class Tensor:
        def __init__(self, name, data):
            self.name, self.data = name, data

        def to_dlpack(self):
            return torch.utils.dlpack.to_dlpack(self.data)

        @staticmethod
        def from_dlpack(name, tensor):
            return Tensor(name, torch.from_dlpack(tensor))

    class Error(str):
        INVALID_ARG = "INVALID_ARG"

        def __new__(cls, message, code=None):
            error = super().__new__(cls, message)
            error.code = code
            return error

    utils = SimpleNamespace(
        Tensor=Tensor,
        TritonError=Error,
        InferenceResponse=SimpleNamespace,
        get_input_tensor_by_name=lambda request, name: request[name],
    )
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", utils)
    name = "examples.asr.export.transducer.waveform_triton_model"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    backend = module.TritonPythonModel()
    calls = []
    validator = _pipeline()

    class Model:
        device = torch.device("cpu")
        validate_inputs = validator.validate_inputs

        def __call__(self, audio, lengths):
            calls.append(lengths.tolist())
            batch = len(lengths)
            return TDTDecoderOutput(
                torch.ones(batch, 1, dtype=torch.long),
                torch.ones(batch, dtype=torch.long),
                torch.zeros(batch, 1, dtype=torch.long),
                torch.ones(batch, 1, dtype=torch.long),
            )

        def texts(self, result):
            return ["hello"] * len(result.lengths)

    backend.model = Model()

    def request(lengths):
        return {
            "AUDIO": Tensor("AUDIO", torch.zeros(len(lengths), 800)),
            "AUDIO_LENGTH": Tensor("AUDIO_LENGTH", torch.tensor(lengths)[:, None]),
        }

    responses = backend.execute([request([800]), request([-1]), request([400, 0])])
    assert calls == [[800, 400, 0]]
    assert responses[1].error.code == Error.INVALID_ARG
    assert [t.name for t in responses[0].output_tensors] == [
        "TEXT",
        "TOKENS",
        "TOKEN_LENGTH",
        "TIMESTAMPS",
        "TOKEN_DURATIONS",
    ]
    assert responses[2].output_tensors[0].data.tolist() == [["hello"], ["hello"]]
    assert backend.execute([request([800])])[0].output_tensors[0].data.tolist() == [["hello"]]


@pytest.mark.unit
def test_waveform_package_config(tmp_path, monkeypatch):
    from google.protobuf import text_format

    pytest.importorskip("grpc")
    pytest.importorskip("rapidjson")
    config_pb2 = pytest.importorskip("tritonclient.grpc.model_config_pb2")
    from examples.asr.export.transducer import package_trt_asr

    monkeypatch.setattr(package_trt_asr, "_profile_limits", lambda *args: (800, 8))
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    tokenizer = object.__new__(SentencePieceTokenizer)
    tokenizer.legacy = False
    tokenizer.tokenizer = SimpleNamespace(serialized_model_proto=lambda: b"tokenizer")
    model = SimpleNamespace(
        tokenizer=tokenizer,
        preprocessor=SimpleNamespace(
            featurizer=SimpleNamespace(exact_pad=False, pad_to=0, hop_length=160, normalize="per_feature")
        ),
        cfg=SimpleNamespace(preprocessor=SimpleNamespace(sample_rate=16000)),
        decoding=SimpleNamespace(supported_punctuation=False, strip_lang_tags=False),
    )
    plan = tmp_path / "input.plan"
    plan.write_bytes(b"plan")
    decoder = tmp_path / "decoder"
    decoder.mkdir()
    (decoder / "config.json").write_text('{"max_batch": 16}')
    output = tmp_path / "asr"
    package_trt_asr.package(model, output, plan, plan, decoder)
    config = text_format.Parse((output / "config.pbtxt").read_text(), config_pb2.ModelConfig())
    assert [item.name for item in config.input] == ["AUDIO", "AUDIO_LENGTH"]
    assert [item.name for item in config.output] == ["TEXT", "TOKENS", "TOKEN_LENGTH", "TIMESTAMPS", "TOKEN_DURATIONS"]
    assert config.max_batch_size == 16
    assert config.dynamic_batching.max_queue_delay_microseconds == 1000


@pytest.mark.unit
def test_waveform_profile_compatibility(monkeypatch):
    from pathlib import Path

    from examples.asr.export.transducer.package_trt_asr import _profile_limits
    from nemo.collections.asr.parts.submodules import tensorrt_plan

    class Plan:
        def __init__(self, path):
            self.frontend = str(path) == "preprocessor"
            self.inputs = ["input_signal", "length"] if self.frontend else ["audio_signal", "length"]
            self.outputs = (
                ["processed_signal", "processed_length"] if self.frontend else ["outputs", "encoded_lengths"]
            )
            self.dtypes = {"input_signal": torch.float32, "length": torch.int64}
            self.engine = self
            self.shapes = {}

        def get_tensor_profile_shape(self, name, profile):
            if name == "length":
                return (1,), (16,), (16,)
            if self.frontend:
                return (1, 1200), (16, 160000), (16, 640000)
            return (1, 128, 16), (16, 128, 1000), (16, 128, 4000)

        def create_execution_context(self):
            return self

        def set_input_shape(self, name, shape):
            self.shapes[name] = shape
            return True

        def infer_shapes(self):
            return []

        def get_tensor_shape(self, name):
            shape = self.shapes[self.inputs[0]]
            if name in ("processed_length", "encoded_lengths"):
                return (shape[0],)
            if self.frontend:
                return shape[0], 128, shape[1] // 160 + 1
            return shape[0], 2048, (shape[2] + 7) // 8

    monkeypatch.setattr(tensorrt_plan, "TensorRTPlan", Plan)
    decoder = {"encoder_hidden": 2048, "max_frames": 500}
    assert _profile_limits(Path("preprocessor"), Path("encoder"), decoder, 16, 639999) == (1200, 16)
    for batch, samples, config in (
        (17, 639999, decoder),
        (16, 640000, decoder),
        (16, 1199, decoder),
        (16, 639999, {**decoder, "encoder_hidden": 1024}),
        (16, 639999, {**decoder, "max_frames": 499}),
    ):
        with pytest.raises(ValueError):
            _profile_limits(Path("preprocessor"), Path("encoder"), config, batch, samples)
