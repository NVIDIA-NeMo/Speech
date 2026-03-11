# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Functional tests for nvidia/sr_ssl_flowmatching_16k_430m."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/sr_ssl_flowmatching_16k_430m"
NEMO_FILE = "nvidia__sr_ssl_flowmatching_16k_430m.nemo"

MODEL_DIR = os.environ.get(
    "NEMO_MODEL_SUPPORT_DIR",
    os.environ.get("NEMO_MODEL_SUPPORT_DIR_CI", "/home/TestData/nemo-speech-ci-models"),
)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from nemo.collections.audio.models import AudioToAudioModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = AudioToAudioModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    model = _load_model()
    model.train()
    d = _DEVICE

    # FlowMatchingAudioToAudioModel.training_step expects a batch dict with
    # input_signal (B, C, T), input_length (B,), and optionally target_signal.
    # For the SSL variant the model uses input as its own target when
    # target_signal is absent, so we only need to supply input_signal and
    # input_length.
    T = 16000
    input_signal = torch.randn(1, 1, T, device=d)
    input_length = torch.tensor([T], device=d)

    # Call _step directly to avoid the self.log() calls that require a Trainer.
    loss = model._step(
        target_signal=input_signal.clone(),
        input_signal=input_signal,
        input_length=input_length,
    )

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    loss.backward()


def test_model_inference():
    model = _load_model()
    model.eval()
    d = _DEVICE

    T = 16000
    input_signal = torch.randn(1, 1, T, device=d)
    input_length = torch.tensor([T], device=d)

    # forward() is decorated with @torch.inference_mode() so no_grad is implicit.
    output_signal, output_length = model(
        input_signal=input_signal,
        input_length=input_length,
    )

    assert output_signal is not None, "Expected non-None output signal"
    assert (
        output_signal.shape == input_signal.shape
    ), f"Expected output shape {input_signal.shape}, got {output_signal.shape}"
    assert output_length is not None, "Expected non-None output length"
