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

"""Functional tests for nvidia/se_den_sb_16k_small."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/se_den_sb_16k_small"
NEMO_FILE = "nvidia__se_den_sb_16k_small.nemo"

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
    # SchroedingerBridgeAudioToAudioModel uses loss_encoded + loss_time (no single self.loss).
    # training_step() calls self.log() which requires a Lightning trainer context, so we call
    # _step() directly with the same batch format: (input_signal, input_length, target_signal, _).
    # Input shape is (B, C, T) — batch, channel, time.
    model = _load_model()
    model.train()
    d = _DEVICE
    B, C, T = 1, 1, 16000
    input_signal = torch.randn(B, C, T, device=d)
    input_length = torch.tensor([T], device=d)
    target_signal = torch.randn(B, C, T, device=d)
    loss, _loss_enc, _loss_time = model._step(
        target_signal=target_signal,
        input_signal=input_signal,
        input_length=input_length,
    )
    assert loss.shape == torch.Size([]), f"Expected scalar loss, got shape {loss.shape}"
    loss.backward()


def test_model_inference():
    # forward() is decorated with @torch.inference_mode(), so no_grad wrapper is not needed.
    # It returns (output_signal, output_length); output_signal has shape (B, C, T).
    model = _load_model()
    model.eval()
    d = _DEVICE
    B, C, T = 1, 1, 16000
    input_signal = torch.randn(B, C, T, device=d)
    input_length = torch.tensor([T], device=d)
    output_signal, output_length = model(
        input_signal=input_signal,
        input_length=input_length,
    )
    assert output_signal.ndim == 3, f"Expected 3D output (B, C, T), got shape {output_signal.shape}"
    assert output_signal.shape[0] == B
    assert output_signal.shape[-1] == T
