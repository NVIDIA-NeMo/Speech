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

"""Functional tests for nvidia/se_der_sb_16k_small."""

import os

import torch

MODEL_NAME = "nvidia/se_der_sb_16k_small"
NEMO_FILE = "nvidia__se_der_sb_16k_small.nemo"

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
    # Call _step() directly to avoid self.log() / self._optimizer accesses that require a Trainer.
    # Batch format (tuple path): (input_signal, input_length, target_signal, target_length)
    # Signals are shape (B, C, T); the model internally rearranges 2-D inputs to (B, 1, T).
    model = _load_model()
    model.train()
    d = _DEVICE

    batch_size = 2
    num_samples = 16000  # 1 second at 16 kHz
    # Build synthetic multi-channel (C=1) audio tensors in the correct (B, C, T) layout
    input_signal = torch.randn(batch_size, 1, num_samples, device=d)
    input_length = torch.tensor([num_samples, num_samples], device=d)
    target_signal = torch.randn(batch_size, 1, num_samples, device=d)

    loss, loss_encoded, loss_time = model._step(
        target_signal=target_signal,
        input_signal=input_signal,
        input_length=input_length,
    )

    assert isinstance(loss, torch.Tensor), "loss must be a tensor"
    assert loss.ndim == 0, "loss must be a scalar tensor"
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()


def test_model_inference():
    # forward() is decorated with @torch.inference_mode() and returns (output_signal, output_length).
    model = _load_model()
    model.eval()
    d = _DEVICE

    input_signal = torch.randn(1, 1, 16000, device=d)
    input_length = torch.tensor([16000], device=d)

    with torch.no_grad():
        output_signal, output_length = model(
            input_signal=input_signal,
            input_length=input_length,
        )

    assert isinstance(output_signal, torch.Tensor), "output_signal must be a tensor"
    assert (
        output_signal.shape == input_signal.shape
    ), f"output shape {output_signal.shape} must match input shape {input_signal.shape}"
    assert isinstance(output_length, torch.Tensor), "output_length must be a tensor"
    assert (
        output_length.shape == input_length.shape
    ), f"output_length shape {output_length.shape} must match input_length shape {input_length.shape}"
