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

"""Functional tests for nvidia/ssl_en_nest_xlarge_v1.0."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/ssl_en_nest_xlarge_v1.0"
NEMO_FILE = "nvidia__ssl_en_nest_xlarge_v1.0.nemo"

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
    from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = EncDecDenoiseMaskedTokenPredModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Run one training step via direct training_step() call."""
    from conftest import prepare_for_training_step

    from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch

    model = _load_model()
    prepare_for_training_step(model)
    d = next(model.parameters()).device
    torch.manual_seed(0)  # fixed seed to avoid intermittent NaN with random inputs
    batch = AudioNoiseBatch(
        audio=torch.randn(2, 16000, device=d),
        audio_len=torch.tensor([16000, 12000], device=d),
        noise=torch.randn(2, 16000, device=d),
        noise_len=torch.tensor([16000, 12000], device=d),
        noisy_audio=torch.randn(2, 16000, device=d),
        noisy_audio_len=torch.tensor([16000, 12000], device=d),
    )
    result = model.training_step(batch, 0)
    loss = result if isinstance(result, torch.Tensor) else result['loss']
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()


def test_model_inference():
    model = _load_model()
    model.eval()
    d = _DEVICE
    with torch.no_grad():
        log_probs, encoded_len, masks, tokens = model.forward(
            input_signal=torch.randn(1, 16000, device=d),
            input_signal_length=torch.tensor([16000], device=d),
            noisy_input_signal=torch.randn(1, 16000, device=d),
            noisy_input_signal_length=torch.tensor([16000], device=d),
        )
    assert log_probs is not None
    assert encoded_len is not None
    assert masks is not None
    assert tokens is not None
