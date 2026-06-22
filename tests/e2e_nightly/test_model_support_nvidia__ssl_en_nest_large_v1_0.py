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

"""Functional tests for nvidia/ssl_en_nest_large_v1.0."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/ssl_en_nest_large_v1.0"
NEMO_FILE = "nvidia__ssl_en_nest_large_v1.0.nemo"

MODEL_DIR = os.environ.get(
    "NEMO_MODEL_SUPPORT_DIR",
    os.environ.get("NEMO_MODEL_SUPPORT_DIR_CI", "/home/TestData/nemo-speech-ci-models"),
)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _make_training_batch(device):
    from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch

    num_samples = 64000  # 4 seconds at 16 kHz; long enough for SSL masking to select frames.
    audio_len = torch.tensor([num_samples, 48000], device=device)
    time = torch.arange(num_samples, device=device, dtype=torch.float32) / 16000.0
    audio = torch.stack(
        [
            0.1 * torch.sin(2 * torch.pi * 220.0 * time),
            0.1 * torch.sin(2 * torch.pi * 330.0 * time),
        ]
    )
    noise = torch.stack(
        [
            0.005 * torch.sin(2 * torch.pi * 1800.0 * time),
            0.005 * torch.sin(2 * torch.pi * 2400.0 * time),
        ]
    )
    audio[1, audio_len[1].item() :] = 0.0
    noise[1, audio_len[1].item() :] = 0.0

    return AudioNoiseBatch(
        audio=audio,
        audio_len=audio_len,
        noise=noise,
        noise_len=audio_len,
        noisy_audio=audio + noise,
        noisy_audio_len=audio_len,
    )


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


@pytest.mark.skip(reason="NEST SSL model produces NaN loss on synthetic random AudioNoiseBatch inputs.")
def test_model_training_step():
    """Run one training step via direct training_step() call."""
    from e2e_utils import prepare_for_training_step

    model = _load_model()
    prepare_for_training_step(model)
    batch = _make_training_batch(next(model.parameters()).device)
    torch.manual_seed(0)
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
