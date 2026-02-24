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

"""Functional tests for nvidia/diar_streaming_sortformer_4spk-v2.1."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"
NEMO_FILE = "nvidia__diar_streaming_sortformer_4spk-v2.1.nemo"

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
    from nemo.collections.asr.models import SortformerEncLabelModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = SortformerEncLabelModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
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

    model = _load_model()
    d = next(model.parameters()).device

    # Discover output frame count and speaker count by running a forward pass.
    num_samples = 64000  # 4 seconds at 16 kHz
    model.eval()
    with torch.no_grad():
        preds_ref = model.forward(
            audio_signal=torch.randn(1, num_samples, device=d),
            audio_signal_length=torch.tensor([num_samples], device=d),
        )
    num_frames = preds_ref.shape[1]
    num_spks = preds_ref.shape[2]

    prepare_for_training_step(model)
    # Build batch as a list: [audio_signal, audio_signal_length, targets, target_lens]
    batch = [
        torch.randn(1, num_samples, device=d),
        torch.tensor([num_samples], dtype=torch.long, device=d),
        torch.zeros(1, num_frames, num_spks, device=d),
        torch.tensor([num_frames], dtype=torch.long, device=d),
    ]
    result = model.training_step(batch, 0)
    loss = result if isinstance(result, torch.Tensor) else result['loss']
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()


def test_model_inference():
    model = _load_model()
    model.eval()
    d = _DEVICE
    with torch.no_grad():
        preds = model.forward(
            audio_signal=torch.randn(1, 16000, device=d),
            audio_signal_length=torch.tensor([16000], device=d),
        )
    # preds shape: (batch_size, T_frames, num_speakers)
    assert preds.ndim == 3, f"Expected 3-D output (B, T, C), got shape {preds.shape}"
    assert preds.shape[0] == 1
    assert preds.shape[2] > 0  # at least one speaker channel
