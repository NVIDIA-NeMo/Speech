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

"""Functional tests for vad_multilingual_frame_marblenet."""

import os

import pytest
import torch

MODEL_NAME = "vad_multilingual_frame_marblenet"
NEMO_FILE = "vad_multilingual_frame_marblenet.nemo"

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
    from nemo.collections.asr.models.classification_models import EncDecFrameClassificationModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = EncDecFrameClassificationModel.restore_from(filepath, map_location="cpu", strict=False).to(_DEVICE)
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
    # Discover output frame count to build matching labels.
    model.eval()
    with torch.no_grad():
        probe = model.forward(
            input_signal=torch.randn(1, 16000, device=d),
            input_signal_length=torch.tensor([16000], device=d),
        )
    n_frames = probe.shape[1]

    prepare_for_training_step(model)
    batch = (
        torch.randn(2, 16000, device=d),
        torch.tensor([16000, 12000], device=d),
        torch.zeros(2, n_frames, dtype=torch.long, device=d),
        torch.tensor([n_frames, n_frames], dtype=torch.long, device=d),
    )
    result = model.training_step(batch, 0)
    loss = result if isinstance(result, torch.Tensor) else result['loss']
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()


def test_model_inference():
    """Test forward pass in eval mode and assert per-frame logit output."""
    model = _load_model()
    model.eval()
    d = _DEVICE
    batch_size = 1
    n_samples = 16000
    input_signal = torch.randn(batch_size, n_samples, device=d)
    input_signal_length = torch.tensor([n_samples], device=d)

    with torch.no_grad():
        logits = model.forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
        )

    # logits must be a 3-D tensor: [B, T, C] (batch, frames, classes).
    assert logits is not None
    assert logits.ndim == 3, f"Expected 3-D logits [B, T, C], got shape {logits.shape}"
    assert logits.shape[0] == batch_size, f"Expected batch size {batch_size}, got {logits.shape[0]}"
    # At least one output frame must be produced.
    assert logits.shape[1] > 0, "Expected at least one output frame"
    # Number of classes must be positive.
    assert logits.shape[2] > 0, "Expected at least one output class"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"
