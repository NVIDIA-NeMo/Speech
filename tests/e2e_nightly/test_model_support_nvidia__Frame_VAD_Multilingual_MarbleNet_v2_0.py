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

"""Functional tests for nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"
NEMO_FILE = "nvidia__Frame_VAD_Multilingual_MarbleNet_v2.0.nemo"

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
    from nemo.collections.asr.models import EncDecClassificationModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = EncDecClassificationModel.restore_from(filepath, map_location="cpu", strict=False).to(_DEVICE)
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
    model = _load_model()
    model.eval()
    d = _DEVICE
    with torch.no_grad():
        logits = model.forward(
            input_signal=torch.randn(1, 16000, device=d),
            input_signal_length=torch.tensor([16000], device=d),
        )
    assert logits is not None
    assert logits.dim() >= 2, f"Expected at least 2-D logits, got shape {logits.shape}"
    assert torch.isfinite(logits).all()
