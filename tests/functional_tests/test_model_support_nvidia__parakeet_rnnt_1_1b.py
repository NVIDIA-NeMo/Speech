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

"""Functional tests for nvidia/parakeet-rnnt-1.1b."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/parakeet-rnnt-1.1b"
NEMO_FILE = "nvidia__parakeet-rnnt-1.1b.nemo"

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
    from nemo.collections.asr.models import ASRModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = ASRModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Run one real training step via Lightning Trainer.fit()."""
    from conftest import run_training_step, swap_rnnt_loss_to_pytorch

    model = _load_model()
    swap_rnnt_loss_to_pytorch(model)
    vocab_size = model.joint.num_classes_with_blank - 1
    batch = (
        torch.randn(2, 16000),
        torch.tensor([16000, 12000]),
        torch.randint(0, max(1, vocab_size), (2, 5), dtype=torch.long),
        torch.tensor([5, 3], dtype=torch.long),
    )
    run_training_step(model, batch)


def test_model_inference():
    """Run encoder-only forward pass in eval mode to verify inference shapes."""
    model = _load_model()
    model.eval()
    d = _DEVICE

    with torch.no_grad():
        encoded, encoded_len = model.forward(
            input_signal=torch.randn(1, 16000, device=d),
            input_signal_length=torch.tensor([16000], device=d),
        )

    assert encoded is not None
    assert encoded.ndim == 3
    assert encoded_len.shape == (1,)
