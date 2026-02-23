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

"""Functional tests for nvidia/speakerverification_en_titanet_large."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/speakerverification_en_titanet_large"
NEMO_FILE = "nvidia__speakerverification_en_titanet_large.nemo"

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
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = EncDecSpeakerLabelModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Run one real training step via Lightning Trainer.fit()."""
    from conftest import run_training_step

    model = _load_model()
    num_classes = model.decoder._num_classes
    batch = (
        torch.randn(2, 16000),
        torch.tensor([16000, 12000]),
        torch.randint(0, num_classes, (2,)),
        torch.tensor([1, 1], dtype=torch.long),
    )
    run_training_step(model, batch)


def test_model_inference():
    model = _load_model()
    model.eval()
    d = _DEVICE

    with torch.no_grad():
        logits, embs = model.forward(
            input_signal=torch.randn(1, 16000, device=d),
            input_signal_length=torch.tensor([16000], device=d),
        )

    assert logits is not None, "Expected logits from forward(), got None"
    assert embs is not None, "Expected embeddings from forward(), got None"
    assert logits.ndim == 2, f"Expected logits to be 2-D (B, num_classes), got shape {logits.shape}"
    assert embs.ndim == 2, f"Expected embeddings to be 2-D (B, emb_dim), got shape {embs.shape}"
    assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"
    assert embs.shape[0] == 1, f"Expected batch size 1, got {embs.shape[0]}"
