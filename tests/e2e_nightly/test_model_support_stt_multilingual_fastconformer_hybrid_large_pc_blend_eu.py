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

"""Functional tests for stt_multilingual_fastconformer_hybrid_large_pc_blend_eu."""

import os

import pytest
import torch

MODEL_NAME = "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu"
NEMO_FILE = "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo"

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
    """Run one training step via direct training_step() call."""
    from conftest import prepare_for_training_step

    model = _load_model()
    prepare_for_training_step(model)
    d = next(model.parameters()).device
    vocab_size = model.joint.num_classes_with_blank - 1
    batch = (
        torch.randn(2, 16000, device=d),
        torch.tensor([16000, 12000], device=d),
        torch.randint(0, max(1, vocab_size), (2, 5), dtype=torch.long, device=d),
        torch.tensor([5, 3], dtype=torch.long, device=d),
    )
    result = model.training_step(batch, 0)
    loss = result if isinstance(result, torch.Tensor) else result['loss']
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()


def test_model_inference():
    """Test full inference pipeline via model.transcribe()."""
    import numpy as np

    model = _load_model()
    model.eval()

    from conftest import prepare_for_transcribe

    prepare_for_transcribe(model)

    audio = np.random.randn(16000).astype(np.float32)

    result = model.transcribe(audio=[audio], batch_size=1)
    assert isinstance(result, list)
    assert len(result) == 1
    # transcribe() may return strings or Hypothesis objects
    text = result[0] if isinstance(result[0], str) else result[0].text
    assert isinstance(text, str)

    hyps = model.transcribe(audio=[audio], batch_size=1, return_hypotheses=True)
    assert isinstance(hyps, list)
    assert len(hyps) == 1
    assert hasattr(hyps[0], 'text')
