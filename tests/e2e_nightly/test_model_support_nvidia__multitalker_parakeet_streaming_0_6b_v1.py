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

"""Functional tests for nvidia/multitalker-parakeet-streaming-0.6b-v1."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
NEMO_FILE = "nvidia__multitalker-parakeet-streaming-0.6b-v1.nemo"

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
    import math

    from e2e_utils import prepare_for_training_step

    model = _load_model()
    prepare_for_training_step(model)
    d = next(model.parameters()).device

    # Multitalker training_step expects a 6-element batch:
    # (signal, signal_len, transcript, transcript_len, spk_targets, bg_spk_targets)
    T_audio = 16000
    T_enc = math.ceil(math.ceil(T_audio / 160) / 8)
    vocab_size = model.joint.num_classes_with_blank - 1
    batch = (
        torch.randn(2, T_audio, device=d),
        torch.tensor([T_audio, 12000], device=d),
        torch.randint(0, max(1, vocab_size), (2, 5), dtype=torch.long, device=d),
        torch.tensor([5, 3], dtype=torch.long, device=d),
        torch.ones(2, T_enc, device=d),
        torch.zeros(2, T_enc, device=d),
    )
    result = model.training_step(batch, 0)
    loss = result if isinstance(result, torch.Tensor) else result['loss']
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()


def test_model_inference():
    """Run encoder-only forward pass in eval mode to verify inference shapes.

    Speaker targets are set to None to trigger single-speaker / all-ones-mask mode.
    """
    model = _load_model()
    model.eval()
    d = _DEVICE

    # Use None speaker targets to trigger single-speaker / all-ones-mask mode.
    model.set_speaker_targets(None, None)

    with torch.no_grad():
        encoded, encoded_len = model.forward(
            input_signal=torch.randn(1, 16000, device=d),
            input_signal_length=torch.tensor([16000], device=d),
        )

    assert encoded is not None
    assert encoded.ndim == 3
    assert encoded_len.shape == (1,)

    model.clear_speaker_targets()
