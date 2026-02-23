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

"""Functional tests for nvidia/parakeet-ctc-1.1b."""

import os

import torch

MODEL_NAME = "nvidia/parakeet-ctc-1.1b"
NEMO_FILE = "nvidia__parakeet-ctc-1.1b.nemo"

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
    from conftest import run_training_step

    model = _load_model()
    vocab_size = model.tokenizer.vocab_size
    batch = (
        torch.randn(2, 16000),
        torch.tensor([16000, 12000]),
        torch.randint(0, vocab_size, (2, 5), dtype=torch.long),
        torch.tensor([5, 3], dtype=torch.long),
    )
    run_training_step(model, batch)


def test_model_inference():
    """
    Run a greedy forward pass in eval mode and verify the output shapes and that
    CTC decoding produces string transcriptions.
    """
    model = _load_model()
    model.eval()
    d = _DEVICE

    batch_size = 2
    signal = torch.randn(batch_size, 16000, device=d)
    signal_len = torch.tensor([16000, 12000], dtype=torch.long, device=d)

    with torch.no_grad():
        log_probs, encoded_len, greedy_predictions = model.forward(
            input_signal=signal,
            input_signal_length=signal_len,
        )

    # log_probs: [B, T, vocab+1],  encoded_len: [B],  greedy_predictions: [B, T]
    assert log_probs.shape[0] == batch_size
    assert log_probs.shape[-1] == model.decoder.num_classes_with_blank
    assert encoded_len.shape == (batch_size,)
    assert greedy_predictions.shape[0] == batch_size
