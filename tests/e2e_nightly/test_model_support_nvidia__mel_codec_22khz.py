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

"""Functional tests for nvidia/mel-codec-22khz."""

import os

import torch

MODEL_NAME = "nvidia/mel-codec-22khz"
NEMO_FILE = "nvidia__mel-codec-22khz.nemo"

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
    from nemo.collections.tts.models import AudioCodecModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = AudioCodecModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Exercise the generator reconstruction loss path without a Lightning Trainer.

    AudioCodecModel uses manual optimization and requires two optimizers, so
    calling training_step() directly outside of a Trainer context would fail at
    self.optimizers(). Instead we drive the equivalent forward computation via
    _process_batch, compute the primary mel reconstruction loss the same way the
    real training_step does, and verify that loss.backward() produces gradients.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # Use one second of audio at the model's native sample rate so that at
    # least one encoded frame is produced regardless of samples_per_frame.
    sr = model.sample_rate
    num_samples = sr  # 1 second
    batch = {
        "audio": torch.randn(1, num_samples, device=d),
        "audio_lens": torch.tensor([num_samples], device=d),
    }

    # _process_batch runs encoder -> (optional) vector quantizer -> decoder.
    audio, audio_len, audio_gen, commit_loss, codes = model._process_batch(batch)

    # Compute the L1 mel loss, which is always enabled (mel_loss_l1_scale > 0
    # by default) and is the primary reconstruction objective.
    loss_mel_l1, loss_mel_l2 = model.mel_loss_fn(
        audio_real=audio.float(),
        audio_gen=audio_gen.float(),
        audio_len=audio_len,
    )
    loss = model.mel_loss_l1_scale * loss_mel_l1
    if model.mel_loss_l2_scale:
        loss = loss + model.mel_loss_l2_scale * loss_mel_l2

    assert loss is not None
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"

    loss.backward()

    # Verify at least one encoder parameter received a gradient.
    grads = [p.grad for p in model.audio_encoder.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found in audio_encoder after backward()"


def test_model_inference():
    """Verify encode -> decode round-trip produces valid audio output."""
    model = _load_model()
    model.eval()
    d = _DEVICE

    # Use one second at the model's native sample rate.
    sr = model.sample_rate
    num_samples = sr
    audio = torch.randn(1, num_samples, device=d)
    audio_len = torch.tensor([num_samples], device=d)

    with torch.no_grad():
        tokens, tokens_len = model.encode(audio=audio, audio_len=audio_len)

        assert tokens is not None
        assert tokens.ndim == 3, f"Expected tokens of shape (B, C, T), got {tokens.shape}"
        assert tokens_len is not None

        # Decode the tokens back to audio.
        audio_out, audio_out_len = model.decode(tokens=tokens, tokens_len=tokens_len)

        assert audio_out is not None
        assert audio_out.ndim == 2, f"Expected decoded audio of shape (B, T), got {audio_out.shape}"
        assert audio_out_len is not None
        assert torch.isfinite(audio_out).all(), "Decoded audio contains non-finite values"
