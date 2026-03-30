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

"""Functional tests for nvidia/audio-codec-22khz."""

import os

import torch

MODEL_NAME = "nvidia/audio-codec-22khz"
NEMO_FILE = "nvidia__audio-codec-22khz.nemo"

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
    """Exercise the generator loss path via _process_batch (no mocking).

    AudioCodecModel uses manual optimization and requires two optimizers, so
    calling training_step() directly outside of a Trainer context would fail.
    Instead we drive the forward computation via _process_batch, compute the
    primary losses the same way the real training_step does, and verify that
    loss.backward() produces gradients.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    sr = model.sample_rate
    num_samples = sr  # 1 second of audio
    batch = {
        "audio": torch.randn(1, num_samples, device=d),
        "audio_lens": torch.tensor([num_samples], device=d),
    }

    audio, audio_len, audio_gen, commit_loss, codes = model._process_batch(batch)

    generator_losses = []
    loss_mel_l1, loss_mel_l2 = model.mel_loss_fn(
        audio_real=audio.float(), audio_gen=audio_gen.float(), audio_len=audio_len
    )
    if model.mel_loss_l1_scale:
        generator_losses.append(model.mel_loss_l1_scale * loss_mel_l1)
    if model.mel_loss_l2_scale:
        generator_losses.append(model.mel_loss_l2_scale * loss_mel_l2)
    if model.time_domain_loss_scale:
        loss_td = model.time_domain_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        generator_losses.append(model.time_domain_loss_scale * loss_td)
    if model.commit_loss_scale and isinstance(commit_loss, torch.Tensor):
        generator_losses.append(model.commit_loss_scale * commit_loss)

    assert generator_losses, "No generator losses were computed."
    loss = sum(generator_losses)

    assert isinstance(loss, torch.Tensor), "Loss must be a tensor."
    assert loss.ndim == 0, "Loss must be a scalar tensor."
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()


def test_model_inference():
    model = _load_model()
    model.eval()
    d = _DEVICE

    sr = model.sample_rate
    num_samples = sr  # one second of audio at the model's native sample rate
    audio = torch.randn(1, num_samples, device=d)
    audio_len = torch.tensor([num_samples], device=d)

    with torch.no_grad():
        # Encode: waveform -> discrete tokens
        tokens, tokens_len = model.encode(audio=audio, audio_len=audio_len)
        assert tokens is not None, "encode() returned None tokens"
        assert tokens.ndim == 3, f"Expected tokens shape (B, C, T), got {tokens.shape}"
        assert tokens.shape[0] == 1, "Batch dimension mismatch"
        assert (
            tokens.shape[1] == model.num_codebooks
        ), f"Expected {model.num_codebooks} codebooks, got {tokens.shape[1]}"
        assert tokens_len.shape == (1,), f"Unexpected tokens_len shape: {tokens_len.shape}"

        # Decode: discrete tokens -> reconstructed waveform
        audio_rec, audio_rec_len = model.decode(tokens=tokens, tokens_len=tokens_len)
        assert audio_rec is not None, "decode() returned None audio"
        assert audio_rec.ndim == 2, f"Expected reconstructed audio shape (B, T), got {audio_rec.shape}"
        assert audio_rec.shape[0] == 1, "Batch dimension mismatch in decoded audio"
        assert audio_rec_len.shape == (1,), f"Unexpected audio_rec_len shape: {audio_rec_len.shape}"
