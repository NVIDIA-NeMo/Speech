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

"""Functional tests for nvidia/low-frame-rate-speech-codec-22khz."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/low-frame-rate-speech-codec-22khz"
NEMO_FILE = "nvidia__low-frame-rate-speech-codec-22khz.nemo"

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
    """Test training forward pass and backward through reconstruction losses.

    AudioCodecModel uses manual optimization (automatic_optimization=False) and
    requires a Lightning trainer for training_step() (optimizers, manual_backward,
    log_dict). Instead we exercise the same computation path directly: call
    _process_batch() to obtain real/generated audio, then compute the mel
    reconstruction loss (l1 + l2) — the primary per-step generator loss — and
    call .backward() on it to verify that gradients flow back through the entire
    encoder-quantizer-decoder chain.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # Build a synthetic batch that matches the format expected by _process_batch:
    #   "audio"     -> [B, T]  float waveform at model.sample_rate
    #   "audio_lens" -> [B]    valid sample counts
    # Use 1 second of audio at the model's native sample rate.
    sample_rate = model.sample_rate  # 22050 for this checkpoint
    num_samples = sample_rate  # 1 s
    batch = {
        "audio": torch.randn(1, num_samples, device=d, requires_grad=False),
        "audio_lens": torch.tensor([num_samples], device=d),
    }

    # Run the shared forward pass used by training_step.
    audio, audio_len, audio_gen, commit_loss, codes = model._process_batch(batch)

    # Compute the mel reconstruction loss (l1 + l2) — same calls as training_step.
    loss_mel_l1, loss_mel_l2 = model.mel_loss_fn(
        audio_real=audio.float(),
        audio_gen=audio_gen.float(),
        audio_len=audio_len,
    )
    loss = model.mel_loss_l1_scale * loss_mel_l1 + model.mel_loss_l2_scale * loss_mel_l2

    assert loss is not None, "mel loss must not be None"
    assert loss.ndim == 0, "mel loss must be a scalar"
    assert torch.isfinite(loss), f"mel loss must be finite, got {loss.item()}"

    loss.backward()

    # Verify that at least some encoder and decoder parameters received gradients.
    enc_params_with_grad = [p for p in model.audio_encoder.parameters() if p.grad is not None]
    dec_params_with_grad = [p for p in model.audio_decoder.parameters() if p.grad is not None]
    assert len(enc_params_with_grad) > 0, "encoder parameters must have gradients after backward"
    assert len(dec_params_with_grad) > 0, "decoder parameters must have gradients after backward"


def test_model_inference():
    """Test encode/decode round-trip: audio -> tokens -> reconstructed audio."""
    model = _load_model()
    model.eval()
    d = _DEVICE

    sample_rate = model.sample_rate  # 22050 for this checkpoint
    num_samples = sample_rate  # 1 s of audio
    audio = torch.randn(1, num_samples, device=d)
    audio_len = torch.tensor([num_samples], device=d)

    with torch.no_grad():
        # Encode: audio -> discrete tokens
        tokens, tokens_len = model.encode(audio=audio, audio_len=audio_len)

        assert tokens is not None, "tokens must not be None"
        assert tokens.ndim == 3, f"tokens must be [B, C, T], got shape {tuple(tokens.shape)}"
        assert tokens.shape[0] == 1, "batch size must be 1"
        assert (
            tokens.shape[1] == model.num_codebooks
        ), f"expected {model.num_codebooks} codebooks, got {tokens.shape[1]}"
        assert tokens_len.shape == (1,), f"tokens_len must have shape (1,), got {tuple(tokens_len.shape)}"

        # Decode: discrete tokens -> reconstructed audio
        audio_out, audio_out_len = model.decode(tokens=tokens, tokens_len=tokens_len)

        assert audio_out is not None, "decoded audio must not be None"
        assert audio_out.ndim == 2, f"decoded audio must be [B, T], got shape {tuple(audio_out.shape)}"
        assert audio_out.shape[0] == 1, "decoded audio batch size must be 1"
        assert audio_out_len.shape == (1,), f"audio_out_len must have shape (1,), got {tuple(audio_out_len.shape)}"
        # The decoded length must cover the original number of input samples.
        assert (
            audio_out_len[0] >= num_samples
        ), f"decoded length {audio_out_len[0].item()} must be >= input length {num_samples}"
