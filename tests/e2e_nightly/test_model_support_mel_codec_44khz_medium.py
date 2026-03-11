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

"""Functional tests for mel_codec_44khz_medium."""

import os

import pytest
import torch

MODEL_NAME = "mel_codec_44khz_medium"
NEMO_FILE = "mel_codec_44khz_medium.nemo"

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
    """Test one generator+discriminator training step without a Lightning trainer.

    AudioCodecModel uses manual optimization (automatic_optimization=False).
    training_step() calls self.optimizers(), self.manual_backward(), self.log_dict(),
    self.log(), and self.lr_schedulers() – all Lightning-trainer methods.  We stub
    those out so we can exercise the actual forward/loss computation end-to-end.

    The discriminator is updated when batch_idx % disc_update_period < disc_updates_per_period.
    With disc_update_period=2 and disc_updates_per_period=1, batch_idx=0 triggers the
    discriminator step; batch_idx=1 skips it and runs the generator step only.
    We run batch_idx=1 to keep the test lighter (no discriminator backward).
    Both generator losses (mel + feature-matching + GAN generator) are computed and
    manual_backward is called with their sum.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # Build two Adam optimizers that mirror configure_optimizers() but without
    # needing a trainer or a data-loader.
    import itertools

    vq_params = list(model.vector_quantizer.parameters()) if model.vector_quantizer else []
    gen_params = itertools.chain(
        model.audio_encoder.parameters(),
        model.audio_decoder.parameters(),
        vq_params,
    )
    optim_gen = torch.optim.Adam(gen_params, lr=2e-4, betas=(0.8, 0.99))
    optim_disc = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99))

    # Capture losses passed to manual_backward so we can assert on them.
    captured_losses = []

    def _manual_backward(loss):
        captured_losses.append(loss)
        loss.backward()

    # Stub out all Lightning-specific methods used inside training_step.
    model.optimizers = lambda: (optim_gen, optim_disc)
    model.manual_backward = _manual_backward
    model.log_dict = lambda *args, **kwargs: None
    model.log = lambda *args, **kwargs: None
    model.lr_schedulers = lambda: None  # update_lr becomes a no-op
    # current_epoch and global_step are Lightning properties that return 0 when
    # no Trainer is attached (self._trainer is None after restore_from), so no
    # further patching is required.

    # mel_codec_44khz_medium uses sample_rate=44100 and samples_per_frame=512.
    # Use ~0.37 s of audio (16384 samples, an integer multiple of 512).
    n_samples = 16384
    batch = {
        "audio": torch.randn(1, n_samples, device=d),
        "audio_lens": torch.tensor([n_samples], device=d),
    }

    # batch_idx=1 skips the discriminator update (1 % 2 >= 1) so only the
    # generator losses are computed and backward is called exactly once.
    model.training_step(batch, 1)

    assert len(captured_losses) >= 1, "manual_backward was never called – no generator loss was computed"
    gen_loss = captured_losses[-1]
    assert gen_loss.dim() == 0, f"Expected scalar generator loss, got shape {gen_loss.shape}"
    assert torch.isfinite(gen_loss), f"Generator loss is not finite: {gen_loss.item()}"


def test_model_inference():
    """Test encode -> decode round-trip at 44.1 kHz.

    model.encode(audio, audio_len) returns (tokens, tokens_len) where
      tokens has shape (batch, num_codebooks, num_frames).
    model.decode(tokens, tokens_len) returns (audio, audio_len) where
      audio has shape (batch, num_output_samples).
    """
    model = _load_model()
    model.eval()
    d = _DEVICE

    # Use 16384 samples at 44100 Hz (~0.37 s), a multiple of samples_per_frame=512.
    n_samples = 16384
    audio_in = torch.randn(1, n_samples, device=d)
    audio_len_in = torch.tensor([n_samples], device=d)

    with torch.no_grad():
        tokens, tokens_len = model.encode(audio=audio_in, audio_len=audio_len_in)

        assert tokens is not None, "encode() returned None tokens"
        assert tokens.dim() == 3, f"Expected tokens shape (B, C, T), got {tokens.shape}"
        assert tokens_len is not None and tokens_len.dim() == 1
        assert tokens_len[0].item() > 0, "tokens_len must be positive"

        audio_out, audio_out_len = model.decode(tokens=tokens, tokens_len=tokens_len)

        assert audio_out is not None, "decode() returned None audio"
        assert audio_out.dim() == 2, f"Expected audio shape (B, T), got {audio_out.shape}"
        assert audio_out.shape[0] == 1, "Batch size mismatch after decode"
        assert audio_out.shape[1] > 0, "Decoded audio has zero length"
        assert audio_out_len is not None and audio_out_len[0].item() > 0
