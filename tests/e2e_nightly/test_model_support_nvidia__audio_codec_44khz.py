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

"""Functional tests for nvidia/audio-codec-44khz."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/audio-codec-44khz"
NEMO_FILE = "nvidia__audio-codec-44khz.nemo"

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
    """Exercise the generator loss path from training_step without a Lightning trainer.

    AudioCodecModel.training_step() relies heavily on Lightning internals
    (manual_backward, optimizers(), log_dict).  We instead replicate the
    generator-loss portion directly: build a synthetic batch, call
    _process_batch(), compute every active loss term exactly as training_step
    does, sum them, verify the result is a finite scalar, and confirm that
    .backward() produces gradients on the encoder/decoder parameters.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # Build a synthetic batch.  _process_batch() expects keys "audio" [B, T]
    # and "audio_lens" [B].  Use 1 second of audio at the model sample rate.
    sr = model.sample_rate  # 44100 for this model
    n_samples = sr
    batch = {
        "audio": torch.randn(1, n_samples, device=d),
        "audio_lens": torch.tensor([n_samples], device=d),
    }

    # Forward pass through encoder -> VQ -> decoder (same as training_step).
    audio, audio_len, audio_gen, commit_loss, codes = model._process_batch(batch)

    generator_losses = []

    # Mel losses (stft does not support bf16 — cast to float32 as training_step does).
    loss_mel_l1, loss_mel_l2 = model.mel_loss_fn(
        audio_real=audio.float(), audio_gen=audio_gen.float(), audio_len=audio_len
    )
    if model.mel_loss_l1_scale:
        generator_losses.append(model.mel_loss_l1_scale * loss_mel_l1)
    if model.mel_loss_l2_scale:
        generator_losses.append(model.mel_loss_l2_scale * loss_mel_l2)

    if model.stft_loss_scale:
        loss_stft = model.stft_loss_fn(audio_real=audio.float(), audio_gen=audio_gen.float(), audio_len=audio_len)
        generator_losses.append(model.stft_loss_scale * loss_stft)

    if model.time_domain_loss_scale:
        loss_td = model.time_domain_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        generator_losses.append(model.time_domain_loss_scale * loss_td)

    if model.si_sdr_loss_scale:
        loss_si_sdr = model.si_sdr_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        generator_losses.append(model.si_sdr_loss_scale * loss_si_sdr)

    # Discriminator scores for generator and feature-matching losses.
    _, disc_scores_gen, fmaps_real, fmaps_gen = model.discriminator(audio_real=audio, audio_gen=audio_gen)
    if model.gen_loss_scale:
        loss_gen = model.gen_loss_fn(disc_scores_gen=disc_scores_gen)
        generator_losses.append(model.gen_loss_scale * loss_gen)

    if model.feature_loss_scale:
        loss_feature = model.feature_loss_fn(fmaps_real=fmaps_real, fmaps_gen=fmaps_gen)
        generator_losses.append(model.feature_loss_scale * loss_feature)

    if model.commit_loss_scale:
        generator_losses.append(model.commit_loss_scale * commit_loss)

    assert generator_losses, "No active loss terms were collected"
    loss = sum(generator_losses)

    assert loss.ndim == 0, "Expected a scalar loss tensor"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    loss.backward()

    # Verify that at least some encoder parameters received gradients.
    enc_params_with_grad = [p for p in model.audio_encoder.parameters() if p.grad is not None]
    assert enc_params_with_grad, "No gradients flowed back to audio_encoder parameters"


def test_model_inference():
    """Encode audio to discrete tokens and decode back to audio.

    Verifies the full encode -> decode roundtrip:
      - encode() returns tokens of shape [B, num_codebooks, T_frames] and
        valid frame lengths.
      - decode() converts those tokens back to a time-domain waveform of
        shape [B, T_samples].
      - The reconstructed audio length is a multiple of samples_per_frame.
    """
    model = _load_model()
    model.eval()
    d = _DEVICE

    sr = model.sample_rate  # 44100
    n_samples = sr  # 1 second of audio
    audio = torch.randn(1, n_samples, device=d)
    audio_len = torch.tensor([n_samples], device=d)

    with torch.no_grad():
        # Encode: audio -> discrete token indices
        tokens, tokens_len = model.encode(audio=audio, audio_len=audio_len)

        assert tokens is not None
        assert tokens.ndim == 3, f"Expected [B, C, T] tokens, got shape {tokens.shape}"
        assert tokens.shape[0] == 1, "Batch size should be 1"
        assert (
            tokens.shape[1] == model.num_codebooks
        ), f"Expected {model.num_codebooks} codebooks, got {tokens.shape[1]}"
        assert tokens_len.shape == (1,)
        assert tokens_len[0] == tokens.shape[2], "tokens_len must match the time dimension"

        # Decode: discrete tokens -> reconstructed waveform
        audio_recon, audio_recon_len = model.decode(tokens=tokens, tokens_len=tokens_len)

        assert audio_recon is not None
        assert audio_recon.ndim == 2, f"Expected [B, T] audio, got shape {audio_recon.shape}"
        assert audio_recon.shape[0] == 1
        assert audio_recon_len.shape == (1,)
        # Reconstructed length must be a multiple of samples_per_frame
        assert audio_recon_len[0] % model.samples_per_frame == 0, (
            f"Reconstructed audio length {audio_recon_len[0]} is not a multiple "
            f"of samples_per_frame {model.samples_per_frame}"
        )
        assert audio_recon.shape[1] == audio_recon_len[0], "Audio tensor width must match reported audio_recon_len"
