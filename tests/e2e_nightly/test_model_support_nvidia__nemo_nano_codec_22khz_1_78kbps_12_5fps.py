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

"""Functional tests for nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps"
NEMO_FILE = "nvidia__nemo-nano-codec-22khz-1.78kbps-12.5fps.nemo"

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
    """Exercise the forward pass used during training.

    AudioCodecModel.training_step() requires a full Lightning Trainer context
    (two optimizers, manual_backward, log_dict, etc.).  Instead we directly
    exercise the core computation: _process_batch() followed by the generator
    reconstruction losses, a summed loss, and loss.backward().
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # One second of audio at the model's native sample rate.
    sample_rate = model.sample_rate  # 22050 for this codec
    num_samples = sample_rate  # 1 s

    audio = torch.randn(1, num_samples, device=d, requires_grad=False)
    audio_len = torch.tensor([num_samples], dtype=torch.long, device=d)
    batch = {"audio": audio, "audio_lens": audio_len}

    # _process_batch runs encoder → (optional) quantizer → decoder and returns
    # the (padded) reference audio, its lengths, the generated audio, the
    # commit loss from the quantizer, and the encoded representation.
    audio_ref, audio_ref_len, audio_gen, commit_loss, _ = model._process_batch(batch)

    # Compute the mel reconstruction losses (the primary generator losses).
    # mel_loss_fn returns (loss_l1, loss_l2); both are scalar tensors.
    loss_mel_l1, loss_mel_l2 = model.mel_loss_fn(
        audio_real=audio_ref.float(),
        audio_gen=audio_gen.float(),
        audio_len=audio_ref_len,
    )

    # Time-domain loss.
    loss_time = model.time_domain_loss_fn(
        audio_real=audio_ref,
        audio_gen=audio_gen,
        audio_len=audio_ref_len,
    )

    # Accumulate scaled losses exactly as training_step does.
    generator_losses = []
    if model.mel_loss_l1_scale:
        generator_losses.append(model.mel_loss_l1_scale * loss_mel_l1)
    if model.mel_loss_l2_scale:
        generator_losses.append(model.mel_loss_l2_scale * loss_mel_l2)
    if model.time_domain_loss_scale:
        generator_losses.append(model.time_domain_loss_scale * loss_time)
    if model.commit_loss_scale and isinstance(commit_loss, torch.Tensor):
        generator_losses.append(model.commit_loss_scale * commit_loss)

    assert generator_losses, "No generator losses were computed — check model config."
    loss = sum(generator_losses)

    assert isinstance(loss, torch.Tensor), "Loss must be a tensor."
    assert loss.ndim == 0, "Loss must be a scalar tensor."
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()


def test_model_inference():
    """Test encode → decode round-trip (inference path)."""
    model = _load_model()
    model.eval()
    d = _DEVICE

    sample_rate = model.sample_rate  # 22050 for this codec
    num_samples = sample_rate  # 1 s

    audio = torch.randn(1, num_samples, device=d)
    audio_len = torch.tensor([num_samples], dtype=torch.long, device=d)

    with torch.no_grad():
        # Encode: audio → discrete tokens.
        tokens, tokens_len = model.encode(audio=audio, audio_len=audio_len)

        assert tokens is not None, "encode() returned None tokens."
        # tokens shape: (batch, num_codebooks, num_frames)
        assert tokens.ndim == 3, f"Expected 3-D tokens tensor, got shape {tokens.shape}."
        assert tokens.shape[0] == 1, "Batch dimension mismatch."
        assert tokens_len.shape == (1,), f"Unexpected tokens_len shape: {tokens_len.shape}."
        assert tokens_len[0] > 0, "tokens_len must be positive."
        # Number of frames must match the last dimension of tokens.
        assert (
            tokens_len[0] == tokens.shape[2]
        ), f"tokens_len {tokens_len[0]} does not match tokens time dimension {tokens.shape[2]}."

        # Decode: discrete tokens → reconstructed audio.
        audio_out, audio_out_len = model.decode(tokens=tokens, tokens_len=tokens_len)

        assert audio_out is not None, "decode() returned None audio."
        assert audio_out.ndim == 2, f"Expected 2-D audio tensor, got shape {audio_out.shape}."
        assert audio_out.shape[0] == 1, "Batch dimension mismatch in decoded audio."
        assert audio_out_len[0] > 0, "Decoded audio length must be positive."
        assert torch.isfinite(audio_out).all(), "Decoded audio contains non-finite values."
