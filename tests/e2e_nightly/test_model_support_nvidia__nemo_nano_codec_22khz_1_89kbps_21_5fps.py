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

"""Functional tests for nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps."""

import os

import torch

MODEL_NAME = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
NEMO_FILE = "nvidia__nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"

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
    """Exercise generator forward+loss without discriminator.

    AudioCodecModel uses manual optimisation (automatic_optimization=False)
    and its discriminator architecture may produce non-leaf tensors that
    prevent changing requires_grad flags.  We bypass the discriminator and
    compute only the generator reconstruction losses (mel + commit), which
    is sufficient to verify the encoder/decoder/quantizer forward+backward.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    sample_rate = model.sample_rate
    num_samples = sample_rate  # 1 second of audio
    batch = {
        "audio": torch.randn(1, num_samples, device=d),
        "audio_lens": torch.tensor([num_samples], dtype=torch.long, device=d),
    }

    audio, audio_len, audio_gen, commit_loss, codes = model._process_batch(batch)

    # Only compute mel reconstruction losses (skip discriminator).
    loss_mel_l1, loss_mel_l2 = model.mel_loss_fn(
        audio_real=audio.float(), audio_gen=audio_gen.float(), audio_len=audio_len
    )
    loss = loss_mel_l1 + loss_mel_l2
    if model.commit_loss_scale and commit_loss is not None:
        loss = loss + model.commit_loss_scale * commit_loss

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_model_inference():
    """Encode audio to discrete tokens then decode back to waveform.

    Checks the full encode -> decode round-trip:
    * ``encode`` returns tokens of shape (B, n_codebooks, T_frames) with
      integer values in [0, codebook_size).
    * ``decode`` produces audio of shape (B, T_audio) with finite values.
    """
    model = _load_model()
    model.eval()
    d = _DEVICE

    # Use 1 second of audio at the model's native sample rate.
    sample_rate = model.sample_rate
    audio = torch.randn(1, sample_rate, device=d)
    audio_len = torch.tensor([sample_rate], device=d)

    with torch.no_grad():
        tokens, tokens_len = model.encode(audio=audio, audio_len=audio_len)

        # Shape checks for tokens.
        assert tokens is not None, "encode() returned None tokens"
        assert tokens.ndim == 3, f"Expected tokens shape (B, C, T), got {tokens.shape}"
        assert tokens.shape[0] == 1, f"Batch dimension mismatch: {tokens.shape}"
        assert (
            tokens.shape[1] == model.num_codebooks
        ), f"Codebook dimension {tokens.shape[1]} != model.num_codebooks {model.num_codebooks}"
        assert tokens_len.shape == (1,), f"Unexpected tokens_len shape: {tokens_len.shape}"
        assert tokens_len[0] > 0, "tokens_len must be positive"

        # Token values must lie within the codebook vocabulary.
        assert tokens.min() >= 0, f"Negative token index found: {tokens.min()}"
        assert tokens.max() < model.codebook_size, f"Token index {tokens.max()} >= codebook_size {model.codebook_size}"

        # Decode back to audio.
        audio_out, audio_out_len = model.decode(tokens=tokens, tokens_len=tokens_len)

        assert audio_out is not None, "decode() returned None audio"
        assert audio_out.ndim == 2, f"Expected decoded audio shape (B, T), got {audio_out.shape}"
        assert audio_out.shape[0] == 1, f"Batch dimension mismatch after decode: {audio_out.shape}"
        assert audio_out_len.shape == (1,), f"Unexpected audio_out_len shape: {audio_out_len.shape}"
        assert audio_out_len[0] > 0, "Decoded audio length must be positive"
        assert torch.isfinite(audio_out).all(), "Decoded audio contains non-finite values"
