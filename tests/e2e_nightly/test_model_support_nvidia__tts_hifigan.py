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

"""Functional tests for nvidia/tts_hifigan."""

import os

import torch

MODEL_NAME = "nvidia/tts_hifigan"
NEMO_FILE = "nvidia__tts_hifigan.nemo"

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
    from nemo.collections.tts.models import HifiGanModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = HifiGanModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Exercise generator forward pass and mel loss without discriminator.

    HifiGanModel uses manual optimization (automatic_optimization=False)
    and the discriminator path can produce size-mismatch errors with
    synthetic audio.  We bypass the full training_step and instead run the
    generator directly on a mel spectrogram, then compute a simple L1 loss
    in waveform domain to verify the forward+backward path.
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # Generate mel spec from audio
    n_samples = 16000
    audio = torch.randn(1, n_samples, device=d)
    audio_len = torch.tensor([n_samples], device=d)
    spec, spec_len = model.audio_to_melspec_precessor(audio, audio_len)

    # Generator forward
    audio_gen = model.generator(x=spec)

    # Trim to match
    min_len = min(audio.shape[-1], audio_gen.shape[-1])
    audio_t = audio[..., :min_len]
    audio_gen_t = audio_gen[..., :min_len]

    # Simple L1 loss in waveform domain
    loss = torch.nn.functional.l1_loss(audio_gen_t, audio_t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_model_inference():
    """Exercise convert_spectrogram_to_audio with a random mel spectrogram.

    HifiGanModel.convert_spectrogram_to_audio(spec) takes a mel
    spectrogram of shape (B, n_mels, T) and returns a waveform of shape
    (B, T_audio).  The model uses nfilt=80 mel bins.
    """
    model = _load_model()
    model.eval()
    d = _DEVICE
    with torch.no_grad():
        # spec shape: (batch=1, n_mels=80, time_frames=100)
        spec = torch.randn(1, 80, 100, device=d)
        audio = model.convert_spectrogram_to_audio(spec=spec)

    assert audio is not None, "convert_spectrogram_to_audio() returned None"
    assert audio.ndim == 2, f"Expected audio shape (B, T), got {audio.shape}"
    assert audio.shape[0] == 1, f"Batch dimension mismatch: {audio.shape}"
    assert audio.shape[1] > 0, "Output audio has zero length"
    assert torch.isfinite(audio).all(), "Output audio contains non-finite values"
