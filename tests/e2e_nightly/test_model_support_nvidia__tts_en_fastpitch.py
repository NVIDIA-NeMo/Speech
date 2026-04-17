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

"""Functional tests for nvidia/tts_en_fastpitch."""

import os

import torch

MODEL_NAME = "nvidia/tts_en_fastpitch"
NEMO_FILE = "nvidia__tts_en_fastpitch.nemo"

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
    from nemo.collections.tts.models import FastPitchModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = FastPitchModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    import math

    model = _load_model()
    model.train()

    # Model parameters from config: sample_rate=22050, hop_length=256, n_mel_channels=80
    sample_rate = 22050
    hop_length = 256
    B = 1
    T_audio = sample_rate  # 1 second of audio

    # T_mel: number of mel frames produced by the preprocessor (pad_to=1)
    T_mel = math.ceil(T_audio / hop_length)
    T_text = 10  # number of text tokens

    audio = torch.randn(B, T_audio).to(_DEVICE)
    audio_lens = torch.tensor([T_audio], dtype=torch.int32).to(_DEVICE)
    text = torch.randint(1, 80, (B, T_text), dtype=torch.long).to(_DEVICE)
    text_lens = torch.tensor([T_text], dtype=torch.int32).to(_DEVICE)
    # pitch: frame-level (per mel frame), shape (B, T_mel)
    pitch = torch.rand(B, T_mel).to(_DEVICE)
    # align_prior_matrix: (B, T_mel, T_text) — attention prior (mel-frames x text-tokens)
    # Normalised so each mel frame sums to 1 over text tokens.
    align_prior = torch.ones(B, T_mel, T_text).to(_DEVICE)
    align_prior = align_prior / align_prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    batch = {
        "audio": audio,
        "audio_lens": audio_lens,
        "text": text,
        "text_lens": text_lens,
        "pitch": pitch,
        "align_prior_matrix": align_prior,
    }

    # Temporarily switch ds_class so training_step treats batch as a plain dict
    # rather than calling process_batch (which requires _train_dl to be set up).
    original_ds_class = model.ds_class
    model.ds_class = "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset"
    try:
        loss = model.training_step(batch, 0)
    finally:
        model.ds_class = original_ds_class

    assert loss.shape == torch.Size([]), f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()


def test_model_inference():
    model = _load_model()
    model.eval()
    with torch.no_grad():
        tokens = model.parse("hello world")
        if _DEVICE.type == "cuda":
            tokens = tokens.to(_DEVICE)
        spec = model.generate_spectrogram(tokens=tokens)
    assert spec is not None, "generate_spectrogram returned None"
    assert spec.ndim == 3, f"Expected 3D spectrogram (B, D, T), got shape {spec.shape}"
    assert spec.shape[1] == 80, f"Expected 80 mel channels, got {spec.shape[1]}"
    assert spec.shape[2] > 0, "Spectrogram time dimension is empty"
