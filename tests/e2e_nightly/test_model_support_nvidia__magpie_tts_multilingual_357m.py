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

"""Functional tests for nvidia/magpie_tts_multilingual_357m."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/magpie_tts_multilingual_357m"
NEMO_FILE = "nvidia__magpie_tts_multilingual_357m.nemo"

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
    from nemo.collections.tts.models import MagpieTTSModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = MagpieTTSModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Exercise MagpieTTSModel's core training computation via process_batch().

    MagpieTTSModel.training_step() calls self.log() / self.log_dict() which
    require an attached Lightning Trainer.  We therefore call process_batch()
    directly — exactly the computation training_step() delegates to — and verify
    that a scalar, finite loss flows correctly through backward().

    The model is of type 'decoder_ce' with a baked context embedding, so no
    context audio is needed in the batch.  The batch only requires:
        text           – phoneme/character token IDs  (B, T_text)
        text_lens      – actual text lengths           (B,)
        audio_codes    – codec token indices           (B, num_codebooks, T_audio)
        audio_codes_lens – actual audio code lengths  (B,)
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    # Derive safe upper bounds for token IDs from the model's embedding tables.
    # text_embedding vocab = num_tokens_tokenizer + 2 (BOS/EOS).
    text_vocab_size = model.text_embedding.num_embeddings
    # audio_codes must be strictly less than codebook_size (special tokens are
    # appended *after* codebook_size inside process_batch / add_special_tokens).
    audio_token_max = model.codebook_size - 1

    B = 1
    T_text = 8  # short phoneme sequence
    T_audio = 20  # short audio sequence (frames)

    # Text tokens: use a safe non-special ID (1) to avoid BOS/EOS collisions.
    text = torch.ones(B, T_text, dtype=torch.long, device=d)
    text_lens = torch.tensor([T_text], dtype=torch.long, device=d)

    # Audio codec tokens: shape (B, num_codebooks, T_audio).
    audio_codes = torch.randint(
        low=0,
        high=audio_token_max,
        size=(B, model.num_audio_codebooks, T_audio),
        dtype=torch.long,
        device=d,
    )
    audio_codes_lens = torch.tensor([T_audio], dtype=torch.long, device=d)

    batch = {
        "text": text,
        "text_lens": text_lens,
        "audio_codes": audio_codes,
        "audio_codes_lens": audio_codes_lens,
    }

    batch_output = model.process_batch(batch)
    loss = batch_output["loss"]

    assert isinstance(loss, torch.Tensor), "process_batch() must return a tensor loss."
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}."
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()


def test_model_inference():
    """Test MagpieTTSModel speech synthesis via do_tts().

    do_tts() returns a tuple of (audio, audio_len) where audio has shape
    (1, T_audio_samples) and audio_len has shape (1,).
    """
    model = _load_model()
    model.eval()
    with torch.no_grad():
        audio, audio_len = model.do_tts(transcript="hello world", language="en")

    assert audio is not None, "do_tts() returned None audio."
    assert isinstance(audio, torch.Tensor), f"Expected Tensor, got {type(audio)}."
    assert audio.ndim == 2, f"Expected 2-D audio tensor (1, T), got shape {audio.shape}."
    assert audio.shape[0] == 1, f"Batch dimension mismatch: {audio.shape}."
    assert audio_len is not None, "do_tts() returned None audio_len."
    assert isinstance(audio_len, torch.Tensor), f"Expected Tensor for audio_len, got {type(audio_len)}."
    assert audio_len.shape == (1,), f"Unexpected audio_len shape: {audio_len.shape}."
    assert audio_len[0] > 0, "Generated audio length must be positive."
