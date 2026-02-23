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

"""Functional tests for nvidia/canary-180m-flash."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/canary-180m-flash"
NEMO_FILE = "nvidia__canary-180m-flash.nemo"

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
    from nemo.collections.asr.models import ASRModel

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = ASRModel.restore_from(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """Run forward + loss using the model's actual code (no mocking)."""
    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch

    model = _load_model()
    model.train()
    d = _DEVICE

    # Build a minimal prompted sequence via the model's prompt formatter.
    # canary2 format requires additional slots: diarize, timestamp, itn, emotion, decodercontext
    turns = [
        {
            "role": "user",
            "slots": {
                "source_lang": "en",
                "target_lang": "en",
                "pnc": "yes",
                "itn": "no",
                "timestamp": "no",
                "diarize": "no",
                "emotion": "<|emo:undefined|>",
                "decodercontext": "",
            },
        },
        {"role": "assistant", "slots": {"text": "hello world", model.prompt.PROMPT_LANGUAGE_SLOT: "en"}},
    ]
    encoded = model.prompt.encode_dialog(turns)
    prompt_ids = encoded["context_ids"]
    answer_ids = encoded["answer_ids"]
    full_ids = encoded["input_ids"]

    audio_len = 16000
    batch = PromptedAudioToTextMiniBatch(
        audio=torch.randn(1, audio_len, device=d),
        audio_lens=torch.tensor([audio_len], dtype=torch.long, device=d),
        transcript=answer_ids.unsqueeze(0).to(d),
        transcript_lens=torch.tensor([answer_ids.shape[0]], dtype=torch.long, device=d),
        prompt=prompt_ids.unsqueeze(0).to(d),
        prompt_lens=torch.tensor([prompt_ids.shape[0]], dtype=torch.long, device=d),
        prompted_transcript=full_ids.unsqueeze(0).to(d),
        prompted_transcript_lens=torch.tensor([full_ids.shape[0]], dtype=torch.long, device=d),
        cuts=None,
    )

    # Replicate the core of training_step: forward + loss.
    input_ids, labels = batch.get_decoder_inputs_outputs()
    input_ids_lens = batch.prompted_transcript_lens - 1

    transf_log_probs, encoded_len, enc_states, enc_mask = model.forward(
        input_signal=batch.audio,
        input_signal_length=batch.audio_lens,
        transcript=input_ids,
        transcript_length=input_ids_lens,
    )

    loss = model.loss(log_probs=transf_log_probs, labels=labels, output_mask=None)

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()


def test_model_inference():
    """Run encoder-decoder forward pass and verify output shapes."""
    model = _load_model()
    model.eval()
    d = _DEVICE

    B = 1
    T_audio = 16000  # 1 second at 16 kHz

    # Use a token id that is valid (not pad) for the decoder prompt.
    tok = 1 if model.tokenizer.pad_id != 1 else 2
    seq_len = 4
    transcript = torch.full((B, seq_len), tok, dtype=torch.long, device=d)
    transcript_length = torch.tensor([seq_len], dtype=torch.long, device=d)

    with torch.no_grad():
        transf_log_probs, encoded_len, enc_states, enc_mask = model.forward(
            input_signal=torch.randn(B, T_audio, device=d),
            input_signal_length=torch.tensor([T_audio], dtype=torch.long, device=d),
            transcript=transcript,
            transcript_length=transcript_length,
        )

    assert transf_log_probs is not None, "transf_log_probs should not be None when transcript is provided"
    assert transf_log_probs.ndim == 3, f"Expected 3-D log_probs [B, T, V], got shape {transf_log_probs.shape}"
    assert transf_log_probs.shape[0] == B
    assert encoded_len.shape == (B,)
    assert enc_states.ndim == 3
