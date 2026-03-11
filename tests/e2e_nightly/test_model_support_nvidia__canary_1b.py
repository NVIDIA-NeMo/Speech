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

"""Functional tests for nvidia/canary-1b."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/canary-1b"
NEMO_FILE = "nvidia__canary-1b.nemo"

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
    """
    EncDecMultiTaskModel.training_step() expects a PromptedAudioToTextMiniBatch.
    We build one using the model's own prompt formatter so the token IDs are valid,
    then exercise the forward + loss path directly (training_step itself also
    accesses self._optimizer which is only wired up by Lightning, so we replicate
    its core logic here without that dependency).
    """
    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch

    model = _load_model()
    model.train()
    d = _DEVICE

    # Build a minimal prompted sequence via the model's own prompt formatter.
    # Canary prompt format: <BOS> <src_lang> <task> <tgt_lang> <pnc> <text> <EOS>
    # The assistant turn needs 'prompt_language' to select the correct sub-tokenizer
    # in the AggregateTokenizer (CanaryTokenizer).
    turns = [
        {"role": "user", "slots": {"source_lang": "en", "task": "asr", "target_lang": "en", "pnc": "yes"}},
        {"role": "assistant", "slots": {"text": "hello world", model.prompt.PROMPT_LANGUAGE_SLOT: "en"}},
    ]
    encoded = model.prompt.encode_dialog(turns)
    # context_ids  = prompt portion (user turn only)
    # answer_ids   = transcript portion (assistant turn, without EOS per canary.py)
    # input_ids    = full prompted sequence (context + answer)
    prompt_ids = encoded["context_ids"]  # 1-D tensor
    answer_ids = encoded["answer_ids"]  # 1-D tensor
    full_ids = encoded["input_ids"]  # prompt + answer concatenated

    # Build batch tensors (batch size = 1).
    audio_len = 16000  # 1 second of audio at 16 kHz
    audio = torch.randn(1, audio_len, device=d)
    audio_lens = torch.tensor([audio_len], device=d, dtype=torch.long)

    prompted_transcript = full_ids.unsqueeze(0).to(d)  # (1, T_full)
    prompted_transcript_lens = torch.tensor([full_ids.shape[0]], device=d, dtype=torch.long)
    transcript = answer_ids.unsqueeze(0).to(d)  # (1, T_ans)
    transcript_lens = torch.tensor([answer_ids.shape[0]], device=d, dtype=torch.long)
    prompt = prompt_ids.unsqueeze(0).to(d)  # (1, T_prompt)
    prompt_lens = torch.tensor([prompt_ids.shape[0]], device=d, dtype=torch.long)

    batch = PromptedAudioToTextMiniBatch(
        audio=audio,
        audio_lens=audio_lens,
        transcript=transcript,
        transcript_lens=transcript_lens,
        prompt=prompt,
        prompt_lens=prompt_lens,
        prompted_transcript=prompted_transcript,
        prompted_transcript_lens=prompted_transcript_lens,
        cuts=None,
    )

    # Replicate the core of training_step without the Lightning optimizer access.
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
    assert loss.item() > 0.0, "Expected positive loss"
    loss.backward()


def test_model_inference():
    """Test full inference pipeline via model.transcribe()."""
    import numpy as np

    model = _load_model()
    model.eval()

    from conftest import prepare_for_transcribe

    prepare_for_transcribe(model)

    audio = np.random.randn(16000).astype(np.float32)

    result = model.transcribe(
        audio=[audio],
        batch_size=1,
        source_lang="en",
        target_lang="en",
        task="asr",
        pnc="yes",
    )
    assert isinstance(result, list)
    assert len(result) == 1
    # transcribe() may return strings or Hypothesis objects
    text = result[0] if isinstance(result[0], str) else result[0].text
    assert isinstance(text, str)
