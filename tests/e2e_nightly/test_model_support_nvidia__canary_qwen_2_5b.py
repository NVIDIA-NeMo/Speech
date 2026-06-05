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

"""Functional tests for nvidia/canary-qwen-2.5b."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/canary-qwen-2.5b"
NEMO_FILE = "nvidia__canary-qwen-2.5b"

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
    from nemo.collections.speechlm2.models import SALM

    filepath = os.path.join(MODEL_DIR, NEMO_FILE)
    _model = SALM.from_pretrained(filepath, map_location="cpu").to(_DEVICE)
    return _model


def test_model_init():
    model = _load_model()
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step():
    """
    Build a minimal batch matching SALMDataset output and run training_step.

    The batch must contain:
      - audios:     (B, T_samples)  float32 raw waveform
      - audio_lens: (B,)            int64 sample counts
      - input_ids:  (B, T_tokens)   int64, contains one audio_locator_tag_id per audio
      - loss_mask:  (B, T_tokens)   bool, True where loss should be computed

    prepare_inputs() calls perception(audios, audio_lens) to get audio embeddings, then
    replaces each audio_locator_tag_id placeholder with the corresponding embedding before
    passing everything to the LLM.  training_step() then computes cross-entropy loss and
    returns a dict with key "loss".
    """
    model = _load_model()
    model.train()
    d = _DEVICE

    B = 1
    sr = model.sampling_rate  # typically 16000
    T_audio = sr * 2  # 2 seconds of audio
    audio_locator_id = model.audio_locator_tag_id

    # Build text token sequence: [<bos>, <audio_locator>, tok, tok, ..., <eos>]
    # The audio_locator_tag must appear exactly once to match the single audio sample.
    T_text = 16
    # Use token id 1 as a generic text token (safe for most LLM tokenizers).
    text_tok = 1
    input_ids = torch.full((B, T_text), text_tok, dtype=torch.long, device=d)
    input_ids[0, 0] = audio_locator_id  # one placeholder at position 0

    # loss_mask=True on the final few tokens (the "response" portion).
    loss_mask = torch.zeros(B, T_text, dtype=torch.bool, device=d)
    loss_mask[0, T_text // 2 :] = True

    audios = torch.randn(B, T_audio, device=d)
    audio_lens = torch.tensor([T_audio], dtype=torch.long, device=d)

    batch = {
        "audios": audios,
        "audio_lens": audio_lens,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }

    # training_step calls self.log_dict which emits a warning without a trainer but does
    # not raise.  We suppress that warning so the test output stays clean.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.training_step(batch, 0)

    assert "loss" in result, f"training_step must return a dict with 'loss', got keys: {list(result.keys())}"
    loss = result["loss"]
    assert loss.ndim == 0, f"loss must be a scalar tensor, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
    loss.backward()


def test_model_inference():
    """
    Run a text-only forward pass through the LLM backbone.

    SALM.forward() takes pre-built input embeddings of shape (B, T, H) — the same
    representation that prepare_inputs() produces after splicing in audio embeddings —
    and returns {"logits": Tensor[B, T, vocab_size]}.
    """
    model = _load_model()
    model.eval()
    d = _DEVICE
    hidden_size = model.llm.config.hidden_size

    B, T = 1, 10
    input_embeds = torch.randn(B, T, hidden_size, device=d)
    attention_mask = torch.ones(B, T, dtype=torch.bool, device=d)

    with torch.no_grad():
        result = model.forward(input_embeds=input_embeds, attention_mask=attention_mask)

    assert isinstance(result, dict), f"forward() must return a dict, got {type(result)}"
    assert "logits" in result, f"forward() output must contain 'logits', got keys: {list(result.keys())}"
    logits = result["logits"]
    assert logits.shape[0] == B, f"logits batch dim mismatch: expected {B}, got {logits.shape[0]}"
    assert logits.shape[1] == T, f"logits time dim mismatch: expected {T}, got {logits.shape[1]}"
    assert (
        logits.shape[2] == model.text_vocab_size
    ), f"logits vocab dim mismatch: expected {model.text_vocab_size}, got {logits.shape[2]}"
