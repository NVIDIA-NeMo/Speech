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

"""Functional tests for nvidia/canary-1b-v2."""

import os

import pytest
import torch

MODEL_NAME = "nvidia/canary-1b-v2"
NEMO_FILE = "nvidia__canary-1b-v2.nemo"

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
    from lhotse import CutSet, MonoCut

    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch

    model = _load_model()
    model.train()
    d = _DEVICE

    # Build a prompted transcript using the model's prompt formatter.
    # encode_dialog with a training turn (user + assistant) returns context_ids, answer_ids, input_ids.
    # The assistant turn needs 'prompt_language' to select the correct sub-tokenizer in the
    # AggregateTokenizer (CanaryTokenizer).  For the user turn, map_manifest_values_to_special_tokens
    # auto-injects CANARY_SPECIAL_TOKENIZER whenever language/boolean special-token slots are present.
    turns = [
        {
            "role": "user",
            "slots": {
                "source_lang": "en",
                "target_lang": "en",
                "pnc": "yes",
                "itn": "yes",
                "timestamp": "notimestamp",
                "diarize": "nodiarize",
                "decodercontext": "",
                "emotion": "<|emo:undefined|>",
            },
        },
        {
            "role": "assistant",
            "slots": {
                "text": "hello world",
                # Required so the AggregateTokenizer knows which sub-tokenizer to use.
                model.prompt.PROMPT_LANGUAGE_SLOT: "en",
            },
        },
    ]
    encoded = model.prompt.encode_dialog(turns)
    # input_ids is the full prompted_transcript [prompt + answer tokens]
    input_ids_1d = encoded["input_ids"]  # 1-D tensor of token IDs
    context_ids_1d = encoded["context_ids"]  # prompt portion only
    answer_ids_1d = encoded["answer_ids"]  # answer portion only

    # Batch dimension = 1
    prompted_transcript = input_ids_1d.unsqueeze(0).to(d)  # [1, T_dec]
    prompted_transcript_lens = torch.tensor([input_ids_1d.shape[0]], dtype=torch.long, device=d)
    prompt_tok = context_ids_1d.unsqueeze(0).to(d)  # [1, T_prompt]
    prompt_lens = torch.tensor([context_ids_1d.shape[0]], dtype=torch.long, device=d)
    transcript = answer_ids_1d.unsqueeze(0).to(d)  # [1, T_ans]
    transcript_lens = torch.tensor([answer_ids_1d.shape[0]], dtype=torch.long, device=d)

    # 1-second of silence as audio
    audio = torch.zeros(1, 16000, device=d)
    audio_lens = torch.tensor([16000], dtype=torch.long, device=d)

    # MultiTaskMetric.update() iterates over batch.cuts to filter per-metric constraints,
    # so we provide a minimal MonoCut with the required 'custom' attributes.
    dummy_cut = MonoCut(
        id="test_cut",
        start=0.0,
        duration=1.0,
        channel=0,
        custom={"source_lang": "en", "target_lang": "en", "taskname": "asr"},
    )
    cuts = CutSet([dummy_cut])

    batch = PromptedAudioToTextMiniBatch(
        audio=audio,
        audio_lens=audio_lens,
        transcript=transcript,
        transcript_lens=transcript_lens,
        prompt=prompt_tok,
        prompt_lens=prompt_lens,
        prompted_transcript=prompted_transcript,
        prompted_transcript_lens=prompted_transcript_lens,
        cuts=cuts,
    )

    # training_step reads self._optimizer.param_groups[0]['lr'], so attach a minimal optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    model._optimizer = optimizer

    output = model.training_step(batch, 0)
    loss = output["loss"]
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    loss.backward()


def test_model_inference():
    """Test full inference pipeline via model.transcribe()."""
    import numpy as np

    model = _load_model()
    model.eval()

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
