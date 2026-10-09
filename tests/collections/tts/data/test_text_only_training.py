# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Regression coverage for phoneme-only text-normalization training."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lhotse import CutSet
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.common.data.lhotse.text_adapters import LhotseTextNormJsonlAdapter
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    MagpieTTSLhotseMultiturnDataset,
    build_phoneme_channel,
    build_token_channel,
)
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import TrainingMode

pytestmark = pytest.mark.unit


class Tokenizer:
    pad = 0
    bos_token_id = 1
    eos_token_id = 2
    tokenizer_pad_ids = {"test": 0}

    def __init__(self):
        self.encoded = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return [3 + ord(c) % 20 for c in text]


@pytest.fixture
def cuts(tmp_path):
    path = tmp_path / "text.jsonl"
    rows = [
        {
            "id": str(i),
            "text": text,
            "text_normalized": "normalized words",
            "ipa": ipa,
            "num_tokens": 1,
            "language_id": language,
        }
        for i, (text, ipa, language) in enumerate(
            [
                ("$1,204.50", "very long IPA sequence" * 10, "en-US"),
                ("7", "short IPA", "vi"),
            ]
        )
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = CutSet.from_cuts(LhotseTextNormJsonlAdapter(str(path), 0.001))
    for cut in result:
        cut.task = "text_only"
        cut.tokenizer_names = ["test"]
    return result


@pytest.fixture
def dataset():
    result = MagpieTTSLhotseMultiturnDataset(
        sample_rate=16000,
        codec_model_samples_per_frame=640,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        num_audio_codebooks=8,
        use_text_conditioning_tokenizer=True,
        text_conditioning_tokenizer_name="test",
        ignore_phoneme_languages=["vi"],
        phoneme_turn_dropout_batch_prob=1.0,
        phoneme_turn_dropout_turn_prob=1.0,
    )
    result.text_tokenizer = Tokenizer()
    result.phoneme_tokenizer = Tokenizer()
    result.pad_id, result.bos_id, result.eos_id = 0, 1, 2
    result.interruption_token_id = 30
    result._initialize_tokenizers = Mock()
    result._collate_audio_channels = Mock(side_effect=AssertionError("audio loading"))
    result._collect_cut_features = Mock(side_effect=AssertionError("audio conditioning"))
    return result


def test_text_only_dataset_reuses_collators_without_audio(dataset, cuts):
    batch = dataset[cuts]
    assert batch["task"] == ["text_only", "text_only"]
    assert "audio" not in batch and "context_audio" not in batch
    assert batch["text_lens"].tolist() == [10, 2]
    assert batch["phoneme_tokens_lens"].tolist() == [222, 11]
    assert not batch["phoneme_turn_dropout"].any()
    assert "$1,204.50" in dataset.text_tokenizer.encoded
    assert "normalized words" not in dataset.text_tokenizer.encoded
    assert "<PHONEME_ONLY><en-US>" in dataset.text_tokenizer.encoded
    assert "<PHONEME_ONLY><vi>" in dataset.text_tokenizer.encoded
    dataset._collate_audio_channels.assert_not_called()
    dataset._collect_cut_features.assert_not_called()


def test_text_only_collators_keep_full_sequences(dataset, cuts):
    text = build_token_channel(
        cuts[0],
        dataset.text_tokenizer,
        0.04,
        {"agent"},
        pad_id=0,
        eos_id=2,
        bos_id=1,
        add_text_bos=False,
        tokenizer_name="test",
    )
    phonemes, dropped = build_phoneme_channel(
        cuts[1],
        dataset.phoneme_tokenizer,
        0.04,
        {"agent"},
        ["vi"],
        pad_id=0,
        eos_id=2,
        bos_id=1,
        phoneme_turn_max_words_to_drop=2,
        apply_turn_dropout=True,
        phoneme_turn_dropout_batch_prob=1.0,
        phoneme_turn_dropout_turn_prob=1.0,
    )
    assert text.tolist() == dataset.text_tokenizer.encode("$1,204.50") + [2]
    assert phonemes.tolist() == [1] + dataset.phoneme_tokenizer.encode("short IPA") + [2]
    assert not dropped


def test_mixed_text_audio_batch_rejected(dataset, cuts):
    cuts[1].task = "tts"
    with pytest.raises(ValueError, match="text_only"):
        dataset[cuts]


@pytest.mark.parametrize("stacking_factor", [1, 2])
def test_text_only_training_has_phoneme_gradients_and_no_audio(dataset, cuts, stacking_factor):
    batch = dataset[cuts]
    model = _text_only_model(stacking_factor)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert model.phoneme_final_proj.weight.grad.abs().sum() > 0
    model.prepare_context_tensors.assert_not_called()
    model.prepare_audio_channel_embeddings.assert_not_called()
    model._codec_helper.audio_to_codes.assert_not_called()


def _text_only_model(stacking_factor=1):
    model = EasyMagpieTTSModel.__new__(EasyMagpieTTSModel)
    nn.Module.__init__(model)
    model._cfg = OmegaConf.create({"embedding_dim": 8, "use_multiturn_dataset": True})
    model.training_modes = [TrainingMode("streaming", 3, 5, 0)]
    model.phoneme_tokenizer = Tokenizer()
    model.phoneme_stacking_factor = stacking_factor
    model.phoneme_vocab_size = 32
    model.phoneme_loss_weight = 1.0
    model.pad_id = 0
    model.disable_cas_for_context_text = True
    model.task_embedding = None
    model.text_embedding = nn.Embedding(32, 8)
    model.phoneme_embedding = nn.Embedding(32, 8)
    model.phoneme_final_proj = nn.Linear(8, 32 * stacking_factor)
    model.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
    model.embed_text_tokens = lambda tokens, **kwargs: model.text_embedding(tokens.long())
    model.embed_phoneme_tokens = lambda tokens: model.phoneme_embedding(tokens.long()).sum(dim=1)
    model.forward = lambda inputs_embeds, attention_mask: SimpleNamespace(last_hidden_state=inputs_embeds)
    model.log = Mock()
    model.prepare_context_tensors = Mock(side_effect=AssertionError("audio context"))
    model.prepare_audio_channel_embeddings = Mock(side_effect=AssertionError("audio decoder"))
    model._codec_helper = SimpleNamespace(audio_to_codes=Mock(side_effect=AssertionError("codec")))
    return model


def test_text_only_validation_skips_audio_inference(dataset, cuts):
    model = _text_only_model()
    model.validation_step_outputs = []
    model.run_val_inference = True
    model.infer_batch = Mock(side_effect=AssertionError("audio inference"))
    output = model.validation_step(dataset[cuts], 0)
    assert torch.isfinite(output["val_loss"])
    assert output["val_codebook_loss"] == 0
    assert model.validation_step_outputs == [output]
    model.infer_batch.assert_not_called()


def test_text_only_process_batch_dispatch(dataset, cuts):
    batch = dataset[cuts]
    model = _text_only_model()
    output = model.process_batch(
        batch["text"],
        batch["text_lens"],
        batch["context_text_tokens"],
        batch["context_text_tokens_lens"],
        None,
        None,
        None,
        None,
        phoneme_tokens=batch["phoneme_tokens"],
        phoneme_tokens_lens=batch["phoneme_tokens_lens"],
        task=batch["task"],
    )
    assert output.logits is None and output.audio_codes_target is None
    assert output.loss == output.phoneme_loss
    assert output.codebook_loss == 0
    assert not batch["phoneme_tokens"].eq(2).all()


def test_text_only_default_tokenizer_and_adapter_task(dataset, cuts):
    for cut in cuts:
        del cut.custom["tokenizer_names"]
    _, names = dataset._prepare_cuts(cuts)
    assert names == ["test", "test"]


def test_text_only_requires_nonempty_ipa(dataset, cuts):
    cuts[0].supervisions[0].custom["ipa"] = ""
    with pytest.raises(ValueError, match="IPA target"):
        dataset[cuts]


def test_regular_audio_collators_keep_existing_behavior(dataset, cuts):
    cut = cuts[1]
    cut.task = "tts"
    text = build_token_channel(
        cut,
        dataset.text_tokenizer,
        0.04,
        {"agent"},
        pad_id=0,
        eos_id=2,
        bos_id=1,
        add_text_bos=False,
        tokenizer_name="test",
    )
    assert "normalized words" in dataset.text_tokenizer.encoded
    assert len(text) == 0  # Ordinary audio channels still use the duration grid.
    phonemes, dropped = build_phoneme_channel(
        cut,
        dataset.phoneme_tokenizer,
        0.04,
        {"agent"},
        ["vi"],
        pad_id=0,
        eos_id=2,
        bos_id=1,
        apply_turn_dropout=False,
    )
    assert len(phonemes) == 0
    assert not dropped


@pytest.mark.parametrize("tasks", [["tts", "text_only"], ["text_only", "tts"]])
def test_model_rejects_mixed_tasks(tasks):
    with pytest.raises(ValueError, match="text_only"):
        EasyMagpieTTSModel._is_text_only_batch(tasks)
