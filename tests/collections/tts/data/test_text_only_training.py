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

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
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
        load_normalized_text_percent=0.0,
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
    assert "[EN-US][TEXT_ONLY]" in dataset.text_tokenizer.encoded
    assert "[VI][TEXT_ONLY]" in dataset.text_tokenizer.encoded
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
        load_normalized_text_percent=0.0,
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


def test_text_only_validation_accuracy_excludes_special_tokens_and_padding(dataset, cuts):
    model = _text_only_model(stacking_factor=2)
    model.validation_step_outputs = []
    targets = torch.tensor([[[3, 5, 2], [4, 2, 0]], [[6, 1, 8], [2, 0, 9]]])
    predictions = targets.clone()
    predictions[0, 0, 1] = 7  # One wrong IPA token out of four.
    predictions[1, :, 2] = 7  # Wrong batch padding must not count.
    logits = torch.full((*targets.shape, 32), -100.0).scatter(-1, predictions.unsqueeze(-1), 100.0)
    logits = logits.permute(0, 2, 1, 3).reshape(2, 3, 64)
    model._process_text_only_batch = Mock(
        return_value=SimpleNamespace(
            loss=torch.tensor(1.0),
            phoneme_loss=torch.tensor(1.0),
            codebook_loss=torch.tensor(0.0),
            phoneme_logits=logits,
            phoneme_tokens_target=targets,
            phoneme_tokens_lens_target=torch.tensor([3, 2]),
        )
    )
    model.validation_step(dataset[cuts], 0)
    args, kwargs = model.log.call_args
    assert args[0] == "val/text_only_phoneme_token_accuracy"
    assert args[1].item() == pytest.approx(0.75)
    assert kwargs["batch_size"] == 4
    assert kwargs["on_epoch"] and kwargs["sync_dist"]
    assert not kwargs["on_step"]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_finite_text_only_validation_is_deterministic_and_sharded(dataset, tmp_path, num_workers):
    path = tmp_path / "validation.jsonl"
    rows = [
        {
            "id": f"validation-{i}",
            "text": str(i),
            "text_normalized": "number",
            "ipa": "number IPA",
            "num_tokens": 4,
        }
        for i in range(8)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    config = {
        "input_cfg": [{"type": "txt_norm_jsonl", "paths": str(path), "language": "en"}],
        "token_equivalent_duration": 0.08,
        "batch_size": 2,
        "use_bucketing": False,
        "use_multimodal_sampling": False,
        "force_finite": True,
        "force_map_dataset": True,
        "shuffle": False,
        "seed": 42,
        "shard_seed": 42,
        "drop_last": False,
        "num_workers": num_workers,
    }
    rank_ids = []
    for rank in range(2):
        loader = get_lhotse_dataloader_from_config(config, global_rank=rank, world_size=2, dataset=dataset)
        batches = list(loader)
        assert len(batches) == 2  # Terminates after one pass rather than repeating.
        assert all(batch["task"] == ["text_only"] * 2 for batch in batches)
        ids = [sample_id for batch in batches for sample_id in batch["sample_id"]]
        assert len(ids) == len(set(ids)) == 4
        assert ids == [sample_id for batch in loader for sample_id in batch["sample_id"]]
        rank_ids.append(set(ids))
    assert rank_ids[0].isdisjoint(rank_ids[1])
    assert rank_ids[0] | rank_ids[1] == {row["id"] for row in rows}


def test_text_only_duration_can_account_for_ipa_tokens(tmp_path, cuts):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import models, pre_tokenizers

    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    backend = BackendTokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "ipa_tokenizer.json"
    backend.save(str(tokenizer_path))
    rows = [json.loads(line) for line in (tmp_path / "text.jsonl").read_text().splitlines()]
    rows[1]["num_tokens"] = 100  # Input can also be longer than the IPA channel.
    (tmp_path / "text.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_norm_jsonl",
                    "paths": str(tmp_path / "text.jsonl"),
                    "duration_phoneme_tokenizer_path": str(tokenizer_path),
                    "duration_padding_tokens": 16,
                }
            ],
            "force_finite": True,
            "token_equivalent_duration": 0.08,
        }
    )
    sampled, _ = read_cutset_from_config(config)
    sampled = list(sampled)
    for cut, row in zip(sampled, rows):
        expected = max(row["num_tokens"], len(backend.encode(row["ipa"]).ids)) + 16
        assert cut.duration == pytest.approx(expected * 0.08)
        assert cut.num_tokens == row["num_tokens"]  # Preserve source metadata.
        assert cut.supervisions[0].ipa == row["ipa"]
    assert sampled[0].duration > rows[0]["num_tokens"] * 0.08


@pytest.mark.parametrize("padding", [-1, 1.5, True])
def test_text_only_duration_rejects_invalid_padding(tmp_path, padding):
    with pytest.raises(ValueError, match="duration_padding_tokens"):
        LhotseTextNormJsonlAdapter(str(tmp_path / "text.jsonl"), 0.08, duration_padding_tokens=padding)


def test_text_only_duration_filters_oversized_ipa_without_fixed_batch_size(dataset, cuts, tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import models, pre_tokenizers

    backend = BackendTokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "ipa_tokenizer.json"
    backend.save(str(tokenizer_path))
    config = {
        "input_cfg": [
            {
                "type": "txt_norm_jsonl",
                "paths": str(tmp_path / "text.jsonl"),
                "duration_phoneme_tokenizer_path": str(tokenizer_path),
                "duration_padding_tokens": 16,
            }
        ],
        "token_equivalent_duration": 0.08,
        "batch_duration": 6.0,
        "batch_size": None,
        "max_duration": 3.0,
        "use_bucketing": False,
        "use_multimodal_sampling": False,
        "force_finite": True,
        "force_map_dataset": True,
        "shuffle": False,
        "seed": 42,
        "shard_seed": 42,
        "num_workers": 0,
    }
    loader = get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=dataset)
    batches = list(loader)
    assert [sample_id for batch in batches for sample_id in batch["sample_id"]] == ["1"]


@pytest.mark.parametrize("world_size", [1, 16])
def test_zero_weight_audio_source_is_not_opened(dataset, cuts, tmp_path, world_size):
    text_config = {
        "input_cfg": [{"type": "txt_norm_jsonl", "paths": str(tmp_path / "text.jsonl")}],
        "token_equivalent_duration": 0.08,
        "batch_size": 1,
        "use_bucketing": False,
        "force_iterable_dataset": True,
    }
    disabled_config = {
        **text_config,
        "input_cfg": [{"type": "txt_norm_jsonl", "paths": str(tmp_path / "disabled_missing.jsonl")}],
    }
    config = {
        "multi_config": True,
        "sampler_fusion": "randomized_round_robin",
        "sampler_weights": {"disabled_audio": 0.0, "text": 1.0},
        "disabled_audio": disabled_config,
        "text": text_config,
        "num_workers": 0,
        "shuffle": False,
        "seed": 42,
        "shard_seed": 42,
    }
    loader = get_lhotse_dataloader_from_config(config, global_rank=0, world_size=world_size, dataset=dataset)
    iterator = iter(loader)
    for _ in range(4):
        batch = next(iterator)
        assert batch["task"] == ["text_only"]


@pytest.mark.parametrize("normalized_text", [None, "normalized words"])
@pytest.mark.parametrize(
    "probability,draw,use_normalized",
    [(0.0, 0.0, False), (1.0, 1.0, True), (0.5, 0.49, True), (0.5, 0.5, False)],
)
def test_text_only_selects_raw_or_normalized_input(
    dataset, cuts, monkeypatch, normalized_text, probability, draw, use_normalized
):
    from nemo.collections.tts.parts.utils import tts_dataset_utils

    for cut in cuts:
        if normalized_text is None:
            cut.supervisions[0].custom.pop("normalized_text")
        else:
            cut.supervisions[0].normalized_text = normalized_text
    dataset.load_normalized_text_percent = probability
    monkeypatch.setattr(tts_dataset_utils.random, "random", lambda: draw)
    batch = dataset[cuts]
    selected = [normalized_text] * 2 if normalized_text is not None and use_normalized else ["$1,204.50", "7"]
    for i, text in enumerate(selected):
        expected = dataset.text_tokenizer.encode(text) + [dataset.eos_id]
        assert batch["text"][i, : batch["text_lens"][i]].tolist() == expected
    assert batch["phoneme_tokens_lens"].tolist() == [222, 11]
    assert cuts[0].supervisions[0].text == "$1,204.50"
    assert cuts[1].supervisions[0].text == "7"
    dataset._collate_audio_channels.assert_not_called()
    dataset._collect_cut_features.assert_not_called()


def test_text_only_duration_accounts_for_normalized_input(tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import models, pre_tokenizers

    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    backend = BackendTokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "text_tokenizer.json"
    backend.save(str(tokenizer_path))
    path = tmp_path / "text.jsonl"
    row = {
        "id": "expanded",
        "text": "123",
        "text_normalized": "one hundred twenty three",
        "ipa": "ipa",
        "num_tokens": 1,
    }
    path.write_text(json.dumps(row) + "\n")
    config = {
        "input_cfg": [
            {
                "type": "txt_norm_jsonl",
                "paths": str(path),
                "duration_text_tokenizer_path": str(tokenizer_path),
                "duration_phoneme_tokenizer_path": str(tokenizer_path),
                "duration_padding_tokens": 16,
            }
        ],
        "force_finite": True,
        "token_equivalent_duration": 0.08,
    }
    sampled, _ = read_cutset_from_config(OmegaConf.create(config))
    cut = next(iter(sampled))
    assert cut.duration == pytest.approx((4 + 16) * 0.08)
    assert cut.sampling_num_tokens == 20
    assert cut.num_tokens == 1
    assert cut.supervisions[0].normalized_text == row["text_normalized"]


def test_text_only_chunk_size_reaches_context_and_input_embeddings(dataset, cuts):
    model = _text_only_model()
    model._cfg.text_only_cas_chunk_size = 3
    model.embed_text_tokens = Mock(wraps=model.embed_text_tokens)
    loss = model.training_step(dataset[cuts], 0)
    assert torch.isfinite(loss)
    assert model.embed_text_tokens.call_count == 2
    assert all(call.kwargs["cas_chunk_size"] == 3 for call in model.embed_text_tokens.call_args_list)
