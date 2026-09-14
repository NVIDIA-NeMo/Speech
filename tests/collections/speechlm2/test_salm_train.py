# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

_SALM_TRAIN_PATH = Path(__file__).parents[3] / "examples" / "speechlm2" / "salm_train.py"
_SPEC = importlib.util.spec_from_file_location("salm_train_for_test", _SALM_TRAIN_PATH)
_SALM_TRAIN = importlib.util.module_from_spec(_SPEC)
with patch("torch.cuda.is_available", return_value=False):
    _SPEC.loader.exec_module(_SALM_TRAIN)


@pytest.mark.unit
def test_create_salm_dataset_omits_unset_multispeaker_config(monkeypatch):
    class LegacySALMDataset:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tokenizer = object()
    monkeypatch.setattr(_SALM_TRAIN, "SALMDataset", LegacySALMDataset)

    dataset = _SALM_TRAIN._create_salm_dataset(tokenizer, {})

    assert dataset.tokenizer is tokenizer


@pytest.mark.unit
def test_create_salm_dataset_does_not_forward_prompt_format_options(monkeypatch):
    class LegacySALMDataset:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tokenizer = object()
    monkeypatch.setattr(_SALM_TRAIN, "SALMDataset", LegacySALMDataset)
    data_cfg = {
        "train_ds": {
            "prompt_format": "nemotron-nano-v3",
            "audio_locator_tag": "<|audio|>",
            "token_equivalent_duration": 0.08,
        }
    }

    dataset = _SALM_TRAIN._create_salm_dataset(tokenizer, data_cfg)

    assert dataset.tokenizer is tokenizer


@pytest.mark.unit
def test_create_salm_dataset_forwards_configured_multispeaker_config(monkeypatch):
    multispeaker_cfg = {"num_speakers": 2}

    class MultiSpeakerSALMDataset:
        def __init__(self, tokenizer, multispeaker_cfg=None):
            self.tokenizer = tokenizer
            self.multispeaker_cfg = multispeaker_cfg

    tokenizer = object()
    monkeypatch.setattr(_SALM_TRAIN, "SALMDataset", MultiSpeakerSALMDataset)

    dataset = _SALM_TRAIN._create_salm_dataset(tokenizer, {"multispeaker_cfg": multispeaker_cfg})

    assert dataset.tokenizer is tokenizer
    assert dataset.multispeaker_cfg is multispeaker_cfg


@pytest.mark.unit
def test_create_salm_dataset_enables_packed_audio_without_a_second_config(monkeypatch):
    class PackedSALMDataset:
        def __init__(self, tokenizer, pack_audio=False):
            self.tokenizer = tokenizer
            self.pack_audio = pack_audio

    tokenizer = object()
    monkeypatch.setattr(_SALM_TRAIN, "SALMDataset", PackedSALMDataset)

    dataset = _SALM_TRAIN._create_salm_dataset(tokenizer, {}, pack_audio=True)

    assert dataset.tokenizer is tokenizer
    assert dataset.pack_audio is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_cfg", "expected_pack_audio", "expected_pack_sequences"),
    [
        ({}, False, False),
        ({"use_nemo_automodel": True, "packed_encoder_sequences": True}, True, False),
        ({"use_nemo_automodel": True, "packed_sequences": True}, False, True),
    ],
)
def test_train_uses_compatible_dataset_factory(
    monkeypatch,
    tmp_path,
    model_cfg,
    expected_pack_audio,
    expected_pack_sequences,
):
    tokenizer = object()
    dataset = object()
    calls = []

    class FakeSALM:
        def __init__(self, model_cfg):
            self.tokenizer = tokenizer

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = []

        def init_module(self):
            return nullcontext()

        def fit(self, model, datamodule):
            pass

    class FakeDataModule:
        def __init__(self, data_cfg, tokenizer, dataset):
            pass

    def create_salm_dataset(tokenizer_arg, data_cfg, *, pack_audio=False, pack_sequences=False):
        calls.append((tokenizer_arg, data_cfg, pack_audio, pack_sequences))
        return dataset

    monkeypatch.setattr(_SALM_TRAIN, "SALM", FakeSALM)
    if model_cfg.get("use_nemo_automodel", False):
        import nemo.collections.speechlm2

        monkeypatch.setattr(nemo.collections.speechlm2, "SALMAutomodel", FakeSALM)
    monkeypatch.setattr(_SALM_TRAIN, "Trainer", FakeTrainer)
    monkeypatch.setattr(_SALM_TRAIN, "DataModule", FakeDataModule)
    monkeypatch.setattr(_SALM_TRAIN, "_create_salm_dataset", create_salm_dataset)
    monkeypatch.setattr(_SALM_TRAIN, "seed_everything", lambda seed: None)
    monkeypatch.setattr(_SALM_TRAIN, "resolve_trainer_cfg", lambda cfg: {})
    monkeypatch.setattr(_SALM_TRAIN, "exp_manager", lambda trainer, cfg: tmp_path)
    monkeypatch.setattr(_SALM_TRAIN.OmegaConf, "save", lambda cfg, path: None)
    monkeypatch.setattr(_SALM_TRAIN.torch.cuda, "is_available", lambda: False)

    cfg = _SALM_TRAIN.OmegaConf.create(
        {
            "data": {"train_ds": {"seed": 0}},
            "model": model_cfg,
            "trainer": {},
        }
    )
    _SALM_TRAIN.train.__wrapped__(cfg)

    assert calls == [(tokenizer, cfg.data, expected_pack_audio, expected_pack_sequences)]
