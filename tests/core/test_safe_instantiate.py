# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.core.classes.common import _is_target_allowed, safe_instantiate


class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def get_class_path(cls):
    return f"{cls.__module__}.{cls.__name__}"


class MockG2p(BaseG2p):
    def __call__(self, text: str) -> str:
        return text


class MockTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__(tokens=["a"])

    def encode(self, text: str) -> list[int]:
        return [0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "config,expected_type",
    [
        ({"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1}, torch.nn.Linear),
        ({"_target_": get_class_path(MockDataset)}, MockDataset),
        ({"_target_": "torch.distributed.fsdp.MixedPrecisionPolicy"}, MixedPrecisionPolicy),
        ({"_target_": "lightning.pytorch.callbacks.ModelCheckpoint"}, ModelCheckpoint),
        ({"_target_": "lightning.pytorch.strategies.DDPStrategy"}, DDPStrategy),
    ],
)
def test_safe_instantiate_allows_approved_targets(config, expected_type):
    obj = safe_instantiate(DictConfig(config))
    assert isinstance(obj, expected_type)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target,target_type",
    [
        ("nemo_text_processing.text_normalization.normalize.Normalizer", type("MockNormalizer", (), {})),
        ("nemo.collections.tts.torch.g2ps.EnglishG2p", MockG2p),
        ("nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer", MockTokenizer),
    ],
)
def test_safe_instantiate_allows_exact_exception_targets(target, target_type):
    sentinel = object()
    config = DictConfig({"_target_": target})

    with (
        patch("hydra.utils.get_class", return_value=target_type),
        patch("hydra.utils.instantiate", return_value=sentinel) as instantiate_mock,
    ):
        obj = safe_instantiate(config)

    assert obj is sentinel
    instantiate_mock.assert_called_once_with(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        "subprocess.Popen",
        "builtins.open",
        "os.system",
    ],
)
def test_safe_instantiate_blocks_unsafe_targets_before_hydra(target):
    config = DictConfig({"_target_": target})
    with patch("hydra.utils.instantiate") as instantiate_mock:
        with pytest.raises(ValueError, match=f"Instantiation of unsafe target '{target}' is blocked"):
            safe_instantiate(config)

    instantiate_mock.assert_not_called()


@pytest.mark.unit
def test_safe_instantiate_validates_nested_targets_before_hydra():
    config = DictConfig(
        {
            "_target_": "torch.nn.ModuleList",
            "modules": [
                {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
                {"_target_": "subprocess.Popen"},
            ],
        }
    )

    with patch("hydra.utils.instantiate") as instantiate_mock:
        with pytest.raises(ValueError, match="Instantiation of unsafe target 'subprocess.Popen' is blocked"):
            safe_instantiate(config)

    instantiate_mock.assert_not_called()


@pytest.mark.unit
def test_safe_instantiate_allows_ssl_batch_augmentation():
    """A Serialization subclass (MultiSpeakerNoiseAugmentation) must pass the guard."""
    pytest.importorskip("nemo.collections.asr.modules.ssl_modules.augmentation")

    assert _is_target_allowed("nemo.collections.asr.modules.ssl_modules.MultiSpeakerNoiseAugmentation") is True
    assert (
        _is_target_allowed("nemo.collections.asr.modules.ssl_modules.augmentation.MultiSpeakerNoiseAugmentation")
        is True
    )
    assert _is_target_allowed("nemo.collections.asr.modules.ssl_modules.augmentation.SpeakerNoiseAugmentation") is True
