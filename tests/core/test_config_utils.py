# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import pytest
import pytorch_lightning as ptl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import config_utils
from nemo.utils.exp_manager import EarlyStoppingParams


@pytest.fixture()
def cls():
    class DummyClass:
        def __init__(self, a, b=5, c: int = 0, d: 'ABC' = None):
            pass

    return DummyClass


class TestConfigUtils:
    @pytest.mark.unit
    def test_all_args_exist(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass)
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_all_args_dont_exist(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass)
        signatures_match, cls_subset, dataclass_subset = result

        assert not signatures_match
        assert len(cls_subset) > 0
        assert len(dataclass_subset) == 0

    @pytest.mark.unit
    def test_extra_args_exist(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None
            e: float = 0.0

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass)
        signatures_match, cls_subset, dataclass_subset = result

        assert not signatures_match
        assert len(cls_subset) == 0
        assert len(dataclass_subset) > 0

    @pytest.mark.unit
    def test_extra_args_exist_but_is_ignored(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None
            e: float = 0.0  # Assume ignored

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass, ignore_args=['e'])
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_args_exist_but_is_remapped(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            e: Any = None  # Assume remapped

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass, remap_args={'e': 'd'})
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_ptl_config(self):
        PTL_DEPRECATED = []
        result = config_utils.assert_dataclass_signature_match(ptl.Trainer, TrainerConfig, PTL_DEPRECATED)
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_early_stopping_config(self,):
        result = config_utils.assert_dataclass_signature_match(EarlyStopping, EarlyStoppingParams)
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None
