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

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules import SpectrogramAugmentation
from nemo.core.classes.common import Serialization


def get_class_path(cls):
    return f"{cls.__module__}.{cls.__name__}"


class MockSerializationImpl(Serialization):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.value = self.__class__.__name__


class MockSerializationImplV2(MockSerializationImpl):
    pass


class TestSerialization:
    @pytest.mark.unit
    def test_from_config_dict_with_cls(self):
        """Here we test that instantiation works for configs with cls class path in them.
        Note that just Serialization.from_config_dict can be used to create an object"""
        config = DictConfig(
            {
                'cls': 'nemo.collections.asr.modules.SpectrogramAugmentation',
                'params': {
                    'rect_freq': 50,
                    'rect_masks': 5,
                    'rect_time': 120,
                },
            }
        )
        obj = Serialization.from_config_dict(config=config)
        assert isinstance(obj, SpectrogramAugmentation)

    @pytest.mark.unit
    def test_from_config_dict_without_cls(self):
        """Here we test that instantiation works for configs without cls class path in them.
        IMPORTANT: in this case, correct class type should call from_config_dict. This should work for Models."""
        preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
        encoder = {
            'cls': 'nemo.collections.asr.modules.ConvASREncoder',
            'params': {
                'feat_in': 64,
                'activation': 'relu',
                'conv_mask': True,
                'jasper': [
                    {
                        'filters': 1024,
                        'repeat': 1,
                        'kernel': [1],
                        'stride': [1],
                        'dilation': [1],
                        'dropout': 0.0,
                        'residual': False,
                        'separable': True,
                        'se': True,
                        'se_context_size': -1,
                    }
                ],
            },
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
            'params': {
                'feat_in': 1024,
                'num_classes': 28,
                'vocabulary': [
                    ' ',
                    'a',
                    'b',
                    'c',
                    'd',
                    'e',
                    'f',
                    'g',
                    'h',
                    'i',
                    'j',
                    'k',
                    'l',
                    'm',
                    'n',
                    'o',
                    'p',
                    'q',
                    'r',
                    's',
                    't',
                    'u',
                    'v',
                    'w',
                    'x',
                    'y',
                    'z',
                    "'",
                ],
            },
        }
        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
        )
        obj = EncDecCTCModel.from_config_dict(config=modelConfig)
        assert isinstance(obj, EncDecCTCModel)

    @pytest.mark.unit
    def test_config_updated(self):
        config = DictConfig(
            {
                'cls': 'nemo.collections.asr.modules.SpectrogramAugmentation',
                'params': {
                    'rect_freq': 50,
                    'rect_masks': 5,
                    'rect_time': 120,
                },
            }
        )
        obj = Serialization.from_config_dict(config=config)
        new_config = obj.to_config_dict()
        assert config != new_config
        assert 'params' not in new_config
        assert 'cls' not in new_config
        assert '_target_' in new_config

    @pytest.mark.unit
    def test_base_class_instantiation(self):
        # Target class is V2 impl, calling class is Serialization (base class)
        config = DictConfig({'target': get_class_path(MockSerializationImplV2)})
        obj = Serialization.from_config_dict(config=config)
        new_config = obj.to_config_dict()
        assert config == new_config
        assert isinstance(obj, MockSerializationImplV2)
        assert obj.value == "MockSerializationImplV2"

    @pytest.mark.unit
    def test_self_class_instantiation(self):
        # Target class is V1 impl, calling class is V1 (same class)
        config = DictConfig({'target': get_class_path(MockSerializationImpl)})
        obj = MockSerializationImpl.from_config_dict(config=config)  # Serialization is base class
        new_config = obj.to_config_dict()
        assert config == new_config
        assert isinstance(obj, MockSerializationImpl)
        assert obj.value == "MockSerializationImpl"

    @pytest.mark.unit
    def test_sub_class_instantiation(self):
        # Target class is V1 impl, calling class is V2 (sub class)
        config = DictConfig({'target': get_class_path(MockSerializationImpl)})
        obj = MockSerializationImplV2.from_config_dict(config=config)  # Serialization is base class
        new_config = obj.to_config_dict()
        assert config == new_config
        assert isinstance(obj, MockSerializationImplV2)
        assert obj.value == "MockSerializationImplV2"
