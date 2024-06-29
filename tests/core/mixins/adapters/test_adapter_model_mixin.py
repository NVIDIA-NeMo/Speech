# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.core import ModelPT, NeuralModule
from nemo.core.classes.mixins import adapter_mixin_strategies, adapter_mixins
from nemo.core.classes.mixins.adapter_mixins import AdapterModelPTMixin, AdapterModuleMixin
from nemo.utils import logging, logging_mode


class DefaultModule(NeuralModule):
    """Define a default neural module (without adapter support)"""

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(50, 50)
        self.bn = torch.nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = x
        return out

    def num_params(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num


class DefaultModuleAdapter(DefaultModule, AdapterModuleMixin):
    """Subclass the DefaultModule, adding adapter module support"""

    def forward(self, x):
        x = super(DefaultModuleAdapter, self).forward(x)

        if self.is_adapter_available():
            # For testing purposes, cache the adapter names
            self._adapter_names = self.get_enabled_adapters()
            # call forward over model adapters, summing them up
            x = self.forward_enabled_adapters(x)

        return x


class DefaultModelAdapterMixin(AdapterModelPTMixin):
    """Mixin class that implements this model's specific overrides to AdapterModelPTMixin
    It will container two modules, an encoder and a decoder, and both can have adapters.
    By default, encoder adapters are enabled, and decoder adapters are diabled. Decoder adapters
    can be enabled via the global_cfg in model.cfg.adapters.

    Checks and forwards functions to the corresponding modules.

    It supports both global adapters and module adapters for testing purpose.
    """

    def setup_adapters(self):
        supports_adapters = False

        # At least the encoder must extend AdapterModuleMixin
        valid_adapter_names = [x for x in self.adapter_module_names if x != '']
        for module_name in valid_adapter_names:
            if hasattr(self, module_name) and isinstance(getattr(self, module_name), AdapterModuleMixin):
                supports_adapters |= True

        # If adapters are supported, setup the adapter config + any modules (pre-existing adapter modules)
        if supports_adapters:
            super().setup_adapters()

    def add_adapter(self, name: str, cfg: DictConfig):
        # Setup the config for adapters
        super().add_adapter(name, cfg)

        # Resolve module name and adapter name
        module_name, adapter_name = self.resolve_adapter_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        valid_module_names = [x for x in self.adapter_module_names if x != '']
        default_module_name = self.default_adapter_module_name

        # Update the model.cfg with information about the new adapter from cfg
        for module_name in module_names:
            # Check if encoder adapters should be added
            if module_name == '':
                for default in default_module_name:  # This model has multiple default modules
                    if hasattr(self, default):
                        # Dispatch the call to the default model.
                        getattr(self, default).add_adapter(name=name, cfg=cfg)

            elif module_name in valid_module_names:
                # Check if module exists
                if hasattr(self, module_name):
                    # Dispatch the call to the module.
                    getattr(self, module_name).add_adapter(name=name, cfg=cfg)

    def set_enabled_adapters(self, name=None, enabled: bool = True):
        # check if valid model with some adapter support
        super().set_enabled_adapters(name, enabled)

        # Resolve the module name and adapter name
        if name is not None:
            module_name, _ = self.resolve_adapter_module_name_(name)
        else:
            module_name = None

        # Use + as a splitter, in order to share one name across multiple modules
        if module_name is not None and '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        valid_module_names = [x for x in self.adapter_module_names if x != '']
        default_module_name = self.default_adapter_module_name

        # Check if default module name is None or not
        if default_module_name is None:
            raise ValueError(
                f"Default module name is None. Class {self.__class__.__name__} must implement "
                f"`default_adapter_module_name`"
            )

        # Forward the method call to the individual modules if they exist
        for module_name in module_names:
            # Check if encoder adapters should be used

            if module_name == '':
                for default in default_module_name:
                    if hasattr(self, default) and isinstance(getattr(self, default), AdapterModuleMixin):
                        if getattr(self, default).is_adapter_available():
                            # Dispatch the call to the default model.
                            getattr(self, default).set_enabled_adapters(name=name, enabled=enabled)

            elif module_name in valid_module_names:
                if hasattr(self, module_name) and isinstance(getattr(self, module_name), AdapterModuleMixin):
                    if getattr(self, module_name).is_adapter_available():
                        # Dispatch the call to the module.
                        getattr(self, module_name).set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> list:
        enabled_adapters = super().get_enabled_adapters()

        valid_module_names = [x for x in self.adapter_module_names if x != '']

        # Check if encoder adapters should be used or are enabled
        for module_name in valid_module_names:
            if hasattr(self, module_name) and isinstance(getattr(self, module_name), AdapterModuleMixin):
                enabled_adapters.extend(getattr(self, module_name).get_enabled_adapters())

        enabled_adapters = list(sorted(list(set(enabled_adapters))))

        return enabled_adapters

    def is_adapter_available(self) -> bool:
        adapters_available = super().is_adapter_available()

        valid_module_names = [x for x in self.adapter_module_names if x != '']

        # Forward the method call to the individual modules
        for module_name in valid_module_names:
            print("Module name", module_name)
            if hasattr(self, module_name) and isinstance(getattr(self, module_name), AdapterModuleMixin):
                adapters_available |= getattr(self, module_name).is_adapter_available()
                print("Adapter available for module", module_name, getattr(self, module_name).is_adapter_available())

        return adapters_available

    def check_valid_model_with_adapter_support_(self):
        global_cfg = DictConfig({})
        if self.adapter_global_cfg_key in self.adapter_cfg:
            global_cfg = self.adapter_cfg[self.adapter_global_cfg_key]

        encoder_adapter = global_cfg.get('encoder_adapter', True)
        decoder_adapter = global_cfg.get('decoder_adapter', False)

        if encoder_adapter and not hasattr(self, 'encoder'):
            logging.warning("Encoder not available", mode=logging_mode.ONCE)
        elif encoder_adapter and not isinstance(self.encoder, AdapterModuleMixin):
            logging.warning("Encoder does not support adapters !", mode=logging_mode.ONCE)

        if decoder_adapter and not hasattr(self, 'decoder'):
            logging.warning("Decoder is not available", mode=logging_mode.ONCE)
        elif decoder_adapter and not isinstance(self.decoder, AdapterModuleMixin):
            logging.warning("Decoder does not support adapters !", mode=logging_mode.ONCE)

    def resolve_adapter_module_name_(self, name: str) -> Tuple[str, str]:
        # resolve name and module
        valid_module_names = self.adapter_module_names
        module_name, adapter_name = super().resolve_adapter_module_name_(name)

        if module_name not in valid_module_names:
            raise ValueError(f"Provided module name `{module_name}` is not in valid list : {valid_module_names}")

        return (module_name, adapter_name)

    def _get_global_cfg(self):
        global_config = DictConfig({})
        if 'adapters' in self.cfg and self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]
        return global_config

    @property
    def adapter_module_names(self) -> list:
        valid_adapter_modules = ['', 'encoder', 'decoder']
        return valid_adapter_modules

    @property
    def default_adapter_module_name(self) -> Optional[List[str]]:
        global_config = self._get_global_cfg()
        default_modules = []
        encoder_adapter = global_config.get('encoder_adapter', True)
        decoder_adapter = global_config.get('decoder_adapter', False)

        if encoder_adapter:
            default_modules.append('encoder')
        if decoder_adapter:
            default_modules.append('decoder')
        return default_modules


class DefaultAdapterModel(ModelPT, DefaultModelAdapterMixin):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer=trainer)

        self.encoder = instantiate(cfg.encoder)  # type: DefaultModuleAdapter
        self.decoder = instantiate(cfg.decoder)  # type: DefaultModuleAdapter

        # Required to be called for adapter support
        self.setup_adapters()

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z

    def list_available_models(cls):
        return []

    def setup_training_data(self, train_data_config):
        self._update_dataset_config('train', train_data_config)
        self._train_dl = None

    def setup_validation_data(self, val_data_config):
        self._update_dataset_config('validation', val_data_config)
        self._validation_dl = None


def get_adapter_cfg(in_features=50, dim=100, norm_pos='pre'):
    cfg = {
        '_target_': 'nemo.collections.common.parts.adapter_modules.LinearAdapter',
        'in_features': in_features,
        'dim': dim,
        'norm_position': norm_pos,
    }
    return cfg


def get_model_config(in_features=50, update_adapter_cfg: bool = True):
    config = OmegaConf.create(
        {
            'in_features': in_features,
            'encoder': {'_target_': get_classpath(DefaultModule)},
            'decoder': {'_target_': get_classpath(DefaultModule)},
        }
    )

    if update_adapter_cfg:
        enc_adapter_metadata = adapter_mixins.get_registered_adapter(config.encoder._target_)
        if enc_adapter_metadata is not None:
            config.encoder._target_ = enc_adapter_metadata.adapter_class_path

        dec_adapter_metadata = adapter_mixins.get_registered_adapter(config.decoder._target_)
        if dec_adapter_metadata is not None:
            config.decoder._target_ = dec_adapter_metadata.adapter_class_path

    return config


def update_adapter_global_cfg(cfg: DictConfig, encoder_adapter=True, decoder_adapter=False):
    if 'adapters' not in cfg:
        cfg.adapters = adapter_mixins._prepare_default_adapter_config(
            global_key=AdapterModuleMixin.adapter_global_cfg_key, meta_key=AdapterModuleMixin.adapter_metadata_cfg_key
        )

    cfg.adapters.global_cfg.encoder_adapter = encoder_adapter
    cfg.adapters.global_cfg.decoder_adapter = decoder_adapter
    return cfg


def get_classpath(cls):
    return f'{cls.__module__}.{cls.__name__}'


if adapter_mixins.get_registered_adapter(DefaultModule) is None:
    adapter_mixins.register_adapter(DefaultModule, DefaultModuleAdapter)


class TestAdapterModelMixin:
    @pytest.mark.unit
    def test_base_model_no_support_for_adapters(self, caplog):
        logging._logger.propagate = True
        original_verbosity = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)
        caplog.set_level(logging.WARNING)

        cfg = get_model_config(in_features=50, update_adapter_cfg=False)
        model = DefaultAdapterModel(cfg)

        with pytest.raises(AttributeError):
            model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # check that warning message indicates that it module is not available
        assert """Encoder does not support adapters !""" in caplog.text

        caplog.clear()
        model.get_enabled_adapters()

        # check that there is not warning message, since it should log only once.
        assert """Encoder does not support adapters !""" not in caplog.text

        logging._logger.propagate = False
        logging.set_verbosity(original_verbosity)

    @pytest.mark.unit
    def test_base_model_replace_adapter_compatible_modules(self, caplog):
        cfg = get_model_config(in_features=50, update_adapter_cfg=False)
        model = DefaultAdapterModel(cfg)

        with pytest.raises(AttributeError):
            model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # Replace the modules of the model dynamically to support adapters
        model.replace_adapter_compatible_modules()

        assert isinstance(model.encoder, AdapterModuleMixin)
        assert model.encoder.is_adapter_available() is False

        model.add_adapter(name='encoder:adapter_0', cfg=get_adapter_cfg())
        assert model.encoder.is_adapter_available() is True

    @pytest.mark.unit
    def test_single_adapter(self):
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_single_encoder_module_adapter(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='encoder:adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        assert model.decoder.is_adapter_available() is False

        adapter_cfg = model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[model.get_enabled_adapters()[0]] == 'encoder'  # encoder

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, 'temp.nemo')
                model.save_to(path)
                shutil.move(path, outer_tmpdir)

            outer_path = os.path.join(outer_tmpdir, 'temp.nemo')
            new_model = DefaultAdapterModel.restore_from(outer_path)  # type: DefaultAdapterModel

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights == new_model.num_weights
        assert new_model.decoder.is_adapter_available() is False

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[model.get_enabled_adapters()[0]] == 'encoder'  # encoder

    @pytest.mark.unit
    def test_single_decoder_module_adapter(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        assert model.encoder.is_adapter_available() is False

        adapter_cfg = model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[model.get_enabled_adapters()[0]] == 'decoder'  # decoder module

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, 'temp.nemo')
                model.save_to(path)
                shutil.move(path, outer_tmpdir)

            outer_path = os.path.join(outer_tmpdir, 'temp.nemo')
            new_model = DefaultAdapterModel.restore_from(outer_path)  # type: DefaultAdapterModel

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights == new_model.num_weights
        assert new_model.encoder.is_adapter_available() is False

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[new_model.get_enabled_adapters()[0]] == 'decoder'  # decoder module

    @pytest.mark.unit
    def test_single_adapter_default_metaconfig(self):
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        adapter_cfg = model.cfg.adapters
        assert model.adapter_global_cfg_key in adapter_cfg
        assert model.adapter_metadata_cfg_key in adapter_cfg[model.adapter_global_cfg_key]

        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        assert meta_cfg is not None
        assert 'modules' in meta_cfg

        modules_cfg = meta_cfg['modules']
        assert modules_cfg is not None
        assert modules_cfg[model.get_enabled_adapters()[0]] == ''  # default module

    @pytest.mark.unit
    def test_all_disabled_adapters(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=False, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights

        assert new_num_params == original_num_params
        assert model.is_adapter_available() is False

    @pytest.mark.unit
    def test_set_enabled_all_adapters_with_no_name(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        model.add_adapter(name='decoder:adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_weights

        model.set_enabled_adapters(enabled=False)

        assert new_num_params > original_num_params
        assert model.is_adapter_available() is True
        assert len(model.get_enabled_adapters()) == 0

    @pytest.mark.unit
    def test_set_enabled_all_adapters_with_no_name_only_decoder(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='decoder:adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_weights

        model.set_enabled_adapters(enabled=False)

        assert new_num_params > original_num_params
        assert model.is_adapter_available() is True
        assert len(model.get_enabled_adapters()) == 0

    @pytest.mark.unit
    def test_enc_dec_enabled_adapters(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)
        model2 = DefaultAdapterModel(cfg)

        model2.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_encdec_num_params = model2.num_weights

        assert new_encdec_num_params > new_num_params

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_multiple_adapter(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        original_num_params = new_num_params
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_multiple_adapter_non_unique_adapter_name(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)

        model = DefaultAdapterModel(cfg)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        with pytest.raises(ValueError):
            model.add_adapter(name='encoder:adapter_0', cfg=get_adapter_cfg())

        with pytest.raises(ValueError):
            model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg())

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_forward_linear_pre(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_forward_linear_post(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(norm_pos='post'))
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_multi_adapter_forward(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_output = model(x)

        if enc:
            assert model.encoder._adapter_names == ['adapter_0', 'adapter_1']
        if dec:
            assert model.decoder._adapter_names == ['adapter_0', 'adapter_1']

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_multi_adapter_partial_forward_global_module_different(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        # add encoder adapters
        if enc:
            model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
            model.add_adapter(name='encoder:adapter_1', cfg=get_adapter_cfg())

        # add decoder adapters
        if dec:
            model.add_adapter(name='decoder:adapter_2', cfg=get_adapter_cfg())
            model.add_adapter(name='decoder:adapter_3', cfg=get_adapter_cfg())

        # disable encoder adapters
        if enc:
            model.set_enabled_adapters(name='adapter_0', enabled=False)

        # disable decoder adapters
        if dec:
            model.set_enabled_adapters(name='adapter_3', enabled=False)

        # perform forward
        new_output = model(x)

        if enc:
            assert model.encoder._adapter_names == ['adapter_1']
        if dec:
            assert model.decoder._adapter_names == ['adapter_2']
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name1', ['adapter_0', 'encoder:adapter_0'])
    @pytest.mark.parametrize('name2', ['adapter_1', 'encoder:adapter_1'])
    def test_multi_adapter_partial_forward_global_module_same_output(self, name1, name2):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        original_output = model(x)

        model.add_adapter(name=name1, cfg=get_adapter_cfg())
        model.add_adapter(name=name2, cfg=get_adapter_cfg())

        model.set_enabled_adapters(name=name1, enabled=False)
        new_output = model(x)

        resolved_name2 = model.resolve_adapter_module_name_(name2)[-1]
        assert model.get_enabled_adapters() == [resolved_name2]
        assert torch.mean(torch.abs(original_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_forward_unfrozen_adapters(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        dim = 10

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(dim=dim))
        model.freeze()
        model.unfreeze_enabled_adapters()

        assert original_num_params == 5300

        original_params = 0
        adapter_params = 0
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                assert param.requires_grad is False
                original_params += param.numel()
            else:
                assert param.requires_grad is True
                adapter_params += param.numel()

        for mname, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                assert module.track_running_stats is False

        assert original_params > adapter_params

        enc_params = model.encoder.num_params()
        dec_params = model.decoder.num_params()
        assert adapter_params == enc_params + dec_params

    @pytest.mark.unit
    def test_forward_linear_no_strategy(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # delete the strategy
        adapter_module = model.encoder.adapter_layer[model.get_enabled_adapters()[0]]
        del adapter_module.adapter_strategy

        with pytest.raises(AttributeError):
            _ = model(x)

    @pytest.mark.unit
    def test_forward_linear_replaced_strategy(self):
        class MultiplyAdapterStrategy(adapter_mixin_strategies.AbstractAdapterStrategy):
            def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: AdapterModuleMixin):
                out = adapter(input)
                return input * out

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        # Use decoder only adapter
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)

        model = DefaultAdapterModel(cfg)
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # modify the strategy of both encoder and decoder
        adapter_module = model.encoder.adapter_layer[model.get_enabled_adapters()[0]]
        adapter_module.adapter_strategy = MultiplyAdapterStrategy()

        adapter_module = model.decoder.adapter_layer[model.get_enabled_adapters()[0]]
        adapter_module.adapter_strategy = MultiplyAdapterStrategy()

        out = model(x)
        # result of adapter is zero tensor, output multiplied by adapter result should be zero
        assert (out > 0.0).any() == torch.tensor(False)

    @pytest.mark.unit
    def test_save_adapter_with_no_adapters_added(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)

        with pytest.raises(AttributeError):
            model.save_adapters(filepath='temp.pt', name=None)

    @pytest.mark.unit
    def test_single_decoder_save_load_adapter_only_exact_name(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg(dim=5))
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        assert model.encoder.is_adapter_available() is False

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                adapter_path = os.path.join(tmpdir, 'temp.pt')
                model.save_adapters(adapter_path, name='decoder:adapter_0')

                model_path = os.path.join('temp.nemo')
                model.save_to(model_path)

                shutil.move(adapter_path, outer_tmpdir)
                shutil.move(model_path, outer_tmpdir)

            outer_adapter_path = os.path.join(outer_tmpdir, 'temp.pt')
            outer_model_path = os.path.join(outer_tmpdir, 'temp.nemo')

            # Assert size of this params
            adapter_filesize = os.path.getsize(outer_adapter_path)
            model_filesize = os.path.getsize(outer_model_path)

            assert model_filesize > adapter_filesize

            # restore adapter to new model (without any decoder adapter)
            new_model = DefaultAdapterModel(cfg)
            new_model.load_adapters(outer_adapter_path, name='decoder:adapter_0')

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights == new_model.num_weights
        assert new_model.encoder.is_adapter_available() is False

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[new_model.get_enabled_adapters()[0]] == 'decoder'  # decoder module

        original_state_dict = model.decoder.adapter_layer.state_dict()
        restored_state_dict = new_model.decoder.adapter_layer.state_dict()

        for ogkey, newkey in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[ogkey] - restored_state_dict[newkey]).abs().mean() < 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize('restore_name', [None, 'adapter_0'])
    def test_single_decoder_save_load_adapter_only_global_name(self, restore_name):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(dim=5))
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        assert model.decoder.is_adapter_available() is False

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                adapter_path = os.path.join(tmpdir, 'temp.pt')
                model.save_adapters(adapter_path, name='adapter_0')

                model_path = os.path.join('temp.nemo')
                model.save_to(model_path)

                shutil.move(adapter_path, outer_tmpdir)
                shutil.move(model_path, outer_tmpdir)

            outer_adapter_path = os.path.join(outer_tmpdir, 'temp.pt')
            outer_model_path = os.path.join(outer_tmpdir, 'temp.nemo')

            # Assert size of this params
            adapter_filesize = os.path.getsize(outer_adapter_path)
            model_filesize = os.path.getsize(outer_model_path)

            assert model_filesize > adapter_filesize

            # restore adapter to new model (without any encoder adapter)
            new_model = DefaultAdapterModel(cfg)
            new_model.load_adapters(outer_adapter_path, name=restore_name)

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights == new_model.num_weights
        assert new_model.decoder.is_adapter_available() is False

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[new_model.get_enabled_adapters()[0]] == ''  # global adapter

        original_state_dict = model.encoder.adapter_layer.state_dict()
        restored_state_dict = new_model.encoder.adapter_layer.state_dict()

        for ogkey, newkey in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[ogkey] - restored_state_dict[newkey]).abs().mean() < 1e-6

    @pytest.mark.unit
    def test_multiple_decoder_save_load_adapter_only_exact_name(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg(dim=5))
        model.add_adapter(name='encoder:adapter_1', cfg=get_adapter_cfg(dim=5))
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                adapter_path = os.path.join(tmpdir, 'temp.pt')
                model.save_adapters(adapter_path, name='decoder:adapter_0')

                model_path = os.path.join('temp.nemo')
                model.save_to(model_path)

                shutil.move(adapter_path, outer_tmpdir)
                shutil.move(model_path, outer_tmpdir)

            outer_adapter_path = os.path.join(outer_tmpdir, 'temp.pt')
            outer_model_path = os.path.join(outer_tmpdir, 'temp.nemo')

            # Assert size of this params
            adapter_filesize = os.path.getsize(outer_adapter_path)
            model_filesize = os.path.getsize(outer_model_path)

            assert model_filesize > adapter_filesize

            # restore adapter to new model (without any decoder adapter)
            new_model = DefaultAdapterModel(cfg)
            new_model.load_adapters(outer_adapter_path, name='decoder:adapter_0')

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights > new_model.num_weights  # the new model has only one adapter not both
        assert new_model.encoder.is_adapter_available() is False  # encoder adaper not available in new model

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[new_model.get_enabled_adapters()[0]] == 'decoder'  # decoder

        original_state_dict = model.decoder.adapter_layer.state_dict()
        restored_state_dict = new_model.decoder.adapter_layer.state_dict()

        for ogkey, newkey in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[ogkey] - restored_state_dict[newkey]).abs().mean() < 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "decoder",
        [
            "adapter_0",
        ],
    )  # "decoder:adapter_0"
    @pytest.mark.parametrize(
        "encoder",
        [
            "adapter_1",
        ],
    )  # "encoder:adapter_1"
    def test_multiple_save_load_adapter_with_multiple_load(self, decoder, encoder):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name=decoder, cfg=get_adapter_cfg(dim=5))
        model.add_adapter(name=encoder, cfg=get_adapter_cfg(dim=5))
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        assert len(model.get_enabled_adapters()) == 2

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                adapter_path = os.path.join(tmpdir, 'temp.pt')
                adapter_path_2 = os.path.join(tmpdir, 'temp-2.pt')
                print("saving adapter ", decoder)
                model.save_adapters(adapter_path, name=decoder)
                print("Saving adapter ", encoder)
                model.save_adapters(adapter_path_2, name=encoder)

                model_path = os.path.join('temp.nemo')
                model.save_to(model_path)

                shutil.move(adapter_path, outer_tmpdir)
                shutil.move(adapter_path_2, outer_tmpdir)
                shutil.move(model_path, outer_tmpdir)

            outer_adapter_path = os.path.join(outer_tmpdir, 'temp.pt')
            outer_adapter_path_2 = os.path.join(outer_tmpdir, 'temp-2.pt')
            outer_model_path = os.path.join(outer_tmpdir, 'temp.nemo')

            # Assert size of this params
            adapter_filesize = os.path.getsize(outer_adapter_path)
            adapter_2_filesize = os.path.getsize(outer_adapter_path_2)
            model_filesize = os.path.getsize(outer_model_path)

            assert model_filesize > adapter_filesize
            assert model_filesize > adapter_2_filesize

            # restore adapter to new model (without any decoder adapter)
            new_model = DefaultAdapterModel(cfg)
            new_model.load_adapters(outer_adapter_path, name=decoder)
            # Seperately load another adapter after the first one
            new_model.load_adapters(outer_adapter_path_2, name=encoder)

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights == new_model.num_weights  # the new model has only one adapter not both
        assert new_model.encoder.is_adapter_available() is True  # encoder adaper is available in new model

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        enabled_adapters = new_model.get_enabled_adapters()
        assert len(enabled_adapters) == 2

        if "decoder:" in decoder:
            original_state_dict = model.decoder.adapter_layer.state_dict()
            restored_state_dict = new_model.decoder.adapter_layer.state_dict()
        else:
            # Default adapter position is on encoder
            original_state_dict = model.encoder.adapter_layer.state_dict()
            restored_state_dict = new_model.encoder.adapter_layer.state_dict()

        for ogkey, newkey in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[ogkey] - restored_state_dict[newkey]).abs().mean() < 1e-6

    @pytest.mark.unit
    def test_multiple_decoder_save_load_adapter_dual_name(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        # one adapter will have module name, other will have global name
        model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg(dim=5))
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg(dim=5))
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                adapter_path = os.path.join(tmpdir, 'temp.pt')
                model.save_adapters(adapter_path, name=None)  # save all adapters

                model_path = os.path.join('temp.nemo')
                model.save_to(model_path)

                shutil.move(adapter_path, outer_tmpdir)
                shutil.move(model_path, outer_tmpdir)

            outer_adapter_path = os.path.join(outer_tmpdir, 'temp.pt')
            outer_model_path = os.path.join(outer_tmpdir, 'temp.nemo')

            # Assert size of this params
            adapter_filesize = os.path.getsize(outer_adapter_path)
            model_filesize = os.path.getsize(outer_model_path)

            assert model_filesize > adapter_filesize

            # restore adapter to new model (without any decoder adapter)
            new_model = DefaultAdapterModel(cfg)
            new_model.load_adapters(outer_adapter_path, name='decoder:adapter_0')  # load just one adapter from 2 saved

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights > new_model.num_weights  # the new model has only one adapter not both
        assert new_model.encoder.is_adapter_available() is False  # encoder adaper not available in new model

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[new_model.get_enabled_adapters()[0]] == 'decoder'  # decoder

        original_state_dict = model.decoder.adapter_layer.state_dict()
        restored_state_dict = new_model.decoder.adapter_layer.state_dict()

        for ogkey, newkey in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[ogkey] - restored_state_dict[newkey]).abs().mean() < 1e-6

    @pytest.mark.unit
    def test_single_decoder_save_load_adapter_only_partial_name(self):
        # create a model config, but do not add global_cfg to it
        # we want to test just module level adapter
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        # build adapter with exact name in decoder module only
        model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg(dim=5))
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        assert model.encoder.is_adapter_available() is False

        # save restore test
        with tempfile.TemporaryDirectory() as outer_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                adapter_path = os.path.join(tmpdir, 'temp.pt')

                # save adapter with partial name- just adapter_0
                model.save_adapters(adapter_path, name='adapter_0')

                model_path = os.path.join('temp.nemo')
                model.save_to(model_path)

                shutil.move(adapter_path, outer_tmpdir)
                shutil.move(model_path, outer_tmpdir)

            outer_adapter_path = os.path.join(outer_tmpdir, 'temp.pt')
            outer_model_path = os.path.join(outer_tmpdir, 'temp.nemo')

            # Assert size of this params
            adapter_filesize = os.path.getsize(outer_adapter_path)
            model_filesize = os.path.getsize(outer_model_path)

            assert model_filesize > adapter_filesize

            # restore adapter to new model (without any decoder adapter)
            new_model = DefaultAdapterModel(cfg)
            # load adapter with partial name only - just adapter_0 - should work
            new_model.load_adapters(outer_adapter_path, name='adapter_0')

            # restore adapter to new model (without any decoder adapter)
            new_model = DefaultAdapterModel(cfg)
            # properly load with correct key
            new_model.load_adapters(outer_adapter_path, name='decoder:adapter_0')

        assert isinstance(new_model, AdapterModelPTMixin)
        assert len(new_model.get_enabled_adapters()) > 0
        assert model.num_weights == new_model.num_weights
        assert new_model.encoder.is_adapter_available() is False

        adapter_cfg = new_model.cfg.adapters
        meta_cfg = adapter_cfg[model.adapter_global_cfg_key][model.adapter_metadata_cfg_key]
        modules_cfg = meta_cfg['modules']

        assert modules_cfg is not None
        assert modules_cfg[new_model.get_enabled_adapters()[0]] == 'decoder'  # decoder module

        original_state_dict = model.decoder.adapter_layer.state_dict()
        restored_state_dict = new_model.decoder.adapter_layer.state_dict()

        for ogkey, newkey in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[ogkey] - restored_state_dict[newkey]).abs().mean() < 1e-6
