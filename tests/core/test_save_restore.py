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
import filecmp
import os
import shutil
import tempfile
from typing import Callable, Dict, Optional, Set, Union

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.core.classes import ModelPT
from nemo.core.connectors import save_restore_connector
from nemo.utils.app_state import AppState
from nemo.utils.exceptions import NeMoBaseException


def classpath(cls):
    return f'{cls.__module__}.{cls.__name__}'


def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def get_size(path='.'):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        return get_dir_size(path)


def getattr2(object, attr):
    if not '.' in attr:
        return getattr(object, attr)
    else:
        arr = attr.split('.')
        return getattr2(getattr(object, arr[0]), '.'.join(arr[1:]))


class MockModel(ModelPT):
    def __init__(self, cfg, trainer=None):
        super(MockModel, self).__init__(cfg=cfg, trainer=trainer)
        self.w = torch.nn.Linear(10, 1)
        # mock temp file
        if 'temp_file' in self.cfg and self.cfg.temp_file is not None:
            self.setup_data_from_file(self.cfg.temp_file)
        else:
            self.temp_file = None
            self.temp_data = None

    def setup_data_from_file(self, temp_file):
        """
        Load data from temp_file to `self.temp_data`
        Allows to test changing resource after instantiation
        """
        with open_dict(self.cfg):
            self.cfg.temp_file = temp_file
        self.temp_file = self.register_artifact('temp_file', self.cfg.temp_file)
        with open(self.temp_file, 'r', encoding='utf-8') as f:
            self.temp_data = f.readlines()

    def change_stub_number(self, new_number: int):
        """
        Change stub number in config, useful for testing nested models,
        since child can mutate config independently
        """
        self.cfg.stub_number = new_number

    def forward(self, x):
        y = self.w(x)
        return y, self.cfg.temp_file

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


class MockModelWithChildren(MockModel):
    """
    Mock Model, can contain 2 children (other NeMo models)
    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        # variant 1 for creating nested NeMo model:
        # load model directly from config

        # variant 2 for creating nested NeMo model:
        # - initialize child model from .nemo checkpoint, subconfig will be automatically saved
        # - after saving model will be restored directly from subconfig (attribute `config_field` of self.cfg)

        # child 1
        self.child1_model: Optional[MockModel]  # annotate type for IDE autocompletion and type checking
        if cfg.get("child1_model") is not None:
            self.register_nemo_submodule(
                "child1_model",
                config_field="child1_model",
                model=MockModel(self.cfg.child1_model),
            )
        elif cfg.get("child1_model_path") is not None:
            self.register_nemo_submodule(
                "child1_model",
                config_field="child1_model",
                model=MockModel.restore_from(self.cfg.child1_model_path),
            )
        else:
            self.child1_model = None

        # child 2
        # can have sub-children
        self.child2_model: Optional[MockModelWithChildren]  # annotate type for IDE autocompletion and type checking
        if cfg.get("child2_model") is not None:
            self.register_nemo_submodule(
                "child2_model",
                config_field="child2_model",
                model=MockModelWithChildren(self.cfg.child2_model),
            )
        elif cfg.get("child2_model_path") is not None:
            self.register_nemo_submodule(
                "child2_model",
                config_field="child2_model",
                model=MockModelWithChildren.restore_from(self.cfg.child2_model_path),
            )
        else:
            self.child2_model = None


class MockModelWithChildEncDecCTCBPE(MockModel):
    """
    Mock Model, will contain EncDecCTC model as a child
    Useful for testing nested models with children initialized from pretrained NeMo models
    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        # variant 3 for creating nested NeMo model:
        # - initialize child model from pretrained NeMo model, subconfig will be automatically saved
        # - after saving model will be restored directly from subconfig (attribute `config_field` of self.cfg)

        self.ctc_model: EncDecCTCModelBPE  # annotate type for IDE autocompletion and type checking

        if cfg.get("ctc_model", None) is not None:
            self.register_nemo_submodule(
                "ctc_model",
                config_field="ctc_model",
                model=EncDecCTCModelBPE(self.cfg.ctc_model),
            )
        else:
            # model is mandatory
            assert cfg.get("ctc_model_pretrained", None) is not None
            self.register_nemo_submodule(
                "ctc_model",
                config_field="ctc_model",
                model=EncDecCTCModelBPE.from_pretrained(self.cfg.ctc_model_pretrained),
            )


class MockModelWithChildCustomConfigPath(MockModel):
    """
    Mock Model, can contain 1 child
    Path in config is not equal to name of the attribute
    Config is stored in `child1_model_config`
    Child model is stored in `child1_model` attribute
    NB: This is not recommended if it's not necessary. But here we test that it works.
    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.child1_model: Optional[MockModel]  # annotate type for IDE autocompletion and type checking
        if cfg.get("child1_model_config") is not None:
            self.register_nemo_submodule(
                "child1_model",
                config_field="child1_model_config",
                model=MockModel(self.cfg.child1_model_config),
            )
        else:
            self.child1_model = None


class MockModelIncorrectWithNemoArtifact(MockModel):
    """
    Incorrect model that tries to use .nemo model checkpoint as an artifact
    Expected to fail, since it is not supported
    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        assert cfg.get("child_model_path") is not None
        # this will fail, since .nemo model checkpoint is not supported as an artifact
        child_model_path = self.register_artifact("child_model_path", cfg.child_model_path)
        self.child_model = ModelPT.restore_from(child_model_path)


def _mock_model_config():
    conf = {'temp_file': None, 'target': classpath(MockModel), 'stub_number': 1}
    conf = OmegaConf.create({'model': conf})
    OmegaConf.set_struct(conf, True)
    return conf


def _mock_model_with_children_config(
    child1_model_path: Optional[str] = None,
    child2_model_path: Optional[str] = None,
    child2_model_cfg: Optional[DictConfig] = None,
) -> DictConfig:
    """
    Child 1 always constructed from .nemo model checkpoint (optional)
    Child 2 can be constructed directly from subconfig (optional) or from .nemo model checkpoint (optional)
    """
    conf = {
        'temp_file': None,
        'target': classpath(MockModelWithChildren),
        'child1_model': None,
        'child1_model_path': child1_model_path,
        'child2_model': child2_model_cfg,
        'child2_model_path': child2_model_path,
        'stub_number': 1,
    }
    conf = OmegaConf.create({'model': conf})
    OmegaConf.set_struct(conf, True)
    return conf


def _mock_model_with_child_encdecctcbpe_config(pretrained_model_name: str) -> DictConfig:
    conf = {'temp_file': None, 'ctc_model_pretrained': pretrained_model_name, 'stub_number': 1}
    conf = OmegaConf.create({'model': conf})
    OmegaConf.set_struct(conf, True)
    return conf


def _mock_model_with_child_custom_config_path_config():
    conf = {
        'temp_file': None,
        'child1_model_config': _mock_model_config().model,
        'target': classpath(MockModelWithChildCustomConfigPath),
        'stub_number': 1,
    }
    conf = OmegaConf.create({'model': conf})
    OmegaConf.set_struct(conf, True)
    return conf


def _mock_model_incorrect_with_nemo_artifact_config(child_model_path: str):
    conf = {'temp_file': None, 'child_model_path': child_model_path, 'stub_number': 1}
    conf = OmegaConf.create({'model': conf})
    OmegaConf.set_struct(conf, True)
    return conf


class TestSaveRestore:
    def __test_restore_elsewhere(
        self,
        model: ModelPT,
        attr_for_eq_check: Set[str] = None,
        override_config_path: Optional[Union[str, DictConfig]] = None,
        map_location: Optional[Union[torch.device, str]] = None,
        strict: bool = False,
        return_config: bool = False,
    ):
        """Test's logic:
        1. Save model into temporary folder (save_folder)
        2. Copy .nemo file from save_folder to restore_folder
        3. Delete save_folder
        4. Attempt to restore from .nemo file in restore_folder and compare to original instance
        """
        # Create a new temporary directory
        with tempfile.TemporaryDirectory() as restore_folder:
            with tempfile.TemporaryDirectory() as save_folder:
                save_folder_path = save_folder
                # Where model will be saved
                model_save_path = os.path.join(save_folder, f"{model.__class__.__name__}.nemo")
                model.save_to(save_path=model_save_path)
                # Where model will be restored from
                model_restore_path = os.path.join(restore_folder, f"{model.__class__.__name__}.nemo")
                shutil.copy(model_save_path, model_restore_path)
            # at this point save_folder should not exist
            assert save_folder_path is not None and not os.path.exists(save_folder_path)
            assert not os.path.exists(model_save_path)
            assert os.path.exists(model_restore_path)
            # attempt to restore
            model_copy = model.__class__.restore_from(
                restore_path=model_restore_path,
                map_location=map_location,
                strict=strict,
                return_config=return_config,
                override_config_path=override_config_path,
            )

            if return_config:
                return model_copy

            assert model.num_weights == model_copy.num_weights
            if attr_for_eq_check is not None and len(attr_for_eq_check) > 0:
                for attr in attr_for_eq_check:
                    assert getattr2(model, attr) == getattr2(model_copy, attr)

            return model_copy

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_EncDecCTCModel(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        qn = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
        self.__test_restore_elsewhere(model=qn, attr_for_eq_check=set(["decoder._feat_in", "decoder._num_classes"]))

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_EncDecCTCModelBPE(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        cn = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_citrinet_256")
        self.__test_restore_elsewhere(model=cn, attr_for_eq_check=set(["decoder._feat_in", "decoder._num_classes"]))

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_EncDecCTCModelBPE_v2(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        cn = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
        self.__test_restore_elsewhere(model=cn, attr_for_eq_check=set(["decoder._feat_in", "decoder._num_classes"]))

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_EncDecCTCModelBPE_v3(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        cn = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_xsmall_ls")
        self.__test_restore_elsewhere(model=cn, attr_for_eq_check=set(["decoder._feat_in", "decoder._num_classes"]))

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_EncDecCTCModelBPE_HF(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        # Specifically use ModelPT instead of EncDecCTCModelBPE in order to test target class resolution.
        cn = ModelPT.from_pretrained(model_name="nvidia/stt_en_citrinet_256_ls")
        self.__test_restore_elsewhere(model=cn, attr_for_eq_check=set(["decoder._feat_in", "decoder._num_classes"]))

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_PunctuationCapitalization(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        pn = PunctuationCapitalizationModel.from_pretrained(model_name='punctuation_en_distilbert')
        self.__test_restore_elsewhere(
            model=pn, attr_for_eq_check=set(["punct_classifier.log_softmax", "punct_classifier.log_softmax"])
        )

    @pytest.mark.unit
    def test_mock_save_to_restore_from(self):
        with tempfile.NamedTemporaryFile('w') as empty_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')

            assert model.temp_file == empty_file.name

            # Save test
            model_copy = self.__test_restore_elsewhere(model, map_location='cpu')

        # Restore test
        diff = model.w.weight - model_copy.w.weight
        # because of caching - cache gets prepended
        assert os.path.basename(model_copy.temp_file).endswith(os.path.basename(model.temp_file))
        assert diff.mean() <= 1e-9
        # assert os.path.basename(model.temp_file) == model_copy.temp_file
        assert model_copy.temp_data == ["*****\n"]

    @pytest.mark.unit
    def test_mock_restore_from_config_only(self):
        with tempfile.NamedTemporaryFile('w') as empty_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = os.path.abspath(empty_file.name)

            # Inject arbitrary config arguments (after creating model)
            with open_dict(cfg.model):
                cfg.model.xyz = "abc"

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')

            assert model.temp_file == empty_file.name
            model_copy = self.__test_restore_elsewhere(model, map_location='cpu', return_config=False)
            # because of caching - cache gets prepended
            assert os.path.basename(model_copy.temp_file).endswith(os.path.basename(model.temp_file))
            # assert filecmp.cmp(model.temp_file, model_copy._cfg.temp_file)
            assert model.cfg.xyz == model_copy.cfg.xyz

    @pytest.mark.unit
    def test_mock_restore_from_config_override_with_OmegaConf(self):
        with tempfile.NamedTemporaryFile('w') as empty_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')

            assert model.temp_file == empty_file.name

            # Inject arbitrary config arguments (after creating model)
            with open_dict(cfg.model):
                cfg.model.xyz = "abc"

            # Save test (with overriden config as OmegaConf object)
            model_copy = self.__test_restore_elsewhere(model, map_location='cpu', override_config_path=cfg)

        # Restore test
        diff = model.w.weight - model_copy.w.weight
        assert diff.mean() <= 1e-9
        assert model_copy.temp_data == ["*****\n"]

        # Test that new config has arbitrary content
        assert model_copy.cfg.xyz == "abc"

    @pytest.mark.unit
    def test_mock_restore_from_config_override_with_yaml(self):
        with tempfile.NamedTemporaryFile('w') as empty_file, tempfile.NamedTemporaryFile('w') as config_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')

            assert model.temp_file == empty_file.name

            # Inject arbitrary config arguments (after creating model)
            with open_dict(cfg.model):
                cfg.model.xyz = "abc"

            # Write new config into file
            OmegaConf.save(cfg, config_file)

            # Save test (with overriden config as OmegaConf object)
            model_copy = self.__test_restore_elsewhere(
                model, map_location='cpu', override_config_path=config_file.name
            )

            # Restore test
            diff = model.w.weight - model_copy.w.weight
            assert diff.mean() <= 1e-9
            assert filecmp.cmp(model.temp_file, model_copy.temp_file)
            assert model_copy.temp_data == ["*****\n"]

            # Test that new config has arbitrary content
            assert model_copy.cfg.xyz == "abc"

    @pytest.mark.unit
    def test_mock_save_to_restore_from_with_target_class(self):
        with tempfile.NamedTemporaryFile('w') as empty_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')  # type: MockModel

            assert model.temp_file == empty_file.name

            # Save file using MockModel
            with tempfile.TemporaryDirectory() as save_folder:
                save_path = os.path.join(save_folder, "temp.nemo")
                model.save_to(save_path)

                # Restore test (using ModelPT as restorer)
                # This forces the target class = MockModel to be used as resolver
                model_copy = ModelPT.restore_from(save_path, map_location='cpu')
            # because of caching - cache gets prepended
            assert os.path.basename(model_copy.temp_file).endswith(os.path.basename(model.temp_file))
            # assert filecmp.cmp(model.temp_file, model_copy.temp_file)
        # Restore test
        diff = model.w.weight - model_copy.w.weight
        assert diff.mean() <= 1e-9
        assert isinstance(model_copy, MockModel)
        assert model_copy.temp_data == ["*****\n"]

    @pytest.mark.unit
    def test_mock_save_to_restore_from_multiple_models(self):
        with tempfile.NamedTemporaryFile('w') as empty_file, tempfile.NamedTemporaryFile('w') as empty_file2:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()
            empty_file2.writelines(["+++++\n"])
            empty_file2.flush()

            # Update config + create ,pde;s
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name
            cfg2 = _mock_model_config()
            cfg2.model.temp_file = empty_file2.name

            # Create models
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')
            model2 = MockModel(cfg=cfg2.model, trainer=None)
            model2 = model2.to('cpu')

            assert model.temp_file == empty_file.name
            assert model2.temp_file == empty_file2.name

            # Save test
            model_copy = self.__test_restore_elsewhere(model, map_location='cpu')
            model2_copy = self.__test_restore_elsewhere(model2, map_location='cpu')

        # Restore test
        assert model_copy.temp_data == ["*****\n"]
        assert model2_copy.temp_data == ["+++++\n"]

    @pytest.mark.unit
    def test_mock_save_to_restore_from_multiple_models_inverted_order(self):
        with tempfile.NamedTemporaryFile('w') as empty_file, tempfile.NamedTemporaryFile('w') as empty_file2:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()
            empty_file2.writelines(["+++++\n"])
            empty_file2.flush()

            # Update config + create ,pde;s
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name
            cfg2 = _mock_model_config()
            cfg2.model.temp_file = empty_file2.name

            # Create models
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')
            model2 = MockModel(cfg=cfg2.model, trainer=None)
            model2 = model2.to('cpu')

            assert model.temp_file == empty_file.name
            assert model2.temp_file == empty_file2.name

            # Save test (inverted order)
            model2_copy = self.__test_restore_elsewhere(model2, map_location='cpu')
            model_copy = self.__test_restore_elsewhere(model, map_location='cpu')

        # Restore test
        assert model_copy.temp_data == ["*****\n"]
        assert model2_copy.temp_data == ["+++++\n"]

    @pytest.mark.unit
    def test_mock_save_to_restore_chained(self):
        with tempfile.NamedTemporaryFile('w') as empty_file, tempfile.NamedTemporaryFile('w') as empty_file2:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config + create ,pde;s
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create models
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to('cpu')

            assert model.temp_file == empty_file.name

            def save_copy(model, save_folder, restore_folder):
                # Where model will be saved
                model_save_path = os.path.join(save_folder, f"{model.__class__.__name__}.nemo")
                model.save_to(save_path=model_save_path)
                # Where model will be restored from
                model_restore_path = os.path.join(restore_folder, f"{model.__class__.__name__}.nemo")
                shutil.copy(model_save_path, model_restore_path)
                return model_restore_path

            # Save test
            with tempfile.TemporaryDirectory() as level4:
                with tempfile.TemporaryDirectory() as level3:
                    with tempfile.TemporaryDirectory() as level2:
                        with tempfile.TemporaryDirectory() as level1:
                            path = save_copy(model, level1, level2)
                        model_copy2 = model.__class__.restore_from(path)
                        path = save_copy(model_copy2, level2, level3)
                    model_copy3 = model.__class__.restore_from(path)
                    path = save_copy(model_copy3, level3, level4)
                model_copy = model.__class__.restore_from(path)

        # Restore test
        assert model_copy.temp_data == ["*****\n"]

        # AppState test
        appstate = AppState()
        metadata = appstate.get_model_metadata_from_guid(model_copy.model_guid)
        assert metadata.guid != model.model_guid
        assert metadata.restoration_path == path

    @pytest.mark.unit
    def test_mock_save_to_multiple_times(self):
        with tempfile.NamedTemporaryFile('w') as empty_file, tempfile.TemporaryDirectory() as tmpdir:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)  # type: MockModel
            model = model.to('cpu')

            assert model.temp_file == empty_file.name

            # Save test
            model.save_to(os.path.join(tmpdir, 'save_0.nemo'))
            model.save_to(os.path.join(tmpdir, 'save_1.nemo'))
            model.save_to(os.path.join(tmpdir, 'save_2.nemo'))

    @pytest.mark.unit
    def test_multiple_model_save_restore_connector(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".nemo", "_XYZ.nemo")
                super(MySaveRestoreConnector, self).save_to(model, save_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()
            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(os.path.join(tmpdir, 'save_custom.nemo'))

            assert os.path.exists(os.path.join(tmpdir, 'save_custom_XYZ.nemo'))
            assert isinstance(model._save_restore_connector, save_restore_connector.SaveRestoreConnector)
            assert isinstance(model_with_custom_connector._save_restore_connector, MySaveRestoreConnector)

            assert type(MockModel._save_restore_connector) == save_restore_connector.SaveRestoreConnector

    @pytest.mark.unit
    def test_restore_from_save_restore_connector(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".nemo", "_XYZ.nemo")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()

            # Create model
            save_path = os.path.join(tmpdir, 'save_custom.nemo')
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(save_path)

            assert os.path.exists(os.path.join(tmpdir, 'save_custom_XYZ.nemo'))

            restored_model = MockModelV2.restore_from(
                save_path.replace(".nemo", "_XYZ.nemo"), save_restore_connector=MySaveRestoreConnector()
            )
            assert type(restored_model) == MockModelV2
            assert type(restored_model._save_restore_connector) == MySaveRestoreConnector

    @pytest.mark.unit
    def test_restore_from_save_restore_connector_return_config(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".nemo", "_XYZ.nemo")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()

            # Create model
            save_path = os.path.join(tmpdir, 'save_custom.nemo')
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(save_path)

            assert os.path.exists(os.path.join(tmpdir, 'save_custom_XYZ.nemo'))

            restored_model_cfg = MockModelV2.restore_from(
                save_path.replace(".nemo", "_XYZ.nemo"),
                save_restore_connector=MySaveRestoreConnector(),
                return_config=True,
            )
            assert isinstance(restored_model_cfg, DictConfig)
            assert model_with_custom_connector.cfg == restored_model_cfg

    @pytest.mark.unit
    def test_restore_from_save_restore_connector_return_config_partial_tar_extraction(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".nemo", "_XYZ.nemo")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()

            # Create model
            save_path = os.path.join(tmpdir, 'save_custom.nemo')
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(save_path)

            true_save_path = os.path.join(tmpdir, 'save_custom_XYZ.nemo')
            assert os.path.exists(true_save_path)

            my_connector = MySaveRestoreConnector()

            with tempfile.TemporaryDirectory() as config_tmpdir:
                config_members = my_connector._filtered_tar_info(
                    true_save_path, filter_fn=lambda name: '.yaml' in name
                )
                my_connector._unpack_nemo_file(true_save_path, out_folder=config_tmpdir, members=config_members)
                current_files = list(os.listdir(config_tmpdir))

                assert len(current_files) == 1  # only config file should have been extracted, no pytorch params
                config_filepath = current_files[0]
                assert config_filepath.endswith(".yaml")

    @pytest.mark.unit
    def test_restore_from_save_restore_connector_unpacked_file(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def __init__(self):
                super().__init__()
                self.pack_nemo_file = False

            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".nemo", "_XYZ.nemo")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()

            # Create model
            save_path = os.path.join(tmpdir, 'temp_model')
            os.makedirs(save_path, exist_ok=True)
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(save_path + "/abc.nemo")

            assert os.path.isdir(save_path)
            assert len(os.listdir(save_path)) == 2  # config and pytorch params

            restore_connector = MySaveRestoreConnector()
            restore_connector.model_extracted_dir = save_path
            restored_model = MockModelV2.restore_from(
                save_path.replace(".nemo", "_XYZ.nemo"), save_restore_connector=restore_connector
            )
            assert type(restored_model) == MockModelV2
            assert type(restored_model._save_restore_connector) == MySaveRestoreConnector

    @pytest.mark.unit
    def test_mock_model_model_collision(self):
        # The usual pipeline is working just fine.
        cfg = _mock_model_config()
        model = MockModel(cfg=cfg.model, trainer=None)  # type: MockModel
        model = model.to('cpu')

        # Let's create a custom config with a 'model.model' node.
        cfg = _mock_model_config()
        OmegaConf.set_struct(cfg, False)
        cfg.model.model = 'aaa'
        OmegaConf.set_struct(cfg, True)

        # Failing due to collision.
        with pytest.raises(ValueError, match="Creating model config node is forbidden"):
            model = MockModel(cfg=cfg.model, trainer=None)  # type: MockModel
            model = model.to('cpu')

    @pytest.mark.unit
    @pytest.mark.parametrize("change_child_number", [False, True])
    @pytest.mark.parametrize("child2_model_from_path", [False, True])
    def test_mock_model_nested(self, change_child_number: bool, child2_model_from_path: bool):
        """
        Test model with 2 children
        Model and each child can be saved/restored separately
        Model is constructed using saved child models (.nemo checkpoints)

        Args:
            change_child_number: if change_child_number is True, child model changes its config
                without notifying parent model, and saved parent model should handle this correctly.
            child2_model_from_path: if child2_model_from_path is True, child2 model is restored from .nemo checkpoint,
                otherwise constructed directly from config. Child1 model always loaded from checkpoint.
        """
        # children - models without sub-children
        cfg_child1 = _mock_model_config()
        cfg_child2 = _mock_model_with_children_config()  # no children

        # Create models
        child1 = MockModel(cfg=cfg_child1.model, trainer=None)
        child1 = child1.to('cpu')
        with tempfile.TemporaryDirectory() as tmpdir_parent:
            parent_path = os.path.join(tmpdir_parent, "parent.nemo")
            with tempfile.TemporaryDirectory() as tmpdir_child:
                # save children
                child1_path = os.path.join(tmpdir_child, 'child1.nemo')
                child1.save_to(child1_path)
                if child2_model_from_path:
                    child2 = MockModelWithChildren(cfg=cfg_child2.model, trainer=None)
                    child2 = child2.to('cpu')
                    child2_path = os.path.join(tmpdir_child, 'child2.nemo')
                    child2.save_to(child2_path)

                    # create model with children using saved .nemo model checkpoints
                    cfg_parent = _mock_model_with_children_config(
                        child1_model_path=child1_path, child2_model_path=child2_path
                    )
                else:
                    # child 2 model will be directly constructed from subconfig
                    cfg_parent = _mock_model_with_children_config(
                        child1_model_path=child1_path, child2_model_path=None, child2_model_cfg=cfg_child2.get("model")
                    )

                parent = MockModelWithChildren(cfg_parent.model)
                if change_child_number:
                    parent.child2_model.change_stub_number(10)
                parent.save_to(parent_path)
            # restore, separate children checkpoints are not available here (tmpdir_child destroyed)
            parent = ModelPT.restore_from(parent_path)
            # check model is transparent, child models can be accessed and can be saved/restored separately
            _ = self.__test_restore_elsewhere(parent.child1_model, map_location='cpu')
            child2 = self.__test_restore_elsewhere(parent.child2_model, map_location='cpu')
            if change_child_number:
                assert child2.cfg.stub_number == 10

            # check model itself can be saved/restored
            parent = self.__test_restore_elsewhere(parent, map_location='cpu')
            if change_child_number:
                assert parent.child2_model.cfg.stub_number == 10

    @pytest.mark.unit
    @pytest.mark.parametrize("change_child_resource", [False, True])
    @pytest.mark.parametrize("child2_model_from_path", [False, True])
    def test_mock_model_nested_with_resources(self, change_child_resource: bool, child2_model_from_path: bool):
        """
        Test nested model with 2 children: model and each child can be saved/restored separately
        child models and parent model itself contain resources

        Args:
            change_child_resource: if change_child_resource is True,
                child model resources are changed after instantiation parent model.
            child2_model_from_path: if child2_model_from_path is True, child2 model is restored from .nemo checkpoint,
                otherwise constructed directly from config. Child1 model always loaded from checkpoint.
        """
        with (
            tempfile.NamedTemporaryFile('w') as file_child1,
            tempfile.NamedTemporaryFile('w') as file_child2,
            tempfile.NamedTemporaryFile('w') as file_child2_other,
            tempfile.NamedTemporaryFile('w') as file_parent,
        ):
            # write text data, use these files as resources
            parent_data = ["*****\n"]
            child1_data = ["+++++\n"]
            child2_data = ["-----\n"]
            child2_data_other = [".....\n"]
            file_parent.writelines(parent_data)
            file_parent.flush()
            file_child1.writelines(child1_data)
            file_child1.flush()
            file_child2.writelines(child2_data)
            file_child2.flush()
            file_child2_other.writelines(child2_data_other)
            file_child2_other.flush()

            # construct child models with resources
            # create configs
            cfg_child1 = _mock_model_config()
            cfg_child1.model.temp_file = file_child1.name
            cfg_child2 = _mock_model_with_children_config()  # no sub-children
            cfg_child2.model.temp_file = file_child2.name
            # create child models
            child1 = MockModel(cfg=cfg_child1.model, trainer=None)
            child1 = child1.to('cpu')

            with tempfile.TemporaryDirectory() as tmpdir_parent:
                parent_path = os.path.join(tmpdir_parent, "parent.nemo")
                with tempfile.TemporaryDirectory() as tmpdir_child:
                    # save children
                    child1_path = os.path.join(tmpdir_child, 'child1.nemo')
                    child1.save_to(child1_path)
                    if child2_model_from_path:
                        child2 = MockModelWithChildren(cfg=cfg_child2.model, trainer=None)
                        child2 = child2.to('cpu')
                        child2_path = os.path.join(tmpdir_child, 'child2.nemo')
                        child2.save_to(child2_path)

                        # create model with children using saved .nemo model checkpoints
                        cfg_parent = _mock_model_with_children_config(
                            child1_model_path=child1_path, child2_model_path=child2_path
                        )
                    else:
                        # child 2 model will be directly constructed from subconfig
                        cfg_parent = _mock_model_with_children_config(
                            child1_model_path=child1_path,
                            child2_model_path=None,
                            child2_model_cfg=cfg_child2.get("model"),
                        )

                    cfg_parent.model.temp_file = file_parent.name  # add resource
                    parent = MockModelWithChildren(cfg_parent.model)
                    if change_child_resource:
                        parent.child2_model.setup_data_from_file(file_child2_other.name)
                    parent.save_to(parent_path)

                # restore, separate children checkpoints are not available here (tmpdir_child destroyed)
                parent = ModelPT.restore_from(parent_path)
                # check model is transparent, child models can be accessed and can be saved/restored separately
                child1 = self.__test_restore_elsewhere(parent.child1_model, map_location='cpu')
                child2 = self.__test_restore_elsewhere(parent.child2_model, map_location='cpu')
                # test parent save/restore
                parent = self.__test_restore_elsewhere(parent, map_location='cpu')

                # test resources
                # check separately restored child models
                assert child1.temp_data == child1_data
                if change_child_resource:
                    assert child2.temp_data == child2_data_other
                else:
                    assert child2.temp_data == child2_data
                # test parent model + child models
                assert parent.temp_data == parent_data
                assert parent.child1_model.temp_data == child1_data
                if change_child_resource:
                    assert parent.child2_model.temp_data == child2_data_other
                else:
                    assert parent.child2_model.temp_data == child2_data

    @pytest.mark.unit
    def test_mock_model_nested_with_resources_multiple_passes(self):
        """
        Test nested model with 2 children: multiple save-restore passes
        child models and parent model itself contain resources
        """
        with (
            tempfile.NamedTemporaryFile('w') as file_child1,
            tempfile.NamedTemporaryFile('w') as file_child2,
            tempfile.NamedTemporaryFile('w') as file_child2_other,
            tempfile.NamedTemporaryFile('w') as file_parent,
        ):
            # write text data, use these files as resources
            parent_data = ["*****\n"]
            child1_data = ["+++++\n"]
            child2_data = ["-----\n"]
            child2_data_other = [".....\n"]
            file_parent.writelines(parent_data)
            file_parent.flush()
            file_child1.writelines(child1_data)
            file_child1.flush()
            file_child2.writelines(child2_data)
            file_child2.flush()
            file_child2_other.writelines(child2_data_other)
            file_child2_other.flush()

            # construct child models with resources
            # create configs
            cfg_child1 = _mock_model_config()
            cfg_child1.model.temp_file = file_child1.name
            cfg_child2 = _mock_model_with_children_config()  # no sub-children
            cfg_child2.model.temp_file = file_child2.name
            # create child models
            child1 = MockModel(cfg=cfg_child1.model, trainer=None)
            child1 = child1.to('cpu')
            child2 = MockModelWithChildren(cfg=cfg_child2.model, trainer=None)
            child2 = child2.to('cpu')

            with (
                tempfile.TemporaryDirectory() as tmpdir_parent1,
                tempfile.TemporaryDirectory() as tmpdir_parent2,
                tempfile.TemporaryDirectory() as tmpdir_parent3,
                tempfile.TemporaryDirectory() as tmpdir_parent4,
            ):
                parent_path1 = os.path.join(tmpdir_parent1, "parent.nemo")
                parent_path2 = os.path.join(tmpdir_parent2, "parent.nemo")
                with tempfile.TemporaryDirectory() as tmpdir_child:
                    # save children
                    child1_path = os.path.join(tmpdir_child, 'child1.nemo')
                    child1.save_to(child1_path)
                    child2_path = os.path.join(tmpdir_child, 'child2.nemo')
                    child2.save_to(child2_path)

                    # create model with children using saved "nemo" checkpoints
                    cfg_parent = _mock_model_with_children_config(
                        child1_model_path=child1_path, child2_model_path=child2_path
                    )
                    cfg_parent.model.temp_file = file_parent.name  # add resource
                    parent = MockModelWithChildren(cfg_parent.model)

                    # save-restore first pass
                    # save to different locations
                    parent.save_to(parent_path1)
                    parent.save_to(parent_path2)

                # restore, separate children checkpoints are not available here (tmpdir_child destroyed)
                parent1 = ModelPT.restore_from(parent_path1)
                parent2 = ModelPT.restore_from(parent_path2)

                # check resources
                for parent in (parent1, parent2):
                    assert parent.temp_data == parent_data
                    assert parent.child1_model.temp_data == child1_data
                    assert parent.child2_model.temp_data == child2_data

                del parent2  # use parent1 for second pass

                # save-restore second pass
                parent_path3 = os.path.join(tmpdir_parent3, "parent.nemo")
                parent_path4 = os.path.join(tmpdir_parent4, "parent.nemo")
                parent1.save_to(parent_path3)
                parent1.save_to(parent_path4)

                parent3 = ModelPT.restore_from(parent_path3)
                parent4 = ModelPT.restore_from(parent_path4)

                # check resources
                for parent in (parent3, parent4):
                    assert parent.temp_data == parent_data
                    assert parent.child1_model.temp_data == child1_data
                    assert parent.child2_model.temp_data == child2_data

    @pytest.mark.unit
    def test_mock_model_nested_double_with_resources(self):
        """
        test nested model: parent -> child_with_child -> child; model and each child can be saved/restored separately
        all models can contain resources
        """
        with (
            tempfile.NamedTemporaryFile('w') as file_child,
            tempfile.NamedTemporaryFile('w') as file_child_with_child,
            tempfile.NamedTemporaryFile('w') as file_parent,
        ):
            # write text data, use these files as resources
            parent_data = ["*****\n"]
            child_with_child_data = ["+++++\n"]
            child_data = ["-----\n"]
            file_parent.writelines(parent_data)
            file_parent.flush()
            file_child_with_child.writelines(child_with_child_data)
            file_child_with_child.flush()
            file_child.writelines(child_data)
            file_child.flush()

            # construct child model (leaf) with resource
            cfg_child = _mock_model_config()
            cfg_child.model.temp_file = file_child.name
            child = MockModel(cfg=cfg_child.model, trainer=None)
            child = child.to('cpu')

            with tempfile.TemporaryDirectory() as tmpdir_parent:
                parent_path = os.path.join(tmpdir_parent, "parent.nemo")
                with tempfile.TemporaryDirectory() as tmpdir_child_with_child:
                    child_with_child_path = os.path.join(tmpdir_child_with_child, 'child_with_child.nemo')
                    with tempfile.TemporaryDirectory() as tmpdir_child:
                        # save child
                        child_path = os.path.join(tmpdir_child, 'child.nemo')
                        child.save_to(child_path)

                        # create child model with child
                        cfg_child_with_child = _mock_model_with_children_config(
                            child1_model_path=None, child2_model_path=child_path
                        )
                        cfg_child_with_child.model.temp_file = file_child_with_child.name
                        child_with_child = MockModelWithChildren(cfg_child_with_child.model)
                        child_with_child.save_to(child_with_child_path)
                    # create parent model with child-with-child, leaf checkpoint is not available here
                    cfg_parent = _mock_model_with_children_config(
                        child1_model_path=None, child2_model_path=child_with_child_path
                    )
                    cfg_parent.model.temp_file = file_parent.name
                    parent = MockModelWithChildren(cfg_parent.model)
                    parent.save_to(parent_path)

                # restore, separate children checkpoints are not available here
                # tmpdir_child, tmpdir_child_with_child are destroyed
                parent = ModelPT.restore_from(parent_path)
                # model is transparent, children and model itself can be saved/restored
                child = self.__test_restore_elsewhere(parent.child2_model.child2_model, map_location='cpu')
                child_with_child = self.__test_restore_elsewhere(parent.child2_model, map_location='cpu')
                parent = self.__test_restore_elsewhere(parent, map_location='cpu')

                # test resources for all restored models
                # leaf model
                assert child.temp_data == child_data
                # child with child
                assert child_with_child.temp_data == child_with_child_data
                assert child_with_child.child2_model.temp_data == child_data
                # parent
                assert parent.temp_data == parent_data
                assert parent.child2_model.temp_data == child_with_child_data
                assert parent.child2_model.child2_model.temp_data == child_data

                # check named_nemo_modules: parent -> child2 -> child2.child2,
                # tuples of (attribute_path, cfg_path, module)
                named_nemo_modules = list(parent.named_nemo_modules())
                etalon_nemo_modules = [
                    ("", "", parent),
                    ("child2_model", "child2_model", parent.child2_model),
                    ("child2_model.child2_model", "child2_model.child2_model", parent.child2_model.child2_model),
                ]
                assert len(named_nemo_modules) == len(etalon_nemo_modules)
                for etalon, actual in zip(etalon_nemo_modules, named_nemo_modules):
                    assert etalon[0] == actual[0]
                    assert etalon[1] == actual[1]
                    assert etalon[2] is actual[2]

    @pytest.mark.unit
    @pytest.mark.with_downloads
    def test_mock_model_nested_child_from_pretrained(self):
        """
        Test nested model with child initialized from pretrained model
        """
        cfg = _mock_model_with_child_encdecctcbpe_config("stt_en_conformer_ctc_small")
        parent = MockModelWithChildEncDecCTCBPE(cfg=cfg.model, trainer=None)
        with tempfile.TemporaryDirectory() as tmpdir_parent:
            parent_path = os.path.join(tmpdir_parent, "parent.nemo")

            # save, then restore
            parent.save_to(parent_path)
            parent = ModelPT.restore_from(parent_path)

            # test child can be saved/restored
            _ = self.__test_restore_elsewhere(parent.ctc_model, map_location='cpu')
            # test parent can be saved/restored
            parent = self.__test_restore_elsewhere(parent, map_location='cpu')
            assert isinstance(parent.ctc_model, EncDecCTCModel)

    @pytest.mark.unit
    def test_mock_model_nested_custom_config_field(self):
        """
        Test nested model with custom config field not equal to attribute name
        Config is stored in `child1_model_config`
        Child model is stored in `child1_model` attribute
        """
        with tempfile.NamedTemporaryFile('w') as file_child1, tempfile.NamedTemporaryFile('w') as file_parent:
            # write text data, use these files as resources
            parent_data = ["*****\n"]
            child1_data = ["+++++\n"]
            file_parent.writelines(parent_data)
            file_parent.flush()
            file_child1.writelines(child1_data)
            file_child1.flush()

            cfg = _mock_model_with_child_custom_config_path_config()
            cfg.model.temp_file = file_parent.name
            cfg.model.child1_model_config.temp_file = file_child1.name

            # construct parent model
            parent = MockModelWithChildCustomConfigPath(cfg=cfg.model, trainer=None)
            with tempfile.TemporaryDirectory() as tmpdir_parent:
                parent_path = os.path.join(tmpdir_parent, "parent.nemo")

                # save, then restore
                parent.save_to(parent_path)
                parent = ModelPT.restore_from(parent_path)
                # test child can be saved/restored
                _ = self.__test_restore_elsewhere(parent.child1_model, map_location='cpu')
                # test parent can be saved/restored
                parent = self.__test_restore_elsewhere(parent, map_location='cpu')

                # check data
                assert parent.temp_data == parent_data
                assert parent.child1_model.temp_data == child1_data

                # check named_nemo_modules: parent -> child, tuples of (attribute_path, cfg_path, module)
                named_nemo_modules = list(parent.named_nemo_modules())
                etalon_nemo_modules = [("", "", parent), ("child1_model", "child1_model_config", parent.child1_model)]
                assert len(named_nemo_modules) == len(etalon_nemo_modules)
                for etalon, actual in zip(etalon_nemo_modules, named_nemo_modules):
                    assert etalon[0] == actual[0]
                    assert etalon[1] == actual[1]
                    assert etalon[2] is actual[2]

    @pytest.mark.unit
    def test_using_nemo_checkpoint_as_artifact_disallowed(self):
        """
        Test that using nemo checkpoint as artifact is disallowed
        """
        cfg_child = _mock_model_config()
        child = MockModel(cfg=cfg_child.model, trainer=None).to("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = os.path.join(tmpdir, "child.nemo")
            child.save_to(child_path)

            cfg_parent = _mock_model_incorrect_with_nemo_artifact_config(child_path)
            with pytest.raises(NeMoBaseException):
                # registering .nemo checkpoint as an artifact is not allowed
                _ = MockModelIncorrectWithNemoArtifact(cfg=cfg_parent.model, trainer=None)

    @pytest.mark.unit
    def test_restore_from_save_restore_connector_extracted_dir(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".nemo", "_XYZ.nemo")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as extracted_tempdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Update config
                cfg = _mock_model_config()

                # Create model
                save_path = os.path.join(tmpdir, 'save_custom.nemo')
                model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
                model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
                model_with_custom_connector.save_to(save_path)

                nemo_filepath = os.path.join(tmpdir, 'save_custom_XYZ.nemo')
                assert os.path.exists(nemo_filepath)

                # extract the contents to this dir apriori
                # simulate by extracting now before calling restore_from
                connector = MySaveRestoreConnector()
                MySaveRestoreConnector._unpack_nemo_file(nemo_filepath, extracted_tempdir)
                assert get_size(extracted_tempdir) > 0

            # delete the old directory and preserve only the new extracted directory (escape scope of old dir)

            # next, set the model's extracted directory path
            connector.model_extracted_dir = extracted_tempdir

            # note, we pass in the "old" nemo_filepath, stored somewhere other than the extracted directory
            # this nemo_filepath is no longer valid, and has been deleted.
            restored_model = MockModelV2.restore_from(nemo_filepath, save_restore_connector=connector)
        assert type(restored_model) == MockModelV2
        assert type(restored_model._save_restore_connector) == MySaveRestoreConnector

        # assert models have correct restoration information and paths
        appstate = AppState()
        original_metadata = appstate.get_model_metadata_from_guid(model_with_custom_connector.model_guid)
        assert original_metadata.restoration_path is None

        restored_metadata = appstate.get_model_metadata_from_guid(restored_model.model_guid)
        assert restored_metadata.restoration_path is not None

        # assert that the restore path was the path of the pre-extracted directory
        # irrespective of whether an old `nemo_filepath` (which doesnt exist anymore) was passed to restore_from.
        assert extracted_tempdir in restored_metadata.restoration_path
        assert extracted_tempdir not in nemo_filepath
        assert not os.path.exists(nemo_filepath)

        # test for parameter equality
        model_with_custom_connector = model_with_custom_connector.to('cpu')
        restored_model = restored_model.to('cpu')

        original_state_dict = model_with_custom_connector.state_dict()
        restored_state_dict = restored_model.state_dict()
        for orig, restored in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[orig] - restored_state_dict[restored]).abs().mean() < 1e-6

    @pytest.mark.unit
    def test_hf_model_filter(self):
        filt = ModelPT.get_hf_model_filter()
        assert isinstance(filt, dict)
        assert filt['library'] == 'nemo'

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_hf_model_info(self):
        filt = ModelPT.get_hf_model_filter()

        # check no override results
        model_infos = ModelPT.search_huggingface_models(model_filter=None)
        model_infos = [next(model_infos) for _ in range(5)]
        assert len(model_infos) > 0

        # check with default override results (should match above)
        default_model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        default_model_infos = [next(default_model_infos) for _ in range(5)]
        assert len(model_infos) == len(default_model_infos)

    @pytest.mark.pleasefixme()
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_hf_model_info_with_card_data(self):
        filt = ModelPT.get_hf_model_filter()

        # check no override results
        model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        model_infos = [next(model_infos) for _ in range(5)]
        assert len(model_infos) > 0

        # check overriden defaults
        filt['cardData'] = True
        model_infos = ModelPT.search_huggingface_models(model_filter=filt)

        for info in model_infos:
            if hasattr(info, 'cardData'):
                assert info.cardData is not None
                break

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_hf_model_info_with_limited_results(self):
        filt = ModelPT.get_hf_model_filter()

        # check no override results
        model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        model_infos = [next(model_infos) for _ in range(6)]
        assert len(model_infos) > 0

        # check overriden defaults
        filt['limit'] = 5
        new_model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        new_model_infos = list(new_model_infos)
        assert len(new_model_infos) <= 5
        assert len(new_model_infos) < len(model_infos)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "filter_method,tar_input",
        [
            (save_restore_connector.SaveRestoreConnector._filtered_recursive_walk, False),
            (save_restore_connector.SaveRestoreConnector._filtered_tar_info, True),
        ],
    )
    def test_filtering_methods(self, filter_method: Callable, tar_input: bool):
        def touch(path):
            with open(path, 'a'):
                os.utime(path, None)

        def filter_even_children(path: str):
            if not path[-1].isdigit():
                return False
            return int(path[-1]) % 2 == 0

        cwd = os.getcwd()
        # Since we os.chdir to a temp directory. Tests can fail if we don't jump back to the CWD.
        # This try:finally block ensures we don't get left in an ephemeral directory
        try:
            with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as nemo_base_dir:
                os.chdir(output_dir)
                os.makedirs('grand/parent', exist_ok=True)
                os.makedirs('grand/aunt', exist_ok=True)
                for i in range(3):
                    touch(f'grand/parent/child_{i}')
                    touch(f'grand/aunt/child_{i}')

                if tar_input:
                    path = f'{nemo_base_dir}/model.nemo'
                    save_restore_connector.SaveRestoreConnector._make_nemo_file_from_folder(
                        filename=path, source_dir=output_dir
                    )
                else:
                    path = '.'

                expected_paths = set(
                    (
                        './grand/aunt/child_0',
                        './grand/aunt/child_2',
                        './grand/parent/child_0',
                        './grand/parent/child_2',
                    )
                )

                observed_paths = filter_method(path, filter_fn=filter_even_children)
                if tar_input:
                    observed_paths = set((p.name for p in observed_paths))
                else:
                    observed_paths = set(observed_paths)

                assert expected_paths == observed_paths
        finally:
            os.chdir(cwd)
