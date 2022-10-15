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
from typing import Dict, Optional, Set, Union

import pytest
import torch
from huggingface_hub.hf_api import ModelFilter, ModelInfo
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.collections.nlp.models import PunctuationCapitalizationModel, TransformerLMModel
from nemo.core.classes import ModelPT
from nemo.core.connectors import save_restore_connector
from nemo.utils.app_state import AppState


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
            self.temp_file = self.register_artifact('temp_file', self.cfg.temp_file)
            with open(self.temp_file, 'r', encoding='utf-8') as f:
                self.temp_data = f.readlines()
        else:
            self.temp_file = None
            self.temp_data = None

    def forward(self, x):
        y = self.w(x)
        return y, self.cfg.temp_file

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    def list_available_models(cls):
        return []


def _mock_model_config():
    conf = {'temp_file': None, 'target': classpath(MockModel)}
    conf = OmegaConf.create({'model': conf})
    OmegaConf.set_struct(conf, True)
    return conf


class TestSaveRestore:
    def __test_restore_elsewhere(
        self,
        model: ModelPT,
        attr_for_eq_check: Set[str] = None,
        override_config_path: Optional[Union[str, DictConfig]] = None,
        map_location: Optional[torch.device] = None,
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
        assert isinstance(filt, ModelFilter)
        assert filt.library == 'nemo'

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_hf_model_info(self):
        filt = ModelPT.get_hf_model_filter()

        # check no override results
        model_infos = ModelPT.search_huggingface_models(model_filter=None)
        assert len(model_infos) > 0

        # check with default override results (should match above)
        default_model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        assert len(model_infos) == len(default_model_infos)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_hf_model_info_with_card_data(self):
        filt = ModelPT.get_hf_model_filter()

        # check no override results
        model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        assert len(model_infos) > 0
        assert not hasattr(model_infos[0], 'cardData')

        # check overriden defaults
        filt.resolve_card_info = True
        model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        assert len(model_infos) > 0

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
        assert len(model_infos) > 0

        # check overriden defaults
        filt.limit_results = 5
        new_model_infos = ModelPT.search_huggingface_models(model_filter=filt)
        assert len(new_model_infos) <= 5
        assert len(new_model_infos) < len(model_infos)
