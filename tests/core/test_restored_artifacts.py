# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from omegaconf import OmegaConf

from nemo.core.classes.modelPT import ModelPT
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import ArtifactPathType
from tests.core.test_save_restore import MockModel


@pytest.fixture(autouse=True)
def restore_state(monkeypatch):
    monkeypatch.setattr(AppState(), 'is_model_being_restored', False)
    monkeypatch.setattr(AppState(), 'nemo_file_folder', None)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['absolute', 'relative', 'symlink'])
@pytest.mark.parametrize('verify', [True, False])
def test_restore_registers_only_bundled_files(tmp_path, monkeypatch, kind, verify):
    archive = tmp_path / 'archive'
    archive.mkdir()
    working = tmp_path / 'working'
    working.mkdir()
    external = working / 'data.txt'
    external.write_text('external data')
    monkeypatch.chdir(working)
    monkeypatch.setattr(AppState(), 'is_model_being_restored', True)
    monkeypatch.setattr(AppState(), 'nemo_file_folder', str(archive))
    source = str(external) if kind == 'absolute' else 'data.txt'
    if kind == 'symlink':
        (archive / 'link.txt').symlink_to(external)
        source = str(archive / 'link.txt')
    model = SimpleNamespace(artifacts={})
    try:
        result = SaveRestoreConnector().register_artifact(model, 'resource', source, verify_src_exists=verify)
    except (ValueError, FileNotFoundError):
        pass
    else:
        assert result is None and not verify
    assert model.artifacts == {}


@pytest.mark.unit
def test_restore_prefers_bundled_legacy_artifact(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    archive.mkdir()
    external = tmp_path / 'data.txt'
    external.write_text('external data')
    bundled = archive / external.name
    bundled.write_text('bundled data')
    monkeypatch.setattr(AppState(), 'is_model_being_restored', True)
    monkeypatch.setattr(AppState(), 'nemo_file_folder', str(archive))
    model = SimpleNamespace(artifacts={})
    assert SaveRestoreConnector().register_artifact(model, 'resource', str(external)) == str(bundled)
    assert model.artifacts['resource'].path_type == ArtifactPathType.TAR_PATH


@pytest.mark.unit
def test_local_model_artifact_registration(tmp_path):
    resource = tmp_path / 'data.txt'
    resource.write_text('local data')
    model = SimpleNamespace(artifacts={})
    assert SaveRestoreConnector().register_artifact(model, 'resource', str(resource)) == str(resource)
    assert model.artifacts['resource'].path_type == ArtifactPathType.LOCAL_PATH


@pytest.mark.unit
def test_restored_archive_does_not_include_external_file(tmp_path):
    external = tmp_path / 'external.txt'
    marker = 'local-fixture-content-5bd650773c'
    external.write_text(marker)
    original = tmp_path / 'original.nemo'
    modified = tmp_path / 'modified.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(original))
    _rewrite_config(original, modified, temp_file=str(external))
    try:
        restored = MockModel.restore_from(str(modified), map_location='cpu')
    except (ValueError, FileNotFoundError):
        return
    saved = tmp_path / 'saved.nemo'
    restored.save_to(str(saved))
    with tarfile.open(saved) as archive:
        contents = [archive.extractfile(member).read() for member in archive if member.isfile()]
    assert not any(marker.encode() in content for content in contents)


@pytest.mark.unit
def test_bundled_artifact_round_trip(tmp_path):
    resource = tmp_path / 'data.txt'
    resource.write_text('bundled data')
    original = tmp_path / 'original.nemo'
    model = MockModel(OmegaConf.create({'temp_file': str(resource)}))
    model.save_to(str(original))
    resource.unlink()
    restored = MockModel.restore_from(str(original), map_location='cpu')
    saved = tmp_path / 'saved.nemo'
    restored.save_to(str(saved))
    copied = MockModel.restore_from(str(saved), map_location='cpu')
    assert copied.temp_data == ['bundled data']


@pytest.mark.unit
@pytest.mark.parametrize('legacy', ['relative', 'absolute', 'parent'])
def test_legacy_bundled_artifact_round_trip(tmp_path, legacy):
    resource = tmp_path / 'data.txt'
    resource.write_text('bundled data')
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    MockModel(OmegaConf.create({'temp_file': str(resource)})).save_to(str(original))
    with tarfile.open(original) as archive:
        config_member = next(member for member in archive if Path(member.name).name == 'model_config.yaml')
        cfg = OmegaConf.create(archive.extractfile(config_member).read().decode())
    name = cfg.temp_file.removeprefix('nemo:')
    source = {'relative': name, 'absolute': '/old/model/' + name, 'parent': '../old/model/' + name}[legacy]
    _rewrite_config(original, modified, temp_file=source)
    restored = MockModel.restore_from(str(modified), map_location='cpu')
    saved = tmp_path / 'saved.nemo'
    restored.save_to(str(saved))
    assert MockModel.restore_from(str(saved), map_location='cpu').temp_data == ['bundled data']


@pytest.mark.unit
def test_runtime_artifact_option_is_scoped_to_the_restore_call(tmp_path):
    external = tmp_path / 'external.txt'
    external.write_text('local data')
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(original))
    _rewrite_config(original, modified, temp_file=str(external))
    connector = SaveRestoreConnector(allow_external_artifacts=True)
    restored = MockModel.restore_from(str(modified), map_location='cpu', save_restore_connector=connector)
    assert restored.temp_data == ['local data']
    with pytest.raises((ValueError, FileNotFoundError)):
        MockModel.restore_from(str(modified), map_location='cpu')
    assert not AppState().is_model_being_restored
    assert AppState().nemo_file_folder is None
    assert MockModel(OmegaConf.create({'temp_file': str(external)})).temp_data == ['local data']


@pytest.mark.unit
def test_model_config_does_not_enable_external_artifact_option(tmp_path):
    external = tmp_path / 'external.txt'
    external.write_text('local data')
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(original))
    _rewrite_config(original, modified, temp_file=str(external), allow_external_artifacts=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        MockModel.restore_from(str(modified), map_location='cpu')


@pytest.mark.unit
@pytest.mark.parametrize('value', ['false', 1, None])
def test_external_artifact_option_requires_bool(value):
    with pytest.raises(TypeError, match='bool'):
        SaveRestoreConnector(allow_external_artifacts=value)


@pytest.mark.unit
def test_legacy_reference_is_normalized_before_saving(tmp_path):
    external = tmp_path / 'resource.txt'
    external.write_text('external data')
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(original))
    _rewrite_config(original, modified, temp_file='old/nemo:' + str(external))
    bundled = b'bundled data'
    with tarfile.open(modified, 'a') as archive:
        member = tarfile.TarInfo(external.name)
        member.size = len(bundled)
        archive.addfile(member, io.BytesIO(bundled))
    restored = MockModel.restore_from(str(modified), map_location='cpu')
    assert restored.artifacts['temp_file'].path == 'nemo:' + external.name
    saved = tmp_path / 'saved.nemo'
    restored.save_to(str(saved))
    assert MockModel.restore_from(str(saved), map_location='cpu').temp_data == ['bundled data']


@pytest.mark.unit
def test_lightning_checkpoint_retains_local_artifacts(tmp_path):
    resource = tmp_path / 'local.txt'
    resource.write_text('local data')
    model = MockModel(OmegaConf.create({'temp_file': str(resource)}))
    path = tmp_path / 'model.ckpt'
    torch.save(
        {
            'state_dict': model.state_dict(),
            'hyper_parameters': {'cfg': model.cfg},
            'pytorch-lightning_version': pl.__version__,
        },
        path,
    )
    restored = MockModel.load_from_checkpoint(str(path), map_location='cpu')
    assert restored.temp_data == ['local data']
    assert restored.artifacts['temp_file'].path_type == ArtifactPathType.LOCAL_PATH
    assert (AppState().is_model_being_restored, AppState().nemo_file_folder) == (False, None)


@pytest.mark.unit
def test_current_restore_option_overrides_a_previous_class_connector(tmp_path, monkeypatch):
    monkeypatch.setattr(EarlyArtifactModel, '_save_restore_connector', SaveRestoreConnector())
    monkeypatch.setattr(ModelPT, '_save_restore_connector', SaveRestoreConnector(), raising=False)
    resource = tmp_path / 'external.txt'
    resource.write_text('local data')
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    EarlyArtifactModel(OmegaConf.create({'temp_file': None, 'early_file': None})).save_to(str(original))
    _rewrite_config(original, modified, early_file=str(resource))
    restored = EarlyArtifactModel.restore_from(
        str(modified), map_location='cpu', save_restore_connector=SaveRestoreConnector(allow_external_artifacts=True)
    )
    assert restored.early_data == 'local data'
    # Generic construction falls back to the abstract base after the artifact is rejected.
    with pytest.raises(TypeError, match='abstract class ModelPT'):
        ModelPT.restore_from(str(modified), map_location='cpu')


@pytest.mark.unit
def test_runtime_option_reaches_a_directly_constructed_child(tmp_path, monkeypatch):
    monkeypatch.setattr(MockModel, '_save_restore_connector', SaveRestoreConnector())
    resource = tmp_path / 'external.txt'
    resource.write_text('local data')
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    ChildArtifactModel(OmegaConf.create({'temp_file': None, 'child_file': None})).save_to(str(original))
    _rewrite_config(original, modified, child_file=str(resource))
    restored = ChildArtifactModel.restore_from(
        str(modified), map_location='cpu', save_restore_connector=SaveRestoreConnector(allow_external_artifacts=True)
    )
    assert restored.child.temp_data == ['local data']


@pytest.mark.unit
@pytest.mark.parametrize('extracted', [False, True])
def test_nested_bundled_reference_round_trip(tmp_path, extracted):
    original, modified = tmp_path / 'original.nemo', tmp_path / 'modified.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(original))
    _rewrite_config(original, modified, temp_file='nested/data.txt')
    with tarfile.open(modified, 'a') as archive:
        for name, contents in [('nested/data.txt', b'bundled data'), ('data.txt', b'other data')]:
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    connector = SaveRestoreConnector()
    if extracted:
        folder = tmp_path / 'extracted'
        folder.mkdir()
        connector._unpack_nemo_file(str(modified), str(folder))
        connector.model_extracted_dir = str(folder)
    restored = MockModel.restore_from(str(modified), map_location='cpu', save_restore_connector=connector)
    saved = tmp_path / 'saved.nemo'
    restored.save_to(str(saved))
    assert MockModel.restore_from(str(saved), map_location='cpu').temp_data == ['bundled data']


@pytest.mark.unit
@pytest.mark.parametrize('failure', ['weights', 'modify'])
def test_restore_failure_recovers_before_local_construction(tmp_path, failure):
    path = tmp_path / 'model.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(path))
    connector = SaveRestoreConnector()
    if failure == 'weights':
        modified = tmp_path / 'modified.nemo'
        _rewrite_weights(path, modified)
        path = modified
    else:

        def reject(*args, **kwargs):
            raise RuntimeError('state transform failed')

        connector.modify_state_dict = reject
    with pytest.raises(RuntimeError):
        MockModel.restore_from(str(path), map_location='cpu', save_restore_connector=connector)
    assert (AppState().is_model_being_restored, AppState().nemo_file_folder) == (False, None)
    resource = tmp_path / 'local.txt'
    resource.write_text('local data')
    assert MockModel(OmegaConf.create({'temp_file': str(resource)})).temp_data == ['local data']


@pytest.mark.unit
@pytest.mark.parametrize('return_config', [False, True])
def test_restore_preserves_an_enclosing_state(tmp_path, monkeypatch, return_config):
    path = tmp_path / 'model.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(path))
    folder = str(tmp_path / 'outer')
    monkeypatch.setattr(AppState(), 'is_model_being_restored', True)
    monkeypatch.setattr(AppState(), 'nemo_file_folder', folder)
    MockModel.restore_from(str(path), map_location='cpu', return_config=return_config)
    assert (AppState().is_model_being_restored, AppState().nemo_file_folder) == (True, folder)


@pytest.mark.unit
def test_nested_restore_recovers_the_parent_state(tmp_path):
    inner = tmp_path / 'inner.nemo'
    MockModel(OmegaConf.create({'temp_file': None})).save_to(str(inner))
    outer = tmp_path / 'outer.nemo'
    NestedRestoreModel(OmegaConf.create({'temp_file': None, 'inner_archive': str(inner)})).save_to(str(outer))
    restored = NestedRestoreModel.restore_from(str(outer), map_location='cpu')
    before, after = restored.nested_restore_state
    assert before == after and before[0]
    assert (AppState().is_model_being_restored, AppState().nemo_file_folder) == (False, None)


class EarlyArtifactModel(MockModel):
    def __init__(self, cfg, trainer=None):
        self.early_data = None
        if cfg.get('early_file'):
            self.early_data = Path(self.register_artifact('early_file', cfg.early_file)).read_text()
        super().__init__(cfg, trainer)


class ChildArtifactModel(MockModel):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer)
        self.child = MockModel(OmegaConf.create({'temp_file': cfg.get('child_file')}))


class NestedRestoreModel(MockModel):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer)
        if cfg.get('inner_archive'):
            state = AppState()
            before = (state.is_model_being_restored, state.nemo_file_folder)
            MockModel.restore_from(cfg.inner_archive, map_location='cpu')
            self.nested_restore_state = (before, (state.is_model_being_restored, state.nemo_file_folder))


def _rewrite_weights(source, destination):
    with tarfile.open(source) as original, tarfile.open(destination, 'w') as updated:
        for member in original:
            if not member.isfile():
                updated.addfile(member)
                continue
            data = original.extractfile(member).read()
            if Path(member.name).name == 'model_weights.ckpt':
                state = torch.load(io.BytesIO(data), weights_only=True)
                del state['w.bias']
                stream = io.BytesIO()
                torch.save(state, stream)
                data = stream.getvalue()
                member.size = len(data)
            updated.addfile(member, io.BytesIO(data))


def _rewrite_config(source, destination, **overrides):
    with tarfile.open(source) as original, tarfile.open(destination, 'w') as updated:
        for member in original:
            if not member.isfile():
                updated.addfile(member)
                continue
            data = original.extractfile(member).read()
            if Path(member.name).name == 'model_config.yaml':
                cfg = OmegaConf.create(data.decode())
                for key, value in overrides.items():
                    cfg[key] = value
                data = OmegaConf.to_yaml(cfg).encode()
                member.size = len(data)
            updated.addfile(member, io.BytesIO(data))
