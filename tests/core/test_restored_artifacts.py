# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

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
