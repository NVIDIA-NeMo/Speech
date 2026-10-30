# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace

import pytest

from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import ArtifactPathType


@pytest.fixture
def artifact_context(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    archive.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('outside data')
    monkeypatch.setattr(AppState(), 'nemo_file_folder', str(archive))
    monkeypatch.setattr(AppState(), 'is_model_being_restored', True)
    return SaveRestoreConnector(), SimpleNamespace(artifacts={}), archive, outside


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['absolute', 'parent', 'sibling', 'symlink'])
@pytest.mark.parametrize('verify', [True, False])
def test_archive_reference_requires_contained_path(artifact_context, kind, verify):
    connector, model, archive, outside = artifact_context
    if kind == 'absolute':
        source = str(outside)
    elif kind == 'parent':
        source = '../outside.txt'
    elif kind == 'sibling':
        sibling = archive.with_name('archive-other')
        sibling.mkdir()
        (sibling / 'data.txt').write_text('sibling data')
        source = '../archive-other/data.txt'
    else:
        (archive / 'link').symlink_to(outside)
        source = 'link'
    with pytest.raises(ValueError, match='artifact'):
        connector.register_artifact(model, 'resource', 'nemo:' + source, verify_src_exists=verify)
    assert model.artifacts == {}


@pytest.mark.unit
@pytest.mark.parametrize('name', ['data.txt', 'nested/data.txt', 'nested/../data.txt'])
def test_archive_reference_resolves_bundled_file(artifact_context, name):
    connector, model, archive, _ = artifact_context
    (archive / 'nested').mkdir()
    (archive / name).write_text('bundled data')
    result = connector.register_artifact(model, 'resource', 'nemo:' + name)
    assert result == os.path.realpath(archive / name)
    assert model.artifacts['resource'].path_type == ArtifactPathType.TAR_PATH


@pytest.mark.unit
@pytest.mark.skipif(os.name == 'nt', reason='Colon is not allowed in Windows filenames')
def test_archive_reference_takes_precedence_over_local_filename(artifact_context, tmp_path, monkeypatch):
    connector, model, archive, _ = artifact_context
    (archive / 'data.txt').write_text('bundled data')
    working = tmp_path / 'working'
    working.mkdir()
    (working / 'nemo:data.txt').write_text('local data')
    monkeypatch.chdir(working)
    assert connector.register_artifact(model, 'resource', 'nemo:data.txt') == str(archive / 'data.txt')
    assert model.artifacts['resource'].path_type == ArtifactPathType.TAR_PATH


@pytest.mark.unit
def test_archive_reference_requires_restore_directory(monkeypatch):
    monkeypatch.setattr(AppState(), 'nemo_file_folder', None)
    with pytest.raises(ValueError, match='artifact'):
        SaveRestoreConnector().register_artifact(SimpleNamespace(artifacts={}), 'resource', 'nemo:data.txt')
