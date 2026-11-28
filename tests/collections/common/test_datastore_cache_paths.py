# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo.utils import data_utils


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv('AIS_ENDPOINT', 'http://localhost:8080')
    monkeypatch.setenv('NEMO_DATA_STORE_CACHE_DIR', str(tmp_path / 'cache'))
    return Path(data_utils.ais_cache_base()) / 'localhost' / '8080'


@pytest.mark.unit
@pytest.mark.parametrize(
    'uri',
    [
        'ais://bucket/../outside',
        'ais://bucket/object#fragment/../../outside',
        'ais://bucket/object?query=1',
        'ais://bucket/%2e%2e/outside',
        'ais://bucket//outside',
        'ais://../outside',
        'ais://bucket/folder/./object',
    ],
)
def test_datastore_rejects_invalid_cache_path(cache, monkeypatch, uri):
    reader = Mock(return_value=BytesIO(b'data'))
    monkeypatch.setattr(data_utils, 'open_best', reader)
    with pytest.raises(ValueError):
        data_utils.get_datastore_object(uri)
    reader.assert_not_called()
    assert not cache.exists()


@pytest.mark.unit
@pytest.mark.parametrize('leaf_symlink', [False, True])
def test_datastore_rejects_cache_symlink_escape(cache, tmp_path, monkeypatch, leaf_symlink):
    outside = tmp_path / 'outside'
    outside.mkdir()
    target = outside / 'object'
    target.write_bytes(b'original')
    (cache / 'bucket').mkdir(parents=True)
    if leaf_symlink:
        (cache / 'bucket' / 'object').symlink_to(target)
        uri = 'ais://bucket/object'
    else:
        (cache / 'bucket' / 'folder').symlink_to(outside, target_is_directory=True)
        uri = 'ais://bucket/folder/object'
    reader = Mock(return_value=BytesIO(b'replacement'))
    monkeypatch.setattr(data_utils, 'open_best', reader)
    with pytest.raises(ValueError):
        data_utils.get_datastore_object(uri, force=True)
    reader.assert_not_called()
    assert target.read_bytes() == b'original'


@pytest.mark.unit
@pytest.mark.parametrize('scheme', ['ais', 's3', 'gs'])
def test_datastore_cache_preserves_nested_objects(cache, monkeypatch, scheme):
    reader = Mock(side_effect=lambda *args, **kwargs: BytesIO(b'features'))
    monkeypatch.setattr(data_utils, 'open_best', reader)
    uri = f'{scheme}://bucket/folder/object with spaces'
    result = data_utils.get_datastore_object(uri)
    assert Path(result) == cache / 'bucket' / 'folder' / 'object with spaces'
    assert Path(result).read_bytes() == b'features'
    assert data_utils.get_datastore_object(uri) == result
    reader.assert_called_once()


@pytest.mark.unit
def test_manifest_audio_paths_use_cache_validation(cache, tmp_path, monkeypatch):
    import json

    from nemo.collections.asr.data.audio_to_text import cache_datastore_manifests

    manifest_uri = 'ais://bucket/data/manifest.json'
    row = {'audio_filepath': 'object#fragment/../../outside', 'duration': 1.0, 'text': 'example'}
    reader = Mock(side_effect=lambda path, **kwargs: BytesIO((json.dumps(row) + '\n').encode()))
    monkeypatch.setattr(data_utils, 'open_best', reader)
    with pytest.raises(ValueError, match='Unsupported store path'):
        cache_datastore_manifests(manifest_uri, cache_audio=True, num_workers=1)
    reader.assert_called_once_with(manifest_uri, num_retries=5)


@pytest.mark.unit
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('suffix', ['?', '#', '\t', '\r', '\n'])
def test_datastore_rejects_ambiguous_cache_aliases(cache, monkeypatch, suffix, cached):
    path = cache / 'bucket' / 'object'
    if cached:
        path.parent.mkdir(parents=True)
        path.write_bytes(b'original')
    reader = Mock(return_value=BytesIO(b'replacement'))
    monkeypatch.setattr(data_utils, 'open_best', reader)
    with pytest.raises(ValueError):
        data_utils.get_datastore_object(
            f'ais://bucket/obj{suffix}ect' if suffix.isspace() else f'ais://bucket/object{suffix}'
        )
    reader.assert_not_called()
    if cached:
        assert path.read_bytes() == b'original'
    else:
        assert not cache.exists()


@pytest.mark.unit
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_datastore_rejects_symlink_above_endpoint_cache(cache, tmp_path, monkeypatch, depth):
    cache_base = Path(data_utils.ais_cache_base())
    path = [cache_base, cache_base / 'localhost', cache][depth]
    outside = tmp_path / 'outside'
    outside.mkdir()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside, target_is_directory=True)
    reader = Mock(return_value=BytesIO(b'features'))
    monkeypatch.setattr(data_utils, 'open_best', reader)
    with pytest.raises(ValueError):
        data_utils.get_datastore_object('ais://bucket/object')
    reader.assert_not_called()
    assert list(outside.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize('scheme', ['ais', 's3', 'gs', 'https'])
@pytest.mark.parametrize('name', ['object;version=2', 'object%3F', 'object%23', 'a%09b'])
def test_datastore_preserves_literal_object_names(cache, monkeypatch, scheme, name):
    monkeypatch.setattr(data_utils, 'open_best', lambda path, **kwargs: BytesIO(path.encode()))
    plain = data_utils.get_datastore_object(f'{scheme}://bucket/object')
    uri = f'{scheme}://bucket/{name}'
    special = data_utils.get_datastore_object(uri)
    assert special != plain
    assert Path(special).name == name
    assert Path(special).read_bytes() == uri.encode()


@pytest.mark.unit
def test_datastore_allows_symlinks_within_cache(cache, monkeypatch):
    target = cache / 'bucket' / 'target'
    target.mkdir(parents=True)
    (cache / 'bucket' / 'alias').symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(data_utils, 'open_best', lambda *args, **kwargs: BytesIO(b'data'))
    path = data_utils.get_datastore_object('ais://bucket/alias/object')
    assert Path(path).read_bytes() == b'data'
    assert (target / 'object').read_bytes() == b'data'


@pytest.mark.unit
def test_datastore_allows_explicitly_symlinked_cache_root(tmp_path, monkeypatch):
    root = tmp_path / 'storage'
    root.mkdir()
    configured = tmp_path / 'configured-cache'
    configured.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv('NEMO_DATA_STORE_CACHE_DIR', str(configured))
    monkeypatch.setenv('AIS_ENDPOINT', 'http://localhost:8080')
    monkeypatch.setattr(data_utils, 'open_best', lambda *args, **kwargs: BytesIO(b'data'))
    path = data_utils.get_datastore_object('ais://bucket/object')
    assert Path(path).is_relative_to(root)
    assert Path(path).read_bytes() == b'data'
