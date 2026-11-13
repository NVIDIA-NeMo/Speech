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
