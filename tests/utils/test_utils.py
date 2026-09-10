# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import io
import os
from unittest import mock

import pytest

from nemo import __version__ as NEMO_VERSION
from nemo.utils.data_utils import (
    ais_binary,
    ais_endpoint_to_dir,
    bucket_and_object_from_uri,
    get_datastore_object,
    is_datastore_path,
    open_datastore_object_with_binary,
    resolve_cache_dir,
)


class TestDataUtils:
    @pytest.mark.unit
    def test_resolve_cache_dir(self):
        """Test cache dir path."""
        TEST_NEMO_ENV_CACHE_DIR = 'TEST_NEMO_ENV_CACHE_DIR'
        with mock.patch('nemo.constants.NEMO_ENV_CACHE_DIR', TEST_NEMO_ENV_CACHE_DIR):

            envar_to_resolved_path = {
                '/path/to/cache': '/path/to/cache',
                'relative/path': os.path.join(os.getcwd(), 'relative/path'),
                '': os.path.expanduser(f'~/.cache/torch/NeMo/NeMo_{NEMO_VERSION}'),
            }

            for envar, expected_path in envar_to_resolved_path.items():
                # Set envar
                os.environ[TEST_NEMO_ENV_CACHE_DIR] = envar
                # Check path
                uut_path = resolve_cache_dir().as_posix()
                assert uut_path == expected_path, f'Expected: {expected_path}, got {uut_path}'

    @pytest.mark.unit
    def test_is_datastore_path(self):
        """Test checking for datastore path."""
        # Positive examples
        assert is_datastore_path('ais://positive/example')
        # Negative examples
        assert not is_datastore_path('ais/negative/example')
        assert not is_datastore_path('/negative/example')
        assert not is_datastore_path('negative/example')

    @pytest.mark.unit
    def test_bucket_and_object_from_uri(self):
        """Test getting bucket and object from URI."""
        # Positive examples
        assert bucket_and_object_from_uri('ais://bucket/object') == ('bucket', 'object')
        assert bucket_and_object_from_uri('ais://bucket_2/object/is/here') == ('bucket_2', 'object/is/here')

        # Negative examples: invalid URI
        with pytest.raises(ValueError):
            bucket_and_object_from_uri('/local/file')

        with pytest.raises(ValueError):
            bucket_and_object_from_uri('local/file')

    @pytest.mark.unit
    def test_ais_endpoint_to_dir(self):
        """Test converting an AIS endpoint to dir."""
        assert ais_endpoint_to_dir('http://local:123') == os.path.join('local', '123')
        assert ais_endpoint_to_dir('http://1.2.3.4:567') == os.path.join('1.2.3.4', '567')

        with pytest.raises(ValueError):
            ais_endpoint_to_dir('local:123')

    @pytest.mark.unit
    def test_ais_binary(self):
        """Test cache dir path."""
        with mock.patch('shutil.which', lambda x: '/test/path/ais'):
            assert ais_binary() == '/test/path/ais'

        # Negative example: AIS binary cannot be found
        with mock.patch('shutil.which', lambda x: None), mock.patch('os.path.isfile', lambda x: None):
            ais_binary.cache_clear()
            assert ais_binary() is None

    @pytest.mark.unit
    def test_get_datastore_object_passes_num_retries_to_datastore_open_on_cache_miss(self, tmp_path):
        """Test datastore cache misses preserve caller retries when downloading through the AIS fallback."""
        local_path = tmp_path / 'cache' / 'object.bin'
        opened_with = {}

        def fake_open_datastore_object_with_binary(path, num_retries=5):
            opened_with['path'] = path
            opened_with['num_retries'] = num_retries
            return io.BytesIO(b'payload')

        with (
            mock.patch('nemo.utils.data_utils.LHOTSE_AVAILABLE', False),
            mock.patch(
                'nemo.utils.data_utils.open_datastore_object_with_binary',
                side_effect=fake_open_datastore_object_with_binary,
            ),
            mock.patch('nemo.utils.data_utils.datastore_path_to_local_path', return_value=str(local_path)),
        ):
            resolved_path = get_datastore_object('ais://bucket/object', num_retries=7)

        assert resolved_path == str(local_path)
        assert local_path.read_bytes() == b'payload'
        assert opened_with == {'path': 'ais://bucket/object', 'num_retries': 7}

    @pytest.mark.unit
    def test_open_datastore_object_with_binary_keeps_stream_open_until_consumed(self, tmp_path):
        """Test the AIS fallback keeps its real subprocess stream readable after returning."""
        ais_binary_path = tmp_path / 'ais'
        ais_binary_path.write_text('#!/bin/sh\nprintf payload\n')
        ais_binary_path.chmod(0o755)

        with (
            mock.patch('nemo.utils.data_utils.ais_endpoint', return_value='http://local:123'),
            mock.patch('nemo.utils.data_utils.ais_binary', return_value=str(ais_binary_path)),
        ):
            with open_datastore_object_with_binary('ais://bucket/object', num_retries=1) as stream:
                assert stream.read() == b'payload'

    @pytest.mark.unit
    def test_get_datastore_object_does_not_leave_partial_file_on_download_failure(self, tmp_path):
        """Test a failed datastore download does not create a cache file."""
        local_path = tmp_path / 'cache' / 'object.bin'

        class FailingStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self, *args):
                raise RuntimeError('download failed')

        with (
            mock.patch('nemo.utils.data_utils.open_best', return_value=FailingStream()),
            mock.patch('nemo.utils.data_utils.datastore_path_to_local_path', return_value=str(local_path)),
        ):
            with pytest.raises(RuntimeError, match='download failed'):
                get_datastore_object('ais://bucket/object')

        assert not local_path.exists()
