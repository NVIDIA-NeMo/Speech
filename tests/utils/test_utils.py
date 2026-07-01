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
import stat
from unittest import mock

import pytest

from nemo import __version__ as NEMO_VERSION
from nemo.utils.data_utils import (
    ais_binary,
    ais_endpoint_to_dir,
    bucket_and_object_from_uri,
    is_datastore_path,
    open_datastore_object_with_binary,
    resolve_cache_dir,
)


class TestDataUtils:
    @staticmethod
    def _write_fake_ais_binary(path):
        path.write_text(
            """#!/usr/bin/env python3
import os
import sys
import time

if sys.argv[1:] != ['get', 'ais://bucket/object', '-']:
    sys.stderr.write(f'unexpected args: {sys.argv[1:]}')
    sys.exit(2)

mode = os.environ['FAKE_AIS_MODE']
if mode == 'success':
    sys.stdout.buffer.write(b'payload')
    sys.stdout.flush()
    time.sleep(0.2)
elif mode == 'error':
    sys.stderr.write('simulated ais failure')
    sys.exit(1)
else:
    sys.stderr.write(f'unsupported mode: {mode}')
    sys.exit(3)
"""
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

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
    def test_open_datastore_object_with_binary_keeps_process_alive_until_close(self, tmp_path):
        """Test datastore streams keep the AIS subprocess alive until the caller closes the stream."""
        fake_ais = tmp_path / 'fake_ais.py'
        self._write_fake_ais_binary(fake_ais)

        with (
            mock.patch('nemo.utils.data_utils.ais_binary', return_value=str(fake_ais)),
            mock.patch('nemo.utils.data_utils.ais_endpoint', return_value='http://local:123'),
            mock.patch.dict(os.environ, {'FAKE_AIS_MODE': 'success'}),
        ):
            with open_datastore_object_with_binary('ais://bucket/object') as stream:
                assert stream.read(1) == b'p'
                assert not stream.closed
                assert stream._proc.poll() is None

            assert stream.closed
            assert stream._proc.poll() == 0

    @pytest.mark.unit
    def test_open_datastore_object_with_binary_raises_if_ais_returns_no_data(self, tmp_path):
        """Test datastore open retries surface the AIS error if the binary never returns data."""
        fake_ais = tmp_path / 'fake_ais.py'
        self._write_fake_ais_binary(fake_ais)

        with (
            mock.patch('nemo.utils.data_utils.ais_binary', return_value=str(fake_ais)),
            mock.patch('nemo.utils.data_utils.ais_endpoint', return_value='http://local:123'),
            mock.patch.dict(os.environ, {'FAKE_AIS_MODE': 'error'}),
        ):
            with pytest.raises(ValueError, match='simulated ais failure'):
                open_datastore_object_with_binary('ais://bucket/object', num_retries=2)
