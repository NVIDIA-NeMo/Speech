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
from unittest import mock

import pytest

from nemo import __version__ as NEMO_VERSION
from nemo.utils.data_utils import (
    ais_binary,
    ais_endpoint_to_dir,
    bucket_and_object_from_uri,
    datastore_path_to_webdataset_url,
    is_datastore_path,
    resolve_cache_dir,
)


class TestDataUtils:
    @pytest.mark.unit
    def test_resolve_cache_dir(self):
        """Test cache dir path.
        """
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
        """Test checking for datastore path.
        """
        # Positive examples
        assert is_datastore_path('ais://positive/example')
        # Negative examples
        assert not is_datastore_path('ais/negative/example')
        assert not is_datastore_path('/negative/example')
        assert not is_datastore_path('negative/example')

    @pytest.mark.unit
    def test_bucket_and_object_from_uri(self):
        """Test getting bucket and object from URI.
        """
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
        """Test converting an AIS endpoint to dir.
        """
        assert ais_endpoint_to_dir('http://local:123') == os.path.join('local', '123')
        assert ais_endpoint_to_dir('http://1.2.3.4:567') == os.path.join('1.2.3.4', '567')

        with pytest.raises(ValueError):
            ais_endpoint_to_dir('local:123')

    @pytest.mark.unit
    def test_ais_binary(self):
        """Test cache dir path.
        """
        with mock.patch('shutil.which', lambda x: '/test/path/ais'):
            assert ais_binary() == '/test/path/ais'

        # Negative example: AIS binary cannot be found
        with mock.patch('shutil.which', lambda x: None), mock.patch('os.path.isfile', lambda x: None):
            with pytest.raises(RuntimeError):
                ais_binary()

    @pytest.mark.unit
    def test_datastore_path_to_webdataset_url(self):
        """Test conversion of data store path to an URL for WebDataset.
        """
        assert datastore_path_to_webdataset_url('ais://test/path') == 'pipe:ais get ais://test/path - || true'
