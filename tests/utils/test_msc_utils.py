# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from unittest import mock

import pytest

from nemo.utils.msc_utils import is_multistorageclient_url


def test_is_multistorageclient_url_with_msc_not_installed():
    with mock.patch('nemo.utils.msc_utils.HAVE_MSC', False):
        assert not is_multistorageclient_url('/tmp/path/to/data.bin')
        assert not is_multistorageclient_url(Path('/tmp/path/to/data.bin'))

        with pytest.raises(ValueError):
            is_multistorageclient_url('msc://profile/path/to/data.bin')


def test_is_multistorageclient_url_with_msc_installed():
    with mock.patch('nemo.utils.msc_utils.HAVE_MSC', True):
        assert is_multistorageclient_url('msc://profile/path/to/data.bin')
        assert not is_multistorageclient_url('/tmp/path/to/data.bin')
