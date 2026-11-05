# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import importlib.util
import urllib.error
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

ROOT_CONFTEST_PATH = Path(__file__).with_name("conftest.py")
ROOT_CONFTEST_SPEC = importlib.util.spec_from_file_location("root_conftest", ROOT_CONFTEST_PATH)
assert ROOT_CONFTEST_SPEC is not None
assert ROOT_CONFTEST_SPEC.loader is not None
root_conftest = importlib.util.module_from_spec(ROOT_CONFTEST_SPEC)
ROOT_CONFTEST_SPEC.loader.exec_module(root_conftest)


class TestRetryUrlOperation:
    @pytest.mark.unit
    def test_retries_with_exponential_backoff(self):
        result = object()
        operation = Mock(side_effect=[urllib.error.URLError("transient"), OSError("transient"), result])

        with patch.object(root_conftest.time, "sleep") as sleep:
            assert root_conftest._retry_url_operation(operation) is result

        assert operation.call_count == 3
        assert sleep.call_args_list == [call(2), call(4)]

    @pytest.mark.unit
    def test_raises_after_five_attempts(self):
        operation = Mock(side_effect=urllib.error.URLError("unavailable"))

        with patch.object(root_conftest.time, "sleep") as sleep, pytest.raises(urllib.error.URLError):
            root_conftest._retry_url_operation(operation)

        assert operation.call_count == 5
        assert sleep.call_args_list == [call(2), call(4), call(8), call(16)]


NIGHTLY_CONFTEST_PATH = Path(__file__).parent / "e2e_nightly" / "conftest.py"
NIGHTLY_CONFTEST_SPEC = importlib.util.spec_from_file_location("nightly_conftest", NIGHTLY_CONFTEST_PATH)
nightly_conftest = importlib.util.module_from_spec(NIGHTLY_CONFTEST_SPEC)
NIGHTLY_CONFTEST_SPEC.loader.exec_module(nightly_conftest)


def _item_at(path, timeout=None):
    item = Mock()
    item.path = path
    item.get_closest_marker.side_effect = lambda name: (
        pytest.mark.timeout(timeout).mark if name == "timeout" and timeout is not None else None
    )
    return item


class TestNightlyTimeout:
    @pytest.mark.unit
    def test_only_unmarked_nightly_items_get_the_longer_timeout(self):
        nightly_dir = NIGHTLY_CONFTEST_PATH.parent
        nightly = _item_at(nightly_dir / "test_model_support_x.py")
        nightly_marked = _item_at(nightly_dir / "test_model_support_y.py", timeout=60)
        ordinary = _item_at(nightly_dir.parent / "test_data_dir.py")
        ordinary_marked = _item_at(nightly_dir.parent / "test_data_dir.py", timeout=30)

        nightly_conftest.pytest_collection_modifyitems(None, [nightly, nightly_marked, ordinary, ordinary_marked])

        nightly.add_marker.assert_called_once_with(pytest.mark.timeout(1200))
        nightly_marked.add_marker.assert_not_called()
        ordinary.add_marker.assert_not_called()
        ordinary_marked.add_marker.assert_not_called()
