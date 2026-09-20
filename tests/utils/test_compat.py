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

import pathlib
import pickle
import sys
from unittest.mock import patch

import pytest

from nemo.utils.compat import python313_pathlib_pickle_compat


PATHLIB_LOCAL = "pathlib._local"
PATHLIB_LOCAL_POSIX_PATH_PICKLE = b"cpathlib._local\nPosixPath\n."


def _missing_pathlib_local():
    return ModuleNotFoundError(name=PATHLIB_LOCAL)


def test_python313_pathlib_pickle_compat_aliases_missing_module(monkeypatch):
    monkeypatch.delitem(sys.modules, PATHLIB_LOCAL, raising=False)

    with patch("nemo.utils.compat.importlib.import_module", side_effect=_missing_pathlib_local()):
        with python313_pathlib_pickle_compat():
            assert sys.modules[PATHLIB_LOCAL] is pathlib
            assert pickle.loads(PATHLIB_LOCAL_POSIX_PATH_PICKLE) is pathlib.PosixPath

    assert PATHLIB_LOCAL not in sys.modules


def test_python313_pathlib_pickle_compat_cleans_up_after_error(monkeypatch):
    monkeypatch.delitem(sys.modules, PATHLIB_LOCAL, raising=False)

    with pytest.raises(RuntimeError, match="load failed"):
        with patch("nemo.utils.compat.importlib.import_module", side_effect=_missing_pathlib_local()):
            with python313_pathlib_pickle_compat():
                raise RuntimeError("load failed")

    assert PATHLIB_LOCAL not in sys.modules


def test_python313_pathlib_pickle_compat_keeps_available_module(monkeypatch):
    existing_module = object()
    monkeypatch.setitem(sys.modules, PATHLIB_LOCAL, existing_module)

    with python313_pathlib_pickle_compat():
        assert sys.modules[PATHLIB_LOCAL] is existing_module

    assert sys.modules[PATHLIB_LOCAL] is existing_module
