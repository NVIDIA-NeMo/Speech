# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import operator

import pytest

from nemo.core.utils.numba_utils import numba_cpu_is_supported, numba_cuda_is_supported


def _stub_check_lib_version(result):
    calls = []

    def stub(libname, checked_version, operator=None):
        calls.append({"libname": libname, "checked_version": checked_version, "operator": operator})
        return result

    return stub, calls


@pytest.mark.unit
def test_cpu_unsupported_when_version_check_fails(monkeypatch):
    stub, _ = _stub_check_lib_version((False, "numba version too old"))
    monkeypatch.setattr("nemo.core.utils.numba_utils.model_utils.check_lib_version", stub)

    assert numba_cpu_is_supported("0.57.0") is False


@pytest.mark.unit
def test_cpu_supported_when_version_check_passes(monkeypatch):
    stub, _ = _stub_check_lib_version((True, "numba >= min_version"))
    monkeypatch.setattr("nemo.core.utils.numba_utils.model_utils.check_lib_version", stub)

    assert numba_cpu_is_supported("0.57.0") is True


@pytest.mark.unit
def test_cpu_unsupported_when_numba_missing(monkeypatch):
    stub, _ = _stub_check_lib_version((None, "Could not import numba, please install it."))
    monkeypatch.setattr("nemo.core.utils.numba_utils.model_utils.check_lib_version", stub)

    assert numba_cpu_is_supported("0.57.0") is False


@pytest.mark.unit
def test_cuda_unsupported_when_version_check_fails(monkeypatch):
    stub, calls = _stub_check_lib_version((False, "too old"))
    monkeypatch.setattr("nemo.core.utils.numba_utils.model_utils.check_lib_version", stub)

    assert numba_cuda_is_supported("0.57.0") is False

    assert len(calls) == 1
    assert calls[0]["libname"] == "numba"
    assert calls[0]["checked_version"] == "0.57.0"
    assert calls[0]["operator"] is operator.ge


@pytest.mark.unit
def test_cuda_unsupported_when_numba_missing(monkeypatch):
    stub, _ = _stub_check_lib_version((None, "Could not import numba, please install it."))
    monkeypatch.setattr("nemo.core.utils.numba_utils.model_utils.check_lib_version", stub)

    assert numba_cuda_is_supported("0.57.0") is False
