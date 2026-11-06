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

"""Validate optional dependencies and precision options without allocating a GPU."""

from unittest.mock import MagicMock

import pytest

from nemo.utils import trt_utils


@pytest.mark.unit
def test_missing_builder_dependencies(monkeypatch):
    monkeypatch.setattr(trt_utils, "HAVE_TRT", False)
    with pytest.raises(ImportError, match="TensorRT and Polygraphy"):
        trt_utils.build_engine("unused.onnx", "unused.plan", fp16=False)


@pytest.mark.unit
@pytest.mark.parametrize("tf32", [None, False, True])
def test_tf32_option(monkeypatch, tf32):
    monkeypatch.setattr(trt_utils, "HAVE_TRT", True)
    mocks = {}
    for name in ("Profile", "CreateConfig", "network_from_onnx_path", "engine_from_network", "save_engine"):
        mocks[name] = MagicMock()
        monkeypatch.setattr(trt_utils, name, mocks[name], raising=False)
    trt_utils.build_engine("model.onnx", "model.plan", fp16=False, tf32=tf32)
    kwargs = mocks["CreateConfig"].call_args.kwargs
    if tf32 is None:
        assert "tf32" not in kwargs
    else:
        assert kwargs["tf32"] is tf32
    mocks["save_engine"].assert_called_once_with(mocks["engine_from_network"].return_value, path="model.plan")
