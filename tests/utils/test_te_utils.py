# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.utils import te_utils


def test_is_mxfp8tensor_uses_imported_tensor_class(monkeypatch):
    class FakeMXFP8Tensor:
        pass

    monkeypatch.setattr(te_utils, "HAVE_TE_MXFP8TENSOR", True)
    monkeypatch.setattr(te_utils, "MXFP8Tensor", FakeMXFP8Tensor)

    assert te_utils.is_mxfp8tensor(FakeMXFP8Tensor())
    assert not te_utils.is_mxfp8tensor(object())
