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
"""CPU-only tests for the CP-helper module."""
from nemo.collections.speechlm2.parts.cp_helpers import get_cp_mesh


def test_get_cp_mesh_none():
    assert get_cp_mesh(None) == (None, 1, 0)


class _DummyCpDim:
    """Stand-in for ``device_mesh['cp']`` whose ``.size()`` is 1 (CP inactive)."""

    def size(self):
        return 1


class _DummyDeviceMesh:
    """Minimal ``DeviceMesh``-like object exposing only the bits ``get_cp_mesh`` reads."""

    def __init__(self, cp_size: int = 1, has_cp: bool = True):
        self.mesh_dim_names = ("dp", "cp", "tp") if has_cp else ("dp", "tp")
        self._cp_size = cp_size

    def __getitem__(self, key):
        if key == "cp":
            class _Dim:
                def __init__(self, size):
                    self._size = size

                def size(self):
                    return self._size

            return _Dim(self._cp_size)
        raise KeyError(key)


def test_get_cp_mesh_cp_size_one():
    assert get_cp_mesh(_DummyDeviceMesh(cp_size=1)) == (None, 1, 0)


def test_get_cp_mesh_no_cp_dim():
    assert get_cp_mesh(_DummyDeviceMesh(has_cp=False)) == (None, 1, 0)
