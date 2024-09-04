# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.lightning import io


class DummyClass(io.IOMixin):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


class TestIOMixin:
    def test_reinit(self):
        dummy = DummyClass(5, 5)
        copied = io.reinit(dummy)
        assert copied is not dummy
        assert copied.a == dummy.a
        assert copied.b == dummy.b

    def test_init(self):
        outputs = []
        for i in range(1001):
            outputs.append(DummyClass(i, i))

        assert len(outputs) == 1001
