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

import pytest

from nemo.collections.common.parts.utils import flatten


class TestListUtils:
    @pytest.mark.unit
    def test_flatten(self):
        """Test flattening an iterable with different values: str, bool, int, float, complex.
        """
        test_cases = []
        test_cases.append({'input': ['aa', 'bb', 'cc'], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', ['bb', 'cc']], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [['bb'], [['cc']]]], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [[1, 2], [[3]], 4]], 'golden': ['aa', 1, 2, 3, 4]})
        test_cases.append({'input': [True, [2.5, 2.0 + 1j]], 'golden': [True, 2.5, 2.0 + 1j]})

        for n, test_case in enumerate(test_cases):
            assert flatten(test_case['input']) == test_case['golden'], f'Test case {n} failed!'
