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
from sdp.utils.edit_spaces import add_start_end_spaces, remove_extra_spaces


@pytest.mark.parametrize("input,expected_output", [("abc xyz   abc xyz", "abc xyz abc xyz"), (" abc xyz ", "abc xyz")])
def test_remove_extra_spaces(input, expected_output):
    assert remove_extra_spaces(input) == expected_output


@pytest.mark.parametrize("input,expected_output", [("abc", " abc "), ("abc xyz", " abc xyz ")])
def test_add_start_end_spaces(input, expected_output):
    assert add_start_end_spaces(input) == expected_output
