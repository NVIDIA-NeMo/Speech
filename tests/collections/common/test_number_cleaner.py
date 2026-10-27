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

import pytest

pytest.importorskip("inflect")

from nemo.collections.common.parts.preprocessing.cleaners import clean_numbers  # noqa: E402


def test_one_dollar_and_one_cent_are_singular():
    assert clean_numbers("$1") == "one dollar"
    assert clean_numbers("$1.01") == "one dollar and one cent"
    assert clean_numbers("$2.01") == "two dollars and one cent"
    assert clean_numbers("$2.02") == "two dollars and two cents"
    assert clean_numbers("$10") == "ten dollars"
    assert clean_numbers("$1,000") == "one thousand dollars"
