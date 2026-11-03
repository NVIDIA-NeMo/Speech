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

from nemo.collections.common.parts.preprocessing import cleaners


class TestCleaners:
    def test_clean_abbreviations_default(self):
        res = cleaners.clean_abbreviations("mr. smith", version=None)
        assert res == "mister smith"

    def test_clean_abbreviations_fastpitch(self):
        res = cleaners.clean_abbreviations("mr. smith", version="fastpitch")
        assert res == "mister smith"

    def test_clean_abbreviations_expanded(self):
        # Common abbreviation handled
        res = cleaners.clean_abbreviations("mr. smith", version="expanded")
        assert res == "mister smith"

        # Expanded abbreviation handled (e.g., 'apt.' -> 'apartment')
        res_apt = cleaners.clean_abbreviations("apt. 5", version="expanded")
        assert "apartment" in res_apt

    def test_clean_abbreviations_immutability(self):
        common_len_before = len(cleaners.ABBREVIATIONS_COMMON)
        _ = cleaners.clean_abbreviations("apt. 5", version="expanded")
        common_len_after = len(cleaners.ABBREVIATIONS_COMMON)
        assert common_len_before == common_len_after
