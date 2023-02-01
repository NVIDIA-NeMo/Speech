# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


from nemo.utils.enum import PrettyStrEnum


class ASRModelType(PrettyStrEnum):
    CTC = "ctc"
    RNNT = "rnnt"


class TestPrettyStrEnum:
    def test_incorrect_value(self):
        """Test pretty error message for invalid value"""
        try:
            ASRModelType("incorrect")
        except ValueError as e:
            assert str(e) == "incorrect is not a valid ASRModelType. Possible choices: ctc, rnnt"

    def test_correct_value(self):
        """Test that correct value is accepted"""
        assert ASRModelType("ctc") == ASRModelType.CTC

    def test_str(self):
        """
        Test that str() returns the source value,
        useful for serialization/deserialization and user-friendly logging
        """
        assert str(ASRModelType("ctc")) == "ctc"
