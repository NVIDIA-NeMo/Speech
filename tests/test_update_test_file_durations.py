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

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("update_test_file_durations.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location("update_test_file_durations", SCRIPT_PATH)
update_test_file_durations = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(update_test_file_durations)


@pytest.mark.unit
def test_parse_file_durations_sums_phases_per_file_and_ignores_repeated_reports():
    report = [
        "L0_Unit_Tests_CPU_ASR_1\tmain\t2026-11-04T16:49:41.2Z 6.80s call     tests/a/test_x.py::TestX::test_one[1-2]",
        "2026-11-04T16:49:41.3Z 1.20s setup    tests/a/test_x.py::TestX::test_one[1-2]",
        "0.50s teardown tests/a/test_x.py::test_two",
        "3.00s call     tests/b/test_y.py::test_three",
        # The CI test template prints the pytest log a second time; repeated entries must not double count.
        "2026-11-04T16:50:00.0Z 6.80s call     tests/a/test_x.py::TestX::test_one[1-2]",
        "not a duration line",
    ]

    assert update_test_file_durations.parse_file_durations(report) == {
        "tests/a/test_x.py": 8.5,
        "tests/b/test_y.py": 3.0,
    }
