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

from nemo.collections.asr.parts.utils.speaker_utils import fl2int, merge_float_intervals


@pytest.mark.unit
class TestFl2IntOverflow:
    """Regression tests for issue #16448: int32 overflow in fl2int."""

    def test_short_recording(self):
        """Values under the old int32 limit should be unchanged."""
        assert fl2int(100.0, 5) == 10000000
        assert fl2int(0.0, 5) == 0

    def test_beyond_int32_limit(self):
        """Times above 21474.8 s must not overflow (was negative with .int())."""
        result = fl2int(21475.0, 5)
        assert result == 2147500000
        assert result > 0

    def test_eight_hour_recording(self):
        """An 8-hour timestamp must round-trip correctly."""
        eight_hours = 8 * 3600.0  # 28800 s
        result = fl2int(eight_hours, 5)
        assert result == 2880000000
        assert result > 0

    def test_merge_float_intervals_long_recording(self):
        """merge_float_intervals must not drop turns past 5.96 h."""
        intervals = [[21470.0, 21474.0], [21480.0, 21490.0]]
        merged = merge_float_intervals(intervals)
        assert len(merged) == 2
        assert merged[1][0] == pytest.approx(21480.0, abs=1e-3)
        assert merged[1][1] == pytest.approx(21490.0, abs=1e-3)
