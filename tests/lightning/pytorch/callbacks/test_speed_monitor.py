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

from nemo.lightning.pytorch.callbacks import SpeedMonitor


def test_speed_monitor():
    monitor = SpeedMonitor(window_size=10, time_unit='seconds')
    assert monitor.divider == 1

    monitor = SpeedMonitor(window_size=100, time_unit='minutes')
    assert monitor.divider == 60

    monitor = SpeedMonitor(window_size=100, time_unit='hours')
    assert monitor.divider == 60 * 60

    monitor = SpeedMonitor(window_size=1000, time_unit='days')
    assert monitor.divider == 60 * 60 * 24
