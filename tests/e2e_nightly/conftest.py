# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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

"""Shared fixtures for per-model functional tests."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    # Model-support tests download and restore released checkpoints of up to a few billion parameters,
    # which can exceed the repository-wide 300-second per-test timeout from pyproject.toml.
    # The hook receives every collected item, so restrict it to the items in this directory.
    nightly_dir = Path(__file__).parent
    for item in items:
        if Path(item.path).is_relative_to(nightly_dir) and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(1200))
