# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "speech_recognition"))

import estimate_duration_bins
import estimate_duration_bins_2d


@pytest.mark.unit
def test_estimate_duration_bins_quiet_flag():
    """Verify default, -q, and --quiet CLI flag parsing for estimate_duration_bins.py."""
    # Test default
    args = estimate_duration_bins.parse_args(["dummy_manifest.json"])
    assert args.quiet is False

    # Test -q flag
    args = estimate_duration_bins.parse_args(["dummy_manifest.json", "-q"])
    assert args.quiet is True

    # Test --quiet flag
    args = estimate_duration_bins.parse_args(["dummy_manifest.json", "--quiet"])
    assert args.quiet is True


@pytest.mark.unit
def test_estimate_duration_bins_2d_quiet_flag():
    """Verify default, -q, and --quiet CLI flag parsing for estimate_duration_bins_2d.py."""
    # Test default
    args = estimate_duration_bins_2d.parse_args(["dummy_manifest.json", "--tokenizer", "dummy.model"])
    assert args.quiet is False

    # Test -q flag
    args = estimate_duration_bins_2d.parse_args(["dummy_manifest.json", "--tokenizer", "dummy.model", "-q"])
    assert args.quiet is True

    # Test --quiet flag
    args = estimate_duration_bins_2d.parse_args(["dummy_manifest.json", "--tokenizer", "dummy.model", "--quiet"])
    assert args.quiet is True
