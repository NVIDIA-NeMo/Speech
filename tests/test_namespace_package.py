# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech and separately installed nemo.* packages must share a namespace."""

import os
import subprocess
import sys
from pathlib import Path


def test_speech_coexists_with_separately_installed_lens(tmp_path):
    sibling = tmp_path / 'nemo' / 'lens'
    sibling.mkdir(parents=True)
    (sibling / '__init__.py').write_text('fixture_marker = "separate-distribution"\n')
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(root), str(tmp_path)))}
    result = subprocess.run(
        [sys.executable, '-c', 'import nemo, nemo.lens; assert nemo.lens.fixture_marker == "separate-distribution"'],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
