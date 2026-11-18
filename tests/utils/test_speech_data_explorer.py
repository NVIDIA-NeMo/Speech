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
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SDE_DIR = Path(__file__).resolve().parents[2] / "tools" / "speech_data_explorer"
SDE_SCRIPT = SDE_DIR / "data_explorer.py"


def test_sde_top_level_imports_are_in_requirements():
    """Every unconditional third-party import of data_explorer.py must be listed in its requirements.txt."""
    requirements = set()
    for line in (SDE_DIR / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            requirements.add(_normalize(re.split(r"[<>=!~;\[ ]", line)[0]))

    missing = []
    for node in ast.parse(SDE_SCRIPT.read_text()).body:  # module-level imports outside try/except guards
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module]
        else:
            continue
        for module in modules:
            top = module.split(".")[0]
            if top not in sys.stdlib_module_names and _normalize(top) not in requirements:
                missing.append(top)
    assert not missing, f"data_explorer.py imports modules missing from requirements.txt: {missing}"


def test_sde_load_manifest_with_predictions(tmp_path, monkeypatch):
    for module in ("dash", "dash_bootstrap_components", "diff_match_patch", "kaldialign", "librosa", "plotly"):
        pytest.importorskip(module)
    import soundfile as sf

    entries = [
        # sub=1 (b->x), ins=1 (e), del=0 -> hits=3
        {"text": "a b c d", "pred_text": "a x c d e"},
        # del=2 (world, foo) -> hits=1
        {"text": "hello world foo", "pred_text": "hello"},
    ]
    manifest = tmp_path / "manifest.json"
    with manifest.open("w") as f:
        for idx, entry in enumerate(entries):
            audio_path = tmp_path / f"{idx}.wav"
            sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
            f.write(json.dumps({"audio_filepath": str(audio_path), "duration": 0.1, **entry}) + "\n")

    # The script parses CLI args and loads the manifest at import time; the server only starts under __main__.
    monkeypatch.setattr(sys, "argv", [str(SDE_SCRIPT), str(manifest)])
    spec = importlib.util.spec_from_file_location("_sde_data_explorer_under_test", SDE_SCRIPT)
    sde = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sde)

    assert sde.metrics_available
    first, second = sde.data
    assert first["WER"] == 50.0
    assert (first["WMR"], first["I"], first["D"], first["D-I"]) == (75.0, 1, 0, -1)
    assert second["WER"] == pytest.approx(66.67)
    assert (second["WMR"], second["I"], second["D"], second["D-I"]) == (33.33, 0, 2, 2)
    # Corpus-level word match rate: (3 + 1) hits over 7 reference words.
    assert sde.wmr == pytest.approx(4 / 7 * 100, abs=0.01)


def _normalize(name):
    return name.strip().lower().replace("-", "_")
