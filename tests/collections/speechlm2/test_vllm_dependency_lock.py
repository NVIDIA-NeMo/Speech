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

"""Check dependency overrides against the installed optional runtime metadata."""

from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.unit
def test_locked_aiohttp_satisfies_vllm_requirement():
    tomllib = pytest.importorskip("tomllib")
    try:
        runtime_version = metadata.version("vllm")
        requirements = metadata.requires("vllm") or []
    except metadata.PackageNotFoundError:
        pytest.skip("vLLM is not installed")

    lock = tomllib.loads((Path(__file__).resolve().parents[3] / "uv.lock").read_text())
    if not any(p["name"] == "vllm" and p["version"] == runtime_version for p in lock["package"]):
        pytest.skip("Installed vLLM does not match the lockfile")

    aiohttp_versions = [p["version"] for p in lock["package"] if p["name"] == "aiohttp"]
    aiohttp_requirements = [Requirement(r) for r in requirements if Requirement(r).name == "aiohttp"]
    assert aiohttp_versions and aiohttp_requirements
    for requirement in aiohttp_requirements:
        if requirement.marker is None or requirement.marker.evaluate():
            for version in aiohttp_versions:
                assert version in requirement.specifier, f"Locked aiohttp {version} does not satisfy {requirement}"
