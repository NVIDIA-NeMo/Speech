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
import importlib
import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["beamforming_flex_channels.yaml"])
def test_example_config_targets_resolve(config_name):
    """Every ``_target_`` in the config must import and accept the keys given next to it (nothing is built)."""
    cfg = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "examples/audio/conf" / config_name), resolve=False)

    targets = list(_iter_targets(cfg))
    assert targets, f"No _target_ entries found in {config_name}"
    for path, node in targets:
        target = _locate(node["_target_"])
        kwargs = {k: v for k, v in node.items() if not k.startswith("_")}
        try:
            inspect.signature(target).bind_partial(**kwargs)
        except TypeError as e:
            raise AssertionError(f"{config_name}: {path} does not match {node['_target_']}: {e}") from e


def _iter_targets(node, path="cfg"):
    if isinstance(node, dict):
        if "_target_" in node:
            yield path, node
        for key, value in node.items():
            yield from _iter_targets(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _iter_targets(value, f"{path}[{idx}]")


def _locate(dotted_path):
    """Import ``a.b.c.Name`` via importlib: longest importable module prefix, then attribute lookup."""
    parts = dotted_path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # Only fall back to a shorter prefix if *this* module is missing, not one of its dependencies.
            if e.name is None or not module_name.startswith(e.name):
                raise
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                raise AssertionError(f"Cannot resolve _target_ '{dotted_path}': '{module_name}' has no '{attr}'")
        return obj
    raise AssertionError(f"Cannot resolve _target_ '{dotted_path}': no importable module prefix")
