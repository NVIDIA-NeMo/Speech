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
"""Recursively inject validator-specific flags into a train_ds-shaped
OmegaConf node and every nested ``input_cfg``."""

import logging
from typing import Any

from omegaconf import DictConfig, ListConfig

LOG = logging.getLogger(__name__)


def inject_validator_flags(cfg: DictConfig, *, force_finite: bool, metadata_only: bool) -> DictConfig:
    """Mutate-in-place: set ``force_finite`` and ``metadata_only`` on ``cfg``
    and on every nested ``input_cfg`` entry (recursively). Logs every
    injection so the user can see exactly what was changed.

    A requested validator mode is authoritative: explicitly false training
    values must be overridden as well as missing values. Otherwise a leaf can
    silently remain infinite or materialize audio despite the corresponding
    validator CLI flag.
    """
    if force_finite:
        _inject_key(cfg, "force_finite", True, ctx="train_ds (top-level)")
    if metadata_only:
        _inject_key(cfg, "metadata_only", True, ctx="train_ds (top-level)")
    _walk_input_cfg(cfg.get("input_cfg"), force_finite=force_finite, metadata_only=metadata_only)
    return cfg


def inject_groundtruth_flags(cfg: DictConfig) -> DictConfig:
    """Select the finite single-process map-style groundtruth loader.

    Training recipes may explicitly pin ``force_iterable_dataset=True``.
    Groundtruth enumeration must override that pin when it enables
    ``force_map_dataset``; leaving both true is an invalid loader contract.
    """
    for key, value in (
        ("num_workers", 0),
        ("use_stateful_dataloader", False),
        ("force_iterable_dataset", False),
        ("force_map_dataset", True),
    ):
        _inject_key(cfg, key, value, ctx="train_ds groundtruth override")
    return cfg


def inject_missing_manifest_policy(cfg: DictConfig, *, skip_missing_manifest_entries: bool) -> DictConfig:
    """Apply one explicit missing-entry policy to the complete input graph.

    Nested input entries normally override inherited values. Validation needs
    an authoritative policy so an individual source cannot silently weaken a
    fail-closed run.
    """
    _inject_key(
        cfg,
        "skip_missing_manifest_entries",
        skip_missing_manifest_entries,
        ctx="train_ds (top-level)",
    )
    _walk_policy_key(
        cfg.get("input_cfg"),
        key="skip_missing_manifest_entries",
        value=skip_missing_manifest_entries,
    )
    return cfg


def _walk_input_cfg(node: Any, *, force_finite: bool, metadata_only: bool, path: str = "input_cfg") -> None:
    if node is None:
        return
    if isinstance(node, (list, ListConfig)):
        for i, sub in enumerate(node):
            _walk_input_cfg(
                sub,
                force_finite=force_finite,
                metadata_only=metadata_only,
                path=f"{path}[{i}]",
            )
        return
    if isinstance(node, str):
        return  # input_cfg reference to a YAML file path — resolved later by NeMo
    if not isinstance(node, (dict, DictConfig)):
        return
    typ = node.get("type", "<no-type>")
    if force_finite:
        _inject_key(node, "force_finite", True, ctx=f"{path} (type={typ})")
    if metadata_only:
        _inject_key(node, "metadata_only", True, ctx=f"{path} (type={typ})")
    if "input_cfg" in node:
        _walk_input_cfg(
            node["input_cfg"],
            force_finite=force_finite,
            metadata_only=metadata_only,
            path=f"{path}.input_cfg",
        )


def _walk_policy_key(node: Any, *, key: str, value: Any, path: str = "input_cfg") -> None:
    if node is None or isinstance(node, str):
        return
    if isinstance(node, (list, ListConfig)):
        for i, sub in enumerate(node):
            _walk_policy_key(sub, key=key, value=value, path=f"{path}[{i}]")
        return
    if not isinstance(node, (dict, DictConfig)):
        return
    typ = node.get("type", "<no-type>")
    _inject_key(node, key, value, ctx=f"{path} (type={typ})")
    if "input_cfg" in node:
        _walk_policy_key(node["input_cfg"], key=key, value=value, path=f"{path}.input_cfg")


def _inject_key(node: Any, key: str, value: Any, *, ctx: str) -> None:
    prev = node.get(key) if isinstance(node, (dict, DictConfig)) else None
    if prev == value:
        return
    node[key] = value
    LOG.info("inject %s=%s into %s (was %r)", key, value, ctx, prev)
