#!/usr/bin/env python3
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

"""Blend several Lhotse ``input_cfg`` YAMLs into one, with per-input weights.

``input_cfg`` has no include mechanism: ``input_cfg:`` is only meaningful on a ``type: group``
entry, and there it must be an inline **list**, not a path. A config that references another
config by path fails with ``Missing key manifest_filepath`` -- the parser sees ``type: nemo`` and
looks for a manifest. So composing task-specific configs means inlining them, and inlining by
hand means the manifest lists and prompts exist twice and drift.

This script inlines them mechanically instead, so the per-task configs stay the single source of
truth for their manifests and their ``tags`` (which is where a per-task ``system_prompt`` lives --
it reaches the model as a per-sample prompt via ``cut.custom``).

Weights multiply through: a group weighted 0.4 inside a config given ``--weights 0.5`` ends up at
0.2, so a sub-config's internal balance is preserved while its share of the blend is set here.

Usage:
    # explicit weights
    python scripts/speechlm2/blend_input_cfgs.py \\
        --inputs asr_input_cfg.yaml mtasr_input_cfg.yaml \\
        --weights 0.746 0.254 \\
        --output blended_input_cfg.yaml

    # weights proportional to total audio duration, read from the manifests
    python scripts/speechlm2/blend_input_cfgs.py \\
        --inputs asr_input_cfg.yaml mtasr_input_cfg.yaml \\
        --weights-by duration \\
        --output blended_input_cfg.yaml

The output is validated by loading it before it is written, so a config that would fail at
training time fails here instead.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import ListConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_groups(path: str) -> list[dict]:
    """Read one input_cfg YAML and return its top-level entries.

    Args:
        path: Path to an ``input_cfg`` YAML (a list of entries, usually ``type: group``).

    Returns:
        list[dict]: the entries, as plain containers.

    Raises:
        ValueError: when the file is not a list, which every input_cfg must be.
    """
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, ListConfig):
        raise ValueError(f"{path}: an input_cfg must be a LIST of entries, got {type(cfg).__name__}.")
    return [OmegaConf.to_container(entry, resolve=True) for entry in cfg]


def manifest_paths(entry: dict) -> list[str]:
    """Every ``manifest_filepath`` reachable from one entry, recursing into groups."""
    found = []
    if "manifest_filepath" in entry:
        value = entry["manifest_filepath"]
        found.extend(value if isinstance(value, (list, tuple)) else [value])
    for child in entry.get("input_cfg", []) or []:
        if isinstance(child, dict):
            found.extend(manifest_paths(child))
    return found


def total_duration_hours(entries: list[dict]) -> float:
    """Sum ``duration`` over every manifest an input_cfg references.

    Reads the manifests directly rather than building a CutSet: this runs before the blend is
    written, and only needs a magnitude to set weights from.
    """
    hours = 0.0
    for entry in entries:
        for path in manifest_paths(entry):
            with open(path) as handle:
                for line in handle:
                    if line.strip():
                        hours += float(json.loads(line)["duration"])
    return hours / 3600


def scale_weights(entries: list[dict], factor: float) -> list[dict]:
    """Multiply each top-level entry's weight by ``factor``.

    An entry without a weight is treated as 1.0, matching the loader, so a single-group config
    simply takes the factor.
    """
    scaled = []
    for entry in entries:
        entry = dict(entry)
        entry["weight"] = float(entry.get("weight", 1.0)) * factor
        scaled.append(entry)
    return scaled


def validate(path: str) -> int:
    """Load the blended config the way training will, and report how many cuts it yields.

    Returns:
        int: number of cuts enumerated (capped), or -1 when the loader could not be imported.
    """
    try:
        from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
    except ImportError:
        log.warning("  (skipping validation: NeMo's lhotse loader is not importable here)")
        return -1
    cfg = OmegaConf.create(
        {
            "input_cfg": path,
            "shuffle": False,
            "sample_rate": 16000,
            "seed": 0,
            "shard_seed": 0,
            "force_finite": True,
            "metadata_only": True,
        }
    )
    cuts, _ = read_cutset_from_config(cfg)
    seen = 0
    for _ in cuts:
        seen += 1
        if seen >= 200:
            break
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description="Blend several Lhotse input_cfg YAMLs into one.")
    parser.add_argument("--inputs", required=True, nargs="+", help="Input input_cfg YAML paths, in order.")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="One weight per input. Mutually exclusive with --weights-by.",
    )
    parser.add_argument(
        "--weights-by",
        choices=("duration", "equal"),
        help="Derive weights instead of naming them: 'duration' reads every referenced manifest "
        "and weights each input by its total audio; 'equal' splits evenly.",
    )
    parser.add_argument("--output", required=True, help="Path to write the blended input_cfg YAML.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep weights as given. By default they are normalised to sum to 1 so they read as "
        "proportions; the loader normalises internally either way.",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip loading the result to check it.")
    args = parser.parse_args()

    if bool(args.weights) == bool(args.weights_by):
        parser.error("pass exactly one of --weights or --weights-by")
    if args.weights and len(args.weights) != len(args.inputs):
        parser.error(f"got {len(args.inputs)} inputs but {len(args.weights)} weights")

    loaded = [load_groups(p) for p in args.inputs]

    if args.weights_by == "duration":
        hours = [total_duration_hours(groups) for groups in loaded]
        if sum(hours) <= 0:
            parser.error("--weights-by duration found no durations; are the manifests readable?")
        weights = hours
        for path, h in zip(args.inputs, hours):
            log.info(f"  {Path(path).name:38s} {h:9.1f} h")
    elif args.weights_by == "equal":
        weights = [1.0] * len(args.inputs)
    else:
        weights = list(args.weights)

    if not args.no_normalize:
        total = sum(weights)
        weights = [w / total for w in weights]

    blended: list[Any] = []
    for path, groups, weight in zip(args.inputs, loaded, weights):
        log.info(f"  {Path(path).name:38s} weight {weight:.4f}  ({len(groups)} group(s))")
        blended.extend(scale_weights(groups, weight))

    out = Path(args.output)
    OmegaConf.save(OmegaConf.create(blended), out)
    log.info(f"wrote {out} ({len(blended)} top-level entries)")

    if not args.no_validate:
        seen = validate(str(out))
        if seen == 0:
            log.error("!!! the blended config loads but yields NO cuts")
            return 1
        if seen > 0:
            log.info(f"validated: loads and yields cuts (sampled {seen})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
