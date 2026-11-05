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
"""
Refresh `tests/test_file_durations.json`, which balances the `--num-file-shards` unit-test shards in CI.

The input is one or more pytest logs that contain the `--durations=0` report (enabled for every run through
`addopts` in `pyproject.toml`), for example the logs of all CPU unit-test jobs (`L0_Unit_Tests_CPU_*`) of one CI run:

    gh run view --repo NVIDIA-NeMo/Speech --job <job-id> --log > cpu_asr_1.log  # repeat for each CPU job
    python tests/update_test_file_durations.py --mode cpu cpu_*.log

Only the given mode ("cpu" or "gpu") is replaced; the other mode is kept.
"""

import argparse
import collections
import json
import re
from pathlib import Path

DURATION_LINE = re.compile(r"(?:^|\s)(\d+\.\d+)s (call|setup|teardown)\s+((tests/\S+?\.py)::\S+)")
DEFAULT_OUTPUT = Path(__file__).with_name("test_file_durations.json")


def parse_file_durations(lines) -> dict:
    """Sum the reported setup, call, and teardown durations of each test file."""
    durations = collections.defaultdict(float)
    seen = set()  # CI job logs print the report twice
    for line in lines:
        match = DURATION_LINE.search(line)
        if match and (match.group(2), match.group(3)) not in seen:
            seen.add((match.group(2), match.group(3)))
            durations[match.group(4)] += float(match.group(1))
    return {path: round(seconds, 1) for path, seconds in sorted(durations.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="pytest logs with a `--durations=0` report")
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True, help="device mode of the logged jobs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lines = []
    for log in args.logs:
        lines.extend(log.read_text(errors="ignore").splitlines())
    durations = parse_file_durations(lines)
    if not durations:
        raise SystemExit("No `--durations` report lines found in the given logs.")

    table = json.loads(args.output.read_text()) if args.output.exists() else {}
    table[args.mode] = durations
    args.output.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(durations)} {args.mode} file durations to {args.output}")


if __name__ == "__main__":
    main()
