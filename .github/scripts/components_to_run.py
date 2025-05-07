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

#!/usr/bin/env python3
import json
import os
import sys
from typing import Any, Dict, List, Set

import click
import git

import nemo_dependencies


def get_changed_files(source_sha: str, target_sha: str) -> List[str]:
    """
    Fetch the changelog between current branch and main.
    Returns a list of dictionaries containing commit information.
    """
    try:
        # Initialize the repo object - go up two levels from this file's location
        repo = git.Repo(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Get the diff between target and source
        diff_index = repo.commit(target_sha).diff(repo.commit(source_sha))

        # Get just the changed filenames
        changed_files = []
        for diff in diff_index:
            changed_files.append(diff.a_path if diff.a_path else diff.b_path)

        return changed_files

    except git.exc.GitCommandError as e:
        print(f"Error fetching changelog: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


@click.command()
@click.option('--source-sha', type=str, required=True, help='Source commit SHA')
@click.option('--target-sha', type=str, required=True, help='Target commit sha')
def main(source_sha: str, target_sha: str):
    """
    Main function to fetch and output the changelog and changed files.
    """

    # Output unique changed files
    print("\nChanged files:")
    changed_files = get_changed_files(source_sha, target_sha)

    print(json.dumps(sorted(list(changed_files)), indent=2))

    nemo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Build dependency graph
    dependencies = nemo_dependencies.build_dependency_graph(nemo_root)

    test_modules: List[str] = []
    for changed_file in changed_files:
        if changed_file in dependencies:
            test_modules.extend(dependencies[changed_file])

    test_modules = list(set(test_modules))

    with open("test_modules.json", "w", encoding="utf-8") as f:
        json.dump(test_modules, f)


if __name__ == "__main__":
    main()
