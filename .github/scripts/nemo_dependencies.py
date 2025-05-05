#!/usr/bin/env python3
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

"""
NeMo dependency structure definition.
This module analyzes the codebase to determine internal dependencies between NeMo collections and core components.
"""

import ast
import json
import os
from typing import Dict, List, Set


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    # Look in nemo directory and other relevant directories
    relevant_dirs = ['nemo', 'scripts', 'examples', 'tests']

    for dir_name in relevant_dirs:
        dir_path = os.path.join(directory, dir_name)
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

    return python_files


def analyze_imports(nemo_root: str, file_path: str) -> Set[str]:
    """Analyze a Python file and return its NeMo package dependencies using AST parsing."""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('nemo.'):
                # Split the module path
                parts = node.module.split('.')

                if len(parts) == 1:
                    continue

                if len(parts) >= 2:
                    module_type = parts[1]

                    if module_type == 'collections':
                        if len(parts) == 2:
                            continue
                        if node.names:
                            for name in node.names:
                                if name.name == '*':
                                    continue

                                imports.add(f"{node.module}.{name.name}")

                    elif module_type in find_top_level_packages(nemo_root):
                        if node.names:
                            for name in node.names:
                                if name.name == '*':
                                    continue
                                imports.add(f"{node.module}.{name.name}")

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

    return imports


def find_top_level_packages(nemo_root: str) -> List[str]:
    """Find all top-level packages under nemo directory."""
    packages: List[str] = []
    nemo_dir = os.path.join(nemo_root, 'nemo')
    tests_dir = os.path.join(nemo_root, 'tests')

    if not os.path.exists(nemo_dir):
        print(f"Warning: nemo directory not found at {nemo_dir}")
        return packages
    if not os.path.exists(tests_dir):
        print(f"Warning: nemo directory not found at {nemo_dir}")
        return packages

    for item in os.listdir(nemo_dir) + os.listdir(tests_dir):
        item_path = os.path.join(nemo_dir, item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            packages.append(item)

    return sorted(packages)


def find_collection_modules(nemo_root: str) -> Dict[str, List[str]]:
    """Find all modules within collections."""
    collection_modules: Dict[str, List[str]] = {}
    collections_dir = os.path.join(nemo_root, 'nemo', 'collections')

    if not os.path.exists(collections_dir):
        print(f"Warning: collections directory not found at {collections_dir}")
        return collection_modules

    for collection in os.listdir(collections_dir):
        collection_path = os.path.join(collections_dir, collection)
        if os.path.isdir(collection_path) and not collection.startswith('__'):
            collection_modules[f"nemo.collections.{collection}"] = []

    return collection_modules


def build_dependency_graph(nemo_root: str) -> Dict[str, List[str]]:
    """Build a dependency graph by analyzing all Python files."""
    # Find all top-level packages
    top_level_packages = find_top_level_packages(nemo_root)
    print(f"Found top-level packages: {top_level_packages}")

    dependencies: Dict[str, List[str]] = {}

    for file_path in find_python_files(nemo_root):
        relative_path = os.path.relpath(file_path, nemo_root)
        parts = relative_path.split(os.sep)

        if len(parts) == 1 or parts[-1] == "__init__.py" or (parts[0] != "nemo" and parts[0] != "tests"):
            continue

        module_path = relative_path.replace(".py", "").replace("/", ".")
        if parts[1] in top_level_packages and parts[1] != 'collections':
            dependencies[module_path] = list(set(analyze_imports(nemo_root, file_path)))
        elif parts[0] == 'tests':
            dependencies[module_path] = [relative_path]
        elif parts[1] == 'collections':
            dependencies[module_path] = list(set(analyze_imports(nemo_root, file_path)))

    # Flip the dependency graph to show reverse dependencies
    reverse_dependencies: Dict[str, List[str]] = {}
    # Handle top-level package dependencies
    for package, deps in dependencies.items():
        for dep in deps:
            if dep not in reverse_dependencies:
                reverse_dependencies[dep] = []
            reverse_dependencies[dep].append(package)
    dependencies = reverse_dependencies

    # Follow and extend records with transitive dependencies
    transitive_dependencies = dependencies.copy()
    # Keep iterating until no new dependencies are added
    while True:
        changes_made = False
        new_dependencies = transitive_dependencies.copy()

        # For each package and its direct dependencies
        for package, deps in transitive_dependencies.items():
            # For each direct dependency
            for dep in deps:
                # If the dependency has its own dependencies
                if dep in transitive_dependencies:
                    # Add those transitive dependencies to the original package
                    for transitive_dep in transitive_dependencies[dep]:
                        if transitive_dep not in new_dependencies[package]:
                            new_dependencies[package].append(transitive_dep)
                            changes_made = True

        # Update dependencies with new transitive ones
        transitive_dependencies = new_dependencies

        # If no new dependencies were added, we're done
        if not changes_made:
            break

    dependencies = transitive_dependencies

    # Simplify values: Either top-level package or collection module
    simplified_dependencies: Dict[str, List[str]] = {}
    for package, deps in dependencies.items():
        package_parts = package.split('.')

        if os.path.isfile((file_path := f"{os.path.join(*package_parts[:-1])}.py")):
            simplified_package_path = file_path
        elif os.path.isdir((file_path := f"{os.path.join(*package_parts[:-1])}")):
            simplified_package_path = file_path
        else:
            simplified_package_path = package

        for dep in deps:
            dep_parts = dep.split('.')

            if simplified_package_path not in simplified_dependencies:
                simplified_dependencies[simplified_package_path] = []

            if (
                len(dep_parts) >= 2
                and dep_parts[1] in find_top_level_packages(nemo_root)
                and dep_parts[1] != 'collections'
            ):
                simplified_dependencies[simplified_package_path].append(f"{dep_parts[0]}.{dep_parts[1]}")

            elif len(dep_parts) >= 3 and (
                simplified_name := f"{dep_parts[0]}.{dep_parts[1]}.{dep_parts[2]}"
            ) in find_collection_modules(nemo_root):
                simplified_dependencies[simplified_package_path].append(simplified_name)

            simplified_dependencies[simplified_package_path].append(package)
            simplified_dependencies[simplified_package_path] = sorted(
                list(set(simplified_dependencies[simplified_package_path]))
            )
    dependencies = simplified_dependencies

    # Bucket
    bucket_deps: Dict[str, List[str]] = {}
    for package, deps in dependencies.items():
        new_deps = []
        for dep in deps:
            if "asr" in dep or "tts" in dep or "speechlm" in dep or "audio" in dep:
                new_deps.append("speech")

            elif "export" in dep or "deploy" in dep:
                new_deps.append("export-deploy")

            elif "llm" in dep or "vlm" in dep or "automodel" in dep:
                new_deps.append("automodel")

            elif "tests/collections" in dep:
                new_deps.append("unit-tests")
                continue

            else:
                new_deps.append("nemo2")

        bucket_deps[package] = sorted(list(set(new_deps)))

    dependencies = bucket_deps

    # Sort dependencies by length of values (number of dependencies)
    dependencies = dict(sorted(dependencies.items(), key=lambda x: len(x[1]), reverse=True))

    return dependencies


def main():
    """Main function to analyze dependencies and output JSON."""
    # Get the root directory of the NeMo project
    nemo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Build dependency graph
    dependencies = build_dependency_graph(nemo_root)

    # Output as JSON
    data = json.dumps(dependencies, indent=4)
    # print(data)
    with open('nemo_dependencies.json', 'w', encoding='utf-8') as f:
        f.write(data)


if __name__ == "__main__":
    main()
