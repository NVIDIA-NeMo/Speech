#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Set, Union


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


def analyze_imports(file_path: str) -> Set[str]:
    """Analyze a Python file and return its NeMo package dependencies using AST parsing."""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)

        # Walk through the AST to find import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('nemo.'):
                # Split the module path
                parts = node.module.split('.')
                if len(parts) >= 2:
                    module_type = parts[1]  # collections, core, utils, or automodel

                    if module_type == 'collections' and len(parts) >= 3:
                        imported_package = f"nemo.collections.{parts[2]}"
                        imports.add(imported_package)
                    elif module_type in ('core', 'utils', 'export', 'deploy', 'lightning', 'automodel'):
                        imported_package = f"nemo.{module_type}"
                        imports.add(imported_package)

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

    return imports


def find_top_level_packages(nemo_root: str) -> List[str]:
    """Find all top-level packages under nemo directory."""
    packages: List[str] = []
    nemo_dir = os.path.join(nemo_root, 'nemo')

    if not os.path.exists(nemo_dir):
        print(f"Warning: nemo directory not found at {nemo_dir}")
        return packages

    for item in os.listdir(nemo_dir):
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
            modules = []
            for root, _, files in os.walk(collection_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        rel_path = os.path.relpath(os.path.join(root, file), collections_dir)
                        module = rel_path.replace(os.sep, '.').replace('.py', '')
                        if module:
                            modules.append(f"nemo.collections.{collection}.{module}")
            collection_modules[f"nemo.collections.{collection}"] = sorted(modules)

    return collection_modules


def build_dependency_graph(nemo_root: str) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """Build a dependency graph by analyzing all Python files."""
    # Find all top-level packages
    top_level_packages = find_top_level_packages(nemo_root)
    print(f"Found top-level packages: {top_level_packages}")

    # Initialize reverse dependency sets for each package
    reverse_deps: Dict[str, Set[str]] = {}

    # Find all Python files
    python_files = find_python_files(nemo_root)

    # First pass: collect all packages
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, nemo_root)
        parts = relative_path.split(os.sep)

        if len(parts) < 2:
            continue

        # Determine which package this file belongs to
        if parts[0] == 'nemo':
            if parts[1] == 'collections' and len(parts) >= 3:
                current_package = f"nemo.collections.{parts[2]}"
            elif parts[1] in top_level_packages:
                current_package = f"nemo.{parts[1]}"
            else:
                continue

            # Initialize reverse dependency set for this package if not exists
            if current_package not in reverse_deps:
                reverse_deps[current_package] = set()

    # Second pass: analyze imports and build reverse dependencies
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, nemo_root)
        parts = relative_path.split(os.sep)

        if len(parts) < 2:
            continue

        # Determine which package this file belongs to
        if parts[0] == 'nemo':
            if parts[1] == 'collections' and len(parts) >= 3:
                current_package = f"nemo.collections.{parts[2]}"
            elif parts[1] in top_level_packages:
                current_package = f"nemo.{parts[1]}"
            else:
                continue

            # Analyze imports in this file
            imports = analyze_imports(file_path)
            # Add current package as a reverse dependency to each imported package
            for imported_pkg in imports:
                if imported_pkg in reverse_deps and imported_pkg != current_package:
                    reverse_deps[imported_pkg].add(current_package)

    # Convert sets to sorted lists and group collections
    dependencies: Dict[str, Union[List[str], Dict[str, List[str]]]] = {}

    # Add collections group
    collections = {}
    for pkg, deps in reverse_deps.items():
        if pkg.startswith('nemo.collections.') and not pkg.endswith('__init__.py'):
            collections[pkg] = sorted(list(deps))
    if collections:
        dependencies['nemo.collections'] = collections

    # Add other packages
    for pkg, deps in reverse_deps.items():
        if not pkg.startswith('nemo.collections.'):
            dependencies[pkg] = sorted(list(deps))

    return dependencies


def main():
    """Main function to analyze dependencies and output JSON."""
    # Get the root directory of the NeMo project
    nemo_root = os.path.dirname(os.path.abspath(__file__))

    # Build dependency graph
    dependencies = build_dependency_graph(nemo_root)

    # Output as JSON
    print(json.dumps(dependencies, indent=4))


if __name__ == "__main__":
    main()
