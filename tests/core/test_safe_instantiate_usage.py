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

import ast
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_INSTANTIATE_WRAPPER = REPO_ROOT / "nemo/core/classes/common.py"
SOURCE_DIRS = ("nemo", "scripts", "examples")


def _iter_python_files():
    for source_dir in SOURCE_DIRS:
        yield from (REPO_ROOT / source_dir).rglob("*.py")


def _name_for_call(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name_for_call(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


class DirectHydraInstantiateVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.function_stack = []
        self.hydra_aliases = set()
        self.hydra_utils_aliases = set()
        self.hydra_instantiate_aliases = set()
        self.violations = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name == "hydra":
                self.hydra_aliases.add(alias.asname or "hydra")
            elif alias.name == "hydra.utils" and alias.asname:
                self.hydra_utils_aliases.add(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "hydra":
            for alias in node.names:
                if alias.name == "utils":
                    self.hydra_utils_aliases.add(alias.asname or alias.name)
        elif node.module == "hydra.utils":
            for alias in node.names:
                if alias.name == "instantiate":
                    self.hydra_instantiate_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Call(self, node: ast.Call):
        call_name = _name_for_call(node.func)
        if self._is_direct_hydra_instantiate(call_name) and not self._is_allowed_wrapper_call():
            self.violations.append((node.lineno, call_name))
        self.generic_visit(node)

    def _is_direct_hydra_instantiate(self, call_name: str | None) -> bool:
        if call_name is None:
            return False
        if call_name in self.hydra_instantiate_aliases:
            return True
        if call_name.endswith(".instantiate"):
            prefix = call_name[: -len(".instantiate")]
            if prefix in self.hydra_utils_aliases:
                return True
            return any(prefix == f"{hydra_alias}.utils" for hydra_alias in self.hydra_aliases)
        return False

    def _is_allowed_wrapper_call(self) -> bool:
        return self.path == SAFE_INSTANTIATE_WRAPPER and self.function_stack[-1:] == ["safe_instantiate"]


@pytest.mark.unit
def test_hydra_instantiate_is_only_called_by_safe_instantiate():
    violations = []
    for path in _iter_python_files():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = DirectHydraInstantiateVisitor(path)
        visitor.visit(tree)
        violations.extend((path.relative_to(REPO_ROOT), lineno, call_name) for lineno, call_name in visitor.violations)

    assert (
        not violations
    ), "Use nemo.core.classes.common.safe_instantiate instead of hydra.utils.instantiate:\n" + "\n".join(
        f"{path}:{lineno} calls {call_name}" for path, lineno, call_name in violations
    )
