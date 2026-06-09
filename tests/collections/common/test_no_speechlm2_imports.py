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
import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
COMMON_ROOT = REPO_ROOT / "nemo/collections/common"


def test_common_collection_has_no_global_speechlm2_imports():
    bad_imports = []
    for path in COMMON_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "nemo.collections.speechlm2" or module.startswith("nemo.collections.speechlm2."):
                    bad_imports.append((path.relative_to(REPO_ROOT), node.lineno, module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "nemo.collections.speechlm2" or alias.name.startswith(
                        "nemo.collections.speechlm2."
                    ):
                        bad_imports.append((path.relative_to(REPO_ROOT), node.lineno, alias.name))

    assert bad_imports == []
