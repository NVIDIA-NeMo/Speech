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

import importlib
import importlib.util
import types


def is_module_available(*modules: str) -> bool:
    """Check whether the given modules are installed without importing them.

    This is safer than a ``try: import ...`` block because some packages have
    side effects at import time (e.g., changing the multiprocessing start
    method).  Use this for lightweight availability checks such as test-skip
    decorators or conditional registration.

    Args:
        *modules: One or more top-level module names to check.

    Returns:
        ``True`` if **all** listed modules are found, ``False`` otherwise.
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def assert_optional_dependency_available(module_name: str, *, pip_name: str | None = None) -> None:
    """Raise an ``ImportError`` if *module_name* is not installed.

    Unlike :func:`import_optional_dependency` this does **not** import the
    module — it only checks availability via :func:`is_module_available` and
    raises with an actionable install hint on failure.  Use this for early
    fail-fast checks (e.g., at the top of a CLI entry-point or ``__init__``).

    Args:
        module_name: The module to check (e.g. ``"lhotse"``).
        pip_name: The pip install name if it differs from *module_name*.
            When *None*, the top-level package name is used.

    Raises:
        ImportError: If the module is not found.
    """
    if not is_module_available(module_name):
        install_name = pip_name if pip_name is not None else module_name.split(".")[0]
        raise ImportError(
            f"Optional dependency '{module_name}' is not installed. " f"Install it with:  pip install {install_name}"
        )


def import_optional_dependency(module_name: str, *, pip_name: str | None = None) -> types.ModuleType:
    """Import an optional dependency, raising a clear error if it is not installed.

    Args:
        module_name: The module to import (e.g. ``"lhotse"`` or ``"torchaudio.transforms"``).
        pip_name: The pip install name if it differs from *module_name*
            (e.g. ``pip_name="Cython"`` for ``module_name="cython"``).
            When *None*, the top-level package name is used.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported, with an actionable
            install hint.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        install_name = pip_name if pip_name is not None else module_name.split(".")[0]
        raise ImportError(
            f"Optional dependency '{module_name}' is not installed. " f"Install it with:  pip install {install_name}"
        ) from None
