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
import pathlib
import sys
from contextlib import contextmanager


@contextmanager
def python313_pathlib_pickle_compat():
    """Allow Python 3.13 ``pathlib`` pickles to load on older Python versions.

    Python 3.13 moved concrete path classes to ``pathlib._local``. Pickles
    created there therefore cannot resolve their module when loaded on older
    Python versions. Temporarily alias the missing module while unpickling.
    """
    module_name = "pathlib._local"
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    else:
        yield
        return

    sys.modules[module_name] = pathlib
    try:
        yield
    finally:
        if sys.modules.get(module_name) is pathlib:
            if previous_module is missing:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = previous_module
