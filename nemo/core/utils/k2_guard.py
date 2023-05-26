# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
Guard for importing optional NeMo dependency k2.
Contains checks for k2 availability and version.
Use `from nemo.core.utils.k2_guard import k2` to import k2 instead of direct import.
If there is an error, the module will raise an exception with a helpful message.
"""

import textwrap

from packaging.version import Version
from pytorch_lightning.utilities.imports import package_available
from nemo.core.utils.k2_utils import K2_INSTALLATION_MESSAGE

__K2_MINIMUM_MAJOR_VERSION = 1
__K2_MINIMUM_MINOR_VERSION = 14

__K2_MINIMUM_VERSION = Version(f"{__K2_MINIMUM_MAJOR_VERSION}.{__K2_MINIMUM_MINOR_VERSION}")


if not package_available("k2"):
    raise ModuleNotFoundError("Module k2 is not available.\n" + K2_INSTALLATION_MESSAGE)

import k2  # noqa: E402

try:
    __k2_version = Version(k2.__dev_version__)
except AttributeError:
    raise ImportError("Module k2 is corrupted.\n" + K2_INSTALLATION_MESSAGE)

if __k2_version < __K2_MINIMUM_VERSION:
    raise ImportError(
        textwrap.dedent(
            f"""
            Minimum required k2 version: {__K2_MINIMUM_VERSION};
            Installed k2 version: {__k2_version}
            """
        )
    )
