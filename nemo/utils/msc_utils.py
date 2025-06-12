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


from pathlib import Path
from typing import Union

try:
    import multistorageclient as msc

    HAVE_MSC = True
except (ImportError, ModuleNotFoundError):
    msc = None

    HAVE_MSC = False


MSC_PROTOCOL = "msc://"


def is_multistorageclient_url(path: Union[str, Path]):
    """
    Check if the path is a multistorageclient URL (e.g. msc://<profile>/<path>).

    Args:
        path: str, the path to check.

    Returns:
        bool, True if the path is a multistorageclient URL, False otherwise.
    """
    if isinstance(path, Path):
        return False

    has_msc_prefix = path and str(path).startswith(MSC_PROTOCOL)

    if HAVE_MSC:
        return has_msc_prefix

    if not HAVE_MSC and has_msc_prefix:
        raise ValueError(
            "Multi-Storage Client is not installed. Please install it with "
            '"pip install multi-storage-client" to handle msc:// URLs.'
        )

    return False


def import_multistorageclient():
    """Import multistorageclient if it is installed."""
    if not HAVE_MSC:
        raise ValueError(
            "Multi-Storage Client is not installed. Please install it with " '"pip install multi-storage-client".'
        )
    return msc
