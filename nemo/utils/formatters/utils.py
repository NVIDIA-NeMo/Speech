# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import sys

from nemo.constants import NEMO_ENV_VARNAME_ENABLE_COLORING
from nemo.utils.env_var_parsing import get_envbool

__all__ = ["check_color_support", "to_unicode"]


def check_color_support():
    # Colors can be forced with an env variable
    if not sys.platform.lower().startswith("win") and get_envbool(NEMO_ENV_VARNAME_ENABLE_COLORING, False):
        return True


def to_unicode(value):
    """
    Converts a string argument to a unicode string.
    If the argument is already a unicode string or None, it is returned
    unchanged.  Otherwise it must be a byte string and is decoded as utf8.
    """
    try:
        if isinstance(value, (str, type(None))):
            return value

        if not isinstance(value, bytes):
            raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))

        return value.decode("utf-8")

    except UnicodeDecodeError:
        return repr(value)
