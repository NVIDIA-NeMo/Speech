# The MIT Licence (MIT)
#
# Copyright (c) 2016 YunoJuno Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Vendored dependency from : https://github.com/yunojuno/python-env-utils/blob/master/env_utils/utils.py
#
#
# Modified by NVIDIA
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import decimal
import json
import os
from datetime import datetime

__all__ = [
    "get_env",
    "get_envbool",
    "get_envint",
    "get_envfloat",
    "get_envdecimal",
    "get_envdate",
    "get_envdatetime",
    "get_envlist",
    "get_envdict",
    "CoercionError",
    "RequiredSettingMissingError",
]


class CoercionError(Exception):
    """Custom error raised when a value cannot be coerced."""

    def __init__(self, key, value, func):
        msg = "Unable to coerce '{}={}' using {}.".format(key, value, func.__name__)
        super(CoercionError, self).__init__(msg)


class RequiredSettingMissingError(Exception):
    """Custom error raised when a required env var is missing."""

    def __init__(self, key):
        msg = "Required env var '{}' is missing.".format(key)
        super(RequiredSettingMissingError, self).__init__(msg)


def _get_env(key, default=None, coerce=lambda x: x, required=False):
    """
    Return env var coerced into a type other than string.
    This function extends the standard os.getenv function to enable
    the coercion of values into data types other than string (all env
    vars are strings by default).
    Args:
        key: string, the name of the env var to look up
    Kwargs:
        default: the default value to return if the env var does not exist. NB the
            default value is **not** coerced, and is assumed to be of the correct type.
        coerce: a function that is used to coerce the value returned into
            another type
        required: bool, if True, then a RequiredSettingMissingError error is raised
            if the env var does not exist.
    Returns the env var, passed through the coerce function
    """
    try:
        value = os.environ[key]
    except KeyError:
        if required is True:
            raise RequiredSettingMissingError(key)
        else:
            return default

    try:
        return coerce(value)
    except Exception:
        raise CoercionError(key, value, coerce)


# standard type coercion functions
def _bool(value):
    if isinstance(value, bool):
        return value

    return not (value is None or value.lower() in ("false", "0", "no", "n", "f", "none"))


def _int(value):
    return int(value)


def _float(value):
    return float(value)


def _decimal(value):
    return decimal.Decimal(value)


def _dict(value):
    return json.loads(value)


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d %B %Y",
    "%B %d, %Y",
)

_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S%z",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S%z",
    "%m/%d/%Y %H:%M:%S",
    "%d %B %Y %H:%M:%S%z",
    "%d %B %Y %H:%M:%S",
    "%B %d, %Y %H:%M:%S%z",
    "%B %d, %Y %H:%M:%S",
)


def _parse_datetime(value):
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(value)


def _datetime(value):
    return _parse_datetime(value)


def _date(value):
    return _parse_datetime(value).date()


def get_env(key, *default, **kwargs):
    """
    Return env var.
    This is the parent function of all other get_foo functions,
    and is responsible for unpacking args/kwargs into the values
    that _get_env expects (it is the root function that actually
    interacts with environ).
    Args:
        key: string, the env var name to look up.
        default: (optional) the value to use if the env var does not
            exist. If this value is not supplied, then the env var is
            considered to be required, and a RequiredSettingMissingError
            error will be raised if it does not exist.
    Kwargs:
        coerce: a func that may be supplied to coerce the value into
            something else. This is used by the default get_foo functions
            to cast strings to builtin types, but could be a function that
            returns a custom class.
    Returns the env var, coerced if required, and a default if supplied.
    """
    assert len(default) in (0, 1), "Too many args supplied."
    func = kwargs.get('coerce', lambda x: x)
    required = len(default) == 0
    default = default[0] if not required else None
    return _get_env(key, default=default, coerce=func, required=required)


def get_envbool(key, *default):
    """Return env var cast as boolean."""
    return get_env(key, *default, coerce=_bool)


def get_envint(key, *default):
    """Return env var cast as integer."""
    return get_env(key, *default, coerce=_int)


def get_envfloat(key, *default):
    """Return env var cast as float."""
    return get_env(key, *default, coerce=_float)


def get_envdecimal(key, *default):
    """Return env var cast as Decimal."""
    return get_env(key, *default, coerce=_decimal)


def get_envdate(key, *default):
    """Return env var as a date."""
    return get_env(key, *default, coerce=_date)


def get_envdatetime(key, *default):
    """Return env var as a datetime."""
    return get_env(key, *default, coerce=_datetime)


def get_envlist(key, *default, **kwargs):
    """Return env var as a list."""
    separator = kwargs.get('separator', ' ')
    return get_env(key, *default, coerce=lambda x: x.split(separator))


def get_envdict(key, *default):
    """Return env var as a dict."""
    return get_env(key, *default, coerce=_dict)
