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


# The "add_port_docs" decorator is needed to nicely generate neural types in Sphynx for input and output ports

__all__ = [
    'add_port_docs',
]

import functools
import sys

import wrapt


def _normalize_docstring(docstring):
    """Normalizes the docstring.
    Replaces tabs with spaces, removes leading and trailing blanks lines, and
    removes any indentation.
    Copied from PEP-257:
    https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation
    Args:
        docstring: the docstring to normalize
    Returns:
        The normalized docstring
    """
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    # (we use sys.maxsize because sys.maxint doesn't exist in Python 3)
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)


def add_port_docs(wrapped=None, instance=None, value=''):
    if wrapped is None:
        return functools.partial(add_port_docs, value=value)

    @wrapt.decorator
    def wrapper(wrapped, instance=None, args=None, kwargs=None):
        return wrapped(*args, **kwargs)

    decorated = wrapper(wrapped)
    try:
        port_2_ntype = decorated(instance)
    except Exception:
        port_2_ntype = None

    port_description = ""
    if port_2_ntype is not None:
        for port, ntype in port_2_ntype.items():
            port_description += "* *" + port + "* : " + str(ntype)
            port_description += "\n\n"

    __doc__ = _normalize_docstring(wrapped.__doc__) + '\n\n' + str(port_description)
    __doc__ = _normalize_docstring(__doc__)

    wrapt.FunctionWrapper.__setattr__(decorated, "__doc__", __doc__)

    return decorated
