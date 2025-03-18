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
Module with guards for optional libraries, that cannot be listed in `requirements.txt`.
Provides helper constants and decorators to check if the library is available in the system.
"""

__all__ = ["KENLM_AVAILABLE", "K2_AVAILABLE", "TRITON_AVAILABLE", "kenlm_required", "k2_required", "triton_required"]

import importlib.util
from functools import wraps
from nemo.core.utils.k2_utils import K2_INSTALLATION_MESSAGE


def is_lib_available(name: str) -> bool:
    """
    Checks if the library/package with `name` is available in the system
    NB: try/catch with importlib.import_module(name) requires importing the library, which can be slow.
    So, `find_spec` should be preferred
    """
    return importlib.util.find_spec(name) is not None


KENLM_AVAILABLE = is_lib_available("kenlm")
KENLM_INSTALLATION_MESSAGE = "Try installing kenlm with `pip install kenlm`"

TRITON_AVAILABLE = is_lib_available("triton")
TRITON_INSTALLATION_MESSAGE = "Try installing triton with `pip install triton`"


try:
    from nemo.core.utils.k2_guard import k2 as _  # noqa: F401

    K2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    K2_AVAILABLE = False


def identity_decorator(f):
    """Identity decorator for further using in conditional decorators"""
    return f


def _lib_required(is_available: bool, name: str, message: str | None = None):
    """
    Decorator factory. Returns identity decorator if lib `is_available`,
    otherwise returns a decorator which returns a function that raises an error when called.
    Such decorator can be used for conditional checks for optional libraries in functions and methods
    with zero computational overhead.
    """
    if is_available:
        return identity_decorator

    # return wrapper that will raise an error when the function is called
    def function_stub_with_error_decorator(f):
        """Decorator that replaces the function and raises an error when called"""

        @wraps(f)
        def wrapper(*args, **kwargs):
            error_msg = f"Module {name} required for the function {f.__name__} is not found."
            if message:
                error_msg += f" {message}"
            raise ModuleNotFoundError(error_msg)

        return wrapper

    return function_stub_with_error_decorator


kenlm_required = _lib_required(is_available=KENLM_AVAILABLE, name="kenlm", message=KENLM_INSTALLATION_MESSAGE)
triton_required = _lib_required(is_available=TRITON_AVAILABLE, name="triton", message=TRITON_INSTALLATION_MESSAGE)
k2_required = _lib_required(is_available=K2_AVAILABLE, name="k2", message=K2_INSTALLATION_MESSAGE)
