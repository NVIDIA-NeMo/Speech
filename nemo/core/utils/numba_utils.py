# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import logging as pylogger
import operator
import os

from typing import Tuple, Union

from nemo.utils import model_utils

# Prevent Numba CUDA logs from showing at info level
cuda_logger = pylogger.getLogger('numba.cuda.cudadrv.driver')
cuda_logger.setLevel(pylogger.ERROR)  # only show error

__NUMBA_DEFAULT_MINIMUM_VERSION__ = "0.53.0"
__NUMBA_MINIMUM_VERSION__ = os.environ.get("NEMO_NUMBA_MINVER", __NUMBA_DEFAULT_MINIMUM_VERSION__)

__NUMBA_MINIMUM_VERSION_FP16_SUPPORTED__ = "0.57.0"


NUMBA_INSTALLATION_MESSAGE = (
    "Could not import `numba`.\n"
    "Please install numba in one of the following ways."
    "1) If using conda, simply install it with conda using `conda install -c numba numba`\n"
    "2) If using pip (not recommended), `pip install --upgrade numba`\n"
    "followed by `export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/'` and \n"
    "`export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so'`.\n"
    "It is advised to always install numba using conda only, "
    "as pip installations might interfere with other libraries such as llvmlite.\n"
    "If pip install does not work, you can also try adding `--ignore-installed` to the pip command,\n"
    "but this is not advised."
)

STRICT_NUMBA_COMPAT_CHECK = True

# Get environment key if available
if 'STRICT_NUMBA_COMPAT_CHECK' in os.environ:
    check_str = os.environ.get('STRICT_NUMBA_COMPAT_CHECK')
    check_bool = str(check_str).lower() in ("yes", "true", "t", "1")
    STRICT_NUMBA_COMPAT_CHECK = check_bool


def is_numba_compat_strict() -> bool:
    """
    Returns strictness level of numba cuda compatibility checks.

    If value is true, numba cuda compatibility matrix must be satisfied.
    If value is false, only cuda availability is checked, not compatibility.
    Numba Cuda may still compile and run without issues in such a case, or it may fail.
    """
    return STRICT_NUMBA_COMPAT_CHECK


def set_numba_compat_strictness(strict: bool):
    """
    Sets the strictness level of numba cuda compatibility checks.

    If value is true, numba cuda compatibility matrix must be satisfied.
    If value is false, only cuda availability is checked, not compatibility.
    Numba Cuda may still compile and run without issues in such a case, or it may fail.

    Args:
        strict: bool value, whether to enforce strict compatibility checks or relax them.
    """
    global STRICT_NUMBA_COMPAT_CHECK
    STRICT_NUMBA_COMPAT_CHECK = strict


@contextlib.contextmanager
def with_numba_compat_strictness(strict: bool):
    initial_strictness = is_numba_compat_strict()
    set_numba_compat_strictness(strict=strict)
    yield
    set_numba_compat_strictness(strict=initial_strictness)


def numba_cpu_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed.

    Args:
        min_version: The minimum version of numba that is required.

    Returns:
        bool, whether numba CPU supported with this current installation or not.
    """
    module_available, msg = model_utils.check_lib_version('numba', checked_version=min_version, operator=operator.ge)

    # If numba is not installed
    if module_available is None:
        return False
    else:
        return True


def numba_cuda_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed, and if it is,
    if cuda is supported properly within it.

    Args:
        min_version: The minimum version of numba that is required.

    Returns:
        bool, whether cuda is supported with this current installation or not.
    """
    module_available = numba_cpu_is_supported(min_version)

    # If numba is not installed
    if module_available is None:
        return False

    # If numba version is installed and available
    if module_available is True:
        from numba import cuda

        try:
            cuda_available = cuda.is_available()
            if cuda_available:
                cuda_compatible = cuda.cudadrv.runtime.get_version()[0] == 13
            else:
                cuda_compatible = False

            if is_numba_compat_strict():
                return cuda_available and cuda_compatible
            else:
                return cuda_available

        except Exception:
            # dlopen(libcudart.dylib) might fail if CUDA was never installed in the first place.
            return False

    else:
        return False


def is_numba_cuda_fp16_supported(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Utility method that returns a bool, stating if FP16 is supported for numba cuda kernels or not.

    Returns:
        bool, whether Numba CUDA will support fp16 or not.
    """
    reason = ""
    use_nvidia_binding = os.environ.get('NUMBA_CUDA_USE_NVIDIA_BINDING', None)
    if use_nvidia_binding is not None:
        use_nvidia_binding = use_nvidia_binding.lower() == "1"
        reason += "Env variable `NUMBA_CUDA_USE_NVIDIA_BINDING` is available and set to `1`. "
    else:
        use_nvidia_binding = False
        reason += "Env variable `NUMBA_CUDA_USE_NVIDIA_BINDING` is not available or has not set to `1`."

    numba_fp16_version_correct = model_utils.check_lib_version(
        'numba', __NUMBA_MINIMUM_VERSION_FP16_SUPPORTED__, operator=operator.ge
    )[0]

    if numba_fp16_version_correct:
        reason += f"Numba CUDA FP16 is supported in installed numba version."
    else:
        reason += f"Numba CUDA FP16 is not supported in installed numba version."

    result = use_nvidia_binding and numba_fp16_version_correct

    if return_reason:
        return result, reason
    else:
        return result


def skip_numba_cuda_test_if_unsupported(min_version: str):
    """
    Helper method to skip pytest test case if numba cuda is not supported.

    Args:
        min_version: The minimum version of numba that is required.
    """
    numba_cuda_support = numba_cuda_is_supported(min_version)
    if not numba_cuda_support:
        import pytest

        pytest.skip(f"Numba cuda test is being skipped. Minimum version required : {min_version}")
