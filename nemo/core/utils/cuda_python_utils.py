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

from packaging.version import Version

__CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__ = (12, 3)  # 12030


def check_cuda_python_cuda_graphs_conditional_nodes_supported():
    try:
        from cuda import cuda
    except ImportError:
        raise ModuleNotFoundError("Please do `pip install cuda-python>=12.3`")

    from cuda import __version__ as cuda_python_version

    if Version(cuda_python_version) < Version("12.3.0"):
        raise ImportError(f"Found cuda-python {cuda_python_version}, but at least version 12.3.0 is needed.")

    error, driver_version = cuda.cuDriverGetVersion()
    if error != cuda.CUresult.CUDA_SUCCESS:
        raise ImportError(f"cuDriverGetVersion() returned {cuda.cuGetErrorString(error)}")

    driver_version_major = driver_version // 1000
    driver_version_minor = (driver_version % 1000) // 10

    driver_version = (driver_version_major, driver_version_minor)
    if driver_version < __CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__:
        required_version = __CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__
        raise ImportError(
            f"""Driver supports cuda toolkit version \
{driver_version_major}.{driver_version_minor}, but the driver needs to support \
at least {required_version[0]},{required_version[1]}. Please update your cuda driver."""
        )


def skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported():
    """
    Helper method to skip pytest test case if cuda graph conditionals nodes are not supported.
    """
    try:
        check_cuda_python_cuda_graphs_conditional_nodes_supported()
    except (ImportError, ModuleNotFoundError) as e:
        import pytest

        pytest.skip(
            f"Test using cuda graphs with conditional nodes is being skipped because cuda graphs with conditional nodes aren't supported. Error message: {e}"
        )


def assert_drv(err):
    """
    Throws an exception if the return value of a cuda-python call is not success.
    """
    from cuda import cuda, cudart, nvrtc

    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cu_call(f_call_out):
    """
    Makes calls to cuda-python's functions inside cuda.cuda more python by throwing an exception if they return a status which is not cudaSuccess
    """
    from cuda import cudart

    error, *others = f_call_out
    if error != cudart.cudaError_t.cudaSuccess:
        raise Exception(f"CUDA failure! {error}")
    else:
        return tuple(others)
