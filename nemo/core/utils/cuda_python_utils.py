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

import contextlib

import numpy as np
import torch
from packaging.version import Version

__CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__ = (12, 6)  # 12060


def check_cuda_python_cuda_graphs_conditional_nodes_supported():
    # for CPU-only environment we need to raise an exception, otherwise cuda-python library will fail
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available")

    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        raise ModuleNotFoundError("No `cuda-python` module. Please do `pip install cuda-python>=12.3`")

    from cuda.bindings import __version__ as cuda_python_version

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
    except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
        import pytest

        pytest.skip(
            "Test using cuda graphs with conditional nodes is being skipped because "
            f"cuda graphs with conditional nodes aren't supported. Error message: {e}"
        )


def assert_drv(err):
    """
    Throws an exception if the return value of a cuda-python call is not success.
    """
    from cuda.bindings import driver as cuda
    from cuda.bindings import nvrtc
    from cuda.bindings import runtime as cudart

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
    Makes calls to cuda-python's functions inside cuda.cuda more python by throwing an exception
    if they return a status which is not cudaSuccess
    """
    from cuda.bindings import runtime as cudart

    error, *others = f_call_out
    if error != cudart.cudaError_t.cudaSuccess:
        raise Exception(f"CUDA failure! {error}")
    else:
        return tuple(others)


@contextlib.contextmanager
def with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle, device):
    """
    Even though we add a conditional node only once, we need to
    capture the kernel that calls cudaGraphSetConditional() both
    before in the parent graph containing the while loop body graph
    and after the rest of the while loop body graph (because we need
    to decide both whether to enter the loop, and also whether to
    execute the next iteration of the loop).
    """
    from cuda.bindings import __version__ as cuda_python_version
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart

    capture_status, _, graph, _, _, _ = cu_call(
        cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=device).cuda_stream)
    )
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    cuda.cuLaunchKernel(
        while_loop_kernel,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        torch.cuda.current_stream(device=device).cuda_stream,
        while_loop_args.ctypes.data,
        0,
    )

    capture_status, _, graph, dependencies, _, _ = cu_call(
        cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=device).cuda_stream)
    )
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    driver_params = cuda.CUgraphNodeParams()
    driver_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    driver_params.conditional.handle = while_loop_conditional_handle
    driver_params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
    driver_params.conditional.size = 1
    if Version(cuda_python_version) == Version("12.3.0"):
        # Work around for https://github.com/NVIDIA/cuda-python/issues/55
        # Originally, cuda-python version 12.3.0 failed to allocate phGraph_out
        # on its own.
        # This bug is fixed in cuda-python version 12.4.0. In fact, we can
        # no longer write to phGraph_out in cuda-python 12.4.0, so we must
        # condition on the version number.
        driver_params.conditional.phGraph_out = [cuda.CUgraph()]
    (ctx,) = cu_call(cuda.cuCtxGetCurrent())
    driver_params.conditional.ctx = ctx

    # Use driver API here because of bug in cuda-python runtime API: https://github.com/NVIDIA/cuda-python/issues/55
    # TODO: Change call to this after fix goes in (and we bump minimum cuda-python version to 12.4.0):
    # node, = cu_call(cudart.cudaGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    (node,) = cu_call(cuda.cuGraphAddNode(graph, dependencies, None, len(dependencies), driver_params))
    body_graph = driver_params.conditional.phGraph_out[0]

    cu_call(
        cudart.cudaStreamUpdateCaptureDependencies(
            torch.cuda.current_stream(device=device).cuda_stream,
            [node],
            None,
            1,
            cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
        )
    )
    body_stream = torch.cuda.Stream(device)
    previous_stream = torch.cuda.current_stream(device=device)
    cu_call(
        cudart.cudaStreamBeginCaptureToGraph(
            body_stream.cuda_stream,
            body_graph,
            None,
            None,
            0,
            cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
        )
    )
    torch.cuda.set_stream(body_stream)

    yield body_stream, body_graph

    cuda.cuLaunchKernel(
        while_loop_kernel, 1, 1, 1, 1, 1, 1, 0, body_stream.cuda_stream, while_loop_args.ctypes.data, 0
    )

    cudart.cudaStreamEndCapture(body_stream.cuda_stream)

    torch.cuda.set_stream(previous_stream)


def run_nvrtc(kernel_string: str, kernel_name: bytes, program_name: bytes):
    from cuda.bindings import driver as cuda
    from cuda.bindings import nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), program_name, 0, [], [])
    assert_drv(err)
    # Compile program
    # Not specifying --gpu-architecture will default us to a fairly low compute capability, which is a safe bet.
    # Otherwise, there are ways to query the current device's compute capability.
    # https://stackoverflow.com/questions/48283009/nvcc-get-device-compute-capability-in-runtime
    opts = []
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert_drv(err)
    err, size = nvrtc.nvrtcGetProgramLogSize(prog)
    assert_drv(err)
    buf = b" " * size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, buf)
    assert_drv(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    assert_drv(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    assert_drv(err)

    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    assert_drv(err)
    err, kernel = cuda.cuModuleGetFunction(module, kernel_name)
    assert_drv(err)

    return kernel
