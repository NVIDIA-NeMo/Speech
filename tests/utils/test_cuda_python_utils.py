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

import enum
import importlib
import sys
from types import ModuleType


class _DummyNvrtcResult(enum.IntEnum):
    """Subset of cuda-python nvrtc result enum used by the helper."""

    NVRTC_SUCCESS = 0


class _DummyCudaResult(enum.IntEnum):
    """Subset of cuda-python driver error enum used by the helper."""

    CUDA_SUCCESS = 0


class _DummyCudartResult(enum.IntEnum):
    """Subset of cuda-python cudart error enum used by the helper."""

    cudaSuccess = 0


def test_run_nvrtc_uses_mutable_log_buffer(monkeypatch):
    # Keep test behavior local by injecting fake cuda-python bindings.
    get_program_log_calls = {"called": False}

    def fake_nvrtc_create_program(*_args):
        return _DummyNvrtcResult.NVRTC_SUCCESS, "prog"

    def fake_nvrtc_compile_program(*_args):
        return (_DummyNvrtcResult.NVRTC_SUCCESS,)

    def fake_nvrtc_get_program_log_size(*_args):
        return _DummyNvrtcResult.NVRTC_SUCCESS, 1

    def fake_nvrtc_get_program_log(*_args):
        _, buf = _args
        get_program_log_calls["called"] = True
        assert isinstance(buf, (bytearray, memoryview))
        if len(buf) > 0:
            buf[0] = 42
        return (_DummyNvrtcResult.NVRTC_SUCCESS,)

    def fake_nvrtc_get_ptx_size(*_args):
        return _DummyNvrtcResult.NVRTC_SUCCESS, 4

    def fake_nvrtc_get_ptx(*_args):
        return (_DummyNvrtcResult.NVRTC_SUCCESS,)

    def fake_cuda_module_load_data(*_args):
        return _DummyCudaResult.CUDA_SUCCESS, "module"

    def fake_cuda_module_get_function(*_args):
        return _DummyCudaResult.CUDA_SUCCESS, "kernel"

    fake_nvrtc = ModuleType("cuda.bindings.nvrtc")
    fake_nvrtc.nvrtcResult = _DummyNvrtcResult
    fake_nvrtc.nvrtcCreateProgram = fake_nvrtc_create_program
    fake_nvrtc.nvrtcCompileProgram = fake_nvrtc_compile_program
    fake_nvrtc.nvrtcGetProgramLogSize = fake_nvrtc_get_program_log_size
    fake_nvrtc.nvrtcGetProgramLog = fake_nvrtc_get_program_log
    fake_nvrtc.nvrtcGetPTXSize = fake_nvrtc_get_ptx_size
    fake_nvrtc.nvrtcGetPTX = fake_nvrtc_get_ptx

    fake_driver = ModuleType("cuda.bindings.driver")
    fake_driver.CUresult = _DummyCudaResult
    fake_driver.cuModuleLoadData = fake_cuda_module_load_data
    fake_driver.cuModuleGetFunction = fake_cuda_module_get_function

    fake_runtime = ModuleType("cuda.bindings.runtime")
    fake_runtime.cudaError_t = _DummyCudartResult

    fake_bindings = ModuleType("cuda.bindings")
    fake_bindings.__version__ = "12.6.0"
    fake_bindings.driver = fake_driver
    fake_bindings.runtime = fake_runtime
    fake_bindings.nvrtc = fake_nvrtc

    fake_cuda = ModuleType("cuda")
    fake_cuda.bindings = fake_bindings

    monkeypatch.setitem(sys.modules, "cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", fake_bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", fake_driver)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", fake_nvrtc)

    import nemo.core.utils.cuda_python_utils as cuda_python_utils
    import nemo.core.utils.optional_libs as optional_libs

    monkeypatch.setattr(optional_libs, "CUDA_PYTHON_AVAILABLE", True, raising=False)
    monkeypatch.setattr(optional_libs, "cuda_python_required", lambda func: func, raising=False)

    cuda_python_utils = importlib.reload(cuda_python_utils)

    kernel = cuda_python_utils.run_nvrtc("kernel", b"kernel_name", b"kernel")
    assert kernel == "kernel"
    assert get_program_log_calls["called"] is True
