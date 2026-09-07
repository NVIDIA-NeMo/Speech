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

import pytest


def test_run_nvrtc_uses_mutable_output_buffers(monkeypatch):
    """Regression test for the ``run_nvrtc`` interned-bytes-singleton corruption.

    ``run_nvrtc`` used to allocate its NVRTC output buffers as ``b" " * size``. NVRTC writes the
    compile log / PTX (a C string, including its NUL terminator) straight into that buffer, but a
    ``bytes`` object is immutable. Worse, for an empty compile log ``size == 1`` and CPython's
    ``b" " * 1`` returns the interned, process-wide 1-byte singleton ``b" "`` -- so NVRTC's NUL
    write would turn every ``b" "`` / ``bytes([32])`` / 1-byte space slice in the whole process
    into ``b"\\x00"`` permanently.

    The fix allocates the buffers as ``bytearray(size)`` (mutable and never interned). This test
    fakes the NVRTC/CUDA bindings -- no GPU required -- and emulates NVRTC writing into the buffer
    via ``memoryview`` (which fails on an immutable ``bytes`` buffer, exactly as the old code was
    broken) to assert the buffers handed to NVRTC are mutable and the interned singleton is intact.
    """
    from nemo.core.utils import cuda_python_utils

    if not cuda_python_utils.CUDA_PYTHON_AVAILABLE:
        pytest.skip("cuda-python is required to test run_nvrtc")

    captured = {}
    fake_ptx = b"//fake-ptx"

    class FakeNvrtc:
        def nvrtcCreateProgram(self, src, name, num_headers, headers, include_names):
            return (0, "prog")

        def nvrtcCompileProgram(self, prog, num_opts, opts):
            return (0,)

        def nvrtcGetProgramLogSize(self, prog):
            # Empty log -> size 1 -> the buggy `b" " * 1` would return the interned singleton.
            return (0, 1)

        def nvrtcGetProgramLog(self, prog, buf):
            captured["log_buf"] = buf
            # Emulate NVRTC writing the C-string NUL terminator; raises on an immutable bytes buffer.
            memoryview(buf)[-1] = 0
            return (0,)

        def nvrtcGetPTXSize(self, prog):
            return (0, len(fake_ptx) + 1)

        def nvrtcGetPTX(self, prog, ptx):
            captured["ptx_buf"] = ptx
            mv = memoryview(ptx)
            mv[: len(fake_ptx)] = fake_ptx
            mv[-1] = 0
            return (0,)

    class FakeCuda:
        def cuModuleLoadData(self, ptr):
            return (0, "module")

        def cuModuleGetFunction(self, module, name):
            return (0, "kernel")

    monkeypatch.setattr(cuda_python_utils, "assert_drv", lambda err: None)
    monkeypatch.setattr(cuda_python_utils, "nvrtc", FakeNvrtc())
    monkeypatch.setattr(cuda_python_utils, "cuda", FakeCuda())

    sentinel = b" "  # the interned 1-byte singleton the bug would clobber process-wide

    kernel = cuda_python_utils.run_nvrtc('extern "C" __global__ void k(){}\n', b"k", b"k.cu")

    assert kernel == "kernel"
    # NVRTC output buffers must be mutable; the old `b" " * size` produced an immutable bytes
    # object (and, for size == 1, the shared interned singleton).
    assert isinstance(captured["log_buf"], bytearray)
    assert isinstance(captured["ptx_buf"], bytearray)
    # The interned 1-byte space must be untouched.
    assert sentinel == b" " and sentinel[0] == 0x20
    assert bytes([32]) == b" "
