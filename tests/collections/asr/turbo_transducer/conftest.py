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

from nemo.core.utils.optional_libs import CUTILE_AVAILABLE, TRITON_AVAILABLE

if TRITON_AVAILABLE:
    import triton
    from triton.runtime.autotuner import Autotuner

    from nemo.collections.asr.parts.turbo_transducer import (
        rnnt_joint_fused_vocab_consistency_triton,
        rnnt_joint_fused_vocab_triton,
    )

if CUTILE_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer import rnnt_joint_fused_vocab_cutile

# Tile size (rows, hidden, vocab) of the kernel config used in tests, see `small_kernel_configs`
TEST_BLOCK_SIZE = 16


@pytest.fixture(scope="session", autouse=True)
def small_kernel_configs():
    """
    Compile the fused Joint kernels with a single small-tile config instead of the production configs.

    Triton autotuning compiles and benchmarks every production config (up to 10 per kernel, large tiles) for each
    new combination of `tl.constexpr` arguments (hidden dim, vocab size, precision flags, pruning range, ...).
    Without the Triton disk cache it takes from 10 seconds up to several minutes per combination, while the tests
    use dozens of them. With a single config Triton skips benchmarking, and a kernel with small tiles compiles in
    0.3-0.6 seconds; the same holds for the fixed production config of the cuTile kernels.
    Small tiles also split the small test tensors into several tiles along every dimension, so the tile boundaries
    are covered without large inputs.

    The production configs are covered by the tests requesting the `production_kernel_configs` fixture.
    They are slow without the cache, marked `nightly` and skipped unless `--nightly` is passed.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        _set_kernel_configs(monkeypatch, production=False)
        yield


@pytest.fixture
def production_kernel_configs(small_kernel_configs, monkeypatch):
    """Restore the production kernel configs for a test; returns the autotuned Triton kernels by name"""
    _set_kernel_configs(monkeypatch, production=True)
    return _get_autotuned_kernels()


# Production configs by the name of the kernel (Triton) or of the module attribute (cuTile),
# stored by `small_kernel_configs` before the first replacement
_PRODUCTION_CONFIGS = {}
_CUTILE_CONFIG_ATTRIBUTES = ("_FORWARD_CONFIG", "_BACKWARD_CONFIG")


def _set_kernel_configs(monkeypatch: pytest.MonkeyPatch, production: bool):
    for name, kernel in _get_autotuned_kernels().items():
        production_configs = _PRODUCTION_CONFIGS.setdefault(name, kernel.configs)
        configs = production_configs if production else [_make_small_triton_config(production_configs[0])]
        monkeypatch.setattr(kernel, "configs", configs)
    if CUTILE_AVAILABLE:
        for name in _CUTILE_CONFIG_ATTRIBUTES:
            production_config = _PRODUCTION_CONFIGS.setdefault(name, getattr(rnnt_joint_fused_vocab_cutile, name))
            config = production_config if production else (TEST_BLOCK_SIZE,) * len(production_config)
            monkeypatch.setattr(rnnt_joint_fused_vocab_cutile, name, config)


def _get_autotuned_kernels() -> dict:
    """All autotuned Triton kernels of the fused Joint modules: `{kernel name: autotuner}`"""
    if not TRITON_AVAILABLE:
        return {}
    return {
        name: kernel
        for module in (rnnt_joint_fused_vocab_triton, rnnt_joint_fused_vocab_consistency_triton)
        for name, kernel in vars(module).items()
        if isinstance(kernel, Autotuner)
    }


def _make_small_triton_config(production_config):
    """Copy of the config with all tile sizes (`*_BLOCK` meta-parameters) replaced by the small test tile size"""
    kwargs = {
        name: TEST_BLOCK_SIZE if name.endswith("_BLOCK") else value for name, value in production_config.kwargs.items()
    }
    return triton.Config(kwargs, num_warps=4, num_stages=2)
