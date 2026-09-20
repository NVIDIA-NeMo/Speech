# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _stub_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_te1_quantize_param_fragment_uses_function_arguments(monkeypatch):
    class DistributedFusedAdam:
        class ParameterBucket:
            pass

        class ParameterFragment:
            pass

        pass

    class ShardedTensor:
        pass

    captured = {}

    def cast_to_fp8(input_, scaling, index, fp8_dtype, *, out):
        captured.update(input=input_, scaling=scaling, index=index, fp8_dtype=fp8_dtype, out=out)

    apex = _stub_module(monkeypatch, "apex")
    apex.__path__ = []
    apex_contrib = _stub_module(monkeypatch, "apex.contrib")
    apex_contrib.__path__ = []
    apex_optimizers = _stub_module(monkeypatch, "apex.contrib.optimizers")
    apex_optimizers.__path__ = []
    _stub_module(
        monkeypatch,
        "apex.contrib.optimizers.distributed_fused_adam",
        DistributedFusedAdam=DistributedFusedAdam,
        _disable_pre_forward_hook=lambda parameter: parameter,
        _multi_tensor_copy=lambda *args, **kwargs: None,
    )
    _stub_module(monkeypatch, "apex.contrib.nccl_allocator")

    megatron = _stub_module(monkeypatch, "megatron")
    megatron.__path__ = []
    megatron_core = _stub_module(monkeypatch, "megatron.core", parallel_state=types.SimpleNamespace())
    megatron_core.__path__ = []
    dist_checkpointing = _stub_module(monkeypatch, "megatron.core.dist_checkpointing")
    dist_checkpointing.__path__ = []
    _stub_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.dict_utils",
        dict_list_map_inplace=lambda *args, **kwargs: None,
    )
    _stub_module(monkeypatch, "megatron.core.dist_checkpointing.mapping", ShardedTensor=ShardedTensor)
    _stub_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.optimizer",
        get_param_id_to_sharded_param_map=lambda *args, **kwargs: {},
        optim_state_to_sharding_state=lambda *args, **kwargs: None,
    )

    transformer_engine = _stub_module(monkeypatch, "transformer_engine")
    transformer_engine.__path__ = []
    transformer_engine_pytorch = _stub_module(monkeypatch, "transformer_engine.pytorch")
    transformer_engine_pytorch.__path__ = []
    _stub_module(monkeypatch, "transformer_engine.pytorch.cpp_extensions", cast_to_fp8=cast_to_fp8)
    _stub_module(
        monkeypatch,
        "nemo.utils.te_utils",
        is_float8tensor=lambda tensor: False,
        is_mxfp8tensor=lambda tensor: False,
        te_version=lambda: (1, 0),
    )

    module_name = "_test_distributed_adam_te1"
    module_path = Path(__file__).parents[2] / "nemo/core/optim/distributed_adam.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    input_ = torch.arange(4, dtype=torch.float32)
    out = torch.empty(4, dtype=torch.uint8)
    scaling = object()
    fp8_dtype = object()
    param = types.SimpleNamespace(
        _fp8_meta={"scaling_fwd": scaling},
        _fp8_meta_index=2,
        _fp8_dtype=fp8_dtype,
    )

    module.quantize_param_fragment(input_, out=out, param=param)

    assert captured["input"].data_ptr() == input_.data_ptr()
    assert captured["out"].data_ptr() == out.data_ptr()
    assert captured["scaling"] is scaling
    assert captured["index"] == 2
    assert captured["fp8_dtype"] is fp8_dtype
