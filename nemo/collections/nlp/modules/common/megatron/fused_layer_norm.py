# coding=utf-8
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


try:
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
    from apex.transformer.layers.layer_norm import FastLayerNorm

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def get_layer_norm(hidden_size, eps=1e-5, persist_layer_norm=False, sequence_parallel=False):
    # List of hiddens sizes supported in the persistent layer norm kernel
    # If the hidden size is not supported, fall back to the non-persistent
    # kernel.
    persist_ln_hidden_sizes = [
        1024,
        1536,
        2048,
        2304,
        3072,
        3840,
        4096,
        5120,
        6144,
        8192,
        10240,
        12288,
        12800,
        15360,
        16384,
        18432,
        20480,
        24576,
        25600,
        30720,
        32768,
        40960,
        49152,
        65536,
    ]
    if hidden_size not in persist_ln_hidden_sizes:
        persist_layer_norm = False

    if persist_layer_norm:
        return FastLayerNorm(hidden_size, eps, sequence_parallel_enabled=sequence_parallel)
    else:
        return MixedFusedLayerNorm(hidden_size, eps, sequence_parallel_enbaled=sequence_parallel)
