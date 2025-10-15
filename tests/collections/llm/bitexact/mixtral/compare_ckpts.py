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

import sys

import torch


def load_dcp(ckpt_dir):
    from pathlib import Path

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == 'TensorStorageMetadata'
    }

    dcp.load(
        state_dict,
        storage_reader=fs_reader,
    )
    return state_dict


def compare_ckpts(a, b, key=''):
    if isinstance(a, dict):
        assert isinstance(b, dict)
        extra_a_keys = a.keys() - b.keys()
        extra_b_keys = b.keys() - a.keys()
        if extra_a_keys or extra_b_keys:
            diff_msg = f"Extra left keys: {extra_a_keys}. Extra right keys: {extra_b_keys}."
            if all(k.startswith("optimizer") for k in extra_a_keys | extra_b_keys):
                print(f"Optimizer keys mismatch (most likely indicating different formats): {diff_msg}")
            else:
                raise RuntimeError(f"Checkpoint keys mismatch: {diff_msg}")
        for key in a.keys() - extra_a_keys:
            compare_ckpts(a[key], b[key], key)
    elif isinstance(a, torch.Tensor):
        try:
            assert a.dtype == b.dtype, f"mismatch\t{key}: different dtypes {a.dtype} {b.dtype}"
            assert a.device == b.device, f"mismatch\t{key}: different device {a.device} {b.device}"
            assert a.shape == b.shape, f"mismatch\t{key}: different shape {a.shape} {b.shape}"
            assert torch.all(a == b), f"mismatch\t{key}: different values {key}\n{a}\n{b}"
            print(f'match\t{key}')
        except Exception as e:
            print(e)
    else:
        print(key, '\t', type(a), '\t', type(b))


def remove_module_from_key(x):
    # module.decoder.layers.mlp.router.weight -> decoder.layers.mlp.router.weight
    # optimizer.state.fp32_param.module.output.weight -> optimizer.state.fp32_param.output.weight
    assert isinstance(x, str)
    return '.'.join(filter(lambda x: x != 'module', x.split('.')))


def remove_module_from_dict_keys(d):
    assert isinstance(d, dict)
    return {remove_module_from_key(k): v for k, v in d.items()}


if __name__ == "__main__":
    load_n_rename = lambda x: remove_module_from_dict_keys(load_dcp(x))
    ckpt = load_n_rename(sys.argv[1])
    ckpt2 = load_n_rename(sys.argv[2])
    compare_ckpts(ckpt, ckpt2)
