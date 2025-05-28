# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager, nullcontext
from typing import Any

import torch


def avoid_bfloat16_autocast_context():
    """
    If the current autocast context is bfloat16,
    cast it to float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
        return torch.amp.autocast('cuda', dtype=torch.float32)
    else:
        return nullcontext()


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast('cuda', dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast('cuda', dtype=torch.bfloat16)
        else:
            return torch.amp.autocast('cuda', dtype=torch.float32)
    else:
        return nullcontext()


def cast_tensor(x, from_dtype=torch.float16, to_dtype=torch.float32):
    return x.to(dtype=to_dtype) if x.dtype == from_dtype else x


def cast_all(x, from_dtype=torch.float16, to_dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return cast_tensor(x, from_dtype=from_dtype, to_dtype=to_dtype)
    else:
        if isinstance(x, dict):
            new_dict = {}
            for k in x.keys():
                new_dict[k] = cast_all(x[k], from_dtype=from_dtype, to_dtype=to_dtype)
            return new_dict
        elif isinstance(x, tuple):
            return tuple(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)


class CastToFloat(torch.nn.Module):
    def __init__(self, mod):
        super(CastToFloat, self).__init__()
        self.mod = mod

    def forward(self, x):
        if torch.is_autocast_enabled() and x.dtype != torch.float32:
            with torch.amp.autocast(x.device.type, enabled=False):
                ret = self.mod.forward(x.to(torch.float32)).to(x.dtype)
        else:
            ret = self.mod.forward(x)
        return ret


class CastToFloatAll(torch.nn.Module):
    def __init__(self, mod):
        super(CastToFloatAll, self).__init__()
        self.mod = mod

    def forward(self, *args):
        if torch.is_autocast_enabled():
            from_dtype = args[0].dtype
            with torch.amp.autocast(self.device.type, enabled=False):
                ret = self.mod.forward(*cast_all(args, from_dtype=from_dtype, to_dtype=torch.float32))
                return cast_all(ret, from_dtype=torch.float32, to_dtype=from_dtype)
        else:
            return self.mod.forward(*args)


@contextmanager
def monkeypatched(object, name, patch):
    """Temporarily monkeypatches an object."""
    pre_patched_value = getattr(object, name)
    setattr(object, name, patch)
    yield object
    setattr(object, name, pre_patched_value)


def maybe_cast_to_type(x: Any, type_: type) -> Any:
    """Try to cast a value to int, if it fails, return the original value.

    Args:
        x (Any): The value to be casted.
        type_ (type): The type to cast to, must be a callable.

    Returns:
        Any: The casted value or the original value if casting fails.
    """
    try:
        return type_(x)
    except Exception:
        return x
