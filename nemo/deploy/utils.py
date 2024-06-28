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

import typing

import numpy as np
import torch
from PIL import Image
from pytriton.model_config import Tensor


def typedict2tensor(
    typedict_class,
    overwrite_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    defaults: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    def _map_type(type_):
        if type_ is int:
            return np.int32
        elif type_ is float:
            return np.float32
        elif type_ is bool:
            return np.bool_
        elif type_ is str:
            return bytes
        else:
            raise Exception(f"Unknown type {type_}")

    def _get_tensor_params(type_):
        count = 0
        while typing.get_origin(type_) is list:
            type_ = typing.get_args(type_)[0]
            count += 1
        count -= 1  # we don't want to count the last dimension
        shape = (-1,) * count if count > 1 else (1,)
        return {"shape": shape, "dtype": _map_type(type_)}

    overwrite_kwargs = overwrite_kwargs or {}
    return tuple(
        Tensor(name=name, **_get_tensor_params(type_), **overwrite_kwargs)
        for name, type_ in typing.get_type_hints(typedict_class).items()
    )


def str_list2numpy(str_list: typing.List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def str_ndarray2list(str_ndarray: np.ndarray) -> typing.List[str]:
    str_ndarray = str_ndarray.astype("bytes")
    str_ndarray = np.char.decode(str_ndarray, encoding="utf-8")
    str_ndarray = str_ndarray.squeeze(axis=-1)
    return str_ndarray.tolist()


def ndarray2img(img_ndarray: np.ndarray) -> typing.List[Image.Image]:
    img_list = [Image.fromarray(i) for i in img_ndarray]
    return img_list


def cast_output(data, required_dtype):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

        data_is_str = required_dtype in (object, np.object_, bytes, np.bytes_)
        if data_is_str:
            data = np.char.encode(data, "utf-8")

    if data.ndim < 2:
        data = data[..., np.newaxis]
    return data.astype(required_dtype)
