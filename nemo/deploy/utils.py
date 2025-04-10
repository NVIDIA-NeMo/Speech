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

import os
import typing
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytriton.model_config import Tensor

from nemo.export.tarutils import TarPath

NEMO2 = "NEMO 2.0"
NEMO1 = "NEMO 1.0"


def typedict2tensor(
    typedict_class,
    overwrite_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    defaults: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    """Converts a type dictionary class into a tuple of PyTriton Tensor objects.

    This function takes a class with type hints and converts each typed field into a PyTriton
    Tensor specification, handling nested list types and mapping Python types to numpy dtypes.

    Args:
        typedict_class: A class with type hints that will be converted to Tensor specs
        overwrite_kwargs: Optional dictionary of kwargs to override default Tensor parameters
        defaults: Optional dictionary of default values (unused)

    Returns:
        tuple: A tuple of PyTriton Tensor objects, one for each typed field in the input class

    Raises:
        Exception: If an unsupported type is encountered during type mapping
    """

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


def nemo_checkpoint_version(path: str) -> str:
    """Determines the version of a NeMo checkpoint from its file structure.

    Examines the provided checkpoint path to determine if it follows the NeMo 2.0
    or NeMo 1.0 format based on the presence of 'context' and 'weights' directories.

    Args:
        path (str): Path to the NeMo checkpoint file or directory

    Returns:
        str: Version string - either NEMO2 or NEMO1 constant indicating the checkpoint version
    """

    if os.path.isdir(path):
        path = Path(path)
    else:
        path = TarPath(path)

    if (path / "context").exists() and (path / "weights").exists():
        return NEMO2
    else:
        return NEMO1


def str_list2numpy(str_list: typing.List[str]) -> np.ndarray:
    """Converts a list of strings to a numpy array of UTF-8 encoded bytes.

    Takes a list of strings and converts it to a numpy array with an additional
    dimension, then encodes the strings as UTF-8 bytes.

    Args:
        str_list (List[str]): List of strings to convert

    Returns:
        np.ndarray: Numpy array of UTF-8 encoded bytes with shape (N, 1) where N is
            the length of the input list
    """
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def str_ndarray2list(str_ndarray: np.ndarray) -> typing.List[str]:
    """Converts a numpy array of UTF-8 encoded bytes back to a list of strings.

    Takes a numpy array of UTF-8 encoded bytes and decodes it back to strings,
    removing any extra dimensions, and returns the result as a Python list.

    Args:
        str_ndarray (np.ndarray): Numpy array of UTF-8 encoded bytes, typically
            with shape (N, 1) where N is the length of the resulting list

    Returns:
        List[str]: List of decoded strings
    """
    str_ndarray = str_ndarray.astype("bytes")
    str_ndarray = np.char.decode(str_ndarray, encoding="utf-8")
    str_ndarray = str_ndarray.squeeze(axis=-1)
    return str_ndarray.tolist()


def ndarray2img(img_ndarray: np.ndarray) -> typing.List[Image.Image]:
    """Converts a numpy array of images to a list of PIL Image objects.

    Takes a numpy array containing one or more images and converts each image
    to a PIL Image object using Image.fromarray().

    Args:
        img_ndarray (np.ndarray): Numpy array of images, where each image is a 2D or 3D array
            representing pixel values

    Returns:
        List[Image.Image]: List of PIL Image objects created from the input array
    """

    img_list = [Image.fromarray(i) for i in img_ndarray]
    return img_list


def cast_output(data, required_dtype):
    """Casts input data to a numpy array with the required dtype.

    Takes input data that may be a torch.Tensor, numpy array, or other sequence type
    and converts it to a numpy array with the specified dtype. For string dtypes,
    the data is encoded as UTF-8 bytes. The output array is ensured to have at least
    2 dimensions.

    Args:
        data: Input data to cast. Can be a torch.Tensor, numpy array, or sequence type
            that can be converted to a numpy array.
        required_dtype: The desired numpy dtype for the output array.

    Returns:
        np.ndarray: A numpy array containing the input data cast to the required dtype,
            with at least 2 dimensions.
    """

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


def broadcast_list(data, src=0, group=None):
    """Broadcasts a list of text data to all processes.

    Args:
        data (list): List of strings to broadcast.
        src (int, optional): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used.
    """

    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    object_list = [data] if torch.distributed.get_rank() == src else [None]
    torch.distributed.broadcast_object_list(object_list, src=src, group=group)
    return object_list[0]
