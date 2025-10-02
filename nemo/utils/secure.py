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

import safetensors
import safetensors.torch as storch
import torch

from nemo.utils import logging

SAFE_EXTENSION = '.safetensor'


def torch_load(filename: str, map_location: str = 'cpu', weights_only: bool = True, safe: bool = False):
    """
    load PyTorch files, for backwards compatability it loads insecurely
    by default because files have to be saved securely to work perfectly,
    however when loading untrusted files, safe should be set to True.

    Args:
        filename: the file to load
        map_location: the device to load the file onto
        safe: whether to load the file with safetensors
    Returns:
        the loaded tensors
    """
    try:
        return storch.load_file(filename + SAFE_EXTENSION, device=map_location)
    except FileNotFoundError as e:
        if safe:
            raise e
        else:
            logging.info(e)

    return torch.load(filename, map_location=map_location, weights_only=weights_only)


def torch_save(tensors, filename: str, safe: bool = False):
    """
    save replacement for pytorch. For backwards compatability it is
    unsafe by default. This causes a pytorch and safetensors file
    to be created. When saving the secure method, only the safetensor
    will be created, however this won't work with old versions of nemo

    Args:
        tensors: the tensors to save
        filename: the file to save to
        safe: whether to save with safetensors
    Returns:
        None
    """
    try:
        storch.save_file(tensors, filename + SAFE_EXTENSION)
    except safetensors.SafetensorError as e:
        if safe:
            raise e
        else:
            logging.info(e)

    # always save when safe is false in order to allow backwards compatability.
    # it will only be used if safe is also false for the load.
    if not safe:
        torch.save(tensors, filename)
