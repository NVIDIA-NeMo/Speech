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

import json
import logging
import os.path
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Union

import numpy

# tenosrstore is needed to register 'bfloat16' dtype with numpy for zarr compatibility
import tensorstore  # noqa: F401 pylint: disable=unused-import
import torch
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

from nemo.export.tarutils import TarPath, ZarrPathStore
from nemo.export.utils._mock_import import _mock_import

LOGGER = logging.getLogger("NeMo")


def nemo_to_path(nemo_checkpoint: Union[Path, str]) -> Union[Path, TarPath]:
    """
    Creates Path / TarPath object suitable for navigating inside the nemo checkpoint.

    Args:
        nemo_checkpoint (Path, str): Path to the NeMo checkpoint.
    Returns:
        Path | TarPath: Suitable Path object for navigating through the checkpoint.
    """
    string_path = str(nemo_checkpoint)

    if os.path.isdir(string_path):
        return Path(string_path)
    return TarPath(string_path)


class TarFileSystemReader(FileSystemReader):
    """Reader that accepts both Path and TarPath checkpoint directory.

    The FileSystemReader works with TarPath, but expects a pure Path.
    It's enough to skip the Path check in __init__.
    """

    def __init__(self, path: Union[Path, TarPath]) -> None:
        """Makes sure that super().__init__ gets a pure path as expected."""
        super_path = str(path) if isinstance(path, TarPath) else path
        super().__init__(super_path)
        if isinstance(path, TarPath):
            self.path = path  # overwrites path set in super().__init__ call


def load_sharded_metadata_torch_dist(
    checkpoint_dir: Union[Path, TarPath], load_extra_states: bool = False
) -> Dict[str, Any]:
    """
    Loads model state dictionary from torch_dist checkpoint.

    Args:
        checkpoint_dir (Path | TarPath): Path to the model weights directory.
        load_extra_states (bool): If set to true, loads BytesIO objects, related to the extra states.
    Returns:
        dict: Loaded model state dictionary (weights are stored in torch tensors).
    """
    fs_reader = TarFileSystemReader(checkpoint_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if isinstance(tp, TensorStorageMetadata)
    }

    if load_extra_states:
        state_dict.update(
            {k: [] for k, tp in metadata.state_dict_metadata.items() if isinstance(tp, BytesStorageMetadata)}
        )

    load(state_dict, storage_reader=fs_reader)
    return state_dict


def load_sharded_pickle_extra_state_scale(dir: Union[Path, TarPath]) -> Dict[str, BytesIO]:
    """
    Loads model extra states from the .pt shards.

    Args:
        dir (Path | TarPath): Path to the directory with sharded extra states.
    Returns:
        dict: State dictionary corresponding to the loaded extra states.
    """
    pt_files = list(dir.glob('shard_*_*.pt'))
    extra_states = {}
    for file in pt_files:
        shard_name = file.name.split('.')[0]
        with file.open('rb') as opened_file:
            extra_states[dir.name + '/' + shard_name] = torch.load(opened_file, weights_only=True)

    return extra_states


def contains_extra_states(subdir: Union[Path, TarPath]) -> bool:
    """
    Checks if zarr directory contains extra states.

    Args:
        subdir (Path | TarPath): Directory inside the zarr checkpoint.
    Returns:
        bool: Is a directory with extra states
    """
    return list(subdir.glob('shard_0_*.pt')) != []


def load_sharded_metadata_zarr(
    checkpoint_dir: Union[Path, TarPath], load_extra_states: bool = False
) -> Dict[str, Any]:
    """
    Loads model dictionary from the zarr format.

    Args:
        checkpoint_dir (Path | TarPath): Path to the NeMo checkpoint.
        load_extra_states (bool): If set to True, the function will load BufferIO objects with extra states.
    Returns:
        dict: Model state dictionary.
    """
    if load_extra_states:
        torch.serialization.add_safe_globals([BytesIO])

    sharded_state_dict = {}
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir():
            continue

        if load_extra_states and contains_extra_states(subdir):
            sharded_state_dict.update(load_sharded_pickle_extra_state_scale(subdir))

        elif (subdir / '.zarray').exists():
            key = subdir.name
            zstore = ZarrPathStore(subdir)

            import zarr

            arr = zarr.open(zstore, 'r')

            if arr.dtype.name == "bfloat16":
                sharded_state_dict[key] = torch.from_numpy(arr[:].view(numpy.int16)).view(torch.bfloat16)
            else:
                sharded_state_dict[key] = torch.from_numpy(arr[:])

    return sharded_state_dict


def nemo_weights_directory(nemo_path: Union[Path, TarPath]) -> Union[Path, TarPath]:
    """
    Returns a Path pointing to the weights directory inside the NeMo checkpoint.

    Args:
        nemo_path (Path | TarPath): Path to the nemo checkpoint.
    Returns:
        Path | TarPath: Path to the weights directory inside the model checkpoint.
    """
    if (nemo_path / "model_weights").exists():
        return nemo_path / "model_weights"

    if (nemo_path / "weights").exists():
        return nemo_path / "weights"

    return nemo_path


def load_model_weights(checkpoint_path: Union[str, Path], load_extra_states: bool = False) -> Dict[str, Any]:
    """
    Loads NeMo state dictionary. Weights are stored in torch.Tensor

    Args:
        checkpoint_path (str | Path): Path to the NeMo checkpoint.
        load_extra_states (bool): If True, loads BytesIO objects, corresponding to the extra states.
    Returns:
        dict: Model state dictionary.
    """

    nemo_path = nemo_to_path(checkpoint_path)
    nemo_weights = nemo_weights_directory(nemo_path)

    with (nemo_weights / 'metadata.json').open(mode='r') as f:
        config_dict = json.load(f)

    if config_dict['sharded_backend'] == 'zarr':
        return load_sharded_metadata_zarr(nemo_weights, load_extra_states=load_extra_states)
    elif config_dict['sharded_backend'] == 'torch_dist':
        # TODO: Remove mocking imports once MCore is available in NIM containers
        with _mock_import("megatron.core.dist_checkpointing.strategies.torch"):
            return load_sharded_metadata_torch_dist(nemo_weights, load_extra_states=load_extra_states)

    raise NotImplementedError(f'Distributed checkpoint backend {config_dict["sharded_backend"]} not supported')
