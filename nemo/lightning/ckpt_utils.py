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

from pathlib import Path
from typing import Union

# NeMo2 checkpoint structure is a checkpoint directory, with a WEIGHTS_PATH and CONTEXT_PATH subdirectory structure.
#  WEIGHTS_PATH stores the weights while CONTEXT_PATH stores the hyper-parameters.
WEIGHTS_PATH: str = "weights"
CONTEXT_PATH: str = "context"
ADAPTER_META_FILENAME = "adapter_metadata.json"

# When saving checkpoints/adapters in HF format we use directories starting with "hf_".
HF_WEIGHTS_PATH: str = "hf_weights"
HF_ADAPTER_PATH: str = "hf_adapter"
HF_ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def idempotent_path_append(base_dir: Union[str, Path], suffix) -> Path:
    """Appends a given suffix to a base directory path only if it is not already present.

    This function takes a base directory (either a string or Path) and ensures that
    the suffix is appended to the path. If the base directory is an AdapterPath instance,
    it also appends the suffix to the AdapterPath's base_model_path if the suffix
    is not already part of that path.

    Args:
        base_dir (Union[str, Path]): The base directory or path object.
        suffix (str): The suffix to append to the base directory.

    Returns:
        Path: The updated path object with the suffix appended if it was not already present.
    """
    from nemo.lightning.resume import AdapterPath
    from nemo.utils.msc_utils import import_multistorageclient, is_multistorageclient_url

    if is_multistorageclient_url(base_dir):
        msc = import_multistorageclient()
        base_dir = msc.Path(base_dir)
    else:
        base_dir = Path(base_dir)

    if base_dir.parts[-1] != suffix:
        base_dir = base_dir / suffix
    if isinstance(base_dir, AdapterPath) and base_dir.base_model_path.parts[-1] != suffix:
        base_dir.base_model_path = base_dir.base_model_path / suffix
    return base_dir


def ckpt_to_context_subdir(filepath: Union[str, Path]) -> Path:
    """Given an input checkpoint filepath, clean it using `ckpt_to_dir` and then return the context subdirectory."""
    base_dir = ckpt_to_dir(filepath=filepath)
    return idempotent_path_append(base_dir, CONTEXT_PATH)


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints
    """
    from nemo.lightning.resume import AdapterPath
    from nemo.utils.msc_utils import import_multistorageclient, is_multistorageclient_url

    if isinstance(filepath, AdapterPath):
        return filepath

    if is_multistorageclient_url(filepath):
        msc = import_multistorageclient()
        filepath = msc.Path(filepath)
    else:
        filepath = Path(filepath)

    if not filepath.suffix == ".ckpt":
        filepath = filepath.with_suffix(filepath.suffix + ".ckpt")

    # adding this assert because we will later remove directories based on the return value of this method
    assert filepath.suffix == ".ckpt", f"filepath: {filepath} must have .ckpt extension"

    # create a new path whose name is the original filepath without the .ckpt extension
    checkpoint_dir = filepath.with_name(filepath.stem)

    return checkpoint_dir
