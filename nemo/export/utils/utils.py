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

import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Union

import torch


def is_nemo2_checkpoint(checkpoint_path: str) -> bool:
    """
    Checks if the checkpoint is in NeMo 2.0 format.
    Args:
        checkpoint_path (str): Path to a checkpoint.
    Returns:
        bool: True if the path points to a NeMo 2.0 checkpoint; otherwise false.
    """

    ckpt_path = Path(checkpoint_path)
    return (ckpt_path / 'context').is_dir()


def prepare_directory_for_export(
    model_dir: Union[str, Path], delete_existing_files: bool, subdir: Optional[str] = None
) -> None:
    """
    Prepares model_dir path for the TensorRTT-LLM / vLLM export.
    Makes sure that the model_dir directory exists and is empty.

    Args:
        model_dir (str): Path to the target directory for the export.
        delete_existing_files (bool): Attempt to delete existing files if they exist.
        subdir (Optional[str]): Subdirectory to create inside the model_dir.

    Returns:
        None
    """
    model_path = Path(model_dir)

    if model_path.exists():
        if delete_existing_files:
            shutil.rmtree(model_path)
        elif any(model_path.iterdir()):
            raise RuntimeError(f"There are files in {model_path} folder: try setting delete_existing_files=True.")

    if subdir is not None:
        model_path /= subdir
    model_path.mkdir(parents=True, exist_ok=True)


def is_nemo_tarfile(path: str) -> bool:
    """
    Checks if the path exists and points to packed NeMo 1 checkpoint.

    Args:
        path (str): Path to possible checkpoint.
    Returns:
        bool: NeMo 1 checkpoint exists and is in '.nemo' format.
    """
    checkpoint_path = Path(path)
    return checkpoint_path.exists() and checkpoint_path.suffix == '.nemo'


# Copied from nemo.collections.nlp.parts.utils_funcs to avoid introducing extra NeMo dependencies:
def torch_dtype_from_precision(precision: Union[int, str], megatron_amp_O2: bool = True) -> torch.dtype:
    """
    Mapping from PyTorch Lighthing (PTL) precision types to corresponding PyTorch parameter data type.

    Args:
        precision (Union[int, str]): The PTL precision type used.
        megatron_amp_O2 (bool): A flag indicating if Megatron AMP O2 is enabled.

    Returns:
        torch.dtype: The corresponding PyTorch data type based on the provided precision.
    """
    if not megatron_amp_O2:
        return torch.float32

    if precision in ['bf16', 'bf16-mixed']:
        return torch.bfloat16
    elif precision in [16, '16', '16-mixed']:
        return torch.float16
    elif precision in [32, '32', '32-true']:
        return torch.float32
    else:
        raise ValueError(f"Could not parse the precision of '{precision}' to a valid torch.dtype")


def get_model_device_type(module: torch.nn.Module) -> str:
    """Find the device type the model is assigned to and ensure consistency."""
    # Collect device types of all parameters and buffers
    param_device_types = {param.device.type for param in module.parameters()}
    buffer_device_types = {buffer.device.type for buffer in module.buffers()}
    all_device_types = param_device_types.union(buffer_device_types)

    if len(all_device_types) > 1:
        raise ValueError(
            f"Model parameters and buffers are on multiple device types: {all_device_types}. "
            "Ensure all parameters and buffers are on the same device type."
        )

    # Return the single device type, or default to 'cpu' if no parameters or buffers
    return all_device_types.pop() if all_device_types else "cpu"


def get_example_inputs(tokenizer) -> Dict[str, torch.Tensor]:
    """Gets example data to feed to the model during ONNX export.

    Returns:
        Dictionary of tokenizer outputs.
    """
    example_inputs = dict(
        tokenizer(
            ["example query one", "example query two"],
            ["example passage one", "example passage two"],
            return_tensors="pt",
        )
    )

    return example_inputs


def validate_fp8_network(network) -> None:
    """Checks the network to ensure it's compatible with fp8 precison.

    Raises:
        ValueError if netowrk doesn't container Q/DQ FP8 layers
    """

    import tensorrt as trt

    quantize_dequantize_layers = []
    for layer in network:
        if layer.type in {trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE}:
            quantize_dequantize_layers.append(layer)
    if not quantize_dequantize_layers:
        error_msg = "No Quantize/Dequantize layers found"
        raise ValueError(error_msg)
    quantize_dequantize_layer_dtypes = Counter(layer.precision for layer in quantize_dequantize_layers)
    if trt.DataType.FP8 not in quantize_dequantize_layer_dtypes:
        error_msg = "Found Quantize/Dequantize layers. But none with FP8 precision."
        raise ValueError(error_msg)
