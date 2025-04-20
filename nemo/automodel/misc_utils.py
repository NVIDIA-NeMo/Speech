#!/usr/bin/python3
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


def calculate_valid_accumulate_grad_batches(
    global_batch_size: int,
    micro_batch_size: int,
    devices: int,
    num_nodes: int,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
) -> int:
    """Calculate valid gradient accumulation steps based on the given parameters.

    Args:
        global_batch_size (int): The desired global batch size
        micro_batch_size (int): The micro batch size per GPU
        devices (int): Number of GPUs per node
        num_nodes (int): Number of nodes
        tp_size (int, optional): Tensor parallel size. Defaults to 1.
        pp_size (int, optional): Pipeline parallel size. Defaults to 1.
        cp_size (int, optional): Context parallel size. Defaults to 1.

    Returns:
        int: The calculated gradient accumulation steps

    Raises:
        ValueError: If the parameters result in invalid configuration
    """

    if any(x <= 0 for x in [global_batch_size, micro_batch_size, devices, num_nodes, tp_size, pp_size, cp_size]):
        raise ValueError("All parameters must be positive")

    # Calculate world size and validate divisibility
    world_size = devices * num_nodes
    model_parallel_size = tp_size * pp_size * cp_size

    if world_size % model_parallel_size != 0:
        raise ValueError(f"World size ({world_size}) must be divisible by model parallel size ({model_parallel_size})")

    # Calculate data parallel size
    data_parallel_size = world_size // model_parallel_size

    # Calculate accumulate_grad_batches
    accumulate_grad_batches = global_batch_size / (micro_batch_size * data_parallel_size)

    # Validate the result
    if not accumulate_grad_batches.is_integer():
        raise ValueError(
            f"Invalid configuration: global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size * data_parallel_size ({micro_batch_size * data_parallel_size})"
        )

    return int(accumulate_grad_batches)
