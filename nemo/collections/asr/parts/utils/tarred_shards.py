# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Sharded (tarred) dataset path expansion, kept free of dataset imports so that any module can use it."""
import braceexpand

from nemo.utils import logging


def expand_sharded_filepaths(sharded_filepaths, shard_strategy: str, world_size: int, global_rank: int):
    valid_shard_strategies = ['scatter', 'replicate']
    if shard_strategy not in valid_shard_strategies:
        raise ValueError(f"`shard_strategy` must be one of {valid_shard_strategies}")

    if isinstance(sharded_filepaths, str):
        # Replace '(' and '[' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in sharded_filepaths:
                sharded_filepaths = sharded_filepaths.replace(bkey, "{")

        # Replace ')' and ']' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in sharded_filepaths:
                sharded_filepaths = sharded_filepaths.replace(bkey, "}")

    if isinstance(sharded_filepaths, str):
        # Brace expand, set escape=False for Windows compatibility
        sharded_filepaths = list(braceexpand.braceexpand(sharded_filepaths, escape=False))

    # Check for distributed and partition shards accordingly
    if world_size > 1:
        if shard_strategy == 'scatter':
            logging.info("All tarred dataset shards will be scattered evenly across all nodes.")

            if len(sharded_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(sharded_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(sharded_filepaths) // world_size) * global_rank
            end_idx = begin_idx + len(sharded_filepaths) // world_size
            sharded_filepaths = sharded_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
        else:
            raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

    return sharded_filepaths
