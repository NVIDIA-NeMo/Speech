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

"""
Megatron Model Parallel Initialization
"""

import os

import megatron.core.parallel_state as ps
import torch


# pylint: disable=C0116
class Utils:
    world_size = torch.cuda.device_count()
    # rank = int(os.environ["LOCAL_RANK"])
    rank = 0

    @staticmethod
    def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1):
        ps.destroy_model_parallel()

        # Torch setup for distributed training
        rank = int(os.environ['LOCAL_RANK'])
        world_size = 1  # torch.cuda.device_count()
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(world_size=world_size, rank=rank)

        # Megatron core distributed training initialization
        ps.initialize_model_parallel(
            tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size=context_parallel_size
        )

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if torch.distributed.is_initialized() and Utils.world_size != torch.distributed.get_world_size():
            torch.distributed.destroy_process_group()

        if rank is None:
            # Utils.rank = int(os.environ["LOCAL_RANK"])
            Utils.rank = 0
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        **kwargs,
    ):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            **kwargs,
        )


# pylint: disable=C0116
