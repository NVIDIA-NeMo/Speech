# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel


def old_fake_initialize_model_parallel(
    world_size,
    rank,
    tensor_model_parallel_size_,
    pipeline_model_parallel_size_,
    pipeline_model_parallel_split_rank_=None,
    virtual_pipeline_model_parallel_size_=None,
    expert_model_parallel_size_=1,
    context_parallel_size_=1,
):
    # Get world size and rank. Ensure some consistencies.
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    context_parallel_size = min(context_parallel_size_, world_size)

    assert (
        world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size) == 0
    ), f'world_size: {world_size} must be divisible by tensor_model_parallel_size: {tensor_model_parallel_size} times pipeline_model_parallel_size {pipeline_model_parallel_size} times context_parallel_size {context_parallel_size}'
    data_parallel_size = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    virtual_pipeline_model_parallel_rank = None
    if virtual_pipeline_model_parallel_size_ is not None:
        virtual_pipeline_model_parallel_rank = 0

    # Build the tensor model-parallel groups.
    tensor_model_parallel_group = None
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        if rank in ranks:
            tensor_model_parallel_group = list(ranks)

    tensor_model_parallel_rank = tensor_model_parallel_group.index(rank)

    # EP rank
    expert_model_parallel_rank = 0
    if expert_model_parallel_size_ is not None and expert_model_parallel_size_ > 1:
        tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size_
        num_expert_groups: int = data_parallel_size // expert_model_parallel_size_
        for i in range(num_tensor_and_data_groups):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                if rank in ranks:
                    expert_model_parallel_rank = list(ranks).index(rank) // tensor_model_parallel_size

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    pipeline_model_parallel_group = None
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        if rank in ranks:
            pipeline_model_parallel_group = list(ranks)

    pipeline_model_parallel_rank = pipeline_model_parallel_group.index(rank)

    return (
        tensor_model_parallel_rank,
        pipeline_model_parallel_rank,
        expert_model_parallel_rank,
        model_parallel_size,
        data_parallel_size,
        pipeline_model_parallel_split_rank_,
        virtual_pipeline_model_parallel_rank,
    )


@pytest.mark.parametrize(
    'nodes, num_gpu, tp, pp, cp, ep',
    [
        (1, 1, 1, 1, 1, 1),
        (4, 8, 2, 4, 1, 1),
        (8, 8, 8, 8, 1, 1),
        (16, 8, 4, 8, 1, 1),
        (16, 8, 4, 8, 4, 1),
        (32, 8, 8, 8, 1, 1),
        (32, 8, 4, 8, 1, 4),
        (32, 8, 8, 8, 4, 1),
    ],
)
def test_fake_initialize(nodes, num_gpu, tp, pp, cp, ep):
    (
        tensor_model_parallel_rank,
        pipeline_model_parallel_rank,
        expert_model_parallel_rank,
        model_parallel_size,
        data_parallel_size,
        pipeline_model_parallel_split_rank,
        virtual_pipeline_model_parallel_rank,
    ) = old_fake_initialize_model_parallel(nodes * num_gpu, 0, tp, pp, None, None, ep, cp)

    (
        m_tensor_model_parallel_rank,
        n_pipeline_model_parallel_rank,
        n_expert_model_parallel_rank,
        n_model_parallel_size,
        n_data_parallel_size,
        n_pipeline_model_parallel_split_rank,
        n_virtual_pipeline_model_parallel_rank,
    ) = fake_initialize_model_parallel(nodes * num_gpu, 0, tp, pp, None, None, ep, cp)
    assert m_tensor_model_parallel_rank == tensor_model_parallel_rank
    assert n_pipeline_model_parallel_rank == pipeline_model_parallel_rank
    assert n_expert_model_parallel_rank == expert_model_parallel_rank
    assert n_model_parallel_size == model_parallel_size
    assert n_data_parallel_size == data_parallel_size
    assert n_pipeline_model_parallel_split_rank == pipeline_model_parallel_split_rank
    assert n_virtual_pipeline_model_parallel_rank == virtual_pipeline_model_parallel_rank
