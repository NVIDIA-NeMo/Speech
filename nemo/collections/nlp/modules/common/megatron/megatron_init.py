# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import random

import numpy as np
import torch

from nemo.utils import AppState, logging

try:
    from apex.transformer.log_util import set_logging_level
    from apex.transformer.microbatches import ConstantNumMicroBatches
    from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import tensor_parallel
    from megatron.core.parallel_state import (
        get_pipeline_model_parallel_rank,
        set_pipeline_model_parallel_rank,
        set_pipeline_model_parallel_split_rank,
        set_pipeline_model_parallel_world_size,
        set_tensor_model_parallel_rank,
        set_tensor_model_parallel_world_size,
        set_virtual_pipeline_model_parallel_rank,
    )

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from apex.transformer.parallel_state import set_virtual_pipeline_model_parallel_world_size

    HAVE_INTERLEAVED = True

except:

    HAVE_INTERLEAVED = False


def initialize_model_parallel_for_nemo(
    world_size,
    global_rank,
    local_rank,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    micro_batch_size=None,
    global_batch_size=None,
    rampup_batch_size=None,
    use_fp8=False,
    init_mpi_proc_group=False,
    seed=1234,
    apex_transformer_log_level=30,
):

    if virtual_pipeline_model_parallel_size is not None and not HAVE_INTERLEAVED:
        raise ValueError("set_virtual_pipeline_model_parallel_world_size is needed in megatron-core for interleaved.")

    # updating NeMo globals
    app_state = AppState()
    app_state.global_rank = global_rank
    app_state.world_size = world_size
    app_state.local_rank = local_rank
    app_state.tensor_model_parallel_size = tensor_model_parallel_size
    app_state.pipeline_model_parallel_size = pipeline_model_parallel_size
    app_state.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    app_state.use_fp8 = use_fp8
    app_state.init_mpi_proc_group = init_mpi_proc_group
    (
        app_state.tensor_model_parallel_rank,
        app_state.pipeline_model_parallel_rank,
        app_state.model_parallel_size,
        app_state.data_parallel_size,
        app_state.pipeline_model_parallel_split_rank,
        app_state.virtual_pipeline_model_parallel_rank,
    ) = fake_initialize_model_parallel(
        world_size=world_size,
        rank=global_rank,
        tensor_model_parallel_size_=tensor_model_parallel_size,
        pipeline_model_parallel_size_=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size_=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank_=pipeline_model_parallel_split_rank,
    )

    # update apex.transformer globals
    set_tensor_model_parallel_world_size(app_state.tensor_model_parallel_size)
    set_tensor_model_parallel_rank(app_state.tensor_model_parallel_rank)

    set_pipeline_model_parallel_rank(app_state.pipeline_model_parallel_rank)
    if HAVE_INTERLEAVED:
        set_virtual_pipeline_model_parallel_world_size(app_state.virtual_pipeline_model_parallel_size)
    set_virtual_pipeline_model_parallel_rank(app_state.virtual_pipeline_model_parallel_rank)
    set_pipeline_model_parallel_world_size(app_state.pipeline_model_parallel_size)
    set_pipeline_model_parallel_split_rank(app_state.pipeline_model_parallel_split_rank)

    _set_random_seed(seed)

    if global_batch_size and micro_batch_size is not None:
        # TODO: add rampup_batch_size here when we have it implemented
        from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
            setup_microbatch_calculator(
                rank=global_rank,
                global_batch_size=global_batch_size,
                micro_batch_size=micro_batch_size,
                data_parallel_size=app_state.data_parallel_size,
                rampup_batch_size=rampup_batch_size,
            )
        else:
            if isinstance(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, ConstantNumMicroBatches):
                assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.current_global_batch_size == global_batch_size
                assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size == micro_batch_size
                assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.num_micro_batches == global_batch_size // (
                    micro_batch_size * app_state.data_parallel_size
                )
            else:
                raise Exception("Microbatch calculator already initialized.")

    app_state._is_megatron_initialized = True

    set_logging_level(apex_transformer_log_level)


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed_))


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # set flags if we are using the 21.10 container
    if torch.__version__ == "1.10.0a0+0aef44c":
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)


def fake_initialize_model_parallel(
    world_size,
    rank,
    tensor_model_parallel_size_,
    pipeline_model_parallel_size_,
    pipeline_model_parallel_split_rank_=None,
    virtual_pipeline_model_parallel_size_=None,
):
    """
    Fake initialize model data parallel groups so that we can instantiate model parallel models before DDP is initialized.
    This is needed because PTL execution flow is init model, init trainer -> call trainer.fit(model). DDP is initialized during .fit.
    This function is taken from megatron.core.parallel_state and modified so that the distributed groups are not created.
    We only need the tensor parallel and pipeline parallel ranks to instantiate the model.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor.
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """

    # Get world size and rank. Ensure some consistencies.
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size

    assert (
        world_size % tensor_model_parallel_size * pipeline_model_parallel_size == 0
    ), f'world_size: {world_size} must be divisible by tensor_model_parallel_size: {tensor_model_parallel_size} times pipeline_model_parallel_size {pipeline_model_parallel_size}'
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    virtual_pipeline_model_parallel_rank = None
    if virtual_pipeline_model_parallel_size_ is not None:
        virtual_pipeline_model_parallel_rank = 0

    # Build the data-parallel groups.
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            if rank in ranks:
                data_parallel_group = list(ranks)
                logging.info(f'Rank {rank} has data parallel group: {data_parallel_group}')

    data_parallel_rank = data_parallel_group.index(rank)
    logging.info(f'All data parallel group ranks: {all_data_parallel_group_ranks}')
    logging.info(f'Ranks {rank} has data parallel rank: {data_parallel_rank}')

    # Build the model-parallel groups.
    all_model_parallel_group_ranks = []
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i] for data_parallel_group_ranks in all_data_parallel_group_ranks]
        all_model_parallel_group_ranks.append(ranks)
        if rank in ranks:
            logging.info(f'Rank {rank} has model parallel group: {list(ranks)}')
    logging.info(f'All model parallel group ranks: {all_model_parallel_group_ranks}')

    # Build the tensor model-parallel groups.
    all_tensor_model_parallel_group_ranks = []
    tensor_model_parallel_group = None
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        all_tensor_model_parallel_group_ranks.append(list(ranks))
        if rank in ranks:
            tensor_model_parallel_group = list(ranks)
            logging.info(f'Rank {rank} has tensor model parallel group: {tensor_model_parallel_group}')

    tensor_model_parallel_rank = tensor_model_parallel_group.index(rank)

    logging.info(f'All tensor model parallel group ranks: {all_tensor_model_parallel_group_ranks}')
    logging.info(f'Rank {rank} has tensor model parallel rank: {tensor_model_parallel_rank}')

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    all_pipeline_model_parallel_group_ranks = []
    all_embedding_group_ranks = []
    pipeline_model_parallel_group = None
    embedding_group = None
    embedding_rank = None
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        all_pipeline_model_parallel_group_ranks.append(list(ranks))
        if rank in ranks:
            pipeline_model_parallel_group = list(ranks)
            logging.info(f'Rank {rank} has pipeline model parallel group: {pipeline_model_parallel_group}')

        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            all_embedding_group_ranks.append(embedding_ranks)
        else:
            embedding_ranks = ranks
            all_embedding_group_ranks.append(list(embedding_ranks))
        if rank in embedding_ranks:
            embedding_group = list(embedding_ranks)
            logging.info(f'Rank {rank} has embedding group: {embedding_group}')

    pipeline_model_parallel_rank = pipeline_model_parallel_group.index(rank)
    if embedding_group is not None:
        embedding_rank = embedding_group.index(rank)

    logging.info(f'All pipeline model parallel group ranks: {all_pipeline_model_parallel_group_ranks}')
    logging.info(f'Rank {rank} has pipeline model parallel rank {pipeline_model_parallel_rank}')
    logging.info(f'All embedding group ranks: {all_pipeline_model_parallel_group_ranks}')
    logging.info(f'Rank {rank} has embedding rank: {embedding_rank}')

    return (
        tensor_model_parallel_rank,
        pipeline_model_parallel_rank,
        model_parallel_size,
        data_parallel_size,
        pipeline_model_parallel_split_rank_,
        virtual_pipeline_model_parallel_rank,
    )
