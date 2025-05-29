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

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import tensor_parallel
    from megatron.core.parallel_state import (
        RankGenerator,
        get_pipeline_model_parallel_rank,
        set_expert_model_parallel_rank,
        set_expert_model_parallel_world_size,
        set_pipeline_model_parallel_rank,
        set_pipeline_model_parallel_split_rank,
        set_pipeline_model_parallel_world_size,
        set_tensor_model_parallel_rank,
        set_tensor_model_parallel_world_size,
    )

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import (
        ConstantNumMicroBatchesCalculator,
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
        init_num_microbatches_calculator,
    )

    MCORE_MB_CALCULATOR = True

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.microbatches import ConstantNumMicroBatches as ConstantNumMicroBatchesCalculator
    from apex.transformer.pipeline_parallel.utils import (
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
    )
    from apex.transformer.pipeline_parallel.utils import (
        setup_microbatch_calculator as init_num_microbatches_calculator,
    )

    MCORE_MB_CALCULATOR = False


def initialize_model_parallel_for_nemo(
    world_size,
    global_rank,
    local_rank,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    pipeline_model_parallel_comm_backend=None,
    context_parallel_size=1,
    encoder_tensor_model_parallel_size=0,
    encoder_pipeline_model_parallel_size=0,
    micro_batch_size=None,
    global_batch_size=None,
    rampup_batch_size=None,
    use_fp8=False,
    init_mpi_proc_group=False,
    seed=1234,
    apex_transformer_log_level=30,
    use_tp_pp_dp_mapping=False,
    use_te_rng_tracker=False,
    num_distributed_optimizer_instances=1,
    nccl_communicator_config_path=None,
    use_sharp=False,
):
    """Initialize model parallel groups in NeMo."""

    # updating NeMo globals
    app_state = AppState()
    app_state.global_rank = global_rank
    app_state.world_size = world_size
    app_state.local_rank = local_rank
    app_state.use_tp_pp_dp_mapping = use_tp_pp_dp_mapping
    app_state.expert_model_parallel_size = expert_model_parallel_size
    app_state.tensor_model_parallel_size = tensor_model_parallel_size
    app_state.pipeline_model_parallel_size = pipeline_model_parallel_size
    app_state.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    app_state.context_parallel_size = context_parallel_size
    app_state.encoder_tensor_model_parallel_size = encoder_tensor_model_parallel_size
    app_state.encoder_pipeline_model_parallel_size = encoder_pipeline_model_parallel_size
    app_state.pipeline_model_parallel_comm_backend = pipeline_model_parallel_comm_backend
    app_state.use_fp8 = use_fp8
    app_state.use_sharp = use_sharp
    app_state.init_mpi_proc_group = init_mpi_proc_group
    app_state.expert_tensor_parallel_size = expert_tensor_parallel_size
    app_state.num_distributed_optimizer_instances = num_distributed_optimizer_instances
    app_state.nccl_communicator_config_path = nccl_communicator_config_path
    (
        app_state.tensor_model_parallel_rank,
        app_state.pipeline_model_parallel_rank,
        app_state.expert_model_parallel_rank,
        app_state.expert_tensor_parallel_rank,
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
        context_parallel_size_=context_parallel_size,
        expert_model_parallel_size_=expert_model_parallel_size,
        expert_tensor_parallel_size_=expert_tensor_parallel_size,
        encoder_tensor_model_parallel_size_=encoder_tensor_model_parallel_size,
        encoder_pipeline_model_parallel_size_=encoder_pipeline_model_parallel_size,
        use_tp_pp_dp_mapping=use_tp_pp_dp_mapping,
    )

    # update apex.transformer globals
    set_tensor_model_parallel_world_size(app_state.tensor_model_parallel_size)
    set_tensor_model_parallel_rank(app_state.tensor_model_parallel_rank)

    set_expert_model_parallel_world_size(app_state.expert_model_parallel_size)
    set_expert_model_parallel_rank(app_state.expert_model_parallel_rank)

    set_pipeline_model_parallel_world_size(
        app_state.pipeline_model_parallel_size + app_state.encoder_pipeline_model_parallel_size
    )
    set_pipeline_model_parallel_split_rank(app_state.pipeline_model_parallel_split_rank)
    set_pipeline_model_parallel_rank(app_state.pipeline_model_parallel_rank)

    tensor_parallel.random.initialize_rng_tracker(use_te_rng_tracker=use_te_rng_tracker)
    if seed is not None:
        # @chcui not setting seed is for model conversion. always set seed for training/inference.
        _set_random_seed(seed)

    if global_batch_size and micro_batch_size is not None:
        # TODO: add rampup_batch_size here when we have it implemented
        if MCORE_MB_CALCULATOR:
            from megatron.core.num_microbatches_calculator import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

            if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
                init_num_microbatches_calculator(
                    rank=global_rank,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    data_parallel_size=app_state.data_parallel_size,
                    rampup_batch_size=rampup_batch_size,
                    decrease_batch_size_if_needed=False,
                )
            else:
                if isinstance(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, ConstantNumMicroBatchesCalculator):
                    assert get_current_global_batch_size() == global_batch_size
                    assert get_micro_batch_size() == micro_batch_size
                    assert get_num_microbatches() == global_batch_size // (
                        micro_batch_size * app_state.data_parallel_size
                    )
                else:
                    raise Exception("Microbatch calculator already initialized.")
        else:
            from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

            if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
                init_num_microbatches_calculator(
                    rank=global_rank,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    data_parallel_size=app_state.data_parallel_size,
                    rampup_batch_size=rampup_batch_size,
                    decrease_batch_size_if_needed=False,
                )
            else:
                if isinstance(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, ConstantNumMicroBatchesCalculator):
                    assert get_current_global_batch_size() == global_batch_size
                    assert get_micro_batch_size() == micro_batch_size
                    assert get_num_microbatches() == global_batch_size // (
                        micro_batch_size * app_state.data_parallel_size
                    )
                else:
                    raise Exception("Microbatch calculator already initialized.")

    app_state._is_megatron_initialized = True

    if HAVE_APEX:
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
    expert_model_parallel_size_=1,
    expert_tensor_parallel_size_=None,
    context_parallel_size_=1,
    encoder_tensor_model_parallel_size_=0,
    encoder_pipeline_model_parallel_size_=0,
    use_tp_pp_dp_mapping=False,
):
    """
    Fake initialize model data parallel groups so that we can instantiate model parallel
    models before DDP is initialized. This is needed because PTL execution flow is init
    model, init trainer -> call trainer.fit(model). DDP is initialized during .fit.
    This function is taken from megatron.core.parallel_state and modified so that the
    distributed groups are not created.
    We only need the tensor parallel and pipeline parallel ranks to instantiate the model.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor.
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline.
        context_parallel_size: number of GPUs used to parallelize tokens of each input.

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
    context_parallel_size = min(context_parallel_size_, world_size)

    if encoder_pipeline_model_parallel_size_ is None:
        encoder_pipeline_model_parallel_size = 0
    else:
        encoder_pipeline_model_parallel_size = encoder_pipeline_model_parallel_size_

    if encoder_tensor_model_parallel_size_ == 0 and encoder_pipeline_model_parallel_size_ > 0:
        encoder_tensor_model_parallel_size = tensor_model_parallel_size
    else:
        encoder_tensor_model_parallel_size = encoder_tensor_model_parallel_size_

    if encoder_tensor_model_parallel_size > 0:
        assert encoder_pipeline_model_parallel_size > 0
        assert (
            encoder_tensor_model_parallel_size <= tensor_model_parallel_size
        ), "We do not support encoders with more TP than the decoder."

    encoder_model_size = (
        encoder_tensor_model_parallel_size * encoder_pipeline_model_parallel_size * context_parallel_size
    )
    decoder_model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    total_model_size = encoder_model_size + decoder_model_size

    assert world_size % total_model_size == 0, (
        f'world_size: {world_size} must be divisible by total world_size: '
        f'(decoder_)tensor_model_parallel_size {tensor_model_parallel_size} '
        f'* (decoder_)pipeline_model_parallel_size {pipeline_model_parallel_size} '
        f'* (decoder_)context_parallel_size {context_parallel_size} + '
        f'encoder_tensor_model_parallel_size {encoder_tensor_model_parallel_size} '
        f'* encoder_pipeline_model_parallel_size {encoder_pipeline_model_parallel_size} '
        f'* context_parallel_size {context_parallel_size}'
    )
    data_parallel_size = world_size // total_model_size

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size
    assert encoder_world_size + decoder_world_size == world_size

    virtual_pipeline_model_parallel_rank = None
    if virtual_pipeline_model_parallel_size_ is not None:
        virtual_pipeline_model_parallel_rank = 0

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order='tp-cp-ep-pp-dp' if use_tp_pp_dp_mapping else 'tp-cp-ep-dp-pp',
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order='tp-cp-ep-pp-dp' if use_tp_pp_dp_mapping else 'tp-cp-ep-dp-pp',
        rank_offset=encoder_world_size,
    )
    # Build expert rank generator
    if expert_tensor_parallel_size_ is None:
        expert_tensor_parallel_size_ = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size_ * expert_model_parallel_size_ * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by "
            f"expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size_,
        ep=expert_model_parallel_size_,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order='tp-cp-ep-pp-dp' if use_tp_pp_dp_mapping else 'tp-cp-ep-dp-pp',
        rank_offset=encoder_world_size,
    )

    assert (
        not use_tp_pp_dp_mapping
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ), "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        from itertools import cycle

        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp."""
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)
        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == 'pp':
            # Map 1 encoder tp rank to several decoder tp ranks, because
            # these won't be the same size.
            for x, y in zip(cycle(e_ranks), d_ranks):
                yield x + y
        elif group_type == 'tp-pp':
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    # Build the data-parallel groups.
    all_data_parallel_group_ranks_with_cp = []
    for ranks in generator_wrapper('dp'):
        if rank in ranks:
            data_parallel_group = list(ranks)
            logging.info(f'Rank {rank} has data parallel group : {data_parallel_group}')

    for ranks_with_cp in generator_wrapper('dp-cp'):
        all_data_parallel_group_ranks_with_cp.append(ranks_with_cp)
        if rank in ranks_with_cp:
            data_parallel_group_with_cp = ranks_with_cp
            logging.info(
                f'Rank {rank} has combined group of data parallel and context parallel : {data_parallel_group_with_cp}'
            )

    data_parallel_rank = data_parallel_group.index(rank)
    logging.info(
        f'All data parallel group ranks with context parallel combined: {all_data_parallel_group_ranks_with_cp}'
    )
    logging.info(f'Ranks {rank} has data parallel rank: {data_parallel_rank}')

    # Build the context-parallel groups.
    all_context_parallel_group_ranks = []
    for ranks in generator_wrapper('cp'):
        all_context_parallel_group_ranks.append(ranks)
        if rank in ranks:
            context_parallel_group = ranks
            logging.info(f'Rank {rank} has context parallel group: {context_parallel_group}')

    context_parallel_rank = context_parallel_group.index(rank)
    logging.info(f'All context parallel group ranks: {all_context_parallel_group_ranks}')
    logging.info(f'Ranks {rank} has context parallel rank: {context_parallel_rank}')

    # Build the model-parallel groups.
    all_model_parallel_group_ranks = []
    for ranks in generator_wrapper('tp-pp'):
        all_model_parallel_group_ranks.append(ranks)
        if rank in ranks:
            logging.info(f'Rank {rank} has model parallel group: {list(ranks)}')
    logging.info(f'All model parallel group ranks: {all_model_parallel_group_ranks}')

    # Build the tensor model-parallel groups.
    all_tensor_model_parallel_group_ranks = []
    tensor_model_parallel_group = None
    for ranks in generator_wrapper('tp'):
        all_tensor_model_parallel_group_ranks.append(ranks)
        if rank in ranks:
            tensor_model_parallel_group = ranks
            logging.info(f'Rank {rank} has tensor model parallel group: {tensor_model_parallel_group}')

    tensor_model_parallel_rank = tensor_model_parallel_group.index(rank)

    logging.info(f'All tensor model parallel group ranks: {all_tensor_model_parallel_group_ranks}')
    logging.info(f'Rank {rank} has tensor model parallel rank: {tensor_model_parallel_rank}')

    # EP rank
    expert_model_parallel_rank = 0
    if expert_model_parallel_size_ is not None and expert_model_parallel_size_ > 1:
        for ranks in generator_wrapper('ep', is_expert=True):
            if rank in ranks:
                expert_model_parallel_rank = list(ranks).index(rank)

    # ETP
    expert_tensor_parallel_rank = 0
    if expert_tensor_parallel_size_ is not None and expert_tensor_parallel_size_ > 1:
        for ranks in generator_wrapper('tp', is_expert=True):
            if rank in ranks:
                expert_tensor_parallel_rank = list(ranks).index(rank)

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    all_pipeline_model_parallel_group_ranks = []
    all_embedding_group_ranks = []
    pipeline_model_parallel_group = None
    embedding_group = None
    embedding_rank = None
    for ranks in generator_wrapper('pp'):
        all_pipeline_model_parallel_group_ranks.append(ranks)
        if rank in ranks:
            pipeline_model_parallel_group = ranks
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
        expert_model_parallel_rank,
        expert_tensor_parallel_rank,
        model_parallel_size,
        data_parallel_size,
        pipeline_model_parallel_split_rank_,
        virtual_pipeline_model_parallel_rank,
    )
