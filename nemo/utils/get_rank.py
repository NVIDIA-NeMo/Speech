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

import torch

from nemo.utils.env_var_parsing import get_envint


def is_global_rank_zero():
    """ Helper function to determine if the current process is global_rank 0 (the main process)
    """
    # Try to get the pytorch RANK env var
    # RANK is set by torch.distributed.launch
    rank = get_envint("RANK", None)
    if rank is not None:
        return rank == 0

    # Try to get the SLURM global rank env var
    # SLURM_PROCID is set by SLURM
    slurm_rank = get_envint("SLURM_PROCID", None)
    if slurm_rank is not None:
        return slurm_rank == 0

    # Try to get the MPI global rank env var
    mpi_rank = get_envint("OMPI_COMM_WORLD_RANK", None)
    if mpi_rank is not None:
        return mpi_rank == 0

    # if neither pytorch, SLURM nor MPI env vars are set
    # check NODE_RANK/GROUP_RANK and LOCAL_RANK env vars
    # assume global_rank is zero if undefined
    node_rank = get_envint("NODE_RANK", get_envint("GROUP_RANK", 0))
    local_rank = get_envint("LOCAL_RANK", 0)
    return node_rank == 0 and local_rank == 0


def get_rank():
    """ Helper function that returns torch.distributed.get_rank() if DDP has been initialized otherwise it returns 0.
    """

    if is_global_rank_zero():
        return 0
    else:
        return torch.distributed.get_rank()
