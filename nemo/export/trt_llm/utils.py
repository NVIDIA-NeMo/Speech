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

from typing import Optional

import tensorrt_llm


def is_rank(rank: Optional[int]) -> bool:
    """
    Check if the current MPI rank matches the specified rank.

    Args:
        rank (Optional[int]): The rank to check against.

    Returns:
        bool: True if the current rank matches the specified rank or if rank is None.
    """
    current_rank = tensorrt_llm.mpi_rank()
    if rank is None:
        return True
    if isinstance(rank, int):
        return current_rank == rank
    raise ValueError(f"Invalid rank argument {rank} of type {type(rank)}.")
