# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from enum import Enum

__all__ = ['NeuralTypeComparisonResult']


class NeuralTypeComparisonResult(Enum):
    """The result of comparing two neural type objects for compatibility.
    When comparing A.compare_to(B):"""

    SAME = 0
    LESS = 1  # A is B
    GREATER = 2  # B is A
    DIM_INCOMPATIBLE = 3  # Resize connector might fix incompatibility
    TRANSPOSE_SAME = 4  # A transpose and/or converting between lists and tensors will make them same
    CONTAINER_SIZE_MISMATCH = 5  # A and B contain different number of elements
    INCOMPATIBLE = 6  # A and B are incompatible
    SAME_TYPE_INCOMPATIBLE_PARAMS = 7  # A and B are of the same type but parametrized differently
    UNCHECKED = 8  # type comparison wasn't done
