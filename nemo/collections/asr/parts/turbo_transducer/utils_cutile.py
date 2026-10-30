# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import cuda.tile as ct


def swizzle_2d_as_triton(i, j, size_i, size_j, size_g):
    """
    Swizzle 2d, following Triton implementation:
    https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html
    https://github.com/triton-lang/triton/blob/609ced5e3f04e55234115524eb734822331a37d7/python/triton/language/standard.py#L80
    """
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = ct.minimum(size_i - off_i, size_g)
    # linear index with respect to the first element in this group
    ij = ij % size_gj
    # new row and column indices
    new_i = off_i + ij % size_g
    new_j = ij // size_g
    return new_i, new_j
