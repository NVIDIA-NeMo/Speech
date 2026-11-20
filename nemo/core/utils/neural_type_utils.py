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

from collections import defaultdict
from typing import Dict, List, Optional
import torch
from nemo.core.neural_types import AxisKind, NeuralType


def get_io_names(types: Optional[Dict[str, NeuralType]], disabled_names: List[str]) -> List[str]:
    if types is None:
        return []
    names = list(types.keys())
    for name in disabled_names:
        if name in names:
            names.remove(name)
    return names


def extract_dynamic_axes(name: str, ntype: NeuralType):
    """
    This method will extract BATCH and TIME dimension ids from each provided input/output name argument.

    For example, if module/model accepts argument named "input_signal" with type corresponding to [Batch, Time, Dim]
    shape, then the returned result should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes
    as they can change from call to call during inference.

    Args:
        name: Name of input or output parameter
        ntype: Corresponding Neural Type

    Returns:

    """

    def unpack_nested_neural_type(neural_type):
        if type(neural_type) in (list, tuple):
            return unpack_nested_neural_type(neural_type[0])
        return neural_type

    dynamic_axes = defaultdict(list)
    if type(ntype) in (list, tuple):
        ntype = unpack_nested_neural_type(ntype)

    if ntype.axes:
        for ind, axis in enumerate(ntype.axes):
            if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                dynamic_axes[name].append(ind)
    return dynamic_axes


def get_dynamic_axes(types, names, use_dynamo=False):
    """Build dynamic-axis mappings from neural type declarations."""
    dynamic_axes = defaultdict(list)
    if names is not None:
        for name in names:
            if name in types:
                dynamic_axes.update(extract_dynamic_axes(name, types[name]))
    if use_dynamo:
        dynamic_shapes = {}
        batch = torch.export.Dim("batch")
        for name, dims in dynamic_axes.items():
            ds = {}
            for d in dims:
                if d == 0:
                    ds[d] = batch
                # this currently has issues: https://github.com/pytorch/pytorch/issues/126127
                else:
                    ds[d] = torch.export.Dim(name + '__' + str(d))
            dynamic_shapes[name] = ds
        dynamic_axes = dynamic_shapes
    return dynamic_axes
