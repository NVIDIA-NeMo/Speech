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

from typing import Any, Optional

import torch

from nemo.core.neural_types.axes import AxisKind, AxisType
from nemo.core.neural_types.comparison import NeuralTypeComparisonResult
from nemo.core.neural_types.elements import ElementType, VoidType

__all__ = [
    'NeuralType',
    'NeuralTypeError',
    'NeuralPortNameMismatchError',
    'NeuralPortNmTensorMismatchError',
]


class NeuralType:
    """This is the main class which would represent neural type concept.
    It is used to represent *the types* of inputs and outputs.

    Args:
        axes (Optional[Tuple]): a tuple of AxisTypes objects representing the semantics of what varying each axis means
            You can use a short, string-based form here. For example: ('B', 'C', 'H', 'W') would correspond to an NCHW
            format frequently used in computer vision. ('B', 'T', 'D') is frequently used for signal processing and
            means [batch, time, dimension/channel].
        elements_type (ElementType): an instance of ElementType class representing the semantics of what is stored
            inside the tensor. For example: logits (LogitsType), log probabilities (LogprobType), etc.
        optional (bool): By default, this is false. If set to True, it would means that input to the port of this
            type can be optional.
    """

    def __str__(self):
        if torch.jit.is_scripting():
            return "SuppressedForTorchScript"
        if self.axes is not None:
            return f"axes: {self.axes}; elements_type: {self.elements_type.__class__.__name__}"
        else:
            return f"axes: None; elements_type: {self.elements_type.__class__.__name__}"

    def __init__(self, axes: Optional[Any] = None, elements_type: Optional[Any] = None, optional: bool = False):
        """
        Args:
            axes: a tuple of AxisTypes objects representing the semantics of what varying each axis means
            elements_type: None or ElementType; we need Any annotation here to avoid problems with TorchScript (it is checked in _init_internal)
            optional: If input to the port of this type can be optional (False by default).
        """
        if not torch.jit.is_scripting():
            self._init_internal(axes=axes, elements_type=elements_type, optional=optional)

    @torch.jit.unused
    def _init_internal(
        self, axes: Optional[Any] = None, elements_type: Optional[ElementType] = None, optional: bool = False
    ):
        """Internals of __init__, separated to make TorchScript and autodoc work"""
        if elements_type is None:
            elements_type = VoidType()
        if not isinstance(elements_type, ElementType):
            raise ValueError(
                "elements_type of NeuralType must be an instance of a class derived from ElementType. "
                "Did you pass a class instead?"
            )
        self.elements_type = elements_type
        if axes is not None:
            NeuralType.__check_sanity(axes)
            axes_list = []
            for axis in axes:
                if isinstance(axis, str):
                    axes_list.append(AxisType(AxisKind.from_str(axis), None))
                elif isinstance(axis, AxisType):
                    axes_list.append(axis)
                else:
                    raise ValueError("axis type must be either str or AxisType instance")
            self.axes = tuple(axes_list)
        else:
            self.axes = None
        self.optional = optional

    def compare(self, second) -> NeuralTypeComparisonResult:
        """Performs neural type comparison of self with second. When you chain two modules' inputs/outputs via
        __call__ method, this comparison will be called to ensure neural type compatibility."""
        if torch.jit.is_scripting():
            # suppress check for TorchScript
            return NeuralTypeComparisonResult.SAME
        # First, handle dimensionality
        axes_a = self.axes
        axes_b = second.axes

        # "Big void" type
        if isinstance(self.elements_type, VoidType) and self.axes is None:
            return NeuralTypeComparisonResult.SAME

        if self.axes is None:
            if second.axes is None:
                return self.elements_type.compare(second.elements_type)
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE

        dimensions_pass = NeuralType.__compare_axes(axes_a, axes_b)
        element_comparison_result = self.elements_type.compare(second.elements_type)

        # SAME DIMS
        if dimensions_pass == 0:
            return element_comparison_result
        # TRANSPOSE_SAME DIMS
        elif dimensions_pass == 1:
            if element_comparison_result == NeuralTypeComparisonResult.SAME:
                return NeuralTypeComparisonResult.TRANSPOSE_SAME
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        # DIM_INCOMPATIBLE DIMS
        elif dimensions_pass == 2:
            if element_comparison_result == NeuralTypeComparisonResult.SAME:
                return NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        else:
            return NeuralTypeComparisonResult.INCOMPATIBLE

    def compare_and_raise_error(self, parent_type_name, port_name, second_object):
        """ Method compares definition of one type with another and raises an error if not compatible. """
        if torch.jit.is_scripting():
            # suppress for TorchScript
            return
        type_comatibility = self.compare(second_object)
        if (
            type_comatibility != NeuralTypeComparisonResult.SAME
            and type_comatibility != NeuralTypeComparisonResult.GREATER
        ):
            raise NeuralPortNmTensorMismatchError(
                parent_type_name, port_name, str(self), str(second_object.ntype), type_comatibility
            )

    def __eq__(self, other):
        if isinstance(other, NeuralType):
            return self.compare(other)

        return False

    @staticmethod
    def __check_sanity(axes):
        # check that list come before any tensor dimension
        are_strings = True
        for axis in axes:
            if not isinstance(axis, str):
                are_strings = False
            if isinstance(axis, str) and not are_strings:
                raise ValueError("Either use full class names or all strings")
        if are_strings:
            return
        checks_passed = True
        saw_tensor_dim = False
        for axis in axes:
            if not axis.is_list:
                saw_tensor_dim = True
            else:  # current axis is a list
                if saw_tensor_dim:  # which is preceded by tensor dim
                    checks_passed = False
        if not checks_passed:
            raise ValueError(
                "You have list dimension after Tensor dimension. All list dimensions must preceed Tensor dimensions"
            )

    @staticmethod
    def __compare_axes(axes_a, axes_b) -> int:
        """
        Compares axes_a and axes_b
        Args:
            axes_a: first axes tuple
            axes_b: second axes tuple

        Returns:
            0 - if they are exactly the same
            1 - if they are "TRANSPOSE_SAME"
            2 - if the are "DIM_INCOMPATIBLE"
            3 - if they are different
        """
        if axes_a is None and axes_b is None:
            return 0
        elif axes_a is None and axes_b is not None:
            return 3
        elif axes_a is not None and axes_b is None:
            return 3
        elif len(axes_a) != len(axes_b):
            return 3
        # After these ifs we know that len(axes_a) == len(axes_b)

        same = True
        kinds_a = dict()
        kinds_b = dict()
        for axis_a, axis_b in zip(axes_a, axes_b):
            kinds_a[axis_a.kind] = axis_a.size
            kinds_b[axis_b.kind] = axis_b.size
            if axis_a.kind == AxisKind.Any:
                same = True
            elif (
                axis_a.kind != axis_b.kind
                or axis_a.is_list != axis_b.is_list
                or (axis_a.size != axis_b.size and axis_a.size is not None)
            ):
                same = False
        if same:
            return 0
        else:
            # can be TRANSPOSE_SAME, DIM_INCOMPATIBLE
            if kinds_a.keys() == kinds_b.keys():
                for key, value in kinds_a.items():
                    if kinds_b[key] != value:
                        return 2
                return 1
            else:
                return 3

    def __repr__(self):
        if torch.jit.is_scripting():
            return "SuppressedForTorchScript"

        if self.axes is not None:
            axes = str(self.axes)
        else:
            axes = "None"

        if self.elements_type is not None:
            element_type = repr(self.elements_type)
        else:
            element_type = "None"

        data = f"axis={axes}, element_type={element_type}"

        if self.optional:
            data = f"{data}, optional={self.optional}"

        final = f"{self.__class__.__name__}({data})"
        return final


class NeuralTypeError(Exception):
    """Base class for neural type related exceptions."""


class NeuralPortNameMismatchError(NeuralTypeError):
    """Exception raised when neural module is called with incorrect port
    names."""

    def __init__(self, input_port_name):
        super().__init__()
        self.message = "Wrong input port name: {0}".format(input_port_name)


class NeuralPortNmTensorMismatchError(NeuralTypeError):
    """Exception raised when a port is fed with a NmTensor of incompatible
    type."""

    def __init__(self, class_name, port_name, first_type, second_type, type_comatibility):
        super().__init__()
        self.message = "\nIn {}. \nPort: {} and a NmTensor it was fed are \n".format(class_name, port_name)
        self.message += "of incompatible neural types:\n\n{} \n\n and \n\n{}".format(first_type, second_type)
        self.message += "\n\nType comparison result: {}".format(type_comatibility)
