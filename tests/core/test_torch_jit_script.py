# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from inspect import isclass
from typing import Any, Type

import pytest
import torch
import torch.nn as nn

import nemo.core.neural_types.elements as nelements
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.classes.mixins import AdapterModuleMixin
from nemo.core.neural_types import FloatType, NeuralType, VoidType


def get_all_neural_types() -> list[Type[nelements.ElementType]]:
    """Get all neural types (elements) by inspecting neural_types.element module"""
    neural_types = []
    for neural_type_str in dir(nelements):
        candidate = getattr(nelements, neural_type_str)
        if (
            isclass(candidate)
            and issubclass(candidate, nelements.ElementType)
            and candidate is not nelements.ElementType
        ):
            neural_types.append(candidate)
    return neural_types


class SimpleLinear(NeuralModule):
    """Simple linear projection. Test use of NeuralModule instead of nn.Module"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearExportable(NeuralModule, Exportable):
    """Simple linear projection. Test use of NeuralModule with Exportable mixin"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearWithAdapterMixin(NeuralModule, AdapterModuleMixin):
    """Simple linear projection. Test use of NeuralModule with AdapterModuleMixin mixin"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearWithTypes(NeuralModule):
    """Simple linear projection. Test use of NeuralModule with input/output types"""

    @property
    def input_types(self) -> dict[str, Any]:
        return {
            "x": NeuralType(None, FloatType()),
        }

    @property
    def output_types(self) -> dict[str, Any]:
        return {
            "output": NeuralType(None, VoidType()),
        }

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    @typecheck()
    def forward(self, x):
        return self.linear(x)


class DummyModuleWithIOTypes(NeuralModule):
    """Identity module. For testing input/output types in a sequence of network calls"""

    @property
    def input_types(self) -> dict[str, NeuralType]:
        return {
            "x": NeuralType(None, VoidType()),
        }

    @property
    def output_types(self) -> dict[str, NeuralType]:
        return {
            "output": NeuralType(None, FloatType()),
        }

    def forward(self, x):
        return x


class TestTorchJitCompatibility:
    @pytest.mark.unit
    def test_simple_linear(self):
        """Test basic module derived from NeuralModule"""
        module = torch.jit.script(SimpleLinear())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    def test_simple_linear_exportable(self):
        """Test basic module derived from NeuralModule and Exportable"""
        module = torch.jit.script(SimpleLinearExportable())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    def test_simple_linear_with_adapter_mixin(self):
        """Test basic module derived from NeuralModule and Adapter Mixin"""
        module = torch.jit.script(SimpleLinearWithAdapterMixin())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    def test_simple_linear_with_types(self):
        """Test basic module derived from NeuralModule containing types"""
        module = torch.jit.script(SimpleLinearWithTypes())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "neural_type", get_all_neural_types(),
    )
    def test_element_compilable(self, neural_type: Type[nelements.ElementType]):
        """
        Tests that all NeuralType Elements are compilable by TorchScript (and be used in modules/functions).
        """

        @torch.jit.script
        def identity(x: torch.Tensor):
            if isinstance(neural_type(), nelements.VoidType):
                return x
            else:
                return x + 1

        _ = identity(torch.tensor(1.0))

    @pytest.mark.unit
    def test_dummy_module_with_io_types(self):
        """Test module with input/output types"""
        module = torch.jit.script(DummyModuleWithIOTypes())
        x = torch.rand(2, 10)
        result = module(x)
        assert result.shape == x.shape
        assert torch.allclose(result, x)

    @pytest.mark.unit
    def test_chain_with_types(self):
        """Test applying 2 modules consecutively with input/output types"""
        dummy_module = torch.jit.script(DummyModuleWithIOTypes())
        module = torch.jit.script(SimpleLinearWithTypes())
        x = torch.zeros(2, 10)
        result = module(dummy_module(x))
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))
