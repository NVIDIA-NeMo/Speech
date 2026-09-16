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

from typing import List, NamedTuple

import pytest
import torch

from nemo.core import Typing, typecheck
from nemo.core.neural_types import *


# Perform recursive shape assert
def recursive_assert_shape(x, shape):
    if isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            recursive_assert_shape(xi, shape)
        return

    assert x.shape == shape


# Perform recursive type assert
def recursive_assert_homogeneous_type(x, type_val):
    if isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            recursive_assert_homogeneous_type(xi, type_val)
        return

    assert x.neural_type.compare(type_val) == NeuralTypeComparisonResult.SAME


class TestNeuralTypeCheckSystem:
    @pytest.mark.unit
    def test_no_types_passthrough(self):
        class NoTypes(Typing):
            @typecheck()
            def __call__(self, x):
                return torch.tensor(1.0)

        obj = NoTypes()
        result = obj(torch.tensor(1.0))

        assert result == torch.tensor(1.0)
        assert not hasattr(result, 'neural_type')

    @pytest.mark.unit
    def test_input_output_types(self):
        class InputOutputTypes(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputOutputTypes()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert result.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        # Test passing wrong key for input
        with pytest.raises(TypeError):
            _ = obj(a=torch.zeros(10))

        # Test using positional args
        with pytest.raises(TypeError):
            _ = obj(torch.zeros(10))

    @pytest.mark.unit
    def test_input_types_only(self):
        class InputTypes(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputTypes()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert hasattr(result, 'neural_type') is False

    @pytest.mark.unit
    def test_multiple_input_types_only(self):
        class InputTypes(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType()), "y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x, y):
                x += y
                return x

        obj = InputTypes()
        result = obj(x=torch.zeros(10), y=torch.ones(10))

        assert result.sum() == torch.tensor(10.0)
        assert hasattr(result, 'neural_type') is False

    @pytest.mark.unit
    def test_output_types_only(self):
        class OutputTypes(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = OutputTypes()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert result.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        # Test passing positional args
        # Positional args allowed if input types is not set !
        result = obj(torch.zeros(10))
        assert result.sum() == torch.tensor(10.0)

    @pytest.mark.unit
    def test_multiple_output_types_only(self):
        class MultipleOutputTypes(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType()), "z": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                y = x + 1
                z = x + 2
                return y, z

        obj = MultipleOutputTypes()
        result_y, result_z = obj(x=torch.zeros(10))

        assert result_y.sum() == torch.tensor(10.0)
        assert result_y.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        assert result_z.sum() == torch.tensor(20.0)
        assert result_z.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_multiple_output_types_only_namedtuple(self):
        class NamedTupleOutputType(NamedTuple):
            y: torch.Tensor
            z: torch.Tensor

        class MultipleOutputTypesWithNamedTuple(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType()), "z": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                y = x + 1
                z = x + 2
                return NamedTupleOutputType(y=y, z=z)

        obj = MultipleOutputTypesWithNamedTuple()
        result = obj(x=torch.zeros(10))

        assert result.y.sum() == torch.tensor(10.0)
        assert result.y.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        assert result.z.sum() == torch.tensor(20.0)
        assert result.z.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_multiple_mixed_output_types_only(self):
        class MultipleMixedOutputTypes(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType()), "z": [NeuralType(('B',), ElementType())]}

            @typecheck()
            def __call__(self, x):
                y = x + 1
                z = x + 2
                return y, [z, z]

        obj = MultipleMixedOutputTypes()
        result_y, result_z = obj(x=torch.zeros(10))

        assert result_y.sum() == torch.tensor(10.0)
        assert result_y.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        assert result_z[0].sum() == torch.tensor(20.0)
        assert result_z[0].neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        assert result_z[1].sum() == torch.tensor(20.0)
        assert result_z[1].neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_multiple_mixed_output_types_only_namedtuple(self):
        class NamedTupleOutputType(NamedTuple):
            y: torch.Tensor
            zs: List[torch.Tensor]

        class MultipleMixedOutputTypes(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType()), "zs": [NeuralType(('B',), ElementType())]}

            @typecheck()
            def __call__(self, x):
                y = x + 1
                z = x + 2
                return NamedTupleOutputType(y=y, zs=[z, z])

        obj = MultipleMixedOutputTypes()
        result_y, result_z = obj(x=torch.zeros(10))

        assert result_y.sum() == torch.tensor(10.0)
        assert result_y.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        assert result_z[0].sum() == torch.tensor(20.0)
        assert result_z[0].neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        assert result_z[1].sum() == torch.tensor(20.0)
        assert result_z[1].neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_multiple_mixed_output_types_only_mismatched(self):
        class MultipleMixedOutputTypes(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType()), "z": [NeuralType(('B',), ElementType())]}

            @typecheck()
            def __call__(self, x):
                # Use list of y, single z, contrary to signature
                y = x + 1
                z = x + 2
                return [y, y], z

        obj = MultipleMixedOutputTypes()
        with pytest.raises(TypeError):
            result_y, result_z = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_multiple_mixed_output_types_only_namedtuple_mismatched(self):
        class NamedTupleOutputType(NamedTuple):
            ys: List[torch.Tensor]
            z: torch.Tensor

        class MultipleMixedOutputTypes(Typing):
            @property
            def output_types(self):
                return {"ys": NeuralType(('B',), ElementType()), "z": [NeuralType(('B',), ElementType())]}

            @typecheck()
            def __call__(self, x):
                # Use list of y, single z, contrary to signature
                y = x + 1
                z = x + 2
                return NamedTupleOutputType(ys=[y, y], z=z)

        obj = MultipleMixedOutputTypes()
        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_incorrect_inheritance(self):
        class IncorrectInheritance(object):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = IncorrectInheritance()

        with pytest.raises(RuntimeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_port_definition_rejection(self):
        class InputPortDefinitionRejection(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"w": NeuralType(('B',), ElementType()), "u": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x, y):
                x += 1
                y -= 1
                return x, y

        # Test input port mismatch
        obj = InputPortDefinitionRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10), y=torch.zeros(10))

        class OutputPortDefinitionRejection(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {
                    "w": NeuralType(('B',), ElementType()),
                }

            @typecheck()
            def __call__(self, x):
                return x + 1, x - 1

        obj = OutputPortDefinitionRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_port_shape_rejection(self):
        class InputPortShapeRejection(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B', 'T'), ElementType())}  # expect rank 2 matrix

            @property
            def output_types(self):
                return {"w": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        # Test input port mismatch
        obj = InputPortShapeRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

        class OutputPortShapeRejection(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {
                    "w": NeuralType(('B', 'T', 'D'), ElementType()),  # expect rank 3 matrix
                }

            @typecheck()
            def __call__(self, x):
                return x + 1

        obj = OutputPortShapeRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_positional_args(self):
        # Test positional check on input type
        class InputPositional(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputPositional()

        with pytest.raises(TypeError):
            _ = obj(torch.zeros(10))

        # Test positional pass-through for only output ports defined
        # NOTE: This is required behaviour to support type checking of NeMo Dataset class
        # during collate_fn() call.
        class OutputPositionalPassthrough(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = OutputPositionalPassthrough()
        result = obj(torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)

    @pytest.mark.unit
    def test_optional_types(self):
        class InputOptionalTypes(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType()), "y": NeuralType(('B',), ElementType(), optional=True)}

            @typecheck()
            def __call__(self, x, y=None):
                if y is None:
                    x += 1
                else:
                    x += y
                return x

        obj = InputOptionalTypes()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert hasattr(result, 'neural_type') is False

        result2 = obj(x=torch.zeros(10), y=torch.full([10], fill_value=5, dtype=torch.int32))

        assert result2.sum() == torch.tensor(10 * 5)
        assert hasattr(result, 'neural_type') is False

    @pytest.mark.unit
    def test_input_output_neural_types(self):
        class NodeA(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": NeuralType(('B', 'D'), LogitsType())}

            @typecheck()
            def __call__(self, x):
                y = torch.randn(x.shape[0], 4)
                return y

        class NodeB(Typing):
            @property
            def input_types(self):
                return {"w": NeuralType(('B', 'D'), LogitsType())}

            @property
            def output_types(self):
                return {"u": NeuralType(('B',), LabelsType())}

            @typecheck()
            def __call__(self, w):
                _, u = w.max(-1)
                return u

        nodeA = NodeA()
        nodeB = NodeB()

        outA = nodeA(x=torch.zeros(10))
        outB = nodeB(w=outA)

        assert outB.shape == torch.Size([10])
        assert outB.neural_type.compare(NeuralType(('B',), LabelsType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_nested_input_output_neural_types(self):
        class NestedNodeA(Typing):
            @property
            def input_types(self):
                return {"x": [[NeuralType(('B',), ElementType())]]}

            @property
            def output_types(self):
                return {
                    "y0": [NeuralType(('B', 'D'), LogitsType())],
                    "y1": [NeuralType(('B', 'D'), LogitsType())],
                }

            @typecheck(ignore_collections=False)
            def __call__(self, x):
                # input x = [[x1, x2], [x3]]
                x0 = x[0][0]
                y = torch.randn(x0.shape[0], 4)
                # Output is same as
                # 1) return ([y, y], [y])
                # 2) return [y, y], [y]
                return [[y, y], [y]]

        # Non-homogeneous output types
        class NestedNodeB(Typing):
            @property
            def input_types(self):
                return {"w": [[NeuralType(('B', 'D'), LogitsType())]]}

            @property
            def output_types(self):
                return {
                    "u0": [NeuralType(('B',), LogprobsType())],  # check non homogeneous type
                    "u1": [NeuralType(('B',), LabelsType())],
                }

            @typecheck(ignore_collections=False)
            def __call__(self, w):
                # input x = [[x1, x2], [x3]]
                _, u00 = w[0][0].max(-1)
                _, u01 = w[0][1].max(-1)
                _, u10 = w[1][0].max(-1)
                # Output is same as
                # 1) return ([u00, u01], [u10])
                # 2) return [u00, u01], [u10]
                return [[u00, u01], [u10]]

        nodeA = NestedNodeA()
        nodeB = NestedNodeB()

        input_nest = [[torch.zeros(10), torch.zeros(10)], [torch.zeros(10)]]
        outA = nodeA(x=input_nest)
        outB = nodeB(w=outA)

        # Perform recursive shape assert
        recursive_assert_shape(outB, torch.Size([10]))

        # Assert non-homogeneous type assertions
        assert outB[0][0].neural_type.compare(NeuralType(('B',), LogprobsType())) == NeuralTypeComparisonResult.SAME
        assert outB[0][1].neural_type.compare(NeuralType(('B',), LogprobsType())) == NeuralTypeComparisonResult.SAME
        assert outB[1][0].neural_type.compare(NeuralType(('B',), LabelsType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_nested_input_output_neural_types_ignore_collections(self):
        class NestedNodeA(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {
                    "y0": NeuralType(('B', 'D'), LogitsType()),
                    "y1": NeuralType(('B', 'D'), LogitsType()),
                }

            @typecheck(ignore_collections=True)
            def __call__(self, x):
                # input x = [[x1, x2], [x3]]
                x0 = x[0][0]
                y = torch.randn(x0.shape[0], 4)
                return [[y, y], [y]]

        # Non-homogeneous output types
        class NestedNodeB(Typing):
            @property
            def input_types(self):
                return {"w": NeuralType(('B', 'D'), LogitsType())}

            @property
            def output_types(self):
                return {
                    "u0": NeuralType(('B',), LogprobsType()),  # check non homogeneous type
                    "u1": NeuralType(('B',), LabelsType()),
                }

            @typecheck(ignore_collections=True)
            def __call__(self, w):
                # input x = [[x1, x2], [x3]]
                _, u00 = w[0][0].max(-1)
                _, u01 = w[0][1].max(-1)
                _, u10 = w[1][0].max(-1)
                return [[u00, u01], [u10]]

        nodeA = NestedNodeA()
        nodeB = NestedNodeB()

        input_nest = [[torch.zeros(10), torch.zeros(10)], [torch.zeros(10)]]
        outA = nodeA(x=input_nest)
        outB = nodeB(w=outA)

        # Perform recursive shape assert
        recursive_assert_shape(outB, torch.Size([10]))

        # Assert non-homogeneous type assertions
        assert outB[0][0].neural_type.compare(NeuralType(('B',), LogprobsType())) == NeuralTypeComparisonResult.SAME
        assert outB[0][1].neural_type.compare(NeuralType(('B',), LogprobsType())) == NeuralTypeComparisonResult.SAME
        assert outB[1][0].neural_type.compare(NeuralType(('B',), LabelsType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_nested_mixed_input_output_neural_types(self):
        class NestedMixedNodeA(Typing):
            @property
            def input_types(self):
                return {"x1": NeuralType(('B',), ElementType()), "x2": [[NeuralType(('B',), ElementType())]]}

            @property
            def output_types(self):
                return {
                    "y0": NeuralType(('B', 'D'), LogprobsType()),
                    "y1": [[NeuralType(('B', 'D'), LogitsType())]],
                }

            @typecheck(ignore_collections=False)
            def __call__(self, x1, x2):
                # input x = [[x1, x2], [x3]]
                x0 = x2[0][0]
                y = torch.randn(x0.shape[0], 4)
                return y, [[y, y], [y]]

        # Non-homogeneous output types
        class NestedMixedNodeB(Typing):
            @property
            def input_types(self):
                return {"w": [[NeuralType(('B', 'D'), LogitsType())]]}

            @property
            def output_types(self):
                return {
                    "u0": [NeuralType(('B',), LogprobsType())],  # check non homogeneous type
                    "u1": NeuralType(('B',), LabelsType()),
                }

            @typecheck(ignore_collections=False)
            def __call__(self, w):
                # input x = [[x1, x2], [x3]]
                _, u00 = w[0][0].max(-1)
                _, u01 = w[0][1].max(-1)
                _, u10 = w[1][0].max(-1)
                return [u00, u01], u10

        nodeA = NestedMixedNodeA()
        nodeB = NestedMixedNodeB()

        input_nest = [[torch.zeros(10), torch.zeros(10)], [torch.zeros(10)]]
        out_y, outA = nodeA(x1=torch.zeros(10), x2=input_nest)
        outB, out_u = nodeB(w=outA)

        # Perform recursive shape assert
        assert out_y.neural_type.compare(NeuralType(('B', 'D'), LogprobsType()))
        recursive_assert_shape(outB, torch.Size([10]))

        # Assert non-homogeneous type assertions
        assert outB[0].neural_type.compare(NeuralType(('B',), LogprobsType())) == NeuralTypeComparisonResult.SAME
        assert outB[1].neural_type.compare(NeuralType(('B',), LogprobsType())) == NeuralTypeComparisonResult.SAME
        assert out_u.neural_type.compare(NeuralType(('B',), LabelsType())) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_multi_forward_type(self):
        class AdaptiveTypeCheck(Typing):
            @property
            def input_types(self):
                if self.mode == 'train':
                    return {"x": NeuralType(('B',), ElementType())}

                elif self.mode == 'infer':
                    return {"y": NeuralType(('B',), ChannelType())}

                elif self.mode == 'eval':
                    return {"x": NeuralType(('B',), ElementType()), "y": NeuralType(('B',), ChannelType())}
                else:
                    raise ValueError("Wrong mode of operation")

            @property
            def output_types(self):
                if self.mode == 'train':
                    return {"u": NeuralType(('B',), ElementType())}

                elif self.mode == 'infer':
                    return {"v": NeuralType(('B',), ChannelType())}

                elif self.mode == 'eval':
                    return {"u": NeuralType(('B',), ElementType()), "v": NeuralType(('B',), ChannelType())}
                else:
                    raise ValueError("Wrong mode of operation")

            def __init__(self):
                self.mode = 'train'

            def __call__(self, **kwargs):
                # Call should call and forward appropriate method in its own mode
                if self.mode == 'train':
                    return self.train_forward(x=kwargs['x'])

                elif self.mode == 'eval':
                    return self.eval_forward(x=kwargs['x'], y=kwargs['y'])

                elif self.mode == 'infer':
                    return self.infer_forward(y=kwargs['y'])

            @typecheck()
            def train_forward(self, x):
                return x + 10

            @typecheck()
            def eval_forward(self, x, y):
                return x - 1, y - 1

            @typecheck()
            def infer_forward(self, y):
                return y - 10

            @property
            def mode(self):
                return self._mode

            @mode.setter
            def mode(self, val):
                if val not in ['train', 'infer', 'eval']:
                    raise ValueError('mode must be either train infer or eval')
                self._mode = val

        obj = AdaptiveTypeCheck()

        x = torch.zeros(10)
        y = torch.full([10], fill_value=5, dtype=torch.int32)

        obj.mode = 'train'
        x = obj(x=x)

        assert torch.all(x == 10)
        assert x.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        obj.mode = 'eval'
        x, y = obj(x=x, y=y)

        assert torch.all(x == 9)
        assert torch.all(y == 4)
        assert x.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME
        assert y.neural_type.compare(NeuralType(('B',), ChannelType())) == NeuralTypeComparisonResult.SAME

        obj.mode = 'infer'
        y = obj(y=y)

        assert torch.all(y == -6)
        assert y.neural_type.compare(NeuralType(('B',), ChannelType())) == NeuralTypeComparisonResult.SAME

        # Now perform assertions of wrong mode with wrong input combinations
        obj.mode = 'train'

        # In train mode, call infer
        with pytest.raises(TypeError):
            _ = obj.eval_forward(x=x, y=y)

        with pytest.raises(TypeError):
            # wrong input + wrong mode
            _ = obj.infer_forward(y=x)

    @pytest.mark.unit
    def test_input_type_override(self):
        class InputTypesOverride(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

            @typecheck(input_types={"y": NeuralType(('B',), CategoricalValuesType())})
            def forward(self, y):
                y -= 1
                return y

        obj = InputTypesOverride()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert hasattr(result, 'neural_type') is False

        # Test override
        result2 = obj.forward(y=torch.zeros(10))

        assert result2.sum() == torch.tensor(-10.0)
        assert hasattr(result2, 'neural_type') is False

    @pytest.mark.unit
    def test_output_type_override(self):
        class OutputTypes(Typing):
            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

            @typecheck(output_types={"z": NeuralType(('B',), CategoricalValuesType())})
            def forward(self, z):
                z -= 1
                return z

        obj = OutputTypes()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert result.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        # Test passing positional args
        # Positional args allowed if input types is not set !
        result = obj(torch.zeros(10))
        assert result.sum() == torch.tensor(10.0)

        # Test override
        result2 = obj.forward(z=torch.zeros(10))

        assert result2.sum() == torch.tensor(-10.0)
        assert hasattr(result2, 'neural_type')
        assert (
            result2.neural_type.compare(NeuralType(('B',), CategoricalValuesType())) == NeuralTypeComparisonResult.SAME
        )

    @pytest.mark.unit
    def test_multi_type_override(self):
        class AdaptiveTypeCheck(Typing):
            @property
            def input_types(self):
                # __call__ assumed to be for inference only,
                # therefore infer types checked at class scope
                return {"y": NeuralType(('B',), ChannelType())}

            @property
            def output_types(self):
                # __call__ assumed to be for inference only,
                # therefore infer types checked at class scope
                return {"v": NeuralType(('B',), ChannelType())}

            def __call__(self, **kwargs):
                # Call should call and forward appropriate method in its own mode
                # Let default "forward" call be the infer mode (this is upto developer)
                # Therefore default class level types == infer types
                return self.infer_forward(y=kwargs['y'])

            @typecheck(
                input_types={"x": NeuralType(('B',), ElementType())},
                output_types={"u": NeuralType(('B',), ElementType())},
            )
            def train_forward(self, x):
                return x + 10

            @typecheck(
                input_types={"x": NeuralType(('B',), ElementType()), "y": NeuralType(('B',), ChannelType())},
                output_types={"u": NeuralType(('B',), ElementType()), "v": NeuralType(('B',), ChannelType())},
            )
            def eval_forward(self, x, y):
                return x - 1, y - 1

            @typecheck(
                input_types={"y": NeuralType(('B',), ChannelType())},
                output_types={"v": NeuralType(('B',), ChannelType())},
            )
            def infer_forward(self, y):
                return y - 10

        obj = AdaptiveTypeCheck()

        x = torch.zeros(10)
        y = torch.full([10], fill_value=5, dtype=torch.int32)

        # infer mode
        y = obj(y=y)

        assert torch.all(y == -5)
        assert y.neural_type.compare(NeuralType(('B',), ChannelType())) == NeuralTypeComparisonResult.SAME

        x, y = obj.eval_forward(x=x, y=y)

        assert torch.all(x == -1)
        assert torch.all(y == -6)
        assert x.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME
        assert y.neural_type.compare(NeuralType(('B',), ChannelType())) == NeuralTypeComparisonResult.SAME

        x = obj.train_forward(x=x)

        assert torch.all(x == 9)
        assert x.neural_type.compare(NeuralType(('B',), ElementType())) == NeuralTypeComparisonResult.SAME

        # In train func, call eval signature
        with pytest.raises(TypeError):
            _ = obj.train_forward(x=x, y=y)

        with pytest.raises(TypeError):
            # wrong input + wrong mode
            _ = obj.infer_forward(x=x)

    @pytest.mark.unit
    def test_disable_typecheck(self):
        class InputOutputTypes(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": NeuralType(('B',), ElementType())}

            @typecheck()
            def __call__(self, x, **kwargs):
                x += 1
                return x

        # Disable typecheck tests
        with typecheck.disable_checks():
            obj = InputOutputTypes()

            # Execute function without kwarg
            result = obj(torch.zeros(10))

            assert result.sum() == torch.tensor(10.0)
            assert hasattr(result, 'neural_type') is False

            # Test passing wrong key for input
            _ = obj(a=torch.zeros(10), x=torch.zeros(5))

    @pytest.mark.unit
    def test_nested_shape_mismatch(self):
        class NestedShapeMismatch(Typing):
            @property
            def input_types(self):
                return {"x": [[NeuralType(('D',), ElementType())]]}  # Each element of nest will have 4 values

            @property
            def output_types(self):
                return {"y": [[NeuralType(('D',), ElementType())]]}  # Each element of nest will have 4 values

            @typecheck()
            def __call__(self, x):
                # v-- this is to satisfy 1 output constraint, python will otherwise interpret x as a 3 output value
                return x

        def bb(dim=4):
            return torch.zeros(dim)

        obj = NestedShapeMismatch()

        # Arbitrary nest 1 (should pass)
        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        result = obj(x=data)

        recursive_assert_shape(result, torch.Size([4]))
        recursive_assert_homogeneous_type(result, NeuralType(('D',), ElementType()))

        # Arbitrary nest 2 (should pass)
        def bb(dim=4):
            return torch.zeros(dim, dim)

        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        # Fails since input shape is incorrect
        with pytest.raises(TypeError):
            _ = obj(x=data)

        # Arbitrary nest 3
        def bb(dim=4):
            return torch.zeros(dim)

        data = [[[bb(), bb(), bb()]], [[bb()], [bb(), bb()]]]
        # Check should fail since nest level is 3!
        with pytest.raises(TypeError):
            result = obj(x=data)

    @pytest.mark.unit
    def test_nested_mixed_shape_mismatch(self):
        class NestedMixedShapeMismatch(Typing):
            @property
            def input_types(self):
                return {"x": [[NeuralType(('D',), ElementType())]]}  # Each element of nest will have 4 values

            @property
            def output_types(self):
                return {"y": [NeuralType(('D',), ElementType())]}  # Each element of nest will have 4 values

            @typecheck()
            def __call__(self, x):
                # v-- this is to satisfy 1 output constraint, python will otherwise interpret x as a 3 output value
                x = x[0]
                return x

        def bb(dim=4):
            return torch.zeros(dim)

        obj = NestedMixedShapeMismatch()

        # Arbitrary nest 1 (should pass)
        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        result = obj(x=data)

        recursive_assert_shape(result, torch.Size([4]))
        recursive_assert_homogeneous_type(result, NeuralType(('D',), ElementType()))

        # Arbitrary nest 2 (should pass)
        def bb(dim=4):
            return torch.zeros(dim, dim)

        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        # Fails since input shape is incorrect
        with pytest.raises(TypeError):
            _ = obj(x=data)

        # Arbitrary nest 3
        def bb(dim=4):
            return torch.zeros(dim)

        data = [[[bb(), bb(), bb()]], [[bb()], [bb(), bb()]]]
        # Check should fail since nest level is 3!
        with pytest.raises(TypeError):
            result = obj(x=data)

    @pytest.mark.unit
    def test_input_container_neural_types(self):
        class NodeA(Typing):
            @property
            def input_types(self):
                return {"x": [NeuralType(('B',), ElementType())]}

            @property
            def output_types(self):
                return {"y": NeuralType(('B', 'D'), LogitsType())}

            @typecheck()
            def __call__(self, x: list()):
                x1, x2, x3 = x  # unpack x
                y = torch.randn(x1.shape[0], 4)
                return y

        nodeA = NodeA()
        outA = nodeA(x=[torch.zeros(10), torch.zeros(10), torch.zeros(10)])

        assert outA.neural_type.compare(NeuralType(('B', 'D'), LogitsType()))

    @pytest.mark.unit
    def test_input_container_neural_types_incorrect(self):
        class NodeA(Typing):
            @property
            def input_types(self):
                # Nest depth level of 2
                return {"x": [[NeuralType(('B',), ElementType())]]}

            @property
            def output_types(self):
                return {"y": NeuralType(('B', 'D'), LogitsType())}

            @typecheck()
            def __call__(self, x: list()):
                x1, x2, x3 = x  # unpack x
                y = torch.randn(x1.shape[0], 4)
                return y

        nodeA = NodeA()
        # Input nest level of 1
        with pytest.raises(TypeError):
            outA = nodeA(x=[torch.zeros(10), torch.zeros(10), torch.zeros(10)])

    @pytest.mark.unit
    def test_output_container_neural_types_incorrect(self):
        class NodeA(Typing):
            @property
            def input_types(self):
                # Nest depth level of 2
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": [[NeuralType(('B', 'D'), LogitsType())]]}

            @typecheck()
            def __call__(self, x):
                y = torch.randn(x.shape[0], 4)
                return y, y, y

        nodeA = NodeA()
        # Input nest level of 1
        with pytest.raises(TypeError):
            outA = nodeA(x=torch.zeros(10))

    @pytest.mark.unit
    def test_output_container_neural_types_no_tuple_wrap(self):
        class NodeA(Typing):
            @property
            def input_types(self):
                # Nest depth level of 2
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": [NeuralType(('B', 'D'), LogitsType())]}

            @typecheck()
            def __call__(self, x):
                y = torch.randn(x.shape[0], 4)
                y = [y, y, y]
                return y

        nodeA = NodeA()
        # Input nest level of 1
        outA = nodeA(x=torch.zeros(10))

        assert len(outA) == 3
        for i in range(len(outA)):
            assert outA[i].neural_type.compare(NeuralType(('B', 'D'), LogitsType()))

    @pytest.mark.unit
    def test_output_container_neural_types_explicit_tuple_wrap(self):
        class NodeA(Typing):
            @property
            def input_types(self):
                # Nest depth level of 2
                return {"x": NeuralType(('B',), ElementType())}

            @property
            def output_types(self):
                return {"y": [NeuralType(('B', 'D'), LogitsType())]}

            @typecheck()
            def __call__(self, x):
                y = torch.randn(x.shape[0], 4)
                y = [y, y, y]
                return (y,)

        nodeA = NodeA()
        # Input nest level of 1
        outA = nodeA(x=torch.zeros(10))

        assert len(outA) == 1
        assert len(outA[0]) == 3
        for i in range(len(outA)):
            assert outA[0][i].neural_type.compare(NeuralType(('B', 'D'), LogitsType()))

    @pytest.mark.unit
    def test_disable_semantic_types_input_output(self):
        class InputOutputTypes(Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B',), LogprobsType())}

            @property
            def output_types(self):
                return {"y": NeuralType(('B',), LabelsType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputOutputTypes()
        result = obj(x=torch.zeros(10))

        assert result.sum() == torch.tensor(10.0)
        assert result.neural_type.compare(NeuralType(('B',), LabelsType())) == NeuralTypeComparisonResult.SAME

        # Test that input is provided with wrong type and semantic checks are not disabled
        with pytest.raises(TypeError):
            input_data = torch.zeros(10)
            input_data.neural_type = NeuralType(('B',), LabelsType())
            _ = obj(x=input_data)

        # Provide input with wrong type after disabling semantic type checks
        with typecheck.disable_semantic_checks():
            input_data = torch.zeros(10)
            input_data.neural_type = NeuralType(('B',), LabelsType())  # Should be LogprobsType()
            result = obj(x=input_data)

            # assert that even if semantic types are disabled, output is attached with appropriate types
            assert result.sum() == torch.tensor(10.0)
            assert result.neural_type.compare(NeuralType(('B',), LabelsType())) == NeuralTypeComparisonResult.SAME
