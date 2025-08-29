# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from types import SimpleNamespace

import torch.nn as nn

from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs


class MockClassWithCudaGraphs(WithOptionalCudaGraphs):
    def __init__(self):
        super().__init__()
        self.cuda_graphs_used = True

    def disable_cuda_graphs(self) -> bool:
        cuda_graphs_were_used = self.cuda_graphs_used
        self.cuda_graphs_used = False
        return cuda_graphs_were_used != self.cuda_graphs_used

    def maybe_enable_cuda_graphs(self) -> bool:
        cuda_graphs_were_used = self.cuda_graphs_used
        self.cuda_graphs_used = True
        return cuda_graphs_were_used != self.cuda_graphs_used


class MockModuleWithCudaGraphs(MockClassWithCudaGraphs, nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)


class MockModuleWithCudaGraphsByPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.decoding = SimpleNamespace(decoding=MockClassWithCudaGraphs())

    def forward(self, x):
        return self.linear(x)


class TestWithOptionalCudaGraphs:
    def test_module_toggle_cuda_graphs(self):
        module_with_graphs = MockModuleWithCudaGraphs()
        assert module_with_graphs.cuda_graphs_used
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(module_with_graphs)
        assert not module_with_graphs.cuda_graphs_used
        WithOptionalCudaGraphs.enable_cuda_graphs_recursive(module_with_graphs)
        assert module_with_graphs.cuda_graphs_used

    def test_module_toggle_cuda_graphs_by_path(self):
        module_with_graphs_by_path = MockModuleWithCudaGraphsByPath()
        assert module_with_graphs_by_path.decoding.decoding.cuda_graphs_used
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(
            module_with_graphs_by_path, attribute_path="decoding.decoding"
        )
        assert not module_with_graphs_by_path.decoding.decoding.cuda_graphs_used
        WithOptionalCudaGraphs.enable_cuda_graphs_recursive(
            module_with_graphs_by_path, attribute_path="decoding.decoding"
        )
        assert module_with_graphs_by_path.decoding.decoding.cuda_graphs_used
