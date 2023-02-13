# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) 2020, Xiaomi CORPORATION.  All rights reserved.
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

from typing import Optional, Tuple

import torch

from nemo.collections.asr.parts.k2.topologies import build_topo
from nemo.collections.asr.parts.k2.utils import compose_with_self_loops, intersect_with_self_loops

from nemo.core.utils.k2_guard import k2  # import k2 from guard module


class CtcTopologyCompiler(object):
    """Default graph compiler.
    It applies its topology to the input token sequence to compile the numerator graph.
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/training/ctc_graph.py
    """

    def __init__(
        self,
        num_classes: int,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.topo_type = topo_type
        self.device = device
        self.base_graph = k2.arc_sort(build_topo(topo_type, list(range(num_classes)), topo_with_self_loops)).to(
            self.device
        )
        self.ctc_topo_inv = k2.arc_sort(self.base_graph.invert())

    def to(self, device: torch.device):
        self.ctc_topo_inv = self.ctc_topo_inv.to(device)
        if self.base_graph is not None:
            self.base_graph = self.base_graph.to(device)
        self.device = device

    def compile(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> 'k2.Fsa':
        token_ids_list = [t[:l].tolist() for t, l in zip(targets, target_lengths)]
        # see https://github.com/k2-fsa/k2/issues/835
        label_graph = k2.linear_fsa(token_ids_list).to(self.device)
        label_graph.aux_labels = label_graph.labels.clone()
        decoding_graphs = compose_with_self_loops(self.base_graph, label_graph)
        decoding_graphs = k2.arc_sort(decoding_graphs).to(self.device)

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs


class CtcNumGraphCompiler(CtcTopologyCompiler):
    """Graph compiler with auxiliary graph to compose with the topology.
    The numerator graph contains the auxiliary graph information.
    """

    def __init__(
        self,
        num_classes: int,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
        aux_graph: Optional['k2.Fsa'] = None,
    ):
        super().__init__(num_classes, topo_type, topo_with_self_loops, device)
        if aux_graph is None:
            self.den_graph = k2.create_fsa_vec([self.ctc_topo_inv.invert()]).to(self.device)
        else:
            self.base_graph = intersect_with_self_loops(self.ctc_topo_inv, aux_graph).invert_()
            self.base_graph = k2.arc_sort(self.base_graph).to(self.device)

    def compile(
        self, targets: torch.Tensor, target_lengths: torch.Tensor, aux_graph: Optional['k2.Fsa'] = None,
    ) -> 'k2.Fsa':
        if aux_graph is None and self.base_graph is None:
            raise ValueError(
                f"At least one of aux_graph and self.base_graph must be set: {aux_graph}, {self.base_graph}"
            )
        elif aux_graph is not None:
            self.base_graph = intersect_with_self_loops(self.ctc_topo_inv, aux_graph).invert()
            self.base_graph = k2.arc_sort(self.base_graph).to(self.device)
        return super().compile(targets, target_lengths)


class MmiGraphCompiler(CtcNumGraphCompiler):
    """Graph compiler for MMI loss.
    The denominator graph is a composition of the auxiliary graph and the topology.
    It is returned along with the numerator graph on every compile() call.
    """

    def __init__(
        self,
        num_classes: int,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
        aux_graph: Optional['k2.Fsa'] = None,
    ):
        super().__init__(num_classes, topo_type, topo_with_self_loops, device, aux_graph)
        if aux_graph is None:
            self.den_graph = k2.create_fsa_vec([self.ctc_topo_inv.invert()]).to(self.device)
        else:
            self.den_graph = k2.create_fsa_vec([self.base_graph.detach()]).to(self.device)

    def to(self, device: torch.device):
        if self.den_graph is not None:
            self.den_graph = self.den_graph.to(device)
        super().to(device)

    def compile(
        self, targets: torch.Tensor, target_lengths: torch.Tensor, aux_graph: Optional['k2.Fsa'] = None,
    ) -> Tuple['k2.Fsa', 'k2.Fsa']:
        num_graphs = super().compile(targets, target_lengths, aux_graph)
        if aux_graph is None and self.den_graph is None:
            raise ValueError(
                f"At least one of aux_graph and self.den_graph must be set: {aux_graph}, {self.den_graph}"
            )
        elif aux_graph is not None:
            self.den_graph = k2.create_fsa_vec([self.base_graph.detach()]).to(self.device)
        return num_graphs, self.den_graph
