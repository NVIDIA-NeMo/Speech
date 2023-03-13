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

from nemo.collections.asr.parts.k2.utils import add_self_loops, compose_with_self_loops, intersect_with_self_loops

from nemo.core.utils.k2_guard import k2  # import k2 from guard module


class CtcTopologyCompiler(object):
    """Default graph compiler.
    It applies its topology to the input token sequence to compile the supervision graph.
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/training/ctc_graph.py
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.topo_type = topo_type
        self.device = device
        from nemo.collections.asr.parts.k2.topologies import build_topo

        self.base_graph = k2.arc_sort(build_topo(topo_type, list(range(num_classes)), blank, topo_with_self_loops)).to(
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
        label_graph = k2.linear_fsa(token_ids_list).to(self.device)
        label_graph.aux_labels = label_graph.labels.clone()
        supervision_graphs = compose_with_self_loops(self.base_graph, label_graph)
        supervision_graphs = k2.arc_sort(supervision_graphs).to(self.device)

        # make sure the gradient is not accumulated
        supervision_graphs.requires_grad_(False)
        return supervision_graphs


class CtcNumGraphCompiler(CtcTopologyCompiler):
    """Graph compiler with auxiliary graph to compose with the topology.
    The supervision graph contains the auxiliary graph information.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
        aux_graph: Optional['k2.Fsa'] = None,
    ):
        super().__init__(num_classes, blank, topo_type, topo_with_self_loops, device)
        if aux_graph is None:
            self.decoding_graph = k2.create_fsa_vec([self.ctc_topo_inv.invert()]).to(self.device)
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
    The decoding graph is a composition of the auxiliary graph and the topology.
    It is returned along with the supervision graph on every compile() call.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
        aux_graph: Optional['k2.Fsa'] = None,
    ):
        super().__init__(num_classes, blank, topo_type, topo_with_self_loops, device, aux_graph)
        if aux_graph is None:
            self.decoding_graph = k2.create_fsa_vec([self.ctc_topo_inv.invert()]).to(self.device)
        else:
            self.decoding_graph = k2.create_fsa_vec([self.base_graph.detach()]).to(self.device)

    def to(self, device: torch.device):
        if self.decoding_graph is not None:
            self.decoding_graph = self.decoding_graph.to(device)
        super().to(device)

    def compile(
        self, targets: torch.Tensor, target_lengths: torch.Tensor, aux_graph: Optional['k2.Fsa'] = None,
    ) -> Tuple['k2.Fsa', 'k2.Fsa']:
        supervision_graphs = super().compile(targets, target_lengths, aux_graph)
        if aux_graph is None and self.decoding_graph is None:
            raise ValueError(
                f"At least one of aux_graph and self.decoding_graph must be set: {aux_graph}, {self.decoding_graph}"
            )
        elif aux_graph is not None:
            self.decoding_graph = k2.create_fsa_vec([self.base_graph.detach()]).to(self.device)
        return supervision_graphs, self.decoding_graph


class RnntTopologyCompiler(CtcTopologyCompiler):
    """Default graph compiler for RNNT loss.
    Each supervision graph is composed with the corresponding RNNT emission adapter.

    If max_adapter_length is provided, the maximum adapter length is limited.

    Note:
      The actual number of classes is `num_classes` + 1 with <eps> as the class 0.

    Warning:
      It is currently not recommended to use topologies other than "minimal".
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        topo_type: str = "minimal",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
        max_adapter_length: int = 0,
    ):
        if topo_type == "compact":
            raise NotImplementedError(f"This compiler does not support topo_type==`compact`.")
        super().__init__(num_classes, blank, topo_type, topo_with_self_loops, device)
        from nemo.collections.asr.parts.k2.topologies import RnntEmissionAdapterBuilder

        self.max_adapter_length = max_adapter_length
        self._builder = RnntEmissionAdapterBuilder(list(range(num_classes)), blank, num_classes)

    def compile(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> 'k2.Fsa':
        supervision_graphs = add_self_loops(super().compile(targets, target_lengths), self._builder.eps_num, "input")

        adapters = self._builder(
            torch.where(target_lengths > self.max_adapter_length, self.max_adapter_length, target_lengths)
            if self.max_adapter_length > 0 and self.max_adapter_length < target_lengths.max()
            else target_lengths
        ).to(device=self.device)
        return k2.intersect(adapters, supervision_graphs, treat_epsilons_specially=False)
