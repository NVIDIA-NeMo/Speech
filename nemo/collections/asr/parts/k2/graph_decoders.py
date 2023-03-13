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

from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.k2.classes import GraphIntersectDenseConfig
from nemo.collections.asr.parts.k2.loss_mixins import CtcK2Mixin, RnntK2Mixin
from nemo.collections.asr.parts.k2.utils import invert_permutation, load_graph
from nemo.utils import logging


class BaseDecoder(object):
    """Base graph decoder with topology for decoding graph.
    Typically uses the same parameters as for the corresponding loss function.

    Can do decoding and forced alignment.

    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    @abstractmethod
    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):

        if cfg is not None:
            intersect_pruned = cfg.get("intersect_pruned", intersect_pruned)
            intersect_conf = cfg.get("intersect_conf", intersect_conf)
            topo_type = cfg.get("topo_type", topo_type)
            topo_with_self_loops = cfg.get("topo_with_self_loops", topo_with_self_loops)

        self.num_classes = num_classes
        self.blank = blank
        self.intersect_pruned = intersect_pruned
        self.device = device
        self.topo_type = topo_type
        self.topo_with_self_loops = topo_with_self_loops
        self.pad_fsavec = self.topo_type == "ctc_compact"
        self.intersect_conf = intersect_conf
        self.graph_compiler = None  # expected to be initialized in child classes
        self.base_graph = None  # expected to be initialized in child classes
        self.decoding_graph = None

    def to(self, device: torch.device):
        if self.graph_compiler.device != device:
            self.graph_compiler.to(device)
        if self.base_graph.device != device:
            self.base_graph = self.base_graph.to(device)
        if self.decoding_graph is not None and self.decoding_graph.device != device:
            self.decoding_graph = self.decoding_graph.to(device)
        self.device = device

    def update_graph(self, graph: 'k2.Fsa'):
        raise NotImplementedError

    def _decode_impl(
        self,
        log_probs: torch.Tensor,
        supervisions: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union['k2.Fsa', Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self.decoding_graph is None:
            self.decoding_graph = self.base_graph

        if log_probs.device != self.device:
            self.to(log_probs.device)
        emissions_graphs = self._prepare_emissions_graphs(log_probs, supervisions)

        if self.intersect_pruned:
            lats = k2.intersect_dense_pruned(
                a_fsas=self.decoding_graph,
                b_fsas=emissions_graphs,
                search_beam=self.intersect_conf.search_beam,
                output_beam=self.intersect_conf.output_beam,
                min_active_states=self.intersect_conf.min_active_states,
                max_active_states=self.intersect_conf.max_active_states,
            )
        else:
            indices = torch.zeros(emissions_graphs.dim0(), dtype=torch.int32, device=self.device)
            dec_graphs = (
                k2.index_fsa(self.decoding_graph, indices)
                if self.decoding_graph.shape[0] == 1
                else self.decoding_graph
            )
            lats = k2.intersect_dense(dec_graphs, emissions_graphs, self.intersect_conf.output_beam)
        self.decoding_graph = None

        order = supervisions[:, 0]
        if return_lattices:
            lats = k2.index_fsa(lats, invert_permutation(order).to(device=log_probs.device))
            if self.blank != 0:
                # change only ilabels
                # suppose self.blank == self.num_classes - 1
                lats.labels = torch.where(lats.labels == 0, self.blank, lats.labels - 1)
            return lats
        else:
            shortest_path_fsas = k2.index_fsa(
                k2.shortest_path(lats, True), invert_permutation(order).to(device=log_probs.device),
            )
            return self._extract_labels_and_probabilities(shortest_path_fsas, return_ilabels, output_aligned)

    def decode(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union['k2.Fsa', Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        log_probs, supervisions, _, _ = self._prepare_log_probs_and_targets(log_probs, log_probs_length, None, None)
        return self._decode_impl(
            log_probs,
            supervisions,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )

    def align(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        log_probs, supervisions, targets, target_lengths = self._prepare_log_probs_and_targets(
            log_probs, log_probs_length, targets, target_lengths
        )
        order = supervisions[:, 0].to(dtype=torch.long)
        self.decoding_graph = self.graph_compiler.compile(targets[order], target_lengths[order])
        return self._decode_impl(
            log_probs,
            supervisions,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )


class CtcDecoder(BaseDecoder, CtcK2Mixin):
    """Regular CTC graph decoder with custom topologies.
    Available topologies:
        -   `default`, with or without self-loops
        -   `compact`, with or without self-loops
        -   `shared_blank`, with or without self-loops
        -   `minimal`, without self-loops

    Can do decoding and forced alignment.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            num_classes, blank, cfg, intersect_pruned, intersect_conf, topo_type, topo_with_self_loops, device
        )
        from nemo.collections.asr.parts.k2.graph_compilers import CtcTopologyCompiler

        self.graph_compiler = CtcTopologyCompiler(
            self.num_classes, self.blank, self.topo_type, self.topo_with_self_loops, self.device
        )
        self.base_graph = k2.create_fsa_vec([self.graph_compiler.ctc_topo_inv.invert()]).to(self.device)


class RnntAligner(BaseDecoder, RnntK2Mixin):
    """RNNT graph decoder with the `minimal` topology.
    If predictor_window_size is not provided, this decoder works as a Viterbi over regular RNNT lattice.
    With predictor_window_size provided, it applies uniform pruning when compiling Emission FSAs
    to reduce memory and compute consumption.

    Can only do forced alignment.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        predictor_window_size: int = 0,
        predictor_step_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        if cfg is not None:
            topo_type = cfg.get("topo_type", topo_type)
            predictor_window_size = cfg.get("predictor_window_size", predictor_window_size)
            predictor_step_size = cfg.get("predictor_step_size", predictor_step_size)
        if topo_type != "minimal":
            raise NotImplementedError(f"Only topo_type=`minimal` is supported at the moment.")
        super().__init__(
            num_classes, blank, cfg, intersect_pruned, intersect_conf, topo_type, topo_with_self_loops, device
        )
        self.predictor_window_size = predictor_window_size
        self.predictor_step_size = predictor_step_size
        from nemo.collections.asr.parts.k2.graph_compilers import RnntTopologyCompiler

        self.graph_compiler = RnntTopologyCompiler(
            self.num_classes,
            self.blank,
            self.topo_type,
            self.topo_with_self_loops,
            self.device,
            max_adapter_length=self.predictor_window_size,
        )
        self.base_graph = self.graph_compiler.base_graph

    def decode(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union['k2.Fsa', Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        raise NotImplementedError("RNNT decoding is not implemented. Only .align(...) method is supported.")

    def align(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert self.predictor_window_size == 0 or log_probs.size(2) <= self.predictor_window_size + 1

        return super().align(
            log_probs,
            log_probs_length,
            targets,
            target_lengths,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )


class TokenLMDecoder(BaseDecoder):
    """Graph decoder with token_lm-based decoding graph.
    Available topologies:
        -   `default`, with or without self-loops
        -   `compact`, with or without self-loops
        -   `shared_blank`, with or without self-loops
        -   `minimal`, without self-loops

    Can do decoding and forced alignment.

    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        token_lm: Optional[Union['k2.Fsa', str]] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            num_classes, blank, cfg, intersect_pruned, intersect_conf, topo_type, topo_with_self_loops, device
        )
        if cfg is not None:
            token_lm = cfg.get("token_lm", token_lm)
        if token_lm is not None:
            self.token_lm = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            if self.token_lm is not None:
                self.update_graph(self.token_lm)
            else:
                logging.warning(
                    f"""token_lm was set to None. Use this for debug 
                                purposes only or call .update_graph(token_lm) before using."""
                )
        else:
            logging.warning(
                f"""token_lm was set to None. Use this for debug
                            purposes only or call .update_graph(token_lm) before using."""
            )
            self.token_lm = None

    def update_graph(self, graph: 'k2.Fsa'):
        self.token_lm = graph
        token_lm = self.token_lm.clone()
        if hasattr(token_lm, "aux_labels"):
            delattr(token_lm, "aux_labels")
        labels = token_lm.labels
        if labels.max() != self.num_classes - 1:
            raise ValueError(f"token_lm is not compatible with the num_classes: {labels.unique()}, {self.num_classes}")
        self.graph_compiler = CtcNumGraphCompiler(
            self.num_classes, self.blank, self.topo_type, self.topo_with_self_loops, self.device, token_lm
        )
        self.base_graph = k2.create_fsa_vec([self.graph_compiler.base_graph]).to(self.device)
