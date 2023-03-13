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

from abc import abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.k2.classes import GraphIntersectDenseConfig
from nemo.collections.asr.parts.k2.loss_mixins import CtcK2Mixin
from nemo.collections.asr.parts.k2.ml_loss import MLLoss
from nemo.collections.asr.parts.k2.utils import (
    create_sparse_wrapped,
    get_tot_objf_and_finite_mask,
    invert_permutation,
    load_graph,
)
from nemo.core.utils.k2_guard import k2  # import k2 from guard module
from nemo.utils import logging


class MAPLoss(MLLoss):
    """
    Maximum a Posteriori Probability criterion.
    It implements Lattice-Free Maximum Mutual Information (LF-MMI) and LF-boosted-MMI (LF-bMMI) losses.
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/mmi.py
    
    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    @abstractmethod
    def __init__(
        self,
        num_classes: int,
        blank: int,
        reduction: str,
        cfg: Optional[DictConfig] = None,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        token_lm: Optional[Union['k2.Fsa', str]] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        boost_coeff: float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            blank=blank,
            reduction=reduction,
            cfg=cfg,
            topo_type=topo_type,
            topo_with_self_loops=topo_with_self_loops,
        )
        if cfg is not None:
            token_lm = cfg.get("token_lm", token_lm)
            intersect_pruned = cfg.get("intersect_pruned", intersect_pruned)
            intersect_conf = cfg.get("intersect_conf", intersect_conf)
            boost_coeff = cfg.get("boost_coeff", boost_coeff)
        self.boost_coeff = boost_coeff
        self._intersect_calc_scores_impl = (
            self._intersect_calc_scores_impl_pruned if intersect_pruned else self._intersect_calc_scores_impl_exact_opt
        )
        self.intersect_conf = intersect_conf
        self.graph_compiler = None  # expected to be initialized in .update_graph(...)
        if token_lm is None:
            logging.warning(
                f"""token_lm is empty. 
                            Trainable token_lm is not supported yet. 
                            Please call .update_graph(token_lm) before using."""
            )
        else:
            self.lm_graph = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            if self.lm_graph is None:
                raise ValueError(f"""lm_graph is empty.""")
            else:
                self.update_graph(self.lm_graph)

    @abstractmethod
    def update_graph(self, graph: 'k2.Fsa'):
        # expected to be set in child classes
        raise NotImplementedError

    def _intersect_calc_scores_impl_exact_opt(
        self, dense_fsa_vec: 'k2.DenseFsaVec', num_graphs: 'k2.Fsa', den_graph: 'k2.Fsa', return_lats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional['k2.Fsa'], Optional['k2.Fsa']]:
        """Inner intersection method.
        Does joint (simultaneous) exact intersection of dense_fsa_vec against num_graphs and den_graph.
        
        Optiolally returns the numerator and the denominator lattices.
        """
        device = dense_fsa_vec.device
        assert device == num_graphs.device and device == den_graph.device

        num_fsas = num_graphs.shape[0]
        assert dense_fsa_vec.dim0() == num_fsas

        den_graph = den_graph.clone()
        num_graphs = num_graphs.clone()

        num_den_graphs = k2.cat([num_graphs, den_graph])

        # NOTE: The a_to_b_map in k2.intersect_dense must be sorted
        # so the following reorders num_den_graphs.

        # [0, 1, 2, ... ]
        num_graphs_indexes = torch.arange(num_fsas, dtype=torch.int32)

        # [num_fsas, num_fsas, num_fsas, ... ]
        den_graph_indexes = torch.tensor([num_fsas] * num_fsas, dtype=torch.int32)

        # [0, num_fsas, 1, num_fsas, 2, num_fsas, ... ]
        num_den_graphs_indexes = torch.stack([num_graphs_indexes, den_graph_indexes]).t().reshape(-1).to(device)

        num_den_reordered_graphs = k2.index_fsa(num_den_graphs, num_den_graphs_indexes)

        # [[0, 1, 2, ...]]
        a_to_b_map = torch.arange(num_fsas, dtype=torch.int32).reshape(1, -1)

        # [[0, 1, 2, ...]] -> [0, 0, 1, 1, 2, 2, ... ]
        a_to_b_map = a_to_b_map.repeat(2, 1).t().reshape(-1).to(device)

        num_den_lats = k2.intersect_dense(
            a_fsas=num_den_reordered_graphs,
            b_fsas=dense_fsa_vec,
            output_beam=self.intersect_conf.output_beam,
            a_to_b_map=a_to_b_map,
            seqframe_idx_name="seqframe_idx" if return_lats else None,
        )

        num_den_tot_scores = num_den_lats.get_tot_scores(log_semiring=True, use_double_scores=False)
        num_tot_scores = num_den_tot_scores[::2]
        den_tot_scores = num_den_tot_scores[1::2]

        if return_lats:
            lat_slice = torch.arange(num_fsas, dtype=torch.int32).to(device) * 2
            return (
                num_tot_scores,
                den_tot_scores,
                k2.index_fsa(num_den_lats, lat_slice),
                k2.index_fsa(num_den_lats, lat_slice + 1),
            )
        else:
            return num_tot_scores, den_tot_scores, None, None

    def _intersect_calc_scores_impl_pruned(
        self, dense_fsa_vec: 'k2.DenseFsaVec', num_graphs: 'k2.Fsa', den_graph: 'k2.Fsa', return_lats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional['k2.Fsa'], Optional['k2.Fsa']]:
        """Inner intersection method.
        Does exact intersection of dense_fsa_vec against num_graphs and pruned intersection against den_graph.
        
        Optiolally returns the numerator and the denominator lattices.
        """
        device = dense_fsa_vec.device
        assert device == num_graphs.device and device == den_graph.device

        num_fsas = num_graphs.shape[0]
        assert dense_fsa_vec.dim0() == num_fsas

        num_lats = k2.intersect_dense(
            a_fsas=num_graphs,
            b_fsas=dense_fsa_vec,
            output_beam=self.intersect_conf.output_beam,
            seqframe_idx_name="seqframe_idx" if return_lats else None,
        )
        den_lats = k2.intersect_dense_pruned(
            a_fsas=den_graph,
            b_fsas=dense_fsa_vec,
            search_beam=self.intersect_conf.search_beam,
            output_beam=self.intersect_conf.output_beam,
            min_active_states=self.intersect_conf.min_active_states,
            max_active_states=self.intersect_conf.max_active_states,
            seqframe_idx_name="seqframe_idx" if return_lats else None,
        )

        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=False)
        den_tot_scores = den_lats.get_tot_scores(log_semiring=True, use_double_scores=False)

        if return_lats:
            return num_tot_scores, den_tot_scores, num_lats, den_lats
        else:
            return num_tot_scores, den_tot_scores, None, None

    def _intersect_calc_scores(
        self, emissions_graphs: 'k2.DenseFsaVec', supervision_graphs: Any, supervisions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Intersects emissions_graphs with supervision_graphs and calculates lattice scores.
        This version implicitly assumes supervision_graphs to be a pair of the numerator and the denominator FSAs.

        It can also calculate accuracy between the numerator and the denominator lattices to use it as additional loss.

        Can be overridden.
        """
        boosted = self.boost_coeff != 0.0
        num_tot_scores, den_tot_scores, num_lats, den_lats = self._intersect_calc_scores_impl(
            emissions_graphs, supervision_graphs[0], supervision_graphs[1], boosted
        )

        inverted_batch_order = invert_permutation(supervisions[:, 0].to(dtype=torch.long))
        self.__batch_order = None
        tot_scores = (num_tot_scores - den_tot_scores)[inverted_batch_order]
        mmi_tot_scores, mmi_valid_mask = get_tot_objf_and_finite_mask(tot_scores, self.reduction)

        if boosted:
            assert num_lats is not None and den_lats is not None

            size = (
                emissions_graphs.dim0(),
                emissions_graphs.scores.shape[0],
                emissions_graphs.scores.shape[1] - 1,
            )
            row_ids = emissions_graphs.emissions_graphs.shape().row_ids(1)
            num_sparse = create_sparse_wrapped(
                indices=[k2.index_select(row_ids, num_lats.seqframe_idx), num_lats.seqframe_idx, num_lats.phones,],
                values=num_lats.get_arc_post(False, True).exp(),
                size=size,
                min_col_index=0,
            )
            del num_lats
            den_sparse = create_sparse_wrapped(
                indices=[k2.index_select(row_ids, den_lats.seqframe_idx), den_lats.seqframe_idx, den_lats.phones,],
                values=den_lats.get_arc_post(False, True).exp(),
                size=size,
                min_col_index=0,
            )
            del den_lats

            acc_loss = torch.sparse.sum((num_sparse - den_sparse).coalesce().abs(), (1, 2)).to_dense()
            del num_sparse, den_sparse

            acc_tot_scores, acc_valid_mask = get_tot_objf_and_finite_mask(acc_loss, self.reduction)
            valid_mask = mmi_valid_mask & acc_valid_mask
            total_loss = (
                (self.boost_coeff * acc_tot_scores[inverted_batch_order][valid_mask] - mmi_tot_scores[valid_mask])
                if self.reduction == "none"
                else self.boost_coeff * acc_tot_scores - mmi_tot_scores
            )
        else:
            valid_mask = mmi_valid_mask
            total_loss = -mmi_tot_scores[valid_mask] if self.reduction == "none" else -mmi_tot_scores
        return total_loss, valid_mask


class CtcMmiLoss(MAPLoss, CtcK2Mixin):
    """MMI loss with custom CTC topologies.
    Available topologies:
        -   `default`, with or without self-loops
        -   `compact`, with or without self-loops
        -   `shared_blank`, with or without self-loops
        -   `minimal`, without self-loops

    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        reduction: str,
        cfg: Optional[DictConfig] = None,
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        token_lm: Optional[Union['k2.Fsa', str]] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        boost_coeff: float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            blank=blank,
            reduction=reduction,
            cfg=cfg,
            topo_type=topo_type,
            topo_with_self_loops=topo_with_self_loops,
            token_lm=token_lm,
            intersect_pruned=intersect_pruned,
            intersect_conf=intersect_conf,
            boost_coeff=boost_coeff,
        )

    def update_graph(self, graph: 'k2.Fsa'):
        self.lm_graph = graph
        lm_graph = self.lm_graph.clone()
        if hasattr(lm_graph, "aux_labels"):
            delattr(lm_graph, "aux_labels")
        labels = lm_graph.labels
        if labels.max() != self.num_classes - 1:
            raise ValueError(f"lm_graph is not compatible with the num_classes: {labels.unique()}, {self.num_classes}")
        from nemo.collections.asr.parts.k2.graph_compilers import MmiGraphCompiler as compiler

        self.graph_compiler = compiler(
            self.num_classes, self.blank, self.topo_type, self.topo_with_self_loops, aux_graph=lm_graph
        )
