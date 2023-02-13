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

from typing import Optional, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.k2.autograd import sparse_abs
from nemo.collections.asr.parts.k2.classes import GraphIntersectDenseConfig
from nemo.collections.asr.parts.k2.utils import (
    create_sparse_wrapped,
    create_supervision,
    get_tot_objf_and_finite_mask,
    load_graph,
    make_blank_first,
    prep_padded_densefsavec,
    shift_labels_inpl,
)
from nemo.core.utils.k2_guard import k2  # import k2 from guard module
from nemo.utils import logging


class MAPLoss(torch.nn.Module):
    """
    Maximum a Posteriori Probability criterion.
    It is implemented as Lattice-Free Maximum Mutual Information (LF-MMI) 
    and LF-boosted-MMI (LF-bMMI) losses.
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/mmi.py
    
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
        loss_type: str = "mmi",
        token_lm: Optional[Union['k2.Fsa', str]] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        boost_coeff: float = 0.0,
    ):
        super().__init__()
        if cfg is not None:
            topo_type = cfg.get("topo_type", topo_type)
            topo_with_self_loops = cfg.get("topo_with_self_loops", topo_with_self_loops)
            loss_type = cfg.get("loss_type", loss_type)
            token_lm = cfg.get("token_lm", token_lm)
            intersect_pruned = cfg.get("intersect_pruned", intersect_pruned)
            intersect_conf = cfg.get("intersect_conf", intersect_conf)
            boost_coeff = cfg.get("boost_coeff", boost_coeff)
        self.num_classes = num_classes
        self.blank = blank
        self.reduction = reduction
        self.loss_type = loss_type
        self.boost_coeff = boost_coeff
        self.intersect_calc_scores = (
            self._intersect_calc_scores_mmi_pruned if intersect_pruned else self._intersect_calc_scores_mmi_exact
        )
        self.intersect_conf = intersect_conf
        self.topo_type = topo_type
        self.topo_with_self_loops = topo_with_self_loops
        self.pad_fsavec = topo_type == "compact"
        self.graph_compiler = None
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

    def update_graph(self, graph: 'k2.Fsa'):
        self.lm_graph = graph
        lm_graph = self.lm_graph.clone()
        if hasattr(lm_graph, "aux_labels"):
            delattr(lm_graph, "aux_labels")
        labels = lm_graph.labels
        if labels.max() != self.num_classes - 1:
            raise ValueError(f"lm_graph is not compatible with the num_classes: {labels.unique()}, {self.num_classes}")
        if self.pad_fsavec:
            shift_labels_inpl([lm_graph], 1)
        if self.loss_type == "mmi":
            from nemo.collections.asr.parts.k2.graph_compilers import MmiGraphCompiler as compiler
        else:
            raise ValueError(f"Invalid value of `loss_type`: {self.loss_type}.")
        self.graph_compiler = compiler(self.num_classes, self.topo_type, self.topo_with_self_loops, aux_graph=lm_graph)

    def _intersect_calc_scores_mmi_exact(
        self, dense_fsa_vec: k2.DenseFsaVec, num_graphs: 'k2.Fsa', den_graph: 'k2.Fsa', return_lats: bool = True,
    ):
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

        num_den_tot_scores = num_den_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
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

    def _intersect_calc_scores_mmi_pruned(
        self, dense_fsa_vec: k2.DenseFsaVec, num_graphs: 'k2.Fsa', den_graph: 'k2.Fsa', return_lats: bool = True,
    ):
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

        # use_double_scores=True does matter
        # since otherwise it sometimes makes rounding errors
        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        den_tot_scores = den_lats.get_tot_scores(log_semiring=True, use_double_scores=True)

        if return_lats:
            return num_tot_scores, den_tot_scores, num_lats, den_lats
        else:
            return num_tot_scores, den_tot_scores, None, None

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert self.graph_compiler is not None
        boosted = self.boost_coeff != 0.0
        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            # and shift targets to emulate blank = 0
            log_probs, targets = make_blank_first(self.blank, log_probs, targets)
        supervisions, order = create_supervision(input_lengths)
        order = order.long()
        targets = targets[order]
        target_lengths = target_lengths[order]

        if log_probs.device != self.graph_compiler.device:
            self.graph_compiler.to(log_probs.device)

        num_graphs, den_graph = self.graph_compiler.compile(
            targets + 1 if self.pad_fsavec else targets, target_lengths
        )

        dense_fsa_vec = (
            prep_padded_densefsavec(log_probs, supervisions)
            if self.pad_fsavec
            else k2.DenseFsaVec(log_probs, supervisions)
        )

        num_tot_scores, den_tot_scores, num_lats, den_lats = self.intersect_calc_scores(
            dense_fsa_vec, num_graphs, den_graph, boosted
        )

        tot_scores = num_tot_scores - den_tot_scores
        mmi_tot_scores, mmi_valid_mask = get_tot_objf_and_finite_mask(tot_scores, self.reduction)

        if boosted:
            assert num_lats is not None and den_lats is not None

            size = (
                dense_fsa_vec.dim0(),
                dense_fsa_vec.scores.shape[0],
                dense_fsa_vec.scores.shape[1] - 1,
            )
            row_ids = dense_fsa_vec.dense_fsa_vec.shape().row_ids(1)
            num_sparse = create_sparse_wrapped(
                indices=[k2.index_select(row_ids, num_lats.seqframe_idx), num_lats.seqframe_idx, num_lats.phones,],
                values=num_lats.get_arc_post(False, True).exp(),
                size=size,
                min_col_index=0,
            )
            den_sparse = create_sparse_wrapped(
                indices=[k2.index_select(row_ids, den_lats.seqframe_idx), den_lats.seqframe_idx, den_lats.phones,],
                values=den_lats.get_arc_post(False, True).exp(),
                size=size,
                min_col_index=0,
            )

            # NOTE: Due to limited support of PyTorch's autograd for sparse tensors,
            # we cannot use (num_sparse - den_sparse) here
            # TODO (alaptev): propose sparse_abs to k2
            acc_loss = torch.sparse.sum(sparse_abs((num_sparse + (-den_sparse)).coalesce()), (1, 2)).to_dense()
            acc_tot_scores, acc_valid_mask = get_tot_objf_and_finite_mask(acc_loss, self.reduction)
            valid_mask = mmi_valid_mask & acc_valid_mask
            total_loss = self.boost_coeff * acc_tot_scores[valid_mask] - mmi_tot_scores[valid_mask]
        else:
            valid_mask = mmi_valid_mask
            total_loss = -mmi_tot_scores[mmi_valid_mask]
        return total_loss, valid_mask
