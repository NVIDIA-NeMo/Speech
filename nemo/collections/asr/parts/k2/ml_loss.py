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
from typing import Any, Optional, Tuple

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.k2.graph_compilers import CtcTopologyCompiler, RnntTopologyCompiler
from nemo.collections.asr.parts.k2.loss_mixins import CtcK2Mixin, RnntK2Mixin
from nemo.collections.asr.parts.k2.utils import get_tot_objf_and_finite_mask, invert_permutation
from nemo.core.utils.k2_guard import k2  # import k2 from guard module


class MLLoss(torch.nn.Module):
    """
    Maximum Likelihood criterion.
    It implements Connectionist Temporal Classification (CTC) loss,
    but can be extended to support other loss functions (ASG, HMM, RNNT, ...).
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/ctc.py
    
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
    ):
        super().__init__()
        if cfg is not None:
            topo_type = cfg.get("topo_type", topo_type)
            topo_with_self_loops = cfg.get("topo_with_self_loops", topo_with_self_loops)
        self.blank = blank
        self.num_classes = num_classes
        self.reduction = reduction
        self.topo_type = topo_type
        self.topo_with_self_loops = topo_with_self_loops
        self.pad_fsavec = topo_type == "compact"
        self.graph_compiler = None  # expected to be initialized in child classes

    def _prepare_graphs_for_intersection(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple['k2.DenseFsaVec', Any, torch.Tensor]:
        """Converts input tensors to FST graphs:
            log_probs to supervision_graphs (DenseFsaVec)
            targets to supervision_graphs
        Can be overridden.
        """
        log_probs, supervisions, targets, target_lengths = self._prepare_log_probs_and_targets(
            log_probs, input_lengths, targets, target_lengths
        )
        log_probs = self._maybe_normalize_gradients(log_probs, supervisions[:, -1].to(dtype=torch.long))
        emissions_graphs = self._prepare_emissions_graphs(log_probs, supervisions)
        del log_probs

        if emissions_graphs.device != self.graph_compiler.device:
            self.graph_compiler.to(emissions_graphs.device)
        order = supervisions[:, 0].to(dtype=torch.long)
        supervision_graphs = self.graph_compiler.compile(targets[order], target_lengths[order])

        return emissions_graphs, supervision_graphs, supervisions

    def _intersect_calc_scores(
        self, emissions_graphs: 'k2.DenseFsaVec', supervision_graphs: Any, supervisions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Intersects emissions_graphs with supervision_graphs and calculates lattice scores.
        Can be overridden.
        """
        lats = k2.intersect_dense(supervision_graphs, emissions_graphs, torch.finfo(torch.float32).max / 10)
        del emissions_graphs

        num_tot_scores = lats.get_tot_scores(log_semiring=True, use_double_scores=False)
        del lats
        tot_scores = num_tot_scores[invert_permutation(supervisions[:, 0].to(dtype=torch.long))]
        tot_scores, valid_mask = get_tot_objf_and_finite_mask(tot_scores, self.reduction)
        return -tot_scores[valid_mask] if self.reduction == "none" else -tot_scores, valid_mask

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.graph_compiler is not None

        emissions_graphs, supervision_graphs, supervisions = self._prepare_graphs_for_intersection(
            log_probs, targets, input_lengths, target_lengths
        )
        scores, mask = self._intersect_calc_scores(emissions_graphs, supervision_graphs, supervisions)
        return scores, mask


class CtcLoss(MLLoss, CtcK2Mixin):
    """Regular CTC loss with custom topologies.
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
    ):
        super().__init__(
            num_classes=num_classes,
            blank=blank,
            reduction=reduction,
            cfg=cfg,
            topo_type=topo_type,
            topo_with_self_loops=topo_with_self_loops,
        )
        self.graph_compiler = CtcTopologyCompiler(
            self.num_classes, self.blank, self.topo_type, self.topo_with_self_loops
        )


class RnntLoss(MLLoss, RnntK2Mixin):
    """RNNT loss with the `minimal` topology.
    If predictor_window_size is not provided, this loss works as regular RNNT.
    With predictor_window_size provided, it applies uniform pruning when compiling Emission FSAs
    to reduce memory and compute consumption.
    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        reduction: str,
        cfg: Optional[DictConfig] = None,
        topo_type: str = "minimal",
        topo_with_self_loops: bool = True,
        predictor_window_size: int = 0,
        predictor_step_size: int = 1,
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
            topo_type = cfg.get("topo_type", topo_type)
            predictor_window_size = cfg.get("predictor_window_size", predictor_window_size)
            predictor_step_size = cfg.get("predictor_step_size", predictor_step_size)
        if topo_type != "minimal":
            raise NotImplementedError(f"Only topo_type=`minimal` is supported at the moment.")
        self.predictor_window_size = predictor_window_size
        self.predictor_step_size = predictor_step_size
        self.graph_compiler = RnntTopologyCompiler(
            self.num_classes,
            self.blank,
            self.topo_type,
            self.topo_with_self_loops,
            max_adapter_length=self.predictor_window_size,
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.predictor_window_size == 0 or log_probs.size(2) <= self.predictor_window_size + 1

        return super().forward(
            log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
