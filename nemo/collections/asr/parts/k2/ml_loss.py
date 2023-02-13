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

from typing import Optional, Tuple, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.k2.grad_utils import GradExpNormalize
from nemo.collections.asr.parts.k2.utils import (
    create_supervision,
    get_tot_objf_and_finite_mask,
    load_graph,
    make_blank_first,
    prep_padded_densefsavec,
)

from nemo.core.utils.k2_guard import k2  # import k2 from guard module


class MLLoss(torch.nn.Module):
    """
    Maximum Likelihood criterion.
    It is implemented as Connectionist Temporal Classification (CTC) loss,
    but can be extended to support other loss functions (ASG, HMM, ...).
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/ctc.py
    
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
        graph_type: str = "topo",
        token_lm: Optional[Union['k2.Fsa', str]] = None,
    ):
        super().__init__()
        if cfg is not None:
            topo_type = cfg.get("topo_type", topo_type)
            topo_with_self_loops = cfg.get("topo_with_self_loops", topo_with_self_loops)
            graph_type = cfg.get("graph_type", graph_type)
            token_lm = cfg.get("token_lm", token_lm)
        self.blank = blank
        self.num_classes = num_classes
        self.reduction = reduction
        self.pad_fsavec = topo_type == "compact"
        if graph_type == "topo":
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTopologyCompiler as compiler

            self.graph_compiler = compiler(self.num_classes, topo_type, topo_with_self_loops)
        elif graph_type == "token_lm":
            from nemo.collections.asr.parts.k2.graph_compilers import CtcNumGraphCompiler as compiler

            if isinstance(token_lm, str):
                token_lm = load_graph(token_lm)
            self.graph_compiler = compiler(self.num_classes, topo_type, topo_with_self_loops, aux_graph=token_lm)

            raise NotImplementedError("Not tested yet")
        else:
            raise ValueError(f"Invalid value of `graph_type`: {graph_type}.")

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            # and shift targets to emulate blank = 0
            log_probs, targets = make_blank_first(self.blank, log_probs, targets)
        supervisions, order = create_supervision(input_lengths)
        order = order.long()
        targets = targets[order]
        target_lengths = target_lengths[order]
        # PyTorch is doing the log-softmax normalization as part of the CTC computation.
        # More: https://github.com/k2-fsa/k2/issues/575
        log_probs = GradExpNormalize.apply(log_probs, input_lengths, "mean" if self.reduction != "sum" else "none")

        if log_probs.device != self.graph_compiler.device:
            self.graph_compiler.to(log_probs.device)
        num_graphs = self.graph_compiler.compile(targets + 1 if self.pad_fsavec else targets, target_lengths)

        dense_fsa_vec = (
            prep_padded_densefsavec(log_probs, supervisions)
            if self.pad_fsavec
            else k2.DenseFsaVec(log_probs, supervisions)
        )

        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, torch.finfo(torch.float32).max)

        # use_double_scores=True does matter
        # since otherwise it sometimes makes rounding errors
        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        tot_scores = num_tot_scores
        tot_scores, valid_mask = get_tot_objf_and_finite_mask(tot_scores, self.reduction)
        return -tot_scores[valid_mask], valid_mask
