# ! /usr/bin/python
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

from typing import Optional

import torch
from omegaconf import DictConfig

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType


class LatticeLoss(Loss):
    """Family of loss functions based on various lattice scores.

    Note:
        Requires k2 v1.14 or later to be installed to use this loss function.

    Losses can be selected via the config, and optionally be passed keyword arguments as follows.

    Examples:
        .. code-block:: yaml

            model:  # Model config
                ...
                graph_module_cfg:  # Config for graph modules, e.g. LatticeLoss
                    criterion_type: "map"
                    loss_type: "mmi"
                    split_batch_size: 0
                    backend_cfg:
                        topo_type: "default"       # other options: "compact", "shared_blank", "minimal"
                        topo_with_self_loops: true
                        token_lm: <token_lm_path>  # must be provided for criterion_type: "map"

    Args:
        num_classes: Number of target classes for the decoder network to predict.
            (Excluding the blank token).

        reduction: Type of reduction to perform on loss. Possible values are `mean_batch`, `mean`, `sum`, or None.
            None will return a torch vector comprising the individual loss values of the batch.

        backend: Which backend to use for loss calculation. Currently only `k2` is supported.

        criterion_type: Type of criterion to use. Choices: `ml` and `map`, 
            with `ml` standing for Maximum Likelihood and `map` for Maximum A Posteriori Probability.

        loss_type: Type of the loss function to use. Choices: `ctc` and `rnnt` for `ml`, and `mmi` for `map`.

        split_batch_size: Local batch size. Used for memory consumption reduction at the cost of speed performance.
            Effective if complies 0 < split_batch_size < batch_size.

        graph_module_cfg: Optional Dict of (str, value) pairs that are passed to the backend loss function.
    """

    @property
    def input_types(self):
        """Input types definitions for LatticeLoss.
        """
        return {
            "log_probs": NeuralType(("B", "T", "D") if self._3d_input else ("B", "T", "T", "D"), LogprobsType()),
            "targets": NeuralType(("B", "T"), LabelsType()),
            "input_lengths": NeuralType(tuple("B"), LengthsType()),
            "target_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for LatticeLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        num_classes: int,
        reduction: str = "mean_batch",
        backend: str = "k2",
        criterion_type: str = "ml",
        loss_type: str = "ctc",
        split_batch_size: int = 0,
        graph_module_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self._blank = num_classes
        self.split_batch_size = split_batch_size
        inner_reduction = None
        if reduction == "mean_batch":
            inner_reduction = "none"
            self._apply_batch_mean = True
        elif reduction in ["sum", "mean", "none"]:
            inner_reduction = reduction
            self._apply_batch_mean = False

        # we assume that self._blank + 1 == num_classes
        if backend == "k2":
            if criterion_type == "ml":
                if loss_type == "ctc":
                    from nemo.collections.asr.parts.k2.ml_loss import CtcLoss as K2Loss
                elif loss_type == "rnnt":
                    from nemo.collections.asr.parts.k2.ml_loss import RnntLoss as K2Loss
                else:
                    raise ValueError(f"Unsupported `loss_type`: {loss_type}.")
            elif criterion_type == "map":
                if loss_type == "ctc":
                    from nemo.collections.asr.parts.k2.map_loss import CtcMmiLoss as K2Loss
                else:
                    raise ValueError(f"Unsupported `loss_type`: {loss_type}.")
            else:
                raise ValueError(f"Unsupported `criterion_type`: {criterion_type}.")

            self._loss = K2Loss(
                num_classes=self._blank + 1, blank=self._blank, reduction=inner_reduction, cfg=graph_module_cfg,
            )
        elif backend == "gtn":
            raise NotImplementedError(f"Backend {backend} is not supported.")
        else:
            raise ValueError(f"Invalid value of `backend`: {backend}.")

        self.criterion_type = criterion_type
        self.loss_type = loss_type
        self._3d_input = self.loss_type != "rnnt"

        if self.split_batch_size > 0:
            # don't need to guard grad_utils
            from nemo.collections.asr.parts.k2.grad_utils import PartialGrad

            self._partial_loss = PartialGrad(self._loss)

    def update_graph(self, graph):
        """Updates graph of the backend loss function.
        """
        if self.criterion_type != "ml":
            self._loss.update_graph(graph)

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary

        assert not (torch.isnan(log_probs).any() or torch.isinf(log_probs).any())

        log_probs = log_probs.float()
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        batch_size = log_probs.shape[0]
        if self.split_batch_size > 0 and self.split_batch_size <= batch_size:
            loss_list = []
            for batch_idx in range(0, batch_size, self.split_batch_size):
                begin = batch_idx
                end = min(begin + self.split_batch_size, batch_size)
                input_lengths_part = input_lengths[begin:end]
                log_probs_part = log_probs[begin:end, : input_lengths_part.max()]
                target_lengths_part = target_lengths[begin:end]
                targets_part = targets[begin:end, : target_lengths_part.max()]
                loss_part, _ = (
                    self._partial_loss(log_probs_part, targets_part, input_lengths_part, target_lengths_part)
                    if log_probs_part.requires_grad
                    else self._loss(log_probs_part, targets_part, input_lengths_part, target_lengths_part)
                )
                del log_probs_part, targets_part, input_lengths_part, target_lengths_part
                loss_list.append(loss_part)
            loss = torch.cat(loss_list, 0)
        else:
            loss, _ = self._loss(
                log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths,
            )
        if self._apply_batch_mean:
            # torch.mean gives nan if loss is empty
            loss = torch.mean(loss) if loss.nelement() > 0 else torch.sum(loss)
        return loss
