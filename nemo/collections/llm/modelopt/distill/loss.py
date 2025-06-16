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

from abc import ABCMeta
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer import MegatronModule
from torch import Tensor
from torch.nn.modules.loss import _Loss

from nemo.utils import logging
from nemo.utils.import_utils import safe_import, safe_import_from

mtd, _ = safe_import("modelopt.torch.distill")
DistillationLossBalancer, _ = safe_import_from("modelopt.torch.distill", "DistillationLossBalancer", alt=object)

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(self, model_config: "TransformerConfig", projection_layer: Optional[nn.Module] = None):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__()
        self._config = model_config
        self._projection = projection_layer

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        if self._projection is not None:
            predictions = self._projection(predictions)
        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor, tp_reduce: bool = False, is_sequence_parallel: bool = False) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return (loss, tp_reduce, is_sequence_parallel)


class MSELoss(BaseLoss):
    """Calculates MSE loss between two tensors without reducing the sequence dim."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            MSE loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.mse_loss(predictions, targets, reduction="none")
        loss = loss.sum(dim=-1)

        return self.post_forward(loss)


class HiddenStateCosineLoss(BaseLoss):
    """
    Calculates Cosine loss between two tensors without reducing the sequence dim.

    The tensors are assumed to be intermediate activations, so extra restrictions are in place.
    """

    def __init__(self, model_config: "TransformerConfig", projection_layer: Optional[nn.Module] = None):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__(model_config, projection_layer=projection_layer)

        if self._config.tensor_model_parallel_size > 1 and not self._config.sequence_parallel:
            logging.warning(
                "``HiddenStateCosineLoss`` only works with tensors with full hidden dim. Ensure the "
                "tensor inputs meet this requirement or use `--sequence_parallel` if tensor parallel is enabled."
            )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            Cosine loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.cosine_embedding_loss(
            predictions.view(-1, predictions.size(-1)),
            targets.view(-1, targets.size(-1)),
            targets.new_ones(1),
            reduction="none",
        )
        loss = loss.view(*predictions.shape[:2])

        # NOTE: Tensor sequence length is still split among TP ranks.
        return self.post_forward(loss, is_sequence_parallel=self._config.sequence_parallel)


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(self, model_config: "TransformerConfig", temperature: float = 1.0, reverse: bool = False):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(model_config)
        self._temperature = temperature
        self._reverse = reverse

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            KLD loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # Division by temp should happen prior to finding max for both student and teacher.
        # Currently we don't use temperature in any of ours runs (temp=1.0)
        output_teacher = targets.float() / self._temperature
        output_student = predictions.float() / self._temperature

        # Compute local softmax, and the reweight to compute global softmax.
        if self._config.tensor_model_parallel_size > 1:
            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use standard reduction function here since the computation
            # that follows it isn't identical across TP ranks.
            denom_teacher = all_reduce_autograd(denom_teacher, group=parallel_state.get_tensor_model_parallel_group())

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(denom_student, group=parallel_state.get_tensor_model_parallel_group())

            slen, bsz, sharded_vocab_size = output_student.shape
            student_log_prob = output_student - torch.log(denom_student).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )
            teacher_log_prob = output_teacher - torch.log(denom_teacher).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )

            if self._reverse:
                loss = torch.sum(
                    F.kl_div(teacher_log_prob, student_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )

        else:
            if self._reverse:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_teacher, dim=-1),
                        F.softmax(output_student, dim=-1),
                        reduction="none",
                    ),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_student, dim=-1),
                        F.softmax(output_teacher, dim=-1),
                        reduction="none",
                    ),
                    dim=-1,
                )

        return self.post_forward(loss, tp_reduce=True)


class LogitsAndIntermediatesLossBalancer(DistillationLossBalancer):
    """
    LossBalancer implementation for Logit and Intermediate losses.

    Dynamically weighs distillation and original losses to balance during training.
    """

    def __init__(self, kd_loss_scale: float = 1.0, skip_original_loss: bool = False):
        """Constructor.

        Args:
            kd_loss_scale: Multiply distillation losses by this before weighing.
                (Not used when `skip_original_loss` is True.)
            skip_original_loss: Used to signal whether the original loss should be used, regardless
                of whether it was passed into ``mtd.DistillationModel.compute_kd_loss()`` or not.
        """
        super().__init__()
        self._kd_loss_scale = kd_loss_scale
        self._skip_original_loss = skip_original_loss

    def forward(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        """Forward function.

        Args:
            loss_dict: All individual scalar losses, passed in during ``mtd.DistillationModel.compute_kd_loss()``

        Returns:
            Aggregate total scalar loss.
        """
        original_loss = loss_dict.pop(mtd.loss_balancers.STUDENT_LOSS_KEY)
        for _key in loss_dict:
            if _key.startswith(LogitsKLLoss.__name__):
                logits_key = _key  # should only be one
        logits_loss = loss_dict.pop(logits_key)
        intermediate_loss = sum(loss_dict.values()) / max(len(loss_dict), 1)

        if intermediate_loss > 0:
            dynamic_scale = logits_loss.item() / intermediate_loss.item()
            intermediate_loss_scaled = intermediate_loss * dynamic_scale
        else:
            intermediate_loss = logits_loss.new_tensor(intermediate_loss)
            intermediate_loss_scaled = intermediate_loss

        if self._skip_original_loss:
            total_loss = logits_loss + intermediate_loss_scaled
        else:
            kd_loss = logits_loss + intermediate_loss_scaled
            kd_loss *= original_loss.item() / kd_loss.item()
            total_loss = original_loss + kd_loss * self._kd_loss_scale

        out_dict = {
            "kd_loss": total_loss,
            "logits_loss": logits_loss,
            "intermediate_loss": intermediate_loss,
        }
        return out_dict


class ProjectionLayer(MegatronModule):
    """Module to project student layer activations to teacher's size."""

    def __init__(self, student_config: "TransformerConfig", teacher_config: "TransformerConfig"):
        """
        Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
        """
        super().__init__(config=student_config)
        if student_config.hidden_size == teacher_config.hidden_size:
            self._fit = nn.Identity()
        else:
            self._fit = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
            self.apply(self._init_weights)
            # Attribute below needed to reduce gradients during backward properly.
            setattr(self._fit.weight, "sequence_parallel", self.config.sequence_parallel)
            setattr(self._fit.bias, "sequence_parallel", self.config.sequence_parallel)

    def forward(self, student_tensor: Tensor):
        """
        Forward function.

        Args:
            student_tensor: Tensor to be fit to teacher size.
        """
        return self._fit(student_tensor)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            self.config.init_method(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()


class _AllReduce(torch.autograd.Function):
    """Implementation from old PyTorch `torch.distributed.nn.parallel`."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        # pylint: disable=C0116
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=C0116
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(tensor, op=torch.distributed.ReduceOp.SUM, group=torch.distributed.group.WORLD):
    """Custom all-reduce function.

    Needed instead of other all-reduce functions available when the computation following
    the all-reduce call differs per rank. In KL loss, this corresponds to the different numerators.
    """
    return _AllReduce.apply(op, group, tensor)
