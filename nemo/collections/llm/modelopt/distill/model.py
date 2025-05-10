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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import torch
from megatron.core import parallel_state
from torch import Tensor, nn

from nemo.collections import llm
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.utils.import_utils import safe_import
from nemo.utils.model_utils import unwrap_model

from .utils import adjust_distillation_model_for_mcore, load_distillation_config, teacher_provider

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
    from nemo.lightning.pytorch.optim import OptimizerModule

mtd, HAVE_MODELOPT = safe_import("modelopt.torch.distill")


class _DistillationLossReduction(MaskedTokenLossReduction):
    """Custom masking and reduction callable used only in training mode."""

    def __init__(self, distillation_loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distillation_loss_fn = distillation_loss_fn
        self._cp_size = parallel_state.get_context_parallel_world_size()

    def forward(self, batch: Dict[str, Tensor], forward_out: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if isinstance(forward_out, tuple):
            # neva returns (logits, loss_mask)
            forward_out, batch["loss_mask"] = forward_out

        # [ModelOpt]: KD loss calculation.
        loss_for_ub = self._distillation_loss_fn(
            loss_reduction_fn=lambda x: self._masked_token_loss(x, batch["loss_mask"])
        )

        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub, {"avg": reduced_loss}

    def _masked_token_loss(self, loss_output: Tensor, mask: Tensor):
        """The function takes as input per-token loss and masks non-required values."""
        if isinstance(loss_output, tuple):
            # [ModelOpt]: Losses can return extra flag to indicate additional TP-reduction (often required)
            loss_output, tp_reduce = loss_output
        else:
            tp_reduce = False

        losses = loss_output.view(-1).float()
        loss_mask = mask.view(-1).float()
        loss_sum = torch.sum(losses * loss_mask)
        num_valid_tokens = loss_mask.sum()

        if self._cp_size > 1:
            loss = torch.cat([loss_sum.view(1), num_valid_tokens.view(1)])
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
            loss = loss[0] / loss[1]  # sequence level nll
        else:
            loss = loss_sum / num_valid_tokens  # sequence level nll

        if tp_reduce is True:
            torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())

        return loss


class DistillationGPTModel(llm.GPTModel):
    """Custom GPT subclass for distillation-related modifications."""

    def __init__(
        self,
        config: llm.GPTConfig,
        teacher_config: llm.GPTConfig,
        teacher_ckpt_path: str,
        optim: Optional["OptimizerModule"] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        """Constructor.

        This subclass of GPTModel takes the configs of a student and teacher model and overrides
        the model construction step to create a ModelOpt `DistillationModel` as the underlying
        MCore model. This model abstracts both student and teacher as a single module whose forward
        pass runs both, and whose loss function automatically calculates a distillation loss on the
        output logits.

        NOTE: This class saves checkpoints which will be re-loaded as the student's original class.
        This allows one to continue using the model after distillation without this special class.

        Args:
            config: Config of student model.
            teacher_config: Config of teacher model.
            teacher_ckpt_path: Path to teacher checkpoint (to restore weights).
            optim: Optimizer.
            tokenizer: Tokenizer.
            model_transform: Transform to apply to model during setup.
        """
        if not HAVE_MODELOPT:
            raise RuntimeError("nvidia-modelopt is needed to use DistillationGPTModel")
        super().__init__(config, optim, tokenizer, model_transform)
        self._teacher_config = teacher_config
        self._teacher_ckpt_path = teacher_ckpt_path
        self._train_called = False

        if not isinstance(config, llm.GPTConfig) or not isinstance(teacher_config, llm.GPTConfig):
            raise ValueError("Student and Teacher must both be subclasses of `llm.GPTModel`")
        if self.config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("ModelOpt Distillation incompatible with interleaved pipeline schedule.")

    def configure_model(self):
        if hasattr(self, "module"):
            return

        model = self.config.configure_model(self.tokenizer)

        # Ensure same for both models.
        for attr in [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "sequence_parallel",
            "pipeline_dtype",
        ]:
            setattr(self._teacher_config, attr, getattr(self.config, attr))

        # [ModelOpt] Intialize DistillationModel.
        distill_cfg = load_distillation_config(self.config)
        kd_config = {
            "teacher_model": (
                teacher_provider,
                [self._teacher_config, self._teacher_ckpt_path],
                {"tokenizer": self.tokenizer, "trainer": self.trainer},
            ),
            "criterion": distill_cfg["criterion"],
            "loss_balancer": distill_cfg["loss_balancer"],
        }
        distillation_model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        # Additional MCore-specific tweaks needed.
        adjust_distillation_model_for_mcore(distillation_model, model_cfg=self.config, distill_cfg=distill_cfg)

        self.module = distillation_model

    def get_inference_wrapper(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "Please restore a checkpoint of this model to its original class to call `get_inference_wrapper`"
        )

    @property
    def training_loss_reduction(self) -> _DistillationLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = _DistillationLossReduction(
                distillation_loss_fn=self.core_module.compute_kd_loss
            )
        return self._training_loss_reduction

    def load_state_dict(self, state_dict, *args, **kwargs):
        # pylint: disable=C0116
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # `super()` would go to `nn.Module` and skip the Context Manager in `mtd.DistillationModel.load_state_dict()`
        return self.core_module.load_state_dict(state_dict, *args, **kwargs)

    @property
    def core_module(self):
        # pylint: disable=C0116
        return unwrap_model(self.module)

    def train(self, mode: bool = True):
        # pylint: disable=C0116
        self._train_called = True
        return super().train(mode)

    def __setattr__(self, name, value):
        # HACK: PTL calls `module.training = True` after sanity check, bypassing `module.train()` which we depend on.
        if name == "training":
            if not self._train_called:
                self.train(value)
                return
            self._train_called = False
        return super().__setattr__(name, value)
