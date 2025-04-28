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
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.utils.import_utils import safe_import
from nemo.utils.model_utils import unwrap_model

from .utils import (
    LoopingCachedDataIterator,
    adjust_distillation_model_for_mcore,
    load_distillation_config,
    teacher_provider,
)

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
    from nemo.lightning.pytorch.optim import OptimizerModule

mtd, HAVE_MODELOPT = safe_import("modelopt.torch.distill")


def gpt_distillation_data_step(dataloader_iter, attn_mask_cpu=False) -> Dict[str, Tensor]:
    """Same as base GPT's data step but with ability to move attention mask to CPU."""
    batch = next(dataloader_iter)

    if isinstance(batch, tuple) and len(batch) == 3:
        batch = batch[0]

    required_device_keys = set()
    required_host_keys = set()

    if attn_mask_cpu:
        # [ModelOpt]: We cache data for PP distillation, and save GPU mem by storing masks on CPU mem.
        required_host_keys.add("attention_mask")
    else:
        required_device_keys.add("attention_mask")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            batch_required_keys[key] = val.cpu()
        else:
            batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(batch_required_keys)

    return output


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
            loss_reduction_fn=lambda x: self._masked_token_loss(
                x, batch["loss_mask"], batch.get("num_valid_tokens_in_ub")
            )
        )

        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub * self._cp_size, {"avg": reduced_loss}

    def _masked_token_loss(self, loss_output: Tensor, mask: Tensor, num_valid_tokens_in_ub: Optional[int] = None):
        """The function takes as input per-token loss and masks non-required values."""
        if isinstance(loss_output, tuple):
            # [ModelOpt]: Losses can return extra flag to indicate additional TP-reduction (often required)
            loss_output, tp_reduce = loss_output
        else:
            tp_reduce = False
        losses = loss_output.float()
        loss_mask = mask.view(-1).float()

        if self._cp_size > 1:
            if num_valid_tokens_in_ub is None:
                num_valid_tokens_in_ub = loss_mask.sum()
            if num_valid_tokens_in_ub < 0.5:  # no valid tokens
                num_valid_tokens_in_ub += 1.0
            loss = torch.sum(losses.view(-1) * loss_mask) / num_valid_tokens_in_ub  # sequence level nll
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll

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

    def data_step(self, dataloader_iter, cache_num_batches: Optional[int] = None) -> Dict[str, Tensor]:
        # NOTE: Ignores `self.config.data_step_fn`
        if cache_num_batches:
            batches = [
                gpt_distillation_data_step(dataloader_iter, attn_mask_cpu=True) for _ in range(cache_num_batches)
            ]
            return LoopingCachedDataIterator(batches)
        elif isinstance(dataloader_iter, LoopingCachedDataIterator):
            batch = next(dataloader_iter)
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].cuda(non_blocking=True)  # move back to GPU
            return batch
        else:
            return gpt_distillation_data_step(dataloader_iter)

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
