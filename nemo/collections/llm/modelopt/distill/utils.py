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

from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.validation import StrictHandling, parse_strict_flag
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.utils import get_model_config, get_model_type, get_model_xattn

from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging
from nemo.utils.import_utils import safe_import, safe_import_from

from .loss import LogitsKLLoss

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

mto, HAVE_MODELOPT = safe_import("modelopt.torch.opt")
DistillationModel, _ = safe_import_from("modelopt.torch.distill", "DistillationModel", alt=object)
DistillationLossBalancer, _ = safe_import_from("modelopt.torch.distill", "DistillationLossBalancer", alt=object)


def load_distillation_config(cfg: "TransformerConfig") -> Dict[str, Any]:
    """Create a default distillation config for MCore GPT Models."""
    logit_pair = ("output_layer", "output_layer")  # logit module names for MCoreGPTModel
    distill_cfg = {
        "criterion": {},
        "loss_balancer": _DummyLossBalancer(),  # HACK: to appease ModelOpt until validation relaxed
        "skip_lm_loss": True,
    }
    if cfg.pipeline_model_parallel_size == 1 or parallel_state.is_pipeline_last_stage(ignore_virtual=False):
        distill_cfg["criterion"][logit_pair] = LogitsKLLoss(cfg)

    return distill_cfg


class _DummyLossBalancer(DistillationLossBalancer):
    def forward(self, loss_dict):
        # pylint: disable=C0116
        return next(iter(loss_dict.values()))


def teacher_provider(
    config: llm.GPTConfig, ckpt_path: str, tokenizer: "TokenizerSpec", trainer: nl.Trainer
) -> "MCoreGPTModel":
    """Teacher model factory (must be a non-local function to pickle)."""
    logging.info("Distillation: Loading teacher weights...")

    # TODO(aanoosheh): Replace spec with modelopt one
    model = config.configure_model(tokenizer)

    sharded_state_dict = {"state_dict": model.sharded_state_dict(prefix="module.")}
    strict = trainer.strategy.ckpt_load_strictness
    checkpoint = trainer.strategy.checkpoint_io.load_checkpoint(ckpt_path, sharded_state_dict, strict=strict)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

    # convert from StrictHandling to bool for PTL
    if strict is not None and not isinstance(strict, bool):
        strict = parse_strict_flag(strict)
        strict_options = [
            StrictHandling.ASSUME_OK_UNEXPECTED,
            StrictHandling.RAISE_UNEXPECTED,
            StrictHandling.RAISE_ALL,
        ]
        strict = strict in strict_options
    model.load_state_dict(state_dict, strict=strict)

    torch.cuda.empty_cache()
    logging.info("Distillation: teacher weights loaded.")
    return model


def adjust_distillation_model_for_mcore(
    model: "DistillationModel", model_cfg: "TransformerConfig", distill_cfg: Dict[str, Any]
):
    """Extra modifications to ``mtd.DistillationModel`` required for Megatron-Core."""
    # Get rid of ModelOpt Distillation state
    # NOTE: If re-placed, above losses need modifcation as `TransformerConfig` has non-pickleable elements.
    existing_state = mto.ModeloptStateManager(model).state_dict()
    assert len(existing_state) == 1 and existing_state[0][0] == "kd_loss", f"{existing_state=}"
    # mto.ModeloptStateManager.remove_state(model)
    delattr(model, mto.ModeloptStateManager._state_key)  # Use above method from modelopt 0.27

    # Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> "ShardedStateDict":
        with self.hide_teacher_model():
            return type(self).sharded_state_dict(self, *args, **kwargs)

    model.sharded_state_dict = MethodType(_sharded_state_dict, model)

    # Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> torch.Tensor:
        if distill_cfg["skip_lm_loss"] and self.training:
            return torch.zeros_like(labels, dtype=logits.dtype)
        return type(self).compute_language_model_loss(self, labels, logits)

    model.compute_language_model_loss = MethodType(_compute_language_model_loss, model)

    # Skip `lm_loss` always for teacher.
    def _compute_language_model_loss(self, labels, logits) -> torch.Tensor:
        return torch.zeros_like(labels, dtype=logits.dtype)

    model.teacher_model.compute_language_model_loss = MethodType(_compute_language_model_loss, model.teacher_model)

    # HACK: Pipeline-parallel Distillation requires splitting input tensor into student and teacher parts.
    def _set_student_input_tensor_shape(self, shapes: List[Tuple[int]]):
        self._tensor_split_idx = shapes[0][-1]

    def _set_input_tensor(self, input_tensors: List[torch.Tensor]):
        teacher_inputs = [t[..., self._tensor_split_idx :] if t is not None else t for t in input_tensors]
        student_inputs = [t[..., : self._tensor_split_idx] if t is not None else t for t in input_tensors]
        type(self).set_input_tensor(self.teacher_model, teacher_inputs)
        type(self).set_input_tensor(self, student_inputs)

    model.set_student_input_tensor_shape = MethodType(_set_student_input_tensor_shape, model)
    model.set_input_tensor = MethodType(_set_input_tensor, model)

    # HACK: Concatenate output tensors when PP>1 so they can be passed between ranks.
    def _forward(self, *args, **kwargs):
        if not self.training:
            with self.only_student_forward():
                return type(self).forward(self, *args, **kwargs)

        with torch.no_grad():
            self._teacher_model.eval()
            teacher_output = self._teacher_model(*args, **kwargs)
        with self.only_student_forward():
            student_output = type(self).forward(self, *args, **kwargs)

        if not parallel_state.is_pipeline_last_stage():
            return torch.cat([student_output, teacher_output], dim=-1)
        else:
            return student_output

    model.forward = MethodType(_forward, model)


def get_tensor_shapes_adjust_fn_for_distillation(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
) -> Union[Callable, None]:
    """
    Return the function to adjust tensor shapes for Distillation in Megatron-Core's forward pass.

    Currently only used during non-interleaved pipelining for Distillation.
    Concatenates sizes of student and teacher output tensors for inter-process communication.
    """
    if not HAVE_MODELOPT:
        return None
    if (
        forward_only
        or parallel_state.get_pipeline_model_parallel_world_size() == 1
        or parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        return None
    # Unwrap
    if isinstance(model, list):
        model = model[0]
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, DistillationModel):
        return None

    def adjust_tensor_shapes(recv_tensor_shapes: List[Tuple[int, ...]], send_tensor_shapes: List[Tuple[int, ...]]):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        teacher_config = get_model_config(model.teacher_model)
        teacher_model_type = get_model_type(model.teacher_model)
        teacher_encoder_decoder_xattn = get_model_xattn(model.teacher_model)

        teacher_recv_tensor_shapes = get_tensor_shapes(
            rank=rank - 1,
            model_type=teacher_model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=teacher_config,
            encoder_decoder_xattn=teacher_encoder_decoder_xattn,
        )
        teacher_send_tensor_shapes = get_tensor_shapes(
            rank=rank,
            model_type=teacher_model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=teacher_config,
            encoder_decoder_xattn=teacher_encoder_decoder_xattn,
        )
        model.set_student_input_tensor_shape(recv_tensor_shapes)

        for i, shape in enumerate(recv_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_recv_tensor_shapes[0][-1]
            recv_tensor_shapes[i] = tuple(shape)
        for i, shape in enumerate(send_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_send_tensor_shapes[0][-1]
            send_tensor_shapes[i] = tuple(shape)

        return recv_tensor_shapes, send_tensor_shapes

    return adjust_tensor_shapes
