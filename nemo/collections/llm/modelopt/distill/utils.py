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

import re
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch
import yaml
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.validation import StrictHandling, parse_strict_flag
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.transformer import TransformerLayer
from megatron.core.utils import get_model_config

from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging
from nemo.utils.import_utils import safe_import, safe_import_from

from .loss import HiddenStateCosineLoss, LogitsAndIntermediatesLossBalancer, LogitsKLLoss, ProjectionLayer

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

mto, HAVE_MODELOPT = safe_import("modelopt.torch.opt")
DistillationModel, _ = safe_import_from("modelopt.torch.distill", "DistillationModel", alt=object)
DistillationLossBalancer, _ = safe_import_from("modelopt.torch.distill", "DistillationLossBalancer", alt=object)


@dataclass
class DistillationConfig:
    """Knowledge-Distillation config.

    Args:
        intermediate_layer_pairs: List of tuples of intermediate layer names.
        logit_layers: Tuple of logit layer names.
        skip_lm_loss: Whether to skip computing the standard language model loss (default: ``True``).
        kd_loss_scale: Relative scaling factor for the distillation loss if ``skip_lm_loss`` is ``False``.
    """

    intermediate_layer_pairs: List[Tuple[str, str]] = field(default_factory=list)
    logit_layers: Tuple[str, str] = ("output_layer", "output_layer")
    skip_lm_loss: bool = True
    kd_loss_scale: float = 1.0
    criterion: Optional[Dict[Tuple[str, str], torch.nn.Module]] = None
    loss_balancer: Optional[DistillationLossBalancer] = None

    def __post_init__(self):
        assert len(self.logit_layers) == 2, f"{self.logit_layers=}"
        assert all(len(pair) == 2 for pair in self.intermediate_layer_pairs), f"{self.intermediate_layer_pairs=}"
        assert self.kd_loss_scale > 0, f"{self.kd_loss_scale=}"


def load_distillation_config(
    config_path: Optional[str], student_cfg: "TransformerConfig", teacher_cfg: "TransformerConfig"
) -> DistillationConfig:
    """Read the distillation yaml config file specified by ``args.export_kd_cfg``.

    Args:
        config_path: Path to user-defined distillation settings yaml file.
            If `None`, uses default logits-only distillation mode for GPT models.
        student_cfg: Model config for student model.
        teacher_cfg: Model config for teacher model.

    WARNING: Assumes intermediate hidden sizes are always that found in the model config's ``hidden_size`` attribute.
    """
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg = DistillationConfig(**cfg)
    else:
        logging.warning("Distillation config not provided. Using default.")
        cfg = DistillationConfig()

    criterion = {}
    if student_cfg.pipeline_model_parallel_size == 1 or parallel_state.is_pipeline_last_stage():
        criterion[tuple(cfg.logit_layers)] = LogitsKLLoss(student_cfg)
        # NOTE: Projection layer shared among intermediate layer pairs.
        projection_layer = ProjectionLayer(student_cfg, teacher_cfg)

        for student_layer, teacher_layer in cfg.intermediate_layer_pairs:
            if parallel_state.get_tensor_and_context_parallel_rank() == 0:
                print(
                    "Distillation: Adding intermediate loss between"
                    f" `{student_layer}` of student (hidden size {student_cfg.hidden_size}) and"
                    f" `{teacher_layer}` of teacher (hidden size {teacher_cfg.hidden_size})."
                )
            student_layer = _adjust_layer_index_for_pp(student_layer, student_cfg)
            teacher_layer = _adjust_layer_index_for_pp(teacher_layer, teacher_cfg)
            criterion[(student_layer, teacher_layer)] = HiddenStateCosineLoss(
                student_cfg, projection_layer=projection_layer
            )

    loss_balancer = LogitsAndIntermediatesLossBalancer(
        kd_loss_scale=cfg.kd_loss_scale, skip_original_loss=cfg.skip_lm_loss
    )

    cfg.criterion = criterion
    cfg.loss_balancer = loss_balancer

    return cfg


def _adjust_layer_index_for_pp(submodule_name, model_cfg):
    """Adjust any sequence-based layer indices found in a submodule name for Pipeline Parallelism."""

    match = re.search(r'(?<=\.)\d+(?=\.)', submodule_name)
    if not match:
        return submodule_name

    offset = TransformerLayer._get_layer_offset(model_cfg)
    new_layer_idx = int(match.group(0)) - offset
    if new_layer_idx < 0:
        raise ValueError(f"Layer {submodule_name} does not fall on final PP rank.")

    new_submodule_name = submodule_name.replace(match.group(0), str(new_layer_idx))
    if parallel_state.get_tensor_and_context_parallel_rank() == 0:
        print(f'Distillation: Renamed layer "{submodule_name}" on final PP rank to "{new_submodule_name}"')
    return new_submodule_name


def teacher_provider(
    config: llm.GPTConfig, ckpt_path: str, tokenizer: "TokenizerSpec", trainer: nl.Trainer
) -> "MCoreGPTModel":
    """Teacher model factory (must be a non-local function to pickle)."""
    logging.info("Distillation: Loading teacher weights...")

    # TODO(aanoosheh): Replace spec with modelopt one
    model = config.configure_model(tokenizer)

    sharded_sd_metadata = trainer.strategy.unwrapped_checkpoint_io.load_content_metadata(ckpt_path)
    sharded_state_dict = {"state_dict": model.sharded_state_dict(prefix="module.", metadata=sharded_sd_metadata)}
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


def adjust_distillation_model_for_mcore(model: DistillationModel, distill_cfg: DistillationConfig):
    """Extra modifications to ``mtd.DistillationModel`` required for Megatron-Core."""
    # Get rid of ModelOpt Distillation state
    # NOTE: If re-placed, above losses need modifcation as `TransformerConfig` has non-pickleable elements.
    existing_state = mto.ModeloptStateManager(model).state_dict()
    assert len(existing_state) == 1 and existing_state[0][0] == "kd_loss", f"{existing_state=}"
    mto.ModeloptStateManager.remove_state(model)

    # Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> "ShardedStateDict":
        with self.hide_teacher_model():
            return type(self).sharded_state_dict(self, *args, **kwargs)

    model.sharded_state_dict = MethodType(_sharded_state_dict, model)

    # Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> torch.Tensor:
        if distill_cfg.skip_lm_loss and self.training:
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
        teacher_config = get_model_config(model.teacher_model)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        teacher_recv_tensor_shapes = get_tensor_shapes(
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=teacher_config,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        teacher_send_tensor_shapes = get_tensor_shapes(
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=teacher_config,
            tp_group=tp_group,
            cp_group=cp_group,
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
