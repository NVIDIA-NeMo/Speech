# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from pathlib import Path
from typing import Any, Dict

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
except ImportError:
    raise AssertionError("ModelOpt only supports Megatron-Core.")
import types
from abc import ABCMeta
from importlib.metadata import version
from typing import Tuple

import torch
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.transformer import TransformerConfig
from pkg_resources import packaging
from torch import Tensor
from torch.nn.modules.loss import _Loss

from nemo.collections.common.parts.utils import extend_instance
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    EmbeddingScalingMixin,
    MegatronGPTModel,
    get_specs,
)
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.model_utils import load_config, unwrap_model

mp.set_start_method("spawn", force=True)

# Model config fields which affect the structure of the model.
# These will be the values taken from the teacher's config file,
# while the rest remain the same as student's.
MODEL_ARCHITECHTURE_KEYS = [
    "encoder_seq_length",
    "max_position_embeddings",
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "init_method_std",
    "use_scaled_init_method",
    "hidden_dropout",
    "attention_dropout",
    "ffn_dropout",
    "kv_channels",
    "apply_query_key_layer_scaling",
    "normalization",
    "layernorm_epsilon",
    "do_layer_norm_weight_decay",
    "make_vocab_size_divisible_by",
    "pre_process",
    "post_process",
    "persist_layer_norm",
    "bias",
    "activation",
    "headscale",
    "transformer_block_type",
    "openai_gelu",
    "normalize_attention_scores",
    "position_embedding_type",
    "rotary_percentage",
    "attention_type",
    "share_embeddings_and_output_weights",
    "overlap_p2p_comm",
    "batch_p2p_comm",
    "num_query_groups",
    "seq_len_interpolation_factor",
    "rotary_base",
    "scale_positional_embedding",
]


class DistillationMegatronGPTModel(MegatronGPTModel):
    """ModelOpt Distillation-enabled subclass of `MegatronGPTModel`."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        """
        Constructor.

        Args:
            cfg: Model configuration.
            trainer: Nemo trainer instance.
        """
        logging.info("Distillation: Enabled.")
        assert cfg.kd_teacher_restore_from_path is not None, "Path to teacher weights must be provided."
        assert cfg.pipeline_model_parallel_size == 1, "Distillation mode does not yet support Pipeline Parallel."

        super().__init__(cfg, trainer)

        logging.info("\n\n************** Final model configuration ***********")
        logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if not self.mcore_gpt:
            raise AssertionError("ModelOpt Distillation only supports MCore model edition.")

        model = MCoreGPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_specs(
                self.spec_name,
                self.transformer_config,
                self.transformer_engine,
                self.cfg.get('hyena', None),
            ),
            vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
            max_sequence_length=self.cfg.get('encoder_seq_length', 512),
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
            position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percent=self.cfg.get('rotary_percentage', 1.0),
            seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
            rotary_base=self.cfg.get('rotary_base', 10000),
        )
        if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
            extend_instance(model.embedding, EmbeddingScalingMixin)

        # [ModelOpt] Distillation mode.
        distill_cfg = load_distillation_config(self.transformer_config)
        # Intialize DistillationModel.
        kd_config = {
            "teacher_model": (_teacher_provider, [self.cfg, copy.deepcopy(self.trainer)], {}),
            "criterion": distill_cfg["criterion"],
            "loss_balancer": distill_cfg["loss_balancer"],
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        # Additional tweaks needed for MCore/Nemo.
        adjust_distillation_model_for_mcore(model, distill_cfg)

        return model

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = self.get_batch(dataloader_iter, tuning)

            # Transfer needed data to GPU
            required_keys = set()
            max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None
            cu_seqlens_argmin = batch['cu_seqlens_argmin'] if 'cu_seqlens_argmin' in batch else None
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if 'cu_seqlens' in batch:
                    required_keys.add('cu_seqlens')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion and 'attention_mask' in required_keys:
                required_keys.remove('attention_mask')
            batch = {
                key: val.cuda(non_blocking=True) if key in required_keys and isinstance(val, torch.Tensor) else None
                for key, val in batch.items()
            }

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': None if self.get_attention_mask_from_fusion else batch['attention_mask'],
                'labels': batch['labels'] if 'labels' in batch else None,
                'loss_mask': batch['loss_mask'],
            }

            # TODO: @eharper can we add this to mcore?
            forward_args.pop('loss_mask')

            if 'cu_seqlens' in batch:  # packed sequence from GPTSFTPackedDataset
                # these args are passed eventually into TEDotProductAttention.forward()
                cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
                # remove -1 "paddings" added in collate_fn
                if cu_seqlens_argmin is not None:
                    cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
                else:
                    cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

                try:
                    from megatron.core.packed_seq_params import PackedSeqParams
                except (ImportError, ModuleNotFoundError) as e:
                    mcore_version = packaging.version.Version(version('megatron-core'))
                    logging.error(
                        f"megatron-core v{mcore_version} does not support training with packed sequence. "
                        "Please use megatron-core >= 0.5.0, or set model.data.train_ds.packed_sequence=False"
                    )
                    raise e

                forward_args['packed_seq_params'] = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                    qkv_format='thd',
                )

            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                if validation_step:
                    loss_for_ub = self.loss_func(
                        batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor, validation_step=True
                    )
                else:
                    # [ModelOpt] KD Loss for a micro-batch (ub)
                    unwrapped_model = unwrap_model(model, (Float16Module, MCoreFloat16Module))
                    loss_for_ub = unwrapped_model.compute_kd_loss(
                        loss_reduction_fn=lambda x: self.loss_func(
                            batch['loss_mask'], batch['num_valid_tokens_in_ub'], x
                        )
                    )
                cp_size = parallel_state.get_context_parallel_world_size()
                if validation_step and not self.validation_drop_last:
                    num_valid_tokens_in_ub = batch['num_valid_tokens_in_ub']
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(loss_for_ub)
                        num_valid_tokens_in_ub = 0
                    else:
                        if self.sample_weight == 'constant':
                            num_valid_tokens_in_ub = 1
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub * cp_size, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub * cp_size, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor, validation_step=False):
        loss = super().loss_func(loss_mask, num_valid_tokens_in_ub, output_tensor)
        if not validation_step and self.cfg.tensor_model_parallel_size > 1:
            # [ModelOpt] KD loss requires extra all-reduce to ensure same values across MP-TP partitions.
            loss = torch.sum(tensor_parallel.gather_from_tensor_model_parallel_region(loss.reshape(1)))
        return loss

    def configure_optimizers(self):
        with self.model.hide_teacher_model():
            return super().configure_optimizers()


########################################################


def load_distillation_config(student_cfg: TransformerConfig) -> Dict[str, Any]:
    """Create a default distillation config for MCore GPT Models.

    Args:
        student_cfg: Model config for student model.
    """
    logit_pair = ("output_layer", "output_layer")  # logit module names for MCoreGPTModel
    tp_enabled = student_cfg.tensor_model_parallel_size > 1

    cfg = {
        "criterion": {tuple(logit_pair): LogitsKLLoss(tensor_parallel=tp_enabled)},
        "loss_balancer": None,
        "skip_lm_loss": True,
    }
    return cfg


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(self, tensor_parallel: bool = False):
        """Constructor.

        Args:
            tensor_parallel: Whether tensor parallelism is enabled or not.
        """
        super().__init__()
        self._tensor_parallel = tensor_parallel

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return loss


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(
        self,
        tensor_parallel: bool = False,
        temperature: float = 1.0,
        reverse: bool = False,
    ):
        """
        Constructor.

        Args:
            tensor_parallel: Whether tensor parallelism is enabled or not.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(tensor_parallel)
        self._temperature = temperature
        self._reverse = reverse

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

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
        if self._tensor_parallel:

            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use `gather_from_tensor_model_parallel_region` here since it discards
            # gradients from other ranks - we need to all_reduce the gradients as well.
            denom_teacher = all_reduce_autograd(denom_teacher, group=get_tensor_model_parallel_group())

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(denom_student, group=get_tensor_model_parallel_group())

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
                        F.log_softmax(output_teacher, dim=-1), F.softmax(output_student, dim=-1), reduction="none"
                    ),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_student, dim=-1), F.softmax(output_teacher, dim=-1), reduction="none"
                    ),
                    dim=-1,
                )

        return self.post_forward(loss)


class _AllReduce(torch.autograd.Function):
    """Implementation from old PyTorch `torch.distributed.nn.parallel`."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(tensor, op=torch.distributed.ReduceOp.SUM, group=torch.distributed.group.WORLD):
    return _AllReduce.apply(op, group, tensor)


def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: Dict[str, Any]):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

    # HACK: Get rid of ModelOpt Distillation state
    mto.ModeloptStateManager(model)._state.pop()

    # HACK: Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        with self.hide_teacher_model():
            return self._sharded_state_dict(*args, **kwargs)

    model._sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = types.MethodType(_sharded_state_dict, model)

    # HACK: Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        if self.training:
            return torch.zeros_like(labels)
        return self._compute_language_model_loss(labels, logits)

    if distill_cfg["skip_lm_loss"]:
        model._compute_language_model_loss = model.compute_language_model_loss
        model.compute_language_model_loss = types.MethodType(_compute_language_model_loss, model)


########################################################


def _teacher_provider(cfg: DictConfig, trainer: Trainer) -> MCoreGPTModel:
    """Teacher model factory (must be a non-local function to pickle)."""
    logging.info("Distillation: Loading teacher weights...")
    teacher_model_cfg = _merge_model_arch_fields(cfg, cfg.kd_teacher_restore_from_path)

    model = MegatronGPTModel.restore_from(
        cfg.kd_teacher_restore_from_path,
        override_config_path=teacher_model_cfg,
        trainer=trainer,
    )
    teacher_model_module_list = model.get_model_module_list()
    logging.info("Distillation: ... teacher weights loaded.")
    return teacher_model_module_list[0]


def _merge_model_arch_fields(cfg: DictConfig, model_load_path: str) -> DictConfig:
    """Overwrite model-architecture fields of a config with a checkpoint's."""
    model_cfg = load_config(model_load_path)
    model_arch_keys = [k for k in MODEL_ARCHITECHTURE_KEYS if k in model_cfg]
    model_arch_cfg = OmegaConf.masked_copy(model_cfg, model_arch_keys)
    with open_dict(cfg):
        cfg = OmegaConf.merge(cfg, model_arch_cfg)
        # Add tokenizer from model if not provided
        if OmegaConf.is_missing(cfg.tokenizer, "model"):
            cfg.tokenizer = model_cfg.tokenizer
    return cfg


########################################################


@hydra_runner(config_path="conf", config_name="megatron_llama_distill")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    with open_dict(cfg):
        cfg.model.name = "modelopt"  # Convert TE layernorm spec to unfused format
        # HACK: Checkpoint-loading process hangs/loops if this isn't present here for some reason.
        cfg.model.target = "nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel"

    # Continual training
    if cfg.model.get("restore_from_path") is not None:
        # Option 1: Restore only the model weights from a .nemo file
        logging.info(f"Continual training: loading weights from {cfg.model.restore_from_path}")

        # Merge model config's architecture fields with the one from the checkpoint
        cfg.model = _merge_model_arch_fields(cfg.model, cfg.model.restore_from_path)

        model = DistillationMegatronGPTModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            override_config_path=cfg.model,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
        logging.info("... weights loaded.")
    elif cfg.model.get("restore_from_ckpt") is not None:
        # Option 2: Restore both model weights and optimizer states from a PTL checkpoint
        logging.info(f"Continual training: loading weights and optimizer states from {cfg.model.restore_from_ckpt}")
        trainer.ckpt_path = Path(cfg.model.restore_from_ckpt)
        model = DistillationMegatronGPTModel(cfg.model, trainer)
        logging.info("... weights and optimizer states loaded.")

    # Start new pretraining or resume from a checkpoint if it exists
    else:
        logging.info("Instantiating new model ...")
        model = DistillationMegatronGPTModel(cfg.model, trainer)
        logging.info("... model instantiated.")

    trainer.fit(model)

    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
    else:
        logging.warning("Skipping saving final model as no `model.nemo_path` provided.")


if __name__ == '__main__':
    main()
