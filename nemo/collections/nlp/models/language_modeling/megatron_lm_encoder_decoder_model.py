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

import copy
import functools
import inspect
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import (
    MegatronTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    compute_beam_search_len_penalty,
    get_sampling_token_fn,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

__all__ = ["MegatronLMEncoderDecoderModel"]


class MegatronLMEncoderDecoderModel(MegatronBaseModel):
    """
    Megatron encoder-decoder base class
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        if cfg.get('pipeline_model_parallel_size', 1) > 1:
            if cfg.get('pipeline_model_parallel_split_rank', 0) <= 0:
                raise ValueError(
                    f"pipeline_model_parallel_split_rank must be > 0 when using pipeline_model_parallel_size > 1"
                )
        if cfg.get('pipeline_model_parallel_size', 1) > 1:
            if not cfg.get('share_token_embeddings', True) or not cfg.get(
                'share_decoder_tokens_head_embeddings', True
            ):
                raise ValueError(
                    "when pipeline_model_parallel_size > 1 we require share_token_embeddings=True and share_decoder_tokens_head_embeddings=True"
                )

        # Make sure trainer.accumulate_grad_batches is 1.
        self._validate_trainer()

        # TODO: Currently does not support interleaved pipeline parallelism.
        # This means we can only use pipeline parallelism without the interleaved schedule.
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            logging.warning("Using CPUAccelerator, model will be built on CPU.")
            self.enc_dec_model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                on_cpu=True,
                model_type=ModelType.encoder_and_decoder,
            )[0]
        else:
            self.enc_dec_model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                model_type=ModelType.encoder_and_decoder,
            )[0]

        # We don't need to call it explicitly? Since it is a pytorch lightning hook function
        # self.setup_optimizer_param_groups()

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        if self.megatron_amp_O2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                self.enc_dec_model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            self.enc_dec_model = Float16Module(
                config=self.model_parallel_config, module=self.enc_dec_model, precision=self.cfg.precision
            )

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        self.enc_dec_model.model_type = ModelType.encoder_and_decoder

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization([self.enc_dec_model])

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Identify params that require grad reductions between
            # pipeline stages
            # See: allreduce_word_and_position_embeddings
            model_parallel_params = []
            if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
                parallel_state.is_rank_in_embedding_group()
            ):
                if self.cfg.get('share_token_embeddings', True) and self.cfg.get(
                    'share_decoder_tokens_head_embeddings', True
                ):
                    model_parallel_params.append(self.enc_dec_model.word_embeddings_weight())
            if (
                parallel_state.is_rank_in_position_embedding_group()
                and parallel_state.get_pipeline_model_parallel_world_size() > 1
                and parallel_state.get_pipeline_model_parallel_split_rank() is not None
                and self.cfg.encoder.get('position_embedding_type') == 'learned_absolute'
                and self.cfg.decoder.get('position_embedding_type') == 'learned_absolute'
            ):
                if self.cfg.get('share_token_embeddings', True):
                    model_parallel_params.append(self.enc_dec_model.position_embeddings_weight())
            if (
                parallel_state.get_pipeline_model_parallel_world_size() > 2
                and parallel_state.get_pipeline_model_parallel_split_rank() is not None
            ):
                if (
                    self.cfg.encoder.get('position_embedding_type') == 'relative'
                    and parallel_state.is_rank_in_encoder_relative_position_embedding_group()
                    and parallel_state.get_pipeline_model_parallel_split_rank() > 1
                ):
                    model_parallel_params.append(self.enc_dec_model.encoder_relative_position_embeddings_weight())
                if (
                    self.cfg.decoder.get('position_embedding_type') == 'relative'
                    and parallel_state.is_rank_in_decoder_relative_position_embedding_group()
                ):
                    model_parallel_params.append(self.enc_dec_model.decoder_relative_position_embeddings_weight())
                    if not self.cfg.decoder.get('relative_position_bias_self_attention_only', True):
                        model_parallel_params.append(
                            self.enc_dec_model.decoder_cross_attention_relative_position_embeddings_weight()
                        )

            # Disable async grad reductions for params that are
            # synchronized for pipeline parallelism
            for param in model_parallel_params:
                param._disable_greedy_grad_copy = not self.megatron_amp_O2
                param._disable_overlap_grad_sync = True

            # Make sure embedding grads are reduced in FP32
            with_fp32_embedding_grads = self.cfg.get('with_fp32_embedding_grads', True)
            for name, param in self.named_parameters():
                if 'word_embedding' in name or 'position_embedding' in name or 'output_layer' in name:
                    param._with_fp32_optimizer = with_fp32_embedding_grads

        return super().configure_optimizers()

    def _handle_bias_activation_fusion_args(self, cfg):
        # For oldest models, we don't have the option to turn on/off bias activation fusion. It is always on.
        if not hasattr(cfg, 'bias_gelu_fusion') and not hasattr(cfg, 'bias_activation_fusion'):
            # Handle the case where the model can have bias=False
            if cfg.get('bias', True):
                cfg.bias_activation_fusion = True
            else:
                cfg.bias_activation_fusion = False
        # For in-between models, Re-map bias_gelu_fusion to bias_activation_fusion
        elif hasattr(cfg, 'bias_gelu_fusion'):
            logging.warning('bias_gelu_fusion is deprecated. Please use bias_activation_fusion instead.')
            cfg.bias_activation_fusion = cfg.bias_gelu_fusion

    def _populate_encoder_decoder_configs_for_backward_compatibility(self, cfg):
        """
        Populate encoder and decoder configs for backward compatibility with a checkpoint that has a common enc/dec config.
        """
        # TODO: This will not remove redundant args that are already present in the new yaml file's config.model
        encoder_cfg = copy.deepcopy(cfg)
        decoder_cfg = copy.deepcopy(cfg)

        OmegaConf.set_struct(encoder_cfg, True)
        OmegaConf.set_struct(decoder_cfg, True)
        OmegaConf.set_struct(cfg, True)

        with open_dict(encoder_cfg), open_dict(decoder_cfg), open_dict(cfg):
            encoder_cfg.arch = cfg.get('encoder_arch', 'transformer')
            decoder_cfg.arch = cfg.get('decoder_arch', 'transformer')

            self._handle_bias_activation_fusion_args(encoder_cfg)
            self._handle_bias_activation_fusion_args(decoder_cfg)

            cfg.encoder = encoder_cfg
            cfg.decoder = decoder_cfg

            # NOTE: For old models there are two scenarios:
            # 1. If we share decoder embeddings with the output layer, we would always set tokens_head_bias=True
            # 2. If we do not share decoder embeddings with the output layer, we would always set tokens_head_bias=False
            cfg.tokens_head_bias = (
                True if cfg.get('share_decoder_tokens_head_embeddings', True) else False
            )  # For models before separate encoder/decoder configs, tokens_head_bias was always True.

    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        if not hasattr(self.cfg, 'encoder') or not hasattr(self.cfg, 'decoder'):
            logging.warning(
                'Could not find encoder or decoder in config. This is probably because of restoring an old checkpoint. Copying shared model configs to encoder and decoder configs.'
            )
            # After the call below, self.cfg.encoder and self.cfg.decoder will be populated with the cfg.model configs from old checkpoints.
            self._populate_encoder_decoder_configs_for_backward_compatibility(self.cfg)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and self.cfg.encoder.arch == 'perceiver':
            raise ValueError(f"Perceivers with pipeline parallel > 1 is not supported yet.")

        if not hasattr(self.cfg, 'embedding_init_method_std'):
            embedding_init_method_std = self.cfg.encoder.init_method_std
        else:
            embedding_init_method_std = self.cfg.embedding_init_method_std

        if not hasattr(self.cfg, 'embedding_dropout'):
            embedding_dropout = self.cfg.encoder.hidden_dropout
        else:
            embedding_dropout = self.cfg.embedding_dropout

        model = MegatronTokenLevelEncoderDecoderModule(
            config=self.model_parallel_config,
            encoder_cfg=self.cfg.encoder,
            decoder_cfg=self.cfg.decoder,
            vocab_size=self.padded_vocab_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            fp16_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            precision=self.cfg.get('precision', 16),
            embedding_init_method_std=embedding_init_method_std,
            embedding_dropout=embedding_dropout,
            label_smoothing=self.cfg.get('label_smoothing', 0.0),
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            share_token_embeddings=self.cfg.get('share_token_embeddings', True),
            share_decoder_tokens_head_embeddings=self.cfg.get('share_decoder_tokens_head_embeddings', True),
            tokens_head_bias=self.cfg.get('tokens_head_bias', True),
            hiddens_cfg=self.cfg.get('hiddens', None),
        )
        return model

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        token_type_ids=None,
        lm_labels=None,
        enc_output=None,
        enc_output_attn_mask=None,
        output_enc_hidden_only=False,
        enc_input=None,
    ):
        output_tensor = self.enc_dec_model(
            enc_input_ids=encoder_input_ids,
            dec_input_ids=decoder_input_ids,
            enc_attn_mask=encoder_attn_mask,
            dec_attn_mask=decoder_attn_mask,
            token_type_ids=token_type_ids,
            labels=lm_labels,
            enc_output=enc_output,
            enc_output_attn_mask=enc_output_attn_mask,
            output_enc_hidden_only=output_enc_hidden_only,
            enc_input=enc_input,
        )

        return output_tensor

    def _execute_fwd_bwd_function(self, data_iterator, forward_only, tensor_shape, decoder_seq_length):
        """
        An auxiliary function that executes the fwd_bwd_step function and parse the returned values.
        """
        fwd_bwd_function = get_forward_backward_func()

        seq_length = tensor_shape[0]

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iterator,
            model=[self.enc_dec_model],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            decoder_seq_length=decoder_seq_length,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            mean_loss_dict = {}
            for k in losses_reduced_per_micro_batch[0].keys():
                # average loss across micro batches
                mean_loss_dict[k] = torch.stack(
                    [loss_reduced[k] for loss_reduced in losses_reduced_per_micro_batch]
                ).mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()
            mean_loss_dict = {"loss": loss_mean}

        return mean_loss_dict

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        """
        Dataloader produces a global batch which is turned into a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Get seq length of batch
        tensor_shape = [self.max_encoder_seq_length, self.cfg.micro_batch_size, self.cfg.encoder.hidden_size]

        return self._execute_fwd_bwd_function(
            data_iterator=dataloader_iter,
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            decoder_seq_length=self.max_decoder_seq_length,
        )

    def training_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        Batch should be a list of microbatches and those microbatches should on CPU.
        Microbatches are then moved to GPU during the pipeline.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # we zero grads here because we also call backward in the megatron fwd/bwd functions
        self._optimizer.zero_grad()

        loss_dict = self.fwd_bwd_step(dataloader_iter, False)

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            # when using pipeline parallelism grads must be reduced after the pipeline (not asynchronously)
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we allreduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # when using pipeline parallelism, we need keep the word and position embeddings in sync
            self.allreduce_word_and_position_embeddings()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        for k, v in loss_dict.items():
            torch.distributed.broadcast(v, get_last_rank())
            n = f'reduced_train_{k}'
            self.log(n, v, prog_bar=n.endswith("_loss"), rank_zero_only=True, batch_size=1)

        if self.torch_dtype == torch.float16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log(
            'global_step',
            self.trainer.global_step,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples',
            self._compute_consumed_samples_after_training_step(),
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        return loss_dict

    @property
    def max_decoder_seq_length(self) -> int:
        seq_len = self._cfg.data.get('seq_length_dec', None)
        if seq_len is None:
            seq_len = self.cfg.seq_length
        return seq_len

    @property
    def max_encoder_seq_length(self) -> int:
        return self.cfg.seq_length

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
        No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks.
        Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/model/distributed.py#L188
        """
        # Bucketize and all-reduce
        buckets = {}
        # Pack the buckets.
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # param.main_grad = param.grad

        # For each bucket, all-reduce and copy all-reduced grads.
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = torch._utils._flatten_dense_tensors(grads)
            coalesced /= parallel_state.get_data_parallel_world_size()
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_parallel_group())
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def allreduce_word_and_position_embeddings(self):

        # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
        # All-reduce word_embeddings' grad across first, last stages to ensure that word_embeddings parameters stay in sync.
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.is_rank_in_embedding_group()
        ):
            if self.cfg.get('share_token_embeddings', True) and self.cfg.get(
                'share_decoder_tokens_head_embeddings', True
            ):
                word_embeddings_weight = self.enc_dec_model.word_embeddings_weight()
                if self.megatron_amp_O2:
                    # O2 recipe stores a "main" copy of weights and grads
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            else:
                raise ValueError(
                    f"Attempting to allreduce word_embeddings for pipeline parallel size > 1, but found untied word embeddings or token head embeddings. This is not supported yet."
                )

        # All-reduce position embeddings for T5.
        if (
            parallel_state.is_rank_in_position_embedding_group()
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_pipeline_model_parallel_split_rank() is not None
            and self.cfg.encoder.get('position_embedding_type') == 'learned_absolute'
            and self.cfg.decoder.get('position_embedding_type') == 'learned_absolute'
        ):
            if self.cfg.get('share_token_embeddings', True):
                position_embeddings_weight = self.enc_dec_model.position_embeddings_weight()
                if self.megatron_amp_O2:
                    grad = position_embeddings_weight.main_grad
                else:
                    grad = position_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())

        # All-reduce relative position embeddings for T5.
        if (
            parallel_state.get_pipeline_model_parallel_world_size()
            > 2  # This > 2 and not > 1 since with PP=2 encoder RPE can live only on one rank.
            and parallel_state.get_pipeline_model_parallel_split_rank() is not None
        ):
            # For split rank = 1, we have only one encoder rank and so we don't need to allreduce.
            if (
                self.cfg.encoder.get('position_embedding_type') == 'relative'
                and parallel_state.is_rank_in_encoder_relative_position_embedding_group()
                and parallel_state.get_pipeline_model_parallel_split_rank() > 1
            ):
                position_embeddings_weight = self.enc_dec_model.encoder_relative_position_embeddings_weight()
                if self.megatron_amp_O2:
                    grad = position_embeddings_weight.main_grad
                else:
                    grad = position_embeddings_weight.grad
                torch.distributed.all_reduce(
                    grad, group=parallel_state.get_encoder_relative_position_embedding_group()
                )

            # For split rank == pipeline_world_size - 1, we have only one decoder rank and so we don't need to allreduce.
            if (
                self.cfg.decoder.get('position_embedding_type') == 'relative'
                and parallel_state.is_rank_in_decoder_relative_position_embedding_group()
            ):
                position_embeddings_weight = self.enc_dec_model.decoder_relative_position_embeddings_weight()
                if self.megatron_amp_O2:
                    grad = position_embeddings_weight.main_grad
                else:
                    grad = position_embeddings_weight.grad
                torch.distributed.all_reduce(
                    grad, group=parallel_state.get_decoder_relative_position_embedding_group()
                )

                # If the model also has separate RPE weights for decoder cross-attention, allreduce those as well.
                if not self.cfg.decoder.get('relative_position_bias_self_attention_only', True):
                    position_embeddings_weight = (
                        self.enc_dec_model.decoder_cross_attention_relative_position_embeddings_weight()
                    )
                    if self.megatron_amp_O2:
                        grad = position_embeddings_weight.main_grad
                    else:
                        grad = position_embeddings_weight.grad
                    torch.distributed.all_reduce(
                        grad, group=parallel_state.get_decoder_relative_position_embedding_group()
                    )

    def _process_batch(self, global_batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        global_batch['dec_mask'][:, 0] = 1

        return [
            global_batch["text_enc"],
            global_batch["text_dec"],
            global_batch["loss_mask"],
            global_batch["labels"],
            global_batch["enc_mask"],
            global_batch["dec_mask"],
            global_batch.get('data', None),
        ]

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
            batch = next(dataloader_iter)
            if isinstance(batch, tuple):
                batch = batch[0]
            # convert to list if not already converted.
            if isinstance(batch, dict):
                # convert to list if not already converted.
                batch = self._process_batch(batch)
            batch = [x.cuda(non_blocking=True) if torch.is_tensor(x) else x for x in batch]
            (
                encoder_input_ids,
                decoder_input_ids,
                loss_mask,
                lm_labels,
                encoder_attn_mask,
                decoder_attn_mask,
                batch_data,
            ) = batch

            output = model(
                encoder_input_ids,  # enc_input_ids
                encoder_attn_mask,  # enc_attn_mask
                decoder_input_ids,  # dec_input_ids
                decoder_attn_mask,  # dec_attn_mask
                None,  # token_type_ids
                lm_labels,  # labels
                batch_data,  # batch_data
            )

            def loss_func(output_tensor):
                if isinstance(output_tensor, dict):
                    # handle loss of hidden transformations
                    loss_dict = output_tensor
                    output_tensor = loss_dict.pop("output")
                    # compute reconstruction (tokens) only loss from per-token reconstruction loss
                    tokens_loss = self.loss_func(loss_mask, output_tensor)
                    loss_dict["tokens_loss"] = tokens_loss
                    tokens_loss_weight = loss_dict.get("tokens_loss_weight", 1.0)
                    # compute total loss
                    loss = loss_dict["loss"] = loss_dict["hiddens_loss"] + tokens_loss_weight * tokens_loss
                    # average losses across data parallel group
                    loss_dict = {
                        k: average_losses_across_data_parallel_group([v.mean()]) for k, v in loss_dict.items()
                    }
                else:
                    # compute reconstruction (tokens) only loss from per-token reconstruction loss
                    loss = self.loss_func(loss_mask, output_tensor)
                    # average losses across data parallel group
                    reduced_loss = average_losses_across_data_parallel_group([loss])
                    loss_dict = {'loss': reduced_loss}

                return loss, loss_dict

            return output, loss_func

        return fwd_output_and_loss_func

    @functools.lru_cache(maxsize=None)
    def _kwargs_to_arg_idx(self):
        """
        Returns a dict {kwarg name: arg index} to be used when mapping
        kwargs into a list of args.

        Computed on first call, and then cached.
        """
        # build mapping of kwargs to arg index at first run
        module = self.enc_dec_model.forward if not self.megatron_amp_O2 else self.enc_dec_model.module.forward
        args_name = inspect.getfullargspec(module)[0][1:]
        kwargs_to_arg_idx = {k: v for k, v in zip(args_name, range(len(args_name)))}

        return kwargs_to_arg_idx

    def _build_forward_args_from_kwargs(self, args_name, args, **kwargs):
        """
        A helper method that converts arguments into positional arguments (by name)

        args - a list of arguments to pass to self.enc_dec_model (tensors from batch)
        args_name - a list of argument name (to be matched against allowed kwargs)
        kwargs - a dict {arg name: arg value} (used for non-tensor values)
        """
        # sanity checks
        if len(args) != len(args_name):
            raise ValueError(f"Mismatch between length in args_name ({len(args_name)}) and args ({len(args)})")
        if any([n in kwargs for n in args_name]):
            raise ValueError(f"args_name = {args_name} cannot overlap kwargs = {list(kwargs.keys())}")

        # get mapping of kwarg names to arg index
        kwargs_to_arg_idx = self._kwargs_to_arg_idx()

        # collect all arguments
        all_args_name = args_name[:]
        all_args = args[:]
        for k, v in kwargs.items():
            all_args_name.append(k)
            all_args.append(v)

        args_idx = [kwargs_to_arg_idx[n] for n in all_args_name]

        # print(f"all_args_name = {all_args_name}   args_idx = {args_idx}")

        # construct args ordered by name (with None as place-holder)
        forward_args = [None] * (max(args_idx) + 1)
        for i, v in zip(args_idx, all_args):
            forward_args[i] = v

        return forward_args

    def _get_forward_output_only_func(self, arg_names, output_name, **kwargs):
        """
        args_idx - maps batch into index of args (with None filling gaps)
        arg_names - corresponding names for a friendly error message
        output_name - name of output (hiddens for encode, logits for decode)
        kwargs - shared arguments (non tensors)
        """

        def fwd_output_only_func(dataloader_iter, model):
            # Extract batch, batch_idx, dataloader_idx only if dataloader_iter is an object of PTL's _DataFetcherWrapper
            if isinstance(dataloader_iter, _DataFetcherWrapper):
                batch, _, _ = next(dataloader_iter)
            else:
                batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) if torch.is_tensor(x) else x for x in batch]

            # map batch and shared args into forward args
            args = self._build_forward_args_from_kwargs(args_name=arg_names, args=batch, **kwargs)
            output = model(*args).contiguous()

            def id_func(output_tensor):
                if isinstance(output_tensor, dict):
                    # handle loss of hidden transformations ("output" is the default output)
                    output_tensor = output_tensor["output"]

                return output_tensor, {output_name: output_tensor}

            return output, id_func

        return fwd_output_only_func

    def _test_validation_step(self, dataloader_iter):
        """
        Shared code for validation and test step
        """

        loss_dict = self.fwd_bwd_step(dataloader_iter, True)

        return loss_dict

    def validation_step(self, dataloader_iter):
        """
        return_values - if given, returns a dictionary with given keys and corresponding values
        """
        outputs = self._test_validation_step(dataloader_iter=dataloader_iter)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_iter.dataloader_idx].append(outputs)
        else:
            self.validation_step_outputs.append(outputs)

    def test_step(self, dataloader_iter):
        outputs = self._test_validation_step(dataloader_iter=dataloader_iter)
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_iter.dataloader_idx].append(outputs)
        else:
            self.test_step_outputs.append(outputs)

    def _test_validation_epoch_end(self, step_outputs, prefix):
        """
        Shared logging for validation and test
        """
        # NOTE: we need to make sure outputs is not empty (this is a workaround for a bug in pytorch lightning (?))
        if not step_outputs:
            logging.warning(f"{prefix} epoch end: outputs is empty")
            return None

        # only the last pipeline parallel stages return loss
        if parallel_state.is_pipeline_last_stage() and len(step_outputs):
            averaged_loss = {k: torch.stack([x[k] for x in step_outputs]).mean() for k in step_outputs[0].keys()}
        else:
            # if we are here we assume that only loss is available and hidden transforms are disabled (since not supported in pipleline parallel)
            averaged_loss = {'loss': torch.tensor(0.0).cuda()}

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        for k, v in averaged_loss.items():
            torch.distributed.broadcast(v, get_last_rank())
            averaged_loss[k] = v
            n = f'{prefix}_{k}'
            # log only '*_loss' values in progress bar
            self.log(n, v, prog_bar=(n.endswith("_loss")), rank_zero_only=True, batch_size=1)

        # free memory
        step_outputs.clear()

        return averaged_loss

    def on_validation_epoch_end(self):
        # FIXME: do we need this? 'global_step' is logged in training_step
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        return self._test_validation_epoch_end(
            step_outputs=self.validation_step_outputs,
            prefix="val",
        )

    def on_test_epoch_end(self):
        return self._test_validation_epoch_end(
            step_outputs=self.test_step_outputs,
            prefix="test",
        )

    def loss_func(self, loss_mask, tokens_loss):
        """
        This function takes as input per-token loss and masks non-required values.
        """
        losses = tokens_loss.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_micro_batch(self, micro_batch):
        """Micro batch returned by MegatronT5 dataloader"""

        data_b = micro_batch

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask

    def _process_global_batch_without_megatron_batch_sampler(self, global_batch, tokenizer=None):
        """Prepares the global batch for megatron-core fwd/bwd functions.
        Global batch is a list of micro batches.
        """
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        text_enc_list = []
        text_dec_list = []
        labels_list = []
        loss_mask_list = []
        enc_mask_list = []
        dec_mask_list = []

        # Determine the maximum encoder and decoder sequence lengths amongst microbatches and pad each microbatch to the max seq length.
        # NOTE: This should only happen for model finetuning where we pad dynamically. Training uses fixed training shapes.

        max_enc_seq_lenth = max([micro_batch['text_enc'].shape[1] for micro_batch in global_batch])
        max_dec_seq_lenth = max([micro_batch['text_dec'].shape[1] for micro_batch in global_batch])

        for micro_batch in global_batch:
            text_enc, text_dec, loss_mask, labels, enc_mask, dec_mask = self.process_micro_batch(micro_batch)
            # Check if encoder sequence length < max encoder sequence length of the global batch and pad.
            if text_enc.shape[1] < max_enc_seq_lenth:
                text_enc = torch.nn.functional.pad(
                    text_enc, (0, max_enc_seq_lenth - text_enc.shape[1], 0, 0), 'constant', tokenizer.pad_id
                )
                enc_mask = torch.nn.functional.pad(
                    enc_mask, (0, max_enc_seq_lenth - enc_mask.shape[1], 0, 0), 'constant', 0
                )
            if text_dec.shape[1] < max_dec_seq_lenth:
                text_dec = torch.nn.functional.pad(
                    text_dec, (0, max_dec_seq_lenth - text_dec.shape[1], 0, 0), 'constant', tokenizer.pad_id
                )
                dec_mask = torch.nn.functional.pad(
                    dec_mask, (0, max_dec_seq_lenth - dec_mask.shape[1], 0, 0), 'constant', 0
                )
                labels = torch.nn.functional.pad(
                    labels, (0, max_dec_seq_lenth - labels.shape[1], 0, 0), 'constant', tokenizer.pad_id
                )
                loss_mask = torch.nn.functional.pad(
                    loss_mask, (0, max_dec_seq_lenth - loss_mask.shape[1], 0, 0), 'constant', 0
                )
            text_enc_list.append(text_enc)
            text_dec_list.append(text_dec)
            labels_list.append(labels)
            loss_mask_list.append(loss_mask)
            enc_mask_list.append(enc_mask)
            dec_mask_list.append(dec_mask)

        # Concatenate to (num_microbatches x micro_batch_size x seq_len)
        tokens_enc_tensor = torch.concat(text_enc_list, dim=0)
        tokens_dec_tensor = torch.concat(text_dec_list, dim=0)
        labels_tensor = torch.concat(labels_list, dim=0)
        loss_mask_tensor = torch.concat(loss_mask_list, dim=0)
        enc_mask_tensor = torch.concat(enc_mask_list, dim=0)
        dec_mask_tensor = torch.concat(dec_mask_list, dim=0)

        return {
            'text_enc': tokens_enc_tensor,
            'text_dec': tokens_dec_tensor,
            'loss_mask': loss_mask_tensor,
            'labels': labels_tensor,
            'enc_mask': enc_mask_tensor,
            'dec_mask': dec_mask_tensor,
        }

    def build_train_valid_test_datasets(self):
        raise NotImplementedError("Please implement this method in child-class")

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self._cfg.data, 'dataloader_type') and self._cfg.data.dataloader_type is not None:
            if self._cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self._cfg.micro_batch_size,
                    global_batch_size=self._cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self._cfg.get('drop_last', True),
                )
            elif self._cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self._cfg.micro_batch_size,
                    global_batch_size=self._cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self._cfg.get('drop_last', True),
                )
            else:
                raise Exception(f'{self._cfg.dataloader_type} dataloader type is not supported.')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def setup(self, stage=None):
        """
        PTL hook that is executed after DDP spawns.
        We setup datasets here as megatron datasets require DDP to instantiate.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.

        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_enc_dec(
            self.enc_dec_model
        )

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}\n'
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}\n'
            f'Number of model parameters on device: {num_parameters_on_device:.2e}\n'
            f'Total number of model parameters: {total_num_parameters:.2e}\n'
        )
        resume_checkpoint_path = self.trainer.ckpt_path

        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self._cfg.data)
        self.setup_validation_data(self._cfg.data)
        self.setup_test_data(self._cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            assert (
                self.cfg.share_token_embeddings
            ), "share_word_embedding must be True when using pipeline model parallel > 1"
            assert (
                self.cfg.share_decoder_tokens_head_embeddings
            ), "share_decoder_tokens_head_embeddings must be True when using pipeline model parallel > 1"
            self.enc_dec_model.sync_initial_word_embeddings()
            if (
                self.cfg.encoder.get('position_embedding_type') == 'learned_absolute'
                and self.cfg.decoder.get('position_embedding_type') == 'learned_absolute'
            ):
                self.enc_dec_model.sync_initial_position_embeddings()
            # Synchronize RPE embeddings across pipeline parallel ranks.
            else:
                if self.cfg.encoder.get('position_embedding_type', 'learned_absolute') == 'relative':
                    self.enc_dec_model.sync_initial_encoder_relative_position_embeddings()
                if self.cfg.decoder.get('position_embedding_type', 'learned_absolute') == 'relative':
                    self.enc_dec_model.sync_initial_decoder_relative_position_embeddings()
                if self.cfg.decoder.get(
                    'position_embedding_type', 'learned_absolute'
                ) == 'relative' and not self.cfg.decoder.get('relative_position_bias_self_attention_only', True):
                    self.enc_dec_model.sync_initial_decoder_cross_attention_relative_position_embeddings()

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_pretraining_data_loader(
                self._train_ds, consumed_samples, num_workers=self._cfg.data.num_workers
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, num_workers=self._cfg.data.num_workers
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, num_workers=0)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        request = batch
        response = self.complete(request)
        logging.info(f"response: {response}")
        return response

    def encode(self, tokens_enc, enc_mask, encoder_input=None, batch_data=None, reconfigure_microbatch=True):
        """
        Args:
            tokens_enc: encoder input tokens
            enc_mask: corresponding mask
            encoder_input: encoder input (bypass tokens), if given tokens_enc can be None.
            batch_data: passed directly to all hidden transformations and losses.
                Can be used to pass additional data like class label.
                Format is not defined and should match the expected format of the used hiddens modules.
        """
        # Check whether the DDP is initialized. This is needed when running inference outside of training loop.
        if not parallel_state.is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            if reconfigure_microbatch:
                _reconfigure_microbatch_calculator(
                    rank=0,  # This doesn't matter since it is only used for logging
                    rampup_batch_size=None,
                    global_batch_size=1,
                    micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                    data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
                )

        # If classes that inherit from this class are using a different tokenizer,
        app_state = AppState()
        if tokens_enc is not None:
            global_batch_per_gpu = tokens_enc.size(0)
            encoder_seq_length = tokens_enc.size(1)
        else:
            global_batch_per_gpu = encoder_input.size(1)
            encoder_seq_length = encoder_input.size(0)

        num_micro_batches_before_decode = get_num_microbatches()
        # Reconfigure microbatch calculator here to set num microbatches to 1 while decoding since its not clear how to decode with "grad acc".
        # reconfigure back to how things were before encode
        if reconfigure_microbatch:
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_per_gpu,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        tensor_shape = [encoder_seq_length, global_batch_per_gpu, self.cfg.encoder.hidden_size]

        # build input arguments description
        if tokens_enc is not None:
            batch_for_pipeline = [tokens_enc, enc_mask, batch_data]
            arg_names = ['enc_input_ids', 'enc_attn_mask', 'batch_data']
        else:
            if encoder_input is None:
                raise ValueError("At least one of tokens_enc and encoder_input must be provided with not None value")

            batch_for_pipeline = [enc_mask]
            arg_names = ['enc_attn_mask']

        if encoder_input is not None:
            batch_for_pipeline.append(encoder_input)
            arg_names.append('enc_input')

        forward_step_func = self._get_forward_output_only_func(
            arg_names=arg_names, output_name="hiddens", output_enc_hidden_only=True
        )

        fwd_bwd_func = get_forward_backward_func()

        # Counter intuitively, we need to set decoder_sequence_length=encoder_seq_length
        # because while running `.encode()`, the last hidden states from encoder are passed through
        # as identity through the pipeline.
        # Setting it to anything else will cause hanging due to tensor shape mismatches.
        output_tensor = fwd_bwd_func(
            forward_step_func=forward_step_func,
            data_iterator=iter(
                [
                    batch_for_pipeline,
                ]
            ),
            model=[self.enc_dec_model],
            forward_only=True,
            num_microbatches=1,
            seq_length=encoder_seq_length,
            decoder_seq_length=encoder_seq_length,
            micro_batch_size=get_micro_batch_size(),
        )

        if output_tensor:
            output_tensor = output_tensor[0]['hiddens']
        else:
            output_tensor = torch.zeros(tensor_shape, dtype=self.autocast_dtype).cuda()

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # Broadcast from the last pipeline stage to all other model-parallel ranks.
            torch.distributed.broadcast(
                output_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

        # Reset microbatch calculator to what it was before decoding.
        if reconfigure_microbatch:
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_per_gpu // num_micro_batches_before_decode,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        # Return the output tensor of encoder and transpose from [seq_len, batch, hidden] to [batch, seq_len, hidden]
        return output_tensor.transpose(1, 0)

    def decode(
        self,
        tokens_enc,
        enc_mask,
        num_tokens_to_generate,
        encoder_input=None,
        tokenizer=None,
        enc_output=None,
        enc_output_attn_mask=None,
        ignore_ids=[],
        bos_id=None,  # If bos=None, will use tokenizer.bos_id unless explicitly set to something else.
        predicted_tokens_dec=None,
        batch_data=None,
        sampling_method: str = "greedy-search",
        sampling_kwargs: dict = {},
    ):
        """
        Args:
            tokens_enc: a tensor of shape [batch_size, seq_len] that contains the input tokens.
            enc_mask: a tensor of shape [batch_size, seq_len] that contains the input tokens mask (1 for active, 0 for inactive).
            num_tokens_to_generate: the max number of tokens to generate.
            encoder_input: a tensor of shape [batch_size, seq_len, hidden_size] that contains the encoder hidden states (replaces tokens_enc if given).
            tokenizer: a tokenizer object.
            enc_output: a tensor of shape [batch_size, seq_len, hidden_size] that contains the encoder hidden states (replaces tokens_enc and encoder_input if given).
            enc_output_attn_mask: a tensor of shape [batch_size, seq_len] that contains the encoder attention mask (replaces enc_mask if given).
            ignore_ids: a list of token ids to ignore when sampling.
            bos_id: the id of the beginning of sentence token. If None, will use tokenizer.bos_id unless explicitly set to something else.
            predicted_tokens_dec: a tensor of shape [batch_size, seq_len] that contains the tokens that have already been decoded.
            sampling_method: a sampling method to use in the decoding iterations. Currently supported methods are
                "beam-search"/"greedy-search"/"topkp-sampling". The argument specifies the sampling function
                that takes in a tensor of logits [batch_size, vocab_size] and returns a tuple
                (tensor of log_probs [batch_size], tensor of sampled tokens_ids from logits [batch_size]).
                If the beam search is enabled, the sampling function returns tensors [batch_size, beam_size]
            sampling_kwargs: dict with arguments to be passed to the sampling function. Please refer to the method
                get_sampling_token_fn to see which arguments are required for a chosen sampling_method.

        Returns:
            tuple of tensors [batch_size, seq_len +1], [batch_size, seq_len] for predicted tokens and their log probs.
            If sampling_method == 'beam-size' and keep_only_best_tokens is False the shape of the tensors are
            [batch_size, beam_size, seq_len + 1], [batch_size, beam_size, seq_len]
        """
        # Setting up the sampling strategy
        sample_token_fn, sampling_kwargs = get_sampling_token_fn(sampling_method, sampling_kwargs)
        beam_search = sampling_method == 'beam-search'
        if beam_search:
            beam_size = sampling_kwargs['beam_size']
            beam_alpha = sampling_kwargs['beam_alpha']
            keep_only_best_tokens = sampling_kwargs['keep_only_best_tokens']
            return_scores = sampling_kwargs['return_scores']
            logging.info(f'Decoding using the beam search method with beam size={beam_size}...')
            assert beam_size >= 1 and beam_alpha >= 0, 'Beam-search related parameters are misspecified'
        else:
            logging.info(f'Decoding using the {sampling_method} method...')

        # Check whether the DDP is initialized. This is needed when running inference outside of training loop.
        if not parallel_state.model_parallel_is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
            )

        # If classes that inherit from this class are using a different tokenizer,
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        app_state = AppState()
        if tokens_enc is not None:
            global_batch_per_gpu = tokens_enc.size(0)
            device = tokens_enc.device
            encoder_seq_length = tokens_enc.size(1)
        elif encoder_input is not None:
            global_batch_per_gpu = encoder_input.size(0)
            device = encoder_input.device
            encoder_seq_length = encoder_input.size(1)
        else:
            global_batch_per_gpu = enc_output.size(0)
            device = enc_output.device
            encoder_seq_length = enc_output.size(1)

        num_micro_batches_before_decode = get_num_microbatches()
        # Reconfigure microbatch calculator here to set num microbatches to 1 while decoding since its not clear how to decode with "grad acc".
        # reconfigure back to how things were before decode
        # TODO: Check if the user is trying to do gradient acc and maybe throw error
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu,  # Make sure that there is no "grad acc" while decoding.
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        # TODO: Figure out how to handle bos being either <bos> for NeMo-Megatron and <pad> for Huggingface/Google.
        bos_id = tokenizer.bos_id if bos_id is None else bos_id
        # initial prompt can be given
        if predicted_tokens_dec is None:
            predicted_tokens_dec = torch.LongTensor([bos_id] * global_batch_per_gpu).unsqueeze(1).to(device)
        # collect log probs that were used in the sampling
        predicted_log_probs = torch.zeros((global_batch_per_gpu, 0), dtype=self.autocast_dtype).to(device)

        tensor_shape = [encoder_seq_length, global_batch_per_gpu, self.cfg.encoder.hidden_size]
        assert predicted_tokens_dec.size(0) == global_batch_per_gpu

        # get encoder hiddens (output)
        if enc_output is None:
            # Encode returns a tensr of shape [batch, seq_len, hidden]
            # All ranks will call `.encode()`, but only the last rank will have a non-empty output tensor.
            enc_output = self.encode(
                tokens_enc=tokens_enc, enc_mask=enc_mask, encoder_input=encoder_input, reconfigure_microbatch=False
            )
        if enc_output_attn_mask is None:
            enc_output_attn_mask = enc_mask

        for i in range(num_tokens_to_generate):
            # No microbatches in decoding. Just the global batch.
            decoder_seq_length = predicted_tokens_dec.size(1)
            dec_mask = predicted_tokens_dec != tokenizer.pad_id
            dec_mask[:, 0] = 1  # Make sure you never mask the first token even if it is <pad>.

            batch_for_pipeline = [enc_output, enc_output_attn_mask, predicted_tokens_dec, dec_mask, batch_data]
            arg_names = ['enc_output', 'enc_output_attn_mask', 'dec_input_ids', 'dec_attn_mask', 'batch_data']

            forward_step_func = self._get_forward_output_only_func(arg_names=arg_names, output_name="logits")
            fwd_bwd_func = get_forward_backward_func()

            output_tensor = fwd_bwd_func(
                forward_step_func=forward_step_func,
                data_iterator=iter(
                    [
                        batch_for_pipeline,
                    ]
                ),
                model=[self.enc_dec_model],
                forward_only=True,
                num_microbatches=1,
                seq_length=encoder_seq_length,
                decoder_seq_length=encoder_seq_length,
                micro_batch_size=get_micro_batch_size(),
            )
            # get output tensor
            if parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor[0]['logits']
                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
                # make sure it won't sample outside the vocab_size range
                output_tensor[:, :, tokenizer.vocab_size :] = -float('Inf')
                # ignore selected indices
                if ignore_ids:
                    output_tensor = output_tensor.index_fill(
                        dim=-1, index=torch.tensor(ignore_ids, device=output_tensor.device), value=-float('Inf')
                    )

                log_probs, token_ids = sample_token_fn(logits=output_tensor[:, -1, :])
                # enforce valid range of token ids
                token_ids = torch.clamp(token_ids, max=tokenizer.vocab_size - 1)

                if beam_search:
                    # beam search: beam creation in the first iteration
                    if i == 0:
                        # resizing decoder inputs to match tensors augmented with beams
                        log_probs, token_ids = log_probs.view(-1), token_ids.view(-1)
                        scores = log_probs.unsqueeze(1).clone()

                        batch_size, src_length, hidden_size = enc_output.size()
                        enc_output_attn_mask = enc_output_attn_mask.repeat(1, beam_size).view(-1, src_length)
                        enc_output = enc_output.repeat(1, beam_size, 1).view(-1, src_length, hidden_size)

                        # resize tensors that collect predicted tokens and logits per iteration to
                        # match shape of tensors augmented with the beam size
                        predicted_tokens_dec = predicted_tokens_dec.repeat(beam_size, 1)
                        predicted_log_probs = predicted_log_probs.repeat(beam_size, 0)

                        pad_profile = torch.zeros_like(scores).long()
                        decoder_seq_lengths = torch.zeros_like(scores).fill_(predicted_tokens_dec.size(1) + 1)

                        # reconfigure batch size for apex since the tensor have been augmented with beam size
                        global_batch_per_gpu = token_ids.shape[0]
                        tensor_shape[1] = global_batch_per_gpu
                        _reconfigure_microbatch_calculator(
                            rank=app_state.global_rank,
                            rampup_batch_size=None,
                            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                            micro_batch_size=global_batch_per_gpu,
                            data_parallel_size=parallel_state.get_data_parallel_world_size(),
                        )

                        # collect all predicted tokens and log_probs
                        predicted_tokens_dec = torch.cat(
                            [predicted_tokens_dec.to(token_ids.device), token_ids.unsqueeze(1)], dim=1
                        )
                        predicted_log_probs = torch.cat(
                            [predicted_log_probs.to(log_probs.device), log_probs.unsqueeze(1)], dim=1
                        )

                    # beam search: beam selection in the second iteration and on
                    else:
                        # mask all finished hypotheses to exclude them from beam
                        pad_mask = pad_profile.repeat(1, beam_size)

                        # for all prefixes ending with <eos> or <pad> replace generated
                        # continuations with <pad>
                        token_ids = tokenizer.pad_id * pad_mask + token_ids * (1 - pad_mask)

                        # force all hypotheses but one generated from already finished
                        # hypotheses to have extremely low score, so they will not be
                        # considered during beam re-ranking
                        pad_mask[:, 1:] = pad_mask[:, 1:] * -10000.0
                        scores = scores + log_probs * (1 - pad_mask).to(scores.dtype)

                        # choose top-k hypotheses with length penalty applied
                        len_penalties = compute_beam_search_len_penalty(decoder_seq_lengths, beam_alpha)
                        scores = scores / len_penalties
                        scores, indices = sample_token_fn(scores.view(-1, beam_size**2), dim=1, log_softmax=False)
                        scores = scores.view(-1, 1) * len_penalties

                        # select predicted sequences which correspond to the chosen hypotheses
                        predicted_tokens_dec = predicted_tokens_dec.unsqueeze(1).repeat(1, beam_size, 1)
                        predicted_tokens_dec = torch.cat((predicted_tokens_dec, token_ids.unsqueeze(2)), dim=2)
                        predicted_tokens_dec = predicted_tokens_dec.view(batch_size, beam_size**2, -1)
                        p_len = predicted_tokens_dec.size(2)
                        predicted_tokens_dec_ids = indices.unsqueeze(2).repeat(1, 1, p_len)
                        predicted_tokens_dec = predicted_tokens_dec.gather(1, predicted_tokens_dec_ids).view(-1, p_len)

                        # select logits which correspond to the chosen hypotheses
                        predicted_log_probs = predicted_log_probs.unsqueeze(1).repeat(1, beam_size, 1)
                        predicted_log_probs = torch.cat((predicted_log_probs, log_probs.unsqueeze(2)), dim=2)
                        predicted_log_probs = predicted_log_probs.view(batch_size, beam_size**2, -1)
                        predicted_log_probs = predicted_log_probs.gather(1, predicted_tokens_dec_ids[:, :, 1:]).view(
                            -1, p_len - 1
                        )

                        # update decoder_seq_length and pad_profile
                        not_eos_pad = predicted_tokens_dec.ne(tokenizer.eos_id) & predicted_tokens_dec.ne(
                            tokenizer.pad_id
                        )
                        decoder_seq_lengths = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
                        pad_profile = (~not_eos_pad[:, -1:]).long()
                else:
                    # collect all predicted tokens and log_probs
                    predicted_tokens_dec = torch.cat(
                        [predicted_tokens_dec.to(token_ids.device), token_ids.unsqueeze(1)], dim=1
                    )
                    predicted_log_probs = torch.cat(
                        [predicted_log_probs.to(log_probs.device), log_probs.unsqueeze(1)], dim=1
                    )

            else:
                predicted_tokens_dec = torch.zeros(
                    (predicted_tokens_dec.shape[0], predicted_tokens_dec.shape[1] + 1),
                    dtype=predicted_tokens_dec.dtype,
                ).cuda()
                predicted_log_probs = torch.zeros(
                    (predicted_log_probs.shape[0], predicted_log_probs.shape[1] + 1), dtype=self.autocast_dtype
                ).cuda()

            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # Broadcast from the last pipeline stage to all other model-parallel ranks.
                torch.distributed.broadcast(
                    predicted_tokens_dec,
                    parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                torch.distributed.broadcast(
                    predicted_log_probs,
                    parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        # Reset microbatch calculator to what it was before decoding.
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        if beam_search and beam_size > 1:
            if keep_only_best_tokens:
                len_penalties = compute_beam_search_len_penalty(decoder_seq_lengths, 0)
                scores = scores / len_penalties
                scores = scores.view(-1, beam_size)
                best_ids = torch.argmax(scores, dim=1, keepdim=True)
                scores = scores * len_penalties.view(-1, beam_size)
                scores = scores.gather(1, best_ids)
                best_tokens = best_ids.repeat(1, predicted_tokens_dec.size(1)).unsqueeze(1)
                predicted_tokens_dec = (
                    predicted_tokens_dec.view(batch_size, beam_size, -1).gather(1, best_tokens).squeeze(1)
                )
                predicted_log_probs = (
                    predicted_log_probs.view(batch_size, beam_size, -1).gather(1, best_tokens[:, :, 1:]).squeeze(1)
                )
            else:
                predicted_tokens_dec = predicted_tokens_dec.view(batch_size, beam_size, -1)
                predicted_log_probs = predicted_log_probs.view(batch_size, beam_size, -1)
                scores = scores.view(-1, beam_size)

        if beam_search:
            if return_scores:
                return predicted_tokens_dec, predicted_log_probs, scores

        return predicted_tokens_dec, predicted_log_probs

    def complete(self, request: Dict):
        """
        Autoregressively invokes language model in the inference mode

        Args:
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.

        Returns:
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * text: completion text (as a single string)

        """
        app_state = AppState()

        # The complete method only works with global batch = micro batch size = data parallel size = 1.
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
        )
        app_state = AppState()

        response = {}
        self.freeze()
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        bos_id = request['bos_id']
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.tokenizer.ids_to_tokens(tokens_enc[0].cpu().numpy().tolist()))
        enc_mask = tokens_enc != self.tokenizer.pad_id

        predicted_tokens_ids, log_probs = self.decode(
            tokens_enc, enc_mask, int(request['tokens_to_generate']), bos_id=bos_id
        )
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.tokenizer.pad_id]
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
        if hasattr(self.tokenizer, 'special_token_to_id'):
            predicted_tokens_ids = [
                id for id in predicted_tokens_ids if id not in self.tokenizer.special_token_to_id.values()
            ]

        predicted_tokens_dec = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        When using pipeline parallelism, we need the global batch to remain on the CPU,
        since the memory overhead will be too high when using a large number of microbatches.
        Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """Certain trainer configurations can break training.
        Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    def list_available_models(self):
        pass

    def build_model_parallel_config(self):
        """Hidden size needs to be set from the cfg.encoder for the pipeline schedule."""

        model_parallel_config = super().build_model_parallel_config()
        try:
            # hidden size is needed for pipeline schedules but is not currently in ModelParallelConfig
            setattr(model_parallel_config, 'hidden_size', self.cfg.encoder.hidden_size)
        except AttributeError:
            logging.warning(
                f'encoder.hidden_size not found in {self.cfg}. Set this in model_parallel_config if using pipeline parallelism.'
            )
        return model_parallel_config
