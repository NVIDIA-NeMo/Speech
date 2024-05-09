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

import itertools
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer
from tqdm import tqdm

from nemo.collections.multimodal.data.clip.clip_dataset import (
    build_imagenet_validation_dataloader,
    build_train_valid_datasets,
)
from nemo.collections.multimodal.losses.clip_loss import ClipLoss
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank, torch_dtype_from_precision
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class CLIPVisionTransformer(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, model_cfg, model_parallel_config, pre_process=True, post_process=True, skip_head=False):
        super(CLIPVisionTransformer, self).__init__()

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )

        self.config = model_parallel_config
        self.hidden_size = model_cfg.hidden_size
        self.global_average_pool = model_cfg.global_average_pool
        self.pre_process = pre_process
        self.post_process = post_process
        self.skip_head = skip_head

        if model_cfg.get("class_token_length") is None or model_cfg.get("class_token_length") <= 0:
            class_token = False
        else:
            class_token = True
        self.backbone = VitBackbone(
            model_cfg,
            model_parallel_config,
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            class_token=class_token,
            single_token_output=False,
        )

        if self.post_process and not skip_head:
            self.output_dim = model_cfg.output_dim
            self.head = torch.nn.Linear(self.hidden_size, self.output_dim, bias=False,)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process and not self.skip_head:
            if self.global_average_pool:
                hidden_states = hidden_states.mean(dim=1)
            else:
                hidden_states = hidden_states[:, 0]
            hidden_states = self.head(hidden_states)
        # print("vision_head", hidden_states.shape)
        return hidden_states


class CLIPTextTransformer(MegatronModule):
    """Text Transformer Model."""

    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True):
        super(CLIPTextTransformer, self).__init__()

        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = model_cfg.fp16_lm_cross_entropy
        self.sequence_parallel = model_cfg.sequence_parallel
        self.gradient_accumulation_fusion = model_cfg.gradient_accumulation_fusion

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )
        self.language_model, self._language_model_key = get_language_model(
            config=model_parallel_config,
            vocab_size=padded_vocab_size,
            hidden_size=model_cfg.hidden_size,
            hidden_dropout=model_cfg.hidden_dropout,
            attention_dropout=model_cfg.attention_dropout,
            num_tokentypes=0,
            max_position_embeddings=model_cfg.max_position_embeddings,
            num_layers=model_cfg.num_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            apply_query_key_layer_scaling=model_cfg.apply_query_key_layer_scaling,
            kv_channels=model_cfg.kv_channels,
            ffn_hidden_size=model_cfg.ffn_hidden_size,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            position_embedding_type=model_cfg.get("position_embedding_type", "learned_absolute"),
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=model_cfg.init_method_std,
            precision=model_cfg.precision,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            activations_checkpoint_granularity=model_cfg.activations_checkpoint_granularity,
            activations_checkpoint_method=model_cfg.activations_checkpoint_method,
            activations_checkpoint_num_layers=model_cfg.activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=model_cfg.activations_checkpoint_layers_per_pipeline,
            normalization=model_cfg.normalization,
            layernorm_epsilon=model_cfg.layernorm_epsilon,
            bias_activation_fusion=model_cfg.bias_activation_fusion,
            bias_dropout_add_fusion=model_cfg.bias_dropout_add_fusion,
            masked_softmax_fusion=model_cfg.masked_softmax_fusion,
            persist_layer_norm=model_cfg.persist_layer_norm,
            openai_gelu=model_cfg.openai_gelu,
            onnx_safe=model_cfg.onnx_safe,
            megatron_legacy=model_cfg.megatron_legacy,
            transformer_engine=model_cfg.transformer_engine,
            fp8=model_cfg.fp8,
            fp8_e4m3=model_cfg.fp8_e4m3,
            fp8_hybrid=model_cfg.fp8_hybrid,
            fp8_margin=model_cfg.fp8_margin,
            fp8_interval=model_cfg.fp8_interval,
            fp8_amax_history_len=model_cfg.fp8_amax_history_len,
            fp8_amax_compute_algo=model_cfg.fp8_amax_compute_algo,
            reduce_amax=model_cfg.get('reduce_amax', True),
            use_emha=model_cfg.use_emha,
            activation=model_cfg.get('activation', 'gelu'),
            use_flash_attention=model_cfg.get('flash_attention', False),
        )

        self.initialize_word_embeddings(
            init_method=init_method_normal(model_cfg.init_method_std),
            vocab_size=padded_vocab_size,
            hidden_size=model_cfg.hidden_size,
        )

        # TODO (yuya): check this position id
        self.position_ids = None
        if self.pre_process:
            self.position_ids = torch.arange(model_cfg.max_position_embeddings).expand(1, -1).cuda()

        if self.post_process:
            self.output_dim = model_cfg.output_dim
            self.head = torch.nn.Linear(model_cfg.hidden_size, self.output_dim, bias=False,)

        self.attn_mask = self.build_attention_mask(model_cfg.max_position_embeddings)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def build_attention_mask(self, max_position_embeddings):
        # lazily create causal attention mask, with full attention between the tokens
        mask = torch.empty(max_position_embeddings, max_position_embeddings, dtype=bool, device='cuda')
        mask.fill_(True)
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.reshape(1, 1, max_position_embeddings, max_position_embeddings)
        return mask

    def forward(
        self, input_ids,
    ):
        # input_ids: [b, s]
        # position_ids: [b, s]
        # attention_mask: [1, 1, s, s]

        hidden_states = self.language_model(
            input_ids,
            self.position_ids,
            self.attn_mask,
            token_type_ids=None,
            layer_past=None,
            get_key_value=False,
            encoder_input=None,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            checkpoint_activations_all_layers=None,
        )

        if self.post_process:
            # shape = [seq, bsz, hidden]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            hidden_states = hidden_states[input_ids.argmax(dim=-1), torch.arange(hidden_states.shape[1])]
            return self.head(hidden_states)

        return hidden_states


class CLIPModel(MegatronModule):
    """CLIP Model"""

    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True):
        super(CLIPModel, self).__init__()

        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.vision_encoder = CLIPVisionTransformer(
            model_cfg.vision, model_parallel_config, pre_process=self.pre_process, post_process=self.post_process,
        )
        self.text_encoder = CLIPTextTransformer(
            model_cfg.text,
            model_parallel_config,
            padded_vocab_size,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # TODO (yuya): fix this
        pass

    def forward(self, images, captions):
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(captions)

        if self.post_process:
            return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1), self.logit_scale.exp()

        return image_features, text_features


class MegatronCLIPModel(MegatronBaseModel):
    """Megatron CLIP Model."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        self.imagenet_val = None
        super().__init__(cfg, trainer=trainer)

        self._validate_trainer()

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                on_cpu=True,
                virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            )
        else:
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            )

        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None:
            self.model = self.model[0]

        if self.megatron_amp_O2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            # TODO (yuya): check this; FP16 Module might not work; when self.model is a list?
            if isinstance(self.model, list):
                converted_model = []
                for module in self.model:
                    converted_model.append(
                        Float16Module(config=self.model_parallel_config, module=module, precision=cfg.precision)
                    )
                    self.model = converted_model
            else:
                self.model = Float16Module(
                    config=self.model_parallel_config, module=self.model, precision=cfg.precision
                )

        self.autocast_dtype = torch_dtype_from_precision(self.trainer.precision)
        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        self.transformer_engine = cfg.get('transformer_engine', False)

        # Convert the global-batch-based profile index to micro-batch index
        if hasattr(self, '_nsys_profile_enabled') or hasattr(self, '_memory_profile_enabled'):
            mp_size = cfg.get('tensor_model_parallel_size', 1) * cfg.get('pipeline_model_parallel_size', 1)
            data_parallel_world_size = trainer.world_size // mp_size
            grad_accum_steps = cfg.get('global_batch_size') // (cfg.get('micro_batch_size') * data_parallel_world_size)
            if hasattr(self, '_nsys_profile_enabled'):
                self._nsys_profile_start_step *= grad_accum_steps
                self._nsys_profile_end_step *= grad_accum_steps
            if hasattr(self, '_memory_profile_enabled'):
                self._memory_profile_start_step *= grad_accum_steps
                self._memory_profile_end_step *= grad_accum_steps
        self.get_attention_mask_from_fusion = self.cfg.get('get_attention_mask_from_fusion', True)
        self.initialize_ub = self.cfg.get('ub_tp_comm_overlap', False)

    def get_module_list(self):
        if isinstance(self.model, list):
            return [model.module if isinstance(model, Float16Module) else model for model in self.model]
        elif isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = CLIPModel(
            model_cfg=self.cfg,
            model_parallel_config=self.model_parallel_config,
            padded_vocab_size=self.padded_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.get('do_layer_norm_weight_decay', False):
            if isinstance(self.model, list):
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
            else:
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization([self.model])

        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets = []
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, Float16Module):
                        module = module.module
                    stage_bucket = []
                    for layer in itertools.chain(
                        module.vision_encoder.backbone.transformer.layers,
                        module.text_encoder.language_model.encoder.layers,
                    ):
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, Float16Module):
                        module = module.module
                    for layer in itertools.chain(
                        module.vision_encoder.backbone.transformer.layers,
                        module.text_encoder.language_model.encoder.layers,
                    ):
                        buckets.append(
                            [p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)]
                        )
            buckets.reverse()
            used_params = set()
            for bucket in buckets:
                used_params.update(bucket)
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)
            self.distributed_adam_buckets = buckets

        return super().configure_optimizers()

    def forward(self, image, text):
        output_tensor = self.model(image, text)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, forward_only):

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(self._optimizer.no_sync, greedy_grad_copy=self.megatron_amp_O2,)
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        # pipeline schedules will get these from self.model.config
        for module in self.get_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.stack(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                raise NotImplementedError("Losses of micro batches sizes must be uniform!")
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def initialize_ub_func(self):
        ub_cfgs = self.cfg.get('ub_tp_comm_overlap_cfg', None)
        if ub_cfgs is None:
            warnings.warn(
                "Couldn't find TP config. Please check the path correctness. Initializing TP comm overlap with the default config."
            )

        input_shape = [
            self.cfg.get('encoder_seq_length') * self.cfg.get('micro_batch_size'),
            self.cfg.get('hidden_size'),
        ]

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=self.cfg.get('tensor_model_parallel_size'),
            use_fp8=self.cfg.get('fp8'),
            ub_cfgs=ub_cfgs,
        )
        self.initialize_ub = False

    def training_step(self, dataloader_iter):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        if self.with_distributed_adam:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, Float16Module):
                    module = module.module
                module = module.text_encoder.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        loss_mean = self.fwd_bwd_step(dataloader_iter, False)

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            # if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
            #     # main grads are stored in the MainParamsOptimizer wrapper
            self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision in [16, '16', '16-mixed']:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step + 1, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

        return loss_mean

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
            Modified from megatron-lm:
            https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_sequence_parallel_module_grads(module, grads)
        else:
            self._append_sequence_parallel_module_grads(self.model, grads)

        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    def get_forward_output_and_loss_func(self):
        loss_func = ClipLoss(local_loss=self.cfg.local_loss, gather_with_grad=self.cfg.gather_with_grad,)

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                images = batch["images"].cuda(non_blocking=True)
                captions = batch["captions"].cuda(non_blocking=True)
            else:
                # GPT3 uses only causal mask, which doesn't need attention mask
                if parallel_state.is_pipeline_first_stage():
                    # Fist pipeline stage needs only the tokens and position_ids
                    images = batch["images"].cuda(non_blocking=True)
                    captions = batch["captions"].cuda(non_blocking=True)
                else:
                    # Intermediate / Last pipeline stage doesn't need any inputs
                    images, captions = None, None

            output_tensor = model(images, captions)
            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            raise NotImplementedError

        return fwd_output_only_func

    def zero_shot_classifier(self):
        if self.cfg.get("megatron_amp_O2", False):
            text_encoder = self.model.module.text_encoder
        else:
            text_encoder = self.model.text_encoder

        with torch.no_grad():
            zeroshot_weights = []
            for texts in self.imagenet_val["texts"]:
                texts = texts.cuda(non_blocking=True)
                # TODO (yuya): distributed not working
                with torch.cuda.amp.autocast(
                    enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
                ):
                    class_embeddings = text_encoder(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights

    def zero_shot_eval(self):
        def accuracy(output, target, topk=(1,)):
            pred = output.topk(max(topk), 1, True, True)[1].t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

        logging.info('Starting zero-shot imagenet.')

        logging.info('Building zero-shot classifier')
        classifier = self.zero_shot_classifier()

        logging.info('Using classifier')

        if self.cfg.get("megatron_amp_O2", False):
            vision_encoder = self.model.module.vision_encoder
        else:
            vision_encoder = self.model.vision_encoder
        with torch.no_grad():
            top1, top5, n = 0.0, 0.0, 0.0
            for images, target in tqdm(self.imagenet_val["images"], desc="Imagenet Zero-shot Evaluation", leave=False):
                if images is None or target is None:
                    continue

                images = images.cuda(non_blocking=True).to(self.autocast_dtype)
                target = target.cuda(non_blocking=True)
                # predict
                with torch.cuda.amp.autocast(
                    enabled=self.autocast_dtype in (torch.half, torch.bfloat16), dtype=self.autocast_dtype,
                ):
                    image_features = vision_encoder(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100.0 * image_features @ classifier

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        logging.info('Finished zero-shot imagenet.')
        top1 = top1 / n
        top5 = top5 / n
        return top1, top5

    def validation_step(self, dataloader_iter):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        loss = self.fwd_bwd_step(dataloader_iter, True)
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        # TODO (yuya): need fix later, check with Sean
        if not self.validation_step_outputs:
            return

        # Run zero shot imagenet evaluation
        if self.imagenet_val is not None:
            imagenet_metric = torch.zeros(2).cuda()
            imagenet_metric[0], imagenet_metric[1] = self.zero_shot_eval()
            imagenet_metric = average_losses_across_data_parallel_group(imagenet_metric)
            self.log('imagenet_top1', imagenet_metric[0], prog_bar=True, rank_zero_only=True, batch_size=1)
            self.log('imagenet_top5', imagenet_metric[1], prog_bar=True, rank_zero_only=True, batch_size=1)

        if parallel_state.is_pipeline_last_stage():
            averaged_metrics = torch.tensor(
                [torch.stack(self.validation_step_outputs).mean()], dtype=torch.float32, device='cuda'
            )
        else:
            averaged_metrics = torch.tensor([0.0], dtype=torch.float32, device='cuda')

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_metrics, get_last_rank())
        averaged_loss = averaged_metrics

        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for CLIP...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        self._train_ds, self._validation_ds = build_train_valid_datasets(
            model_cfg=self.cfg, consumed_samples=self.compute_consumed_samples(0), tokenizer=self.tokenizer,
        )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building datasets for CLIP.')

        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
        else:
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda()

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        # allowing restored models to optionally setup datasets
        self.build_train_valid_test_datasets()

        # Batch size need to be provided for webdatset
        self._num_micro_batches = get_num_microbatches()
        self._micro_batch_size = self.cfg.micro_batch_size

        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

        if self.cfg.data.get("imagenet_val") is not None:
            self.imagenet_val = build_imagenet_validation_dataloader(self.cfg, self.tokenizer)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = torch.utils.data.DataLoader(
                self._train_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = torch.utils.data.DataLoader(
                self._validation_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = torch.utils.data.DataLoader(
                self._test_ds, batch_size=self._micro_batch_size, num_workers=cfg.num_workers, pin_memory=True,
            )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        raise NotImplementedError

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()
