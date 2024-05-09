# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import queue
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron import dataset_utils
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron.bert.bert_model import (
    MCoreBertModelWrapperWithPostLNSupport,
    NeMoBertModel,
)
from nemo.collections.nlp.models.language_modeling.megatron.bert.bert_spec import (
    bert_layer_with_transformer_engine_spec_postln,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import logging

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    import logging

    from lddl.torch_mp import get_bert_pretrain_data_loader

    HAVE_LDDL = True
except (ImportError, ModuleNotFoundError):
    HAVE_LDDL = False

try:
    from megatron.core import parallel_state
    from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False


class MegatronBertModel(MegatronBaseModel):
    """
    Megatron Bert pretraining.
    Model returns [batch, seq, hidden] shape
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        self.cfg = cfg

        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        super().__init__(cfg, trainer=trainer, no_lm_init=False)

        self._validate_trainer()

        self.transformer_config = self.build_transformer_config()

        self.mcore_bert = cfg.get('mcore_bert', False)

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        # used in NVIDIA NGC PyTorch containers
        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_lm_loss_buffer = []
        self._reduced_sop_loss_buffer = []

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
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
            self._wrap_model_for_O2()

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

    def model_provider_func(self, pre_process, post_process):
        cfg = self.cfg
        num_tokentypes = 2 if cfg.bert_binary_head else 0
        transformer_block_type = cfg.get('transformer_block_type', 'pre_ln')
        if self.mcore_bert:
            if transformer_block_type == 'pre_ln':
                layer_spec = bert_layer_with_transformer_engine_spec
            else:
                layer_spec = bert_layer_with_transformer_engine_spec_postln

            model = MCoreBertModelWrapperWithPostLNSupport(
                config=self.transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=self.padded_vocab_size,
                max_sequence_length=cfg.max_position_embeddings,
                num_tokentypes=num_tokentypes,
                add_binary_head=cfg.bert_binary_head,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                transformer_block_type=transformer_block_type,
                add_pooler=self.cfg.get('add_pooler', False),
            )
        else:
            model = NeMoBertModel(
                config=self.model_parallel_config,
                vocab_size=self.padded_vocab_size,
                hidden_size=cfg.hidden_size,
                max_position_embeddings=cfg.max_position_embeddings,
                num_layers=cfg.num_layers,
                num_attention_heads=cfg.num_attention_heads,
                apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=cfg.get('kv_channels', None),
                ffn_hidden_size=cfg.ffn_hidden_size,
                num_tokentypes=num_tokentypes,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=cfg.get('init_method_std', 0.02),
                fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
                hidden_dropout=cfg.get('hidden_dropout', 0.1),
                precision=cfg.get('precision', 16),
                fp32_residual_connection=cfg.get('fp32_residual_connection', False),
                activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
                activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_layers_per_pipeline=self.cfg.get(
                    'activations_checkpoint_layers_per_pipeline', None
                ),
                layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
                masked_softmax_fusion=cfg.get('masked_softmax_fusion', True),
                normalization=cfg.get('normalization', 'layernorm'),
                transformer_block_type=transformer_block_type,
                bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
                bias_dropout_add_fusion=cfg.get("bias_dropout_add_fusion", True),
                onnx_safe=cfg.get('onnx_safe', False),
                add_binary_head=cfg.bert_binary_head,
                megatron_legacy=cfg.get('megatron_legacy', False),
                position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
                add_pooler=cfg.get('add_pooler', True),
                add_lm_head=cfg.get('add_lm_head', True),
            )

        return model

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                batch, batch_idx, dataloader_idx = next(dataloader_iter)
                tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = (
                    batch['text'].cuda(non_blocking=True),
                    batch['types'].cuda(non_blocking=True),
                    batch['is_random'].cuda(non_blocking=True),
                    batch['loss_mask'].cuda(non_blocking=True),
                    batch['labels'].cuda(non_blocking=True),
                    batch['padding_mask'].cuda(non_blocking=True),
                )
            else:
                batch, batch_idx, dataloader_idx = next(dataloader_iter)
                if parallel_state.is_pipeline_first_stage():
                    tokens = batch['text'].cuda(non_blocking=True)
                    types = batch['types'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    loss_mask, lm_labels = None, None
                elif parallel_state.is_pipeline_last_stage():
                    loss_mask = batch['loss_mask'].cuda(non_blocking=True)
                    lm_labels = batch['labels'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    tokens, types = None, None
                else:
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    tokens, types, loss_mask, lm_labels = None, None, None, None

            dataloader_iter._dataloader_idx = dataloader_idx
            dataloader_iter._batch_idx = batch_idx

            if not self.cfg.bert_binary_head:
                types = None

            forward_args = {
                "input_ids": tokens,
                "attention_mask": padding_mask,
                "lm_labels": lm_labels,
            }

            if not self.mcore_bert:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                forward_args["model"] = model
                forward_args["token_type_ids"] = types
            else:
                forward_args["tokentype_ids"] = types

            output_tensor = None
            if self.mcore_bert:
                output_tensor = model(**forward_args)
            else:
                output_tensor = self.forward(**forward_args)

            def loss_func(output_tensor):
                loss_dict = self.loss_func(loss_mask, sentence_order, output_tensor)
                if 'sop loss' in loss_dict:
                    lm_loss = loss_dict['lm loss']
                    sop_loss = loss_dict['sop loss']
                    loss = lm_loss + sop_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss, sop_loss])
                else:
                    lm_loss = loss_dict['lm loss']
                    loss = lm_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss])
                return loss, {'loss': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
        model=None,
    ):
        if model is None:
            model = self.model

        if self.mcore_bert:
            output_tensor = model(input_ids, attention_mask, tokentype_ids=token_type_ids,)
        else:
            output_tensor = model(
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
                lm_labels=lm_labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            )
        if parallel_state.is_pipeline_last_stage():
            # Return the output tensor of encoder and transpose from [seq_len, batch, hidden] to [batch, seq_len, hidden]
            if torch.is_tensor(output_tensor):
                output_tensor = output_tensor.transpose(1, 0).contiguous()
            else:
                lm_loss_, sop_logits = output_tensor

                lm_loss_ = lm_loss_.transpose(1, 0).contiguous()
                if sop_logits is not None:
                    sop_logits = sop_logits.transpose(1, 0).contiguous()
                output_tensor = (lm_loss_, sop_logits)

        return output_tensor

    def training_step(self, dataloader_iter):

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
                if isinstance(module, (Float16Module, MCoreFloat16Module)):
                    module = module.module
                if not self.mcore_bert:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        if self.cfg.data.dataloader_type == "LDDL":
            # this is of type bert dataset
            seq_length = dataloader_iter.iterator.loaders.get_seqlen()
        else:
            seq_length = self.cfg.encoder_seq_length

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(dataloader_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=False,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        if losses_reduced_per_micro_batch:
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.vstack(loss_tensors_list)
            loss_mean = loss_tensor.mean(axis=0)
        else:
            if self.cfg.bert_binary_head == True:
                loss_mean = torch.tensor([0.0, 0.0, 0.0]).cuda()
            else:
                loss_mean = torch.tensor([0.0, 0.0]).cuda()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
                # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.torch_dtype == torch.float16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        if (dataloader_iter._batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            self.log('reduced_train_loss', loss_mean[0], prog_bar=True, batch_size=1)
            if len(loss_mean) > 2:
                self.log('reduced_lm_train_loss', loss_mean[1], prog_bar=True, batch_size=1)
                self.log('reduced_sop_train_loss', loss_mean[2], prog_bar=True, batch_size=1)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr, batch_size=1)
            self.log('global_step', self.trainer.global_step, prog_bar=True, batch_size=1)
            self.log(
                'consumed_samples', self._compute_consumed_samples_after_training_step(), prog_bar=True, batch_size=1,
            )

        return loss_mean[0]

    def _make_data_iterator_list(self, data_iterator: Iterator) -> List[Iterator]:
        """ Convert data iterator into form expected by Megatron
            With interleaved pipeline parallelism, Megatron expects a
            list of one data iterator per model chunk. Each model
            chunk independently gets data from its data iterator, so
            we need to interact with the data iterator multiple times
            for each microbatch step. Instead of incorporating this
            logic into the data loader, we cache the iterator's output
            to the first model chunk and reuse it in the other model
            chunks.
        """

        if not isinstance(self.model, list) or len(self.model) == 1:
            return data_iterator  # TODO @tmoon: Remove
            # TODO @tmoon: Use once available in Megatron-LM
            # return DataIteratorList([data_iterator])

        class CachingIterator:
            """Iterator wrapper that caches values"""

            class Proxy:
                """Returns values from caching iterator wrapper
                Assumed to never advance past the caching iterator.
                """

                def __init__(self):
                    self.cache = queue.Queue()

                def __iter__(self):
                    return self

                def __next__(self):
                    return self.cache.get_nowait()

            def __init__(self, iterator: Iterator):
                self.iterator = iterator
                self.proxies = []

            def make_proxy(self):
                self.proxies.append(CachingIterator.Proxy())
                return self.proxies[-1]

            def __iter__(self):
                return self

            def __next__(self):
                val = next(self.iterator)
                for proxy in self.proxies:
                    proxy.cache.put(val)
                return val

        # Make list of iterator wrappers
        iters = [CachingIterator(data_iterator)]
        while len(iters) < len(self.model):
            iters.append(iters[0].make_proxy())
        return iters  # TODO @tmoon: Remove
        # TODO @tmoon: Use once available in Megatron-LM
        # return DataIteratorList(iters)

    def allreduce_first_last_embeddings(self):

        # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            module_list = self.get_model_module_list()
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                module = module_list[0]
            elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                module = module_list[-1]

            share_embeddings = (
                module.share_embeddings_and_output_weights if self.mcore_bert else module.share_token_embeddings
            )
            if share_embeddings:
                word_embeddings_weight = (
                    module.shared_embedding_or_output_weight() if self.mcore_bert else module.word_embeddings_weight()
                )

                if self.megatron_amp_O2:
                    # O2 recipe stores a "main" copy of weights and grads
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

    def validation_step(self, dataloader_iter):
        prefix = "test" if self.trainer.testing else "val"
        if self.cfg.data.dataloader_type == "LDDL":
            seq_length = dataloader_iter.iterator.get_seqlen()
        else:
            seq_length = self.cfg.encoder_seq_length

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(dataloader_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        if losses_reduced_per_micro_batch:
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.vstack(loss_tensors_list)
            loss_mean = loss_tensor.mean(axis=0)
        else:
            loss_mean = torch.tensor([0.0]).cuda()

        loss = loss_mean[0]
        self.validation_step_outputs.append(loss) if prefix == 'val' else self.test_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        if parallel_state.is_pipeline_last_stage():
            averaged_loss = torch.stack(self.validation_step_outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, dataloader_iter):
        return self.validation_step(dataloader_iter)

    def on_test_epoch_end(self):
        averaged_loss = average_losses_across_data_parallel_group(self.test_step_outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, sentence_order, output_tensor):
        lm_loss_, sop_logits = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()

        # Sometimes when the number of tokens is very small, none of the tokens get masked for prediction. In that case loss mask is all zeros
        # i.e Happens when the entire batch is masked out (Practically when MBS=1 or 2, and the number of tokens in each batch is < 7 )
        if loss_mask.sum() == 0:
            lm_loss = torch.sum(lm_loss_.view(-1)) * 0.0
        else:
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        if sop_logits is not None:
            sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
            sop_loss = sop_loss.float()
            return {'lm loss': lm_loss, 'sop loss': sop_loss}
            # loss = lm_loss + sop_loss
            # averaged_losses = average_losses_across_data_parallel_group(
            #     [lm_loss, sop_loss])
            # return loss, {'lm loss': averaged_losses[0],
            #               'sop loss': averaged_losses[1]}

        else:
            return {'lm loss': lm_loss}
            # loss = lm_loss
            # averaged_losses = average_losses_across_data_parallel_group(
            #     [lm_loss])
            # return loss, {'lm loss': averaged_losses[0]}

    def build_LDDL_data(self, cfg):
        if not HAVE_LDDL:
            raise ImportError(
                "LDDL was not found. Please see the LDDL README for installation instructions: https://github.com/NVIDIA/LDDL#installation."
            )
        logging.info(f'Starting building LDDL Dataloaders')
        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        num_micro_batches = self.cfg.global_batch_size // (self.cfg.micro_batch_size * data_parallel_size)
        global_batch_size_on_this_data_parallel_rank = num_micro_batches * self.cfg.micro_batch_size
        samples_consumed_dploader = self.compute_consumed_samples(0) // data_parallel_size
        # We run under the assumption that the datapath is the prefix if LDDL dataloader
        train_lddl_data_path = self.cfg.data.data_prefix[0]
        self._train_dl = get_bert_pretrain_data_loader(
            train_lddl_data_path,
            dp_rank=parallel_state.get_data_parallel_rank(),
            local_rank=self.local_rank,
            shuffle_buffer_size=16384,
            shuffle_buffer_warmup_factor=16,
            vocab_file=self.cfg.tokenizer.vocab_file,
            data_loader_kwargs={
                'batch_size': global_batch_size_on_this_data_parallel_rank,
                'num_workers': self.cfg.data.num_workers,
                'prefetch_factor': 2,
            },
            mlm_probability=0.15,
            base_seed=self.cfg.seed,
            log_level=logging.CRITICAL,
            log_dir="/tmp/log",
            return_raw_samples=False,
            start_epoch=0,
            sequence_length_alignment=8,
            ignore_index=-1,
            samples_seen=samples_consumed_dploader,
            micro_batch_size=self.cfg.micro_batch_size,
        )
        logging.info(f'Completed build train LDDL Dataloader')
        if len(self.cfg.data.data_prefix) > 1:
            val_lddl_data_path = self.cfg.data.data_prefix[1]
            self._validation_dl = get_bert_pretrain_data_loader(
                val_lddl_data_path,
                dp_rank=parallel_state.get_data_parallel_rank(),
                local_rank=self.local_rank,
                shuffle_buffer_size=16384,
                shuffle_buffer_warmup_factor=16,
                vocab_file=self.cfg.tokenizer.vocab_file,
                data_loader_kwargs={
                    'batch_size': global_batch_size_on_this_data_parallel_rank,
                    'num_workers': self.cfg.data.num_workers,
                    'prefetch_factor': 2,
                },
                mlm_probability=0.15,
                base_seed=self.cfg.seed,
                log_level=logging.CRITICAL,
                log_dir="/tmp/log",
                return_raw_samples=False,
                start_epoch=0,
                sequence_length_alignment=8,
                ignore_index=-1,
                micro_batch_size=self.cfg.micro_batch_size,
            )
        if len(self.cfg.data.data_prefix) > 2:
            test_lddl_data_path = self.cfg.data.data_prefix[2]
            self._test_dl = get_bert_pretrain_data_loader(
                test_lddl_data_path,
                dp_rank=parallel_state.get_data_parallel_rank(),
                local_rank=self.local_rank,
                shuffle_buffer_size=16384,
                shuffle_buffer_warmup_factor=16,
                vocab_file=self.cfg.tokenizer.vocab_file,
                data_loader_kwargs={
                    'batch_size': global_batch_size_on_this_data_parallel_rank,
                    'num_workers': self.cfg.data.num_workers,
                    'prefetch_factor': 2,
                },
                mlm_probability=0.15,
                base_seed=self.cfg.seed,
                log_level=logging.CRITICAL,
                log_dir="/tmp/log",
                return_raw_samples=False,
                start_epoch=0,
                sequence_length_alignment=8,
                ignore_index=-1,
                micro_batch_size=self.cfg.micro_batch_size,
            )
        logging.info(f'Finished building LDDL Dataloaders')

    def build_train_valid_test_datasets(self):
        logging.info('Building Bert datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.cfg.global_batch_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = dataset_utils.build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self.cfg.data.seq_length,
            masked_lm_prob=self.cfg.data.masked_lm_prob,
            short_seq_prob=self.cfg.data.short_seq_prob,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
            binary_head=self.cfg.bert_binary_head,
            max_seq_length_dec=None,
            dataset_type='standard_bert',
            tokenizer=self.tokenizer.tokenizer,
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')

        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
        self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_dl, 'val')

        return self._train_ds, self._validation_ds, self._test_ds

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def setup(self, stage=None):
        """
        PTL hook that is executed after DDP spawns.
        We setup datasets here as megatron datasets require DDP to instantiate.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.

        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert()

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

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            if self.cfg.data.dataloader_type == "LDDL":
                self.build_LDDL_data(self.cfg.data)
                torch.distributed.barrier()
            else:
                self.build_train_valid_test_datasets()
                self.setup_training_data(self.cfg.data)
                self.setup_validation_data(self.cfg.data)
                self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            for index, module in enumerate(self.get_model_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    sync_embeddings = (
                        module.setup_embeddings_and_output_layer
                        if self.mcore_bert
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_bert', False):
            self.setup_transformer_engine_tp_groups()

    def setup_transformer_engine_tp_groups(self):
        """ This should be called after model parallel groups have been initialized
            and only needs to be called when using Transformer Engine.
        """
        for module in self.get_bert_module_list():
            """Set TP group
               Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py#L398
            """
            # Deep iterate but skip self to avoid infinite recursion.
            for index, child in enumerate(module.modules()):
                if index == 0:
                    continue
                if hasattr(child, "set_tensor_parallel_group"):
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    child.set_tensor_parallel_group(tp_group)

    def get_bert_module_list(self):
        if isinstance(self.model, list):
            return [
                model.module if isinstance(model, (Float16Module, MCoreFloat16Module)) else model
                for model in self.model
            ]
        elif isinstance(self.model, (Float16Module, MCoreFloat16Module)):
            return [self.model.module]
        else:
            return [self.model]

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

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"megatron_bert_345m_{vocab}",
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_bert_345m_{vocab}/versions/1/files/megatron_bert_345m_{vocab}.nemo",
                    description=f"345M parameter BERT Megatron model with {vocab} vocab.",
                )
            )
        for vocab_size in ['50k', '30k']:
            for vocab in ['cased', 'uncased']:
                result.append(
                    PretrainedModelInfo(
                        pretrained_model_name=f"biomegatron345m_biovocab_{vocab_size}_{vocab}",
                        location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m_biovocab_{vocab_size}_{vocab}/versions/1/files/BioMegatron345m-biovocab-{vocab_size}-{vocab}.nemo",
                        description="Megatron 345m parameters model with biomedical vocabulary ({vocab_size} size) {vocab}, pre-trained on PubMed biomedical text corpus.",
                    )
                )
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"biomegatron-bert-345m-{vocab}",
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m{vocab}/versions/1/files/BioMegatron345m{vocab.capitalize()}.nemo",
                    description=f"Megatron pretrained on {vocab} biomedical dataset PubMed with 345 million parameters.",
                )
            )
        return result

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                modules = self.get_model_module_list()
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    module = modules[0]  # only the first virtual rank has the embeddings
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        param = (
                            module.shared_embedding_or_output_weight()
                            if self.mcore_bert
                            else module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_O2
                        param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if len(modules) > 1:
                        module = modules[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = modules[0]
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        param = (
                            module.shared_embedding_or_output_weight()
                            if self.mcore_bert
                            else module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_O2
                        param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

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
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    stage_bucket = []
                    layers = module.encoder.layers if self.mcore_bert else module.language_model.encoder.layers
                    for layer in layers:
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    layers = module.encoder.layers if self.mcore_bert else module.language_model.encoder.layers
                    for layer in layers:
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

    # Required for ONNX export
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    # Required for ONNX export
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=2048, size=sz, device=sample.device)
        token_type_ids = torch.randint(low=0, high=1, size=sz, device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        return tuple([input_dict])

    def sharded_state_dict(self, prefix: str = '') -> Dict[str, Any]:
        """
        Creates the sharded state dict which is used by dist_checkpoint to save the sharded tensors to disk.
        When given the sharded_stated_dict, dist_checkpoint.load will load the tensors corresponding to
        self.state_dict().
        The sharded tensor mapping is defined in the GPTModel class from mcore.
        """
        if self.mcore_bert:
            module_prefix = f'{prefix}model.'
            sharded_state_dict = {}
            for index, module in enumerate(self.get_model_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    # virtual pipline rank must be set so that GPTModel returns the correct sharded state dict
                    parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    module_sharded_state_dict = module.sharded_state_dict(prefix=module_prefix)
                    sharded_state_dict[f'model_{index}'] = module_sharded_state_dict
                else:
                    module_sharded_state_dict = module.sharded_state_dict(prefix=module_prefix)
                    sharded_state_dict.update(module_sharded_state_dict)

            # reset vp rank
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

            return sharded_state_dict

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """
        if self.mcore_bert:
            checkpoint['sharded_state_dict'] = self.sharded_state_dict()
        else:
            if isinstance(self.model, list):
                for i in range(len(self.model)):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if self.mcore_bert:
            if 'state_dict' in checkpoint and checkpoint['state_dict']:
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        checkpoint_state_dict = checkpoint['state_dict'][f'model_{index}']
                    else:
                        checkpoint_state_dict = checkpoint['state_dict']
                    # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                    checkpoint_state_dict = {
                        key.replace('model.', ''): checkpoint_state_dict.pop(key)
                        for key in list(checkpoint_state_dict.keys())
                    }
                    module.load_state_dict(checkpoint_state_dict, strict=True)
            else:
                checkpoint['state_dict'] = {}
        else:
            if isinstance(self.model, list):
                for i in range(len(self.model)):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def build_transformer_config(self) -> TransformerConfig:
        """ Builds the megatron core gpt transformer config for the model.
            For attributes in the nemo model config that are the same
            as the megatron core TransformerConfig, we will use the value from the nemo model config.
            For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """
        activation = self.cfg.get('activation', 'gelu')
        assert activation == 'gelu', "Only gelu activation is support for BERT at the moment."

        normalization = self.cfg.get('normalization', 'layernorm')

        layernorm_zero_centered_gamma = self.cfg.get('normalization', 'layernorm') == 'layernorm1p'
        if normalization == 'layernorm':
            normalization = 'LayerNorm'
        elif normalization == 'rmsnorm':
            normalization = 'RMSNorm'
        elif normalization == 'layernorm1p':
            normalization = 'LayerNorm'
            layernorm_zero_centered_gamma = True
        else:
            logging.warning(
                f"The normalization type: {normalization} might not be supported in megatron core."
                f"Supported types are LayerNorm and RMSNorm."
            )

        # any configs that are not in the nemo model config will be added here
        model_specific_configs = {
            'layernorm_zero_centered_gamma': layernorm_zero_centered_gamma,
            'normalization': normalization,
        }

        transformer_config = super().build_transformer_config()

        for key, value in model_specific_configs.items():
            setattr(transformer_config, key, value)

        # pass mcore customization configs directly to mcore
        mcore_customization_config_dict = self.cfg.get('mcore_customization_config', {})
        for key, value in mcore_customization_config_dict.items():
            setattr(transformer_config, key, value)

        return transformer_config


class MegatronBertTextEmbeddingModel(MegatronBertModel):
    """
    Megatron Bert Text Embedding.
    Model returns [batch, hidden] shape
    """

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
        model=None,
    ):
        outputs = super().forward(
            input_ids, attention_mask, token_type_ids, lm_labels, checkpoint_activations_all_layers, model
        )
        embeddings = self.average_pool(outputs[0], attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
